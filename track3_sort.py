"""
Track 3 "Spot it, sort it" for armfarm14.

Zones (taped, measured in world mm from the home view):
  NEUTRAL   long strip on the right of the wrist-cam image, items start here
            (this is the cam-workspace crop: x 125..555, y -230..-120)
  SAFE      box above the horizontal tape   -> good items
  ANOMALOUS box below the horizontal tape   -> defective items

Loop, one item at a time, re-detecting from home every time (recovery):
  1. home-pose, detect items in NEUTRAL with object-cluster (3D, any shape/colour)
  2. move the wrist camera straight above the item, run the `anomaly`
     vision service (PatchCore) on a few frames, decide good / defective
  3. log: item number, decision, confidence, timestamp (+ first arm motion time)
  4. pick it (jaws turned across the item's short axis) and drop it in its zone

Run with EXECUTE = False first: it detects, inspects nothing, moves nothing.
Locally:  python track3_sort.py [--execute]     Control tab: set EXECUTE below.
"""
import asyncio
import csv
import math
import os
import struct
import sys
import time
from datetime import datetime, timezone

from viam.components.gripper import Gripper
from viam.components.switch import Switch
from viam.proto.common import Pose, PoseInFrame
from viam.robot.client import RobotClient
from viam.services.motion import MotionClient
from viam.services.vision import VisionClient

EXECUTE = False
EXECUTE = EXECUTE or "--execute" in sys.argv
# Training-data capture: python track3_sort.py --capture good   (or: --capture bad)
# Visits every item in the neutral zone and saves inspection images to train_images/<label>/
CAPTURE = sys.argv[sys.argv.index("--capture") + 1] if "--capture" in sys.argv else None
# Hand-placed capture: python track3_sort.py --manual good   (or bad)
# The camera stays where it is (park it over the strip at inspection height first); you put a
# part under it, press Enter, it takes SHOTS_PER_PRESS images and uploads them.
MANUAL = sys.argv[sys.argv.index("--manual") + 1] if "--manual" in sys.argv else None
SHOTS_PER_PRESS = 3
SHOTS_PER_ITEM = 6           # small camera offsets per item for variety
PART_ID = "49d63d4f-4191-4d94-9e43-e379f846dd0e"      # armfarm14-main
DATASET_ID = "6af7997699f5ac3d88a89c33"               # kashyap-armfarm14-clean (audited)
TAG_FOR = {"good": "normal", "bad": "anomalous"}      # tags the anomalib trainer reads

# ---- zones and geometry (world mm) -------------------------------------
TABLE_Z = -1                 # table surface in the neutral strip (measured)
SAFE_XY = (485, 110)         # centre of the SAFE box
ANOM_XY = (205, 100)         # centre of the ANOMALOUS box
DROP_SPREAD = 35             # successive drops shift by this much so items don't stack
DROP_TIP_Z = 25              # fingertips this high above the table when releasing
TRAVEL_Z = 180               # gripper height for moves between zones

# Fingertips past the gripper frame origin. MEASURED 13 Nov by teleoping the closed gripper
# onto the table: gripper origin at z 65.2 with the tips touching (table z -1, gripper tilted
# 6.8 deg) -> 66.7 mm, matching UFactory's 172 mm flange-to-tip spec minus the 105 mm frame.
TIP_MM = 67
GRASP_FRACTION = 0.3      # close at this fraction of the item height (0.5 slipped on flat bolts)
MIN_TIP_ABOVE_TABLE = 2      # fingertips never lower than table + this (hard floor, now on a measured TIP_MM)
EXTRA_DEPTH_MM = 0           # + to go lower if the fingers close above the item
APPROACH_MM = 80
MAX_GRIP_MM = 80

# ---- detection filters ---------------------------------------------------
MIN_POINTS = 25              # drops specks of depth noise (small screws give ~30-60 pts)
MIN_SIZE_MM = 5
MIN_TOP_MM = 4.5             # item top must be this far above the table (table patches sit at ~4)
MIN_ZSPAN_MM = 0.8           # real parts have relief (a tiny screw: ~1 mm); table patches are flat (0)
MIN_BRIGHT = 50              # black tape edges come back at ~30; bare metal is 60-150
MAX_SIZE_MM = 170            # anything longer is paper/tape/background, not a part
# Detection runs from two closer scan poses over the strip instead of from home: at
# ~64 cm a small screw is a handful of noisy depth points; at ~38 cm it is several times
# more points with far less noise. Camera heights are above the table.
SCAN_POSES = [(450, -156, 380), (230, -156, 380)]   # (x, y, cam z); y = strip centre (re-measured)
SAME_ITEM_MM = 20            # detections closer than this from two scans are one item

# ---- inspection ----------------------------------------------------------
INSPECT_DIST_MM = 230        # camera height above the item top; training images use the same
# The fingertips hang ~147 mm below the camera lens (cam at flange +18, gripper TCP at
# +105, tips 60 past that). At 150 mm the tips grazed the parts, and the RealSense also
# loses depth that close. Never let the tips come closer than this above an item:
FINGERS_BELOW_CAM_MM = 147
MIN_TIP_CLEARANCE_MM = 60
INSPECT_CAM = "cam-inspect"  # centre 400x400 crop of the wrist cam (no tape, no fingers)
CAM_THETA = 84.35            # camera roll at home, keeps the same image orientation
N_FRAMES = 3                 # frames averaged per decision
SETTLE_S = 0.4              # pause after small moves before re-detecting
SHOT_SETTLE_S = 1.5         # pause before any photo/inspection: the arm rings for ~1 s after a move
                            # and the RealSense hands back a buffered frame, so wait, then grab and
                            # throw away one frame before the one we keep


async def steady_frame(cam):
    """Wait for the arm to stop ringing, flush the camera buffer, return a fresh frame."""
    await asyncio.sleep(SHOT_SETTLE_S)
    await cam.get_images()                  # stale / mid-motion frame, discarded
    await asyncio.sleep(0.15)
    return await cam.get_images()

THRESHOLD = 0.42             # anomaly score at/above this = defective
PREPROCESS_AT_INFERENCE = True  # match prepared_v3 training crops; set False to restore camera-only path
MAX_ITEMS = 10
GRIPPER = "gripper"
LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "track3_log.csv")


def now_iso():
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="milliseconds")


async def connect():
    opts = RobotClient.Options.with_api_key(
        api_key=os.environ["VIAM_API_KEY"], api_key_id=os.environ["VIAM_API_KEY_ID"])
    return await RobotClient.at_address(os.environ["VIAM_MACHINE_ADDRESS"], opts)


def pose(x, y, z, theta, o=(0, 0, -1)):
    return PoseInFrame(reference_frame="world",
                       pose=Pose(x=x, y=y, z=z, o_x=o[0], o_y=o[1], o_z=o[2], theta=theta))


def pcd_xy(pc_bytes):
    """Binary PCD (x y z rgb, metres) -> list of (x, y, z) in mm."""
    head, _, data = pc_bytes.partition(b"DATA binary\n")
    n = int([l for l in head.split(b"\n") if l.startswith(b"POINTS")][0].split()[1])
    pts = []
    for i in range(n):
        x, y, z, rgb = struct.unpack_from("<fffI", data, i * 16)
        bright = ((rgb >> 16 & 255) + (rgb >> 8 & 255) + (rgb & 255)) / 3
        pts.append((x * 1000, y * 1000, z * 1000, bright))
    return pts


def describe(obj):
    g = obj.geometries.geometries[0]
    pts = pcd_xy(obj.point_cloud)
    n = len(pts)
    mx = sum(p[0] for p in pts) / n
    my = sum(p[1] for p in pts) / n
    top = max(p[2] for p in pts)
    zspan = top - min(p[2] for p in pts)
    bright = sum(p[3] for p in pts) / n
    cxx = sum((p[0] - mx) ** 2 for p in pts) / n
    cyy = sum((p[1] - my) ** 2 for p in pts) / n
    cxy = sum((p[0] - mx) * (p[1] - my) for p in pts) / n
    long_ang = 0.5 * math.atan2(2 * cxy, cxx - cyy)          # long axis, radians
    ux, uy = math.cos(long_ang), math.sin(long_ang)
    along = [(p[0] - mx) * ux + (p[1] - my) * uy for p in pts]
    across = [-(p[0] - mx) * uy + (p[1] - my) * ux for p in pts]
    # centre of the item's extent, not the point mean: shiny metal returns depth unevenly
    # (bright head, sparse shank), which drags the mean toward one end
    ca = (max(along) + min(along)) / 2
    cb = (max(across) + min(across)) / 2
    cx, cy = mx + ca * ux - cb * uy, my + ca * uy + cb * ux
    return {"x": cx, "y": cy, "top": top, "height": top - TABLE_Z, "zspan": zspan, "bright": bright,
            "length": max(along) - min(along), "width": max(across) - min(across),
            "long_deg": math.degrees(long_ang), "points": n,
            "box": (g.box.dims_mm.x, g.box.dims_mm.y)}


def inspect_z(item):
    """Camera height for inspecting an item, never letting the fingertips get close.
    Long parts get a higher camera so the whole part fits in the 400 px cam-inspect crop
    with ~15 mm to spare each end (the crop spans ~0.65 x the camera height)."""
    fit = 1.53 * (item.get("length", 0) + 30)
    return item["top"] + max(INSPECT_DIST_MM, FINGERS_BELOW_CAM_MM + MIN_TIP_CLEARANCE_MM, fit)


def jaw_theta(long_deg):
    # At theta -180 the jaws close along world y; +theta turns them clockwise seen
    # from above. Jaws must close across the long axis, i.e. along long_deg + 90.
    a = ((long_deg + 90) % 180) - 90                         # (-90, 90]
    return -180 - a


def is_part(d):
    """True if a cluster looks like a fastener rather than noise, tape, table or paper."""
    return not (d["points"] < MIN_POINTS
                or max(d["length"], d["width"]) < MIN_SIZE_MM
                or max(d["length"], d["width"]) > MAX_SIZE_MM
                or d["height"] < MIN_TOP_MM or d["zspan"] < MIN_ZSPAN_MM
                or d["bright"] < MIN_BRIGHT)


async def center_on(machine, motion, finder, it, passes=2):
    """Re-detect the item from the current (close) camera pose and shift the camera so the
    item sits on the optical axis. Errors in the camera calibration grow with distance, so
    the position measured from home lands the item off-centre; measuring again up close
    and correcting in the camera's own frame removes that."""
    for _ in range(passes):
        objs = await finder.get_object_point_clouds("cam-workspace")
        if not objs:
            return it
        parts = [d for d in (describe(o) for o in objs) if is_part(d)]
        if not parts:
            return it
        best = min(parts, key=lambda d: (d["x"] - it["x"]) ** 2 + (d["y"] - it["y"]) ** 2)
        if (best["x"] - it["x"]) ** 2 + (best["y"] - it["y"]) ** 2 > 30 ** 2:
            return it                                   # lost it; keep the old estimate
        it = best
        in_cam = await machine.transform_pose(
            PoseInFrame(reference_frame="world",
                        pose=Pose(x=it["x"], y=it["y"], z=it["top"], o_z=1)), "cam")
        dx, dy = in_cam.pose.x, in_cam.pose.y
        if dx * dx + dy * dy < 4 ** 2:                  # within 4 mm of the axis: done
            break
        await motion.move("cam", PoseInFrame(reference_frame="cam",
                                             pose=Pose(x=dx, y=dy, z=0, o_z=1, theta=0)))
        await asyncio.sleep(SETTLE_S)
    return it


async def detect(finder, motion=None):
    """Find items in the neutral strip. With `motion`, scan from the closer SCAN_POSES and
    merge; without it, detect from wherever the camera is."""
    poses = SCAN_POSES if motion else [None]
    items = []
    for sp in poses:
        if sp:
            await motion.move("cam", pose(sp[0], sp[1], sp[2], CAM_THETA))
            await asyncio.sleep(SETTLE_S)
        for o in await finder.get_object_point_clouds("cam-workspace"):
            d = describe(o)
            if not is_part(d):
                continue
            dup = next((e for e in items if (e["x"] - d["x"]) ** 2 + (e["y"] - d["y"]) ** 2
                        < SAME_ITEM_MM ** 2), None)
            if dup is None:
                items.append(d)
            elif d["points"] > dup["points"]:          # keep the better-seen copy
                items[items.index(dup)] = d
    # far end of the strip first, so the arm works toward the drop zones
    return sorted(items, key=lambda d: -d["x"])


async def inspect(machine, motion, finder, anomaly, item):
    await motion.move("cam", pose(item["x"], item["y"], inspect_z(item), CAM_THETA))
    await asyncio.sleep(SETTLE_S)
    await center_on(machine, motion, finder, item)
    await asyncio.sleep(SHOT_SETTLE_S)          # let the arm stop ringing before scoring
    scores = []
    if not PREPROCESS_AT_INFERENCE:
        for _ in range(N_FRAMES):
            cls = await anomaly.get_classifications_from_camera(INSPECT_CAM, 1)
            c = cls[0]
            scores.append(c.confidence if c.class_name == "anomalous" else 1 - c.confidence)
        return sum(scores) / len(scores)

    # Same preprocess as prepared_v3 training: grab frames, clean/centre, classify image
    import cv2
    import numpy as np
    from viam.components.camera import Camera
    from viam.media.video import CameraMimeType, ViamImage
    from prep.preprocess import preprocess
    try:
        cam = Camera.from_robot(machine, INSPECT_CAM)
    except Exception:
        from viam.components.camera.client import CameraClient
        cam = CameraClient(INSPECT_CAM, machine._channel)
    for _ in range(N_FRAMES):
        imgs, _ = await steady_frame(cam)
        raw = imgs[0].data
        arr = np.frombuffer(raw, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            # already decoded RGB bytes unlikely; fall back to camera classifications
            cls = await anomaly.get_classifications_from_camera(INSPECT_CAM, 1)
            c = cls[0]
            scores.append(c.confidence if c.class_name == "anomalous" else 1 - c.confidence)
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        cleaned, _dbg = preprocess(rgb, out_size=256)
        ok, buf = cv2.imencode(".png", cv2.cvtColor(cleaned, cv2.COLOR_RGB2BGR))
        image = ViamImage(buf.tobytes(), CameraMimeType.PNG)
        cls = await anomaly.get_classifications(image, 1)
        c = cls[0]
        scores.append(c.confidence if c.class_name == "anomalous" else 1 - c.confidence)
    score = sum(scores) / len(scores)              # mean anomaly score, 0..1
    return score


async def capture(machine, motion, finder, home, label):
    from viam.components.camera import Camera
    try:
        cam = Camera.from_robot(machine, INSPECT_CAM)
    except Exception:
        # resource list on the client can lag a config change; talk to it directly
        from viam.components.camera.client import CameraClient
        cam = CameraClient(INSPECT_CAM, machine._channel)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_images", label)
    os.makedirs(out, exist_ok=True)
    await home.set_position(2)
    await asyncio.sleep(0.8)
    items = await detect(finder, motion)
    print(f"Capturing {label}: {len(items)} item(s) -> {out}")
    dc = None
    try:
        from viam.app.viam_client import ViamClient
        from viam.rpc.dial import DialOptions
        vc = await ViamClient.create_from_dial_options(DialOptions.with_api_key(
            os.environ["VIAM_API_KEY"], os.environ["VIAM_API_KEY_ID"]))
        dc = vc.data_client
    except Exception as e:
        print(f"  (no cloud upload, saving locally only: {e})")
    uploaded = 0
    base_offsets = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (0, 0)]
    stamp = datetime.now().strftime("%H%M%S")
    for i, it in enumerate(items):
        await motion.move("cam", pose(it["x"], it["y"], inspect_z(it), CAM_THETA))
        await asyncio.sleep(SETTLE_S)
        it = await center_on(machine, motion, finder, it)
        # nudge size: 8 mm for small parts, less for long ones so their ends stay in frame
        step = 8 if it.get("length", 0) < 80 else 4
        offsets = [(dx * step, dy * step) for dx, dy in base_offsets]
        print(f"  item {i + 1}: {it['length']:.0f} x {it['width']:.0f} mm, top {it['top']:.0f} mm, "
              f"camera at z {inspect_z(it):.0f}")
        for k in range(SHOTS_PER_ITEM):
            dx, dy = offsets[k % len(offsets)]
            if k:
                # small nudge in the camera frame from wherever the last shot was
                px, py = offsets[(k - 1) % len(offsets)]
                await motion.move("cam", PoseInFrame(reference_frame="cam", pose=Pose(
                    x=dx - px, y=dy - py, z=0, o_z=1, theta=0)))
                await asyncio.sleep(SETTLE_S)
            t_req = datetime.now(timezone.utc)
            imgs, _ = await steady_frame(cam)
            t_rec = datetime.now(timezone.utc)
            path = os.path.join(out, f"{label}_{stamp}_{i}_{k}.jpg")
            with open(path, "wb") as f:
                f.write(imgs[0].data)
            if dc:
                try:
                    await dc.binary_data_capture_upload(
                        binary_data=imgs[0].data, part_id=PART_ID,
                        component_type="rdk:component:camera", component_name=INSPECT_CAM,
                        method_name="GetImages", file_extension=".jpg",
                        tags=[TAG_FOR.get(label, label)], dataset_ids=[DATASET_ID],
                        data_request_times=(t_req, t_rec))
                    uploaded += 1
                    with open(os.path.join(os.path.dirname(out), "uploaded_clean.txt"), "a") as uf:
                        uf.write(os.path.basename(path) + "\n")
                except Exception as e:
                    print(f"  upload failed, kept locally: {e}")
                    dc = None
        print(f"  item {i + 1}: {SHOTS_PER_ITEM} images")
    await home.set_position(2)
    print(f"Done. Uploaded {uploaded} image(s) to kashyap-armfarm14-clean (tag "
          f"{TAG_FOR.get(label, label)}). Move/rotate the items and run again for variety.")


async def upload_saved():
    """python track3_sort.py --upload : push train_images/good|bad/*.jpg to the dataset."""
    from viam.app.viam_client import ViamClient
    from viam.rpc.dial import DialOptions
    vc = await ViamClient.create_from_dial_options(DialOptions.with_api_key(
        os.environ["VIAM_API_KEY"], os.environ["VIAM_API_KEY_ID"]))
    dc = vc.data_client
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_images")
    done_path = os.path.join(root, "uploaded_clean.txt")
    done = set(open(done_path).read().split()) if os.path.exists(done_path) else set()
    n = 0
    with open(done_path, "a") as done_f:
        for label in ("good", "bad"):
            folder = os.path.join(root, label)
            if not os.path.isdir(folder):
                continue
            for name in sorted(os.listdir(folder)):
                if not name.endswith(".jpg") or name in done:
                    continue
                path = os.path.join(folder, name)
                t = datetime.fromtimestamp(os.path.getmtime(path), timezone.utc)
                with open(path, "rb") as f:
                    data = f.read()
                await dc.binary_data_capture_upload(
                    binary_data=data, part_id=PART_ID,
                    component_type="rdk:component:camera", component_name=INSPECT_CAM,
                    method_name="GetImages", file_extension=".jpg",
                    tags=[TAG_FOR[label]], dataset_ids=[DATASET_ID], data_request_times=(t, t))
                done_f.write(name + "\n")
                n += 1
    vc.close()
    print(f"Uploaded {n} saved image(s) to kashyap-armfarm14-clean.")


async def manual_capture(machine, label):
    from viam.components.camera import Camera
    try:
        cam = Camera.from_robot(machine, INSPECT_CAM)
    except Exception:
        from viam.components.camera.client import CameraClient
        cam = CameraClient(INSPECT_CAM, machine._channel)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_images", label)
    os.makedirs(out, exist_ok=True)
    from viam.app.viam_client import ViamClient
    from viam.rpc.dial import DialOptions
    vc = await ViamClient.create_from_dial_options(DialOptions.with_api_key(
        os.environ["VIAM_API_KEY"], os.environ["VIAM_API_KEY_ID"]))
    dc = vc.data_client
    stamp = datetime.now().strftime("%H%M%S")
    n = 0
    print(f"Manual capture '{label}' (tag {TAG_FOR.get(label, label)}). Camera stays put.")
    print("Place ONE part in the middle of the view, hands out of frame, then press Enter.")
    print("Rotate / flip / move it a little between presses. Type q + Enter to stop.")
    loop = asyncio.get_event_loop()
    while True:
        ans = await loop.run_in_executor(None, input, f"[{n} saved] Enter = capture, q = quit: ")
        if ans.strip().lower() == "q":
            break
        for k in range(SHOTS_PER_PRESS):
            t_req = datetime.now(timezone.utc)
            imgs, _ = await steady_frame(cam) if k == 0 else await cam.get_images()
            t_rec = datetime.now(timezone.utc)
            data = imgs[0].data
            path = os.path.join(out, f"{label}_m{stamp}_{n:03d}.jpg")
            with open(path, "wb") as f:
                f.write(data)
            try:
                await dc.binary_data_capture_upload(
                    binary_data=data, part_id=PART_ID,
                    component_type="rdk:component:camera", component_name=INSPECT_CAM,
                    method_name="GetImages", file_extension=".jpg",
                    tags=[TAG_FOR.get(label, label)], dataset_ids=[DATASET_ID],
                    data_request_times=(t_req, t_rec))
                with open(os.path.join(os.path.dirname(out), "uploaded_clean.txt"), "a") as uf:
                    uf.write(os.path.basename(path) + "\n")
            except Exception as e:
                print(f"  upload failed, kept locally: {e}")
            n += 1
            await asyncio.sleep(0.4)
    vc.close()
    print(f"Saved and uploaded {n} image(s) to kashyap-armfarm14-clean.")


async def main():
    if MANUAL:
        async with await connect() as machine:
            await manual_capture(machine, MANUAL)
        return
    if "--upload" in sys.argv:
        await upload_saved()
        return
    async with await connect() as machine:
        gripper = Gripper.from_robot(machine, GRIPPER)
        home = Switch.from_robot(machine, "home-pose")
        finder = VisionClient.from_robot(machine, "object-cluster")
        anomaly = VisionClient.from_robot(machine, "anomaly")
        motion = MotionClient.from_robot(machine, "builtin")
        # Measured with kashyap-fastener-closeup-v2 at the inspection height: good socket bolt
        # 0.26-0.27, tape-wrapped bolt 0.58-0.62. Cut-off halfway between (also set on the service).
        threshold = THRESHOLD

        if CAPTURE:
            await capture(machine, motion, finder, home, CAPTURE)
            return

        new_log = not os.path.exists(LOG_PATH)
        logf = open(LOG_PATH, "a", newline="")
        log = csv.writer(logf)
        if new_log:
            log.writerow(["item", "decision", "confidence", "timestamp",
                          "first_motion", "latency_ms", "x", "y", "anomaly_score"])

        if EXECUTE:
            await gripper.open()
            await home.set_position(2)
            await asyncio.sleep(0.8)

        counts = {"good": 0, "defective": 0}
        n = 0                                     # item number in the log
        last_xy = None                            # position of an item we just missed
        for attempt in range(1, MAX_ITEMS * 2 + 1):
            items = await detect(finder, motion if EXECUTE else None)
            if not items:
                print("Neutral zone empty. Done.")
                break
            if not EXECUTE:
                print(f"Dry run: {len(items)} item(s) in the neutral zone")
                for d in items:
                    print(f"  ({d['x']:.0f}, {d['y']:.0f}) {d['length']:.0f}x{d['width']:.0f} mm, "
                          f"top {d['top']:.0f}, long axis {d['long_deg']:.0f} deg, "
                          f"theta {jaw_theta(d['long_deg']):.0f}, {d['points']} pts")
                print("Rerun with --execute (or EXECUTE = True) to sort.")
                break

            it = items[0]
            if it["width"] > MAX_GRIP_MM:
                print(f"Item at ({it['x']:.0f},{it['y']:.0f}) too wide, skipping run.")
                break

            score = await inspect(machine, motion, finder, anomaly, it)
            decision = "defective" if score >= threshold else "good"
            conf = score if decision == "defective" else 1 - score
            t_dec = now_iso()
            t_dec_mono = time.monotonic()

            theta = jaw_theta(it["long_deg"])
            tip_z = max(TABLE_Z + it["height"] * GRASP_FRACTION,
                        TABLE_Z + MIN_TIP_ABOVE_TABLE) - EXTRA_DEPTH_MM
            tip_z = max(tip_z, TABLE_Z + MIN_TIP_ABOVE_TABLE)     # EXTRA_DEPTH can't break the floor
            grasp_z = tip_z + TIP_MM

            # same item as a missed grasp (within 70 mm; a failed grasp can nudge it)? keep its number, don't log it twice
            retry = last_xy is not None and (it["x"] - last_xy[0]) ** 2 + (it["y"] - last_xy[1]) ** 2 < 70 ** 2
            if not retry:
                n += 1
                if n > MAX_ITEMS:
                    break

            # first arm motion after the decision: the approach toward the item
            t_motion = now_iso()
            lat_ms = int((time.monotonic() - t_dec_mono) * 1000)
            move = asyncio.ensure_future(
                motion.move(GRIPPER, pose(it["x"], it["y"], grasp_z + APPROACH_MM, theta)))
            if not retry:
                log.writerow([n, decision, f"{conf:.3f}", t_dec, t_motion, lat_ms,
                              f"{it['x']:.0f}", f"{it['y']:.0f}", f"{score:.3f}"])
                logf.flush()
            print(f"LOG {n}, {decision}, {conf:.3f}, {t_dec}  (score {score:.3f}, motion +{lat_ms} ms)"
                  + ("  [retry, not re-logged]" if retry else ""))
            await move

            await motion.move(GRIPPER, pose(it["x"], it["y"], grasp_z, theta))
            await gripper.grab()
            await asyncio.sleep(0.3)
            await motion.move(GRIPPER, pose(it["x"], it["y"], TRAVEL_Z, theta))
            if not await gripper.is_holding_something():
                print("  missed the grasp, re-detecting")
                last_xy = (it["x"], it["y"])
                await gripper.open()
                await home.set_position(2)
                continue
            last_xy = None

            zx, zy = SAFE_XY if decision == "good" else ANOM_XY
            k = counts[decision]
            dx, dy = [(0, 0), (DROP_SPREAD, 0), (-DROP_SPREAD, 0), (0, DROP_SPREAD),
                      (0, -DROP_SPREAD), (DROP_SPREAD, DROP_SPREAD)][k % 6]
            counts[decision] += 1
            await motion.move(GRIPPER, pose(zx + dx, zy + dy, TRAVEL_Z, -180))
            await motion.move(GRIPPER, pose(zx + dx, zy + dy, TABLE_Z + DROP_TIP_Z + TIP_MM, -180))
            await gripper.open()
            await asyncio.sleep(0.3)
            await motion.move(GRIPPER, pose(zx + dx, zy + dy, TRAVEL_Z, -180))
            await home.set_position(2)
            await asyncio.sleep(0.5)

        logf.close()
        print(f"Log: {LOG_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
