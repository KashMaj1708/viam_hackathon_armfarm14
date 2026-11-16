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

import agent_review
import run_metrics

INFER_MS = []                # anomaly-model call times for this run (filled by inspect)

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
# Small parts (screws lying flat) need the fingertips right down on the paper. RECORDED 14 Nov by
# teleop: closed on a small screw with the fingertip centre at world z -6.07 (gripper origin
# (353.0, -176.2, 60.3), tilted ~8 deg), i.e. lower than TABLE_Z suggests. Kashyap: use 0.5 mm above that.
SMALL_GRASP_TIP_Z = -6.07 + 0.2          # teleop height + 0.2 mm (was +0.5, came up ~0.3 mm short)
SMALL_PART_HEIGHT_MM = 10    # parts at most this tall (above TABLE_Z) use SMALL_GRASP_TIP_Z
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
REVIEW_LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agent_review_log.csv")
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

THRESHOLD = float(os.environ.get("ANOMALY_THRESHOLD", 0.42))             # anomaly score at/above this = defective
PREPROCESS_AT_INFERENCE = True  # match prepared_v3 training crops; set False to restore camera-only path
MAX_ITEMS = 10
GRIPPER = "gripper"
LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "track3_log.csv")


def _key(it):
    """Attempt-memory key: the first remembered spot within 40 mm (a failed grasp nudges parts),
    else a new spot."""
    for k in _SPOTS:
        if (k[0] - it["x"]) ** 2 + (k[1] - it["y"]) ** 2 < 40 ** 2:
            return k
    k = (round(it["x"], 1), round(it["y"], 1))
    _SPOTS.append(k)
    return k


_SPOTS = []


def now_iso():
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="milliseconds")


async def connect():
    opts = RobotClient.Options.with_api_key(
        api_key=os.environ["VIAM_API_KEY"], api_key_id=os.environ["VIAM_API_KEY_ID"])
    # a missed session heartbeat (Wi-Fi blip, busy laptop) kills the whole run with SESSION_EXPIRED
    opts.disable_sessions = True
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


# ---- 2D fallback for small screws ---------------------------------------
# The RealSense often returns a small shiny screw as a ~3 mm smear that the 3D crop/filters drop.
# From each scan pose we also segment the colour frame (metal vs white paper), cast each blob's
# pixel ray onto the table plane and add blobs the 3D pass missed.
WS_X = (125, 580)            # same limits as the cam-workspace crop (world mm)
WS_Y = (-215, -97)
BLOB_MIN_PX, BLOB_MAX_PX = 40, 6000
SMALL_EST_TOP_MM = 6         # assumed top height of a 2D-only part above TABLE_Z
# The colour image lands ~11 mm off in world y from the depth data (measured 14 Nov on a small hex
# bolt at x 350: colour (346.4, -123.9) vs depth (346.8, -135.4); a pick at the depth-side position
# succeeded, verified on the overview camera). Applied to every colour-derived position.
COLOR_OFFSET_XY = (0.4, -11.5)
USE_OVERVIEW = False          # overview-cam recalibrated 14 Nov 18:40 (frames registered to overview_ref.jpg)


async def _cam(machine):
    from viam.components.camera import Camera
    try:
        return Camera.from_robot(machine, "cam")
    except Exception:
        from viam.components.camera.client import CameraClient
        return CameraClient("cam", machine._channel)


async def _pixel_to_table(machine, u, v, K):
    fx, fy, cx, cy = K
    pts = []
    for zc in (200.0, 600.0):
        p = await machine.transform_pose(PoseInFrame(reference_frame="cam", pose=Pose(
            x=(u - cx) / fx * zc, y=(v - cy) / fy * zc, z=zc, o_z=1)), "world")
        pts.append((p.pose.x, p.pose.y, p.pose.z))
    (x1, y1, z1), (x2, y2, z2) = pts
    zt = TABLE_Z + 3
    t = (zt - z1) / (z2 - z1)
    return x1 + t * (x2 - x1), y1 + t * (y2 - y1)


async def detect_2d(machine):
    """Small-part blobs in the current colour frame, as items in world mm (src='2d')."""
    import cv2
    import numpy as np
    cam = await _cam(machine)
    props = await cam.get_properties()
    ip = props.intrinsic_parameters
    K = (ip.focal_x_px, ip.focal_y_px, ip.center_x_px, ip.center_y_px)
    await asyncio.sleep(SHOT_SETTLE_S)
    imgs, _ = await cam.get_images()
    img = next((i for i in imgs if "jpeg" in str(i.mime_type) or "png" in str(i.mime_type)), imgs[0])
    bgr = cv2.imdecode(np.frombuffer(img.data, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        return []
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.int16)
    L, A, B = lab[..., 0], lab[..., 1], lab[..., 2]
    paper = L > np.percentile(L, 60)
    pl, pa, pb = np.median(L[paper]), np.median(A[paper]), np.median(B[paper])
    fg = (((pl - L) > 28) | (np.abs(B - pb) > 14) | (np.abs(A - pa) > 14)) & (L > 70)
    fg = fg.astype(np.uint8) * 255
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, lbl, stats, cents = cv2.connectedComponentsWithStats(fg)
    dark = cv2.dilate((L < 60).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0   # tape + its grey edge
    out = []
    for k in range(1, n):
        area = stats[k, cv2.CC_STAT_AREA]
        if not (BLOB_MIN_PX <= area <= BLOB_MAX_PX):
            continue
        ys, xs = np.nonzero(lbl == k)
        if dark[ys, xs].mean() > 0.3:                   # tape edge, not a part
            continue
        pts = np.stack([xs, ys], 1).astype(np.float64)
        m = pts.mean(0)
        evals, evecs = np.linalg.eigh(np.cov((pts - m).T))
        ax = evecs[:, 1]
        proj = (pts - m) @ ax
        e1, e2 = m + ax * proj.min(), m + ax * proj.max()
        wx, wy = await _pixel_to_table(machine, m[0], m[1], K)
        wx, wy = wx + COLOR_OFFSET_XY[0], wy + COLOR_OFFSET_XY[1]
        if not (WS_X[0] <= wx <= WS_X[1] and WS_Y[0] <= wy <= WS_Y[1]):
            continue
        ax1, ay1 = await _pixel_to_table(machine, *e1, K)
        ax2, ay2 = await _pixel_to_table(machine, *e2, K)
        length = math.hypot(ax2 - ax1, ay2 - ay1)
        if length < MIN_SIZE_MM or length > MAX_SIZE_MM:  # thin long screws are often missed by depth too
            continue
        width_mm = length * math.sqrt(max(evals[0], 1e-6) / max(evals[1], 1e-6))
        fill = area / max(1.0, 4 * math.sqrt(evals[0] * evals[1]) * 3.0)
        highlight = L[ys, xs].max() >= pl - 8                 # specular glint on metal
        tinted = np.abs(B[ys, xs] - pb).mean() > 10           # blue zinc plating
        if width_mm < 3 or fill < 0.35 or not (highlight or tinted):
            continue                                          # pencil mark / scuff / shadow
        out.append({"x": wx, "y": wy, "top": TABLE_Z + SMALL_EST_TOP_MM, "height": SMALL_EST_TOP_MM,
                    "zspan": 0.0, "bright": 0.0, "length": length,
                    "width": max(3.0, length * math.sqrt(max(evals[0], 1e-6) / max(evals[1], 1e-6))),
                    "long_deg": math.degrees(math.atan2(ay2 - ay1, ax2 - ax1)),
                    "points": 0, "box": (0, 0), "src": "2d"})
    return out


async def center_on(machine, motion, finder, it, passes=2):
    """Re-detect the item from the current (close) camera pose and shift the camera so the
    item sits on the optical axis. Errors in the camera calibration grow with distance, so
    the position measured from home lands the item off-centre; measuring again up close
    and correcting in the camera's own frame removes that."""
    for _ in range(passes):
        if it.get("src") in ("2d", "ov"):
            parts = await detect_2d(machine)
        else:
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


OV_MATCH_MM = 30             # an item within this of an overview blob's footprint belongs to it


def _dedupe_with_overview(items, ov):
    """The overview camera sees each part as one blob. Wrist detections that land on the same
    blob (a long screw split into head and tail clusters, or a 2D hit on its end) are one part:
    keep the best-seen copy. 2D hits on no blob at all are ghosts (pencil marks, glare)."""
    if not ov:
        return items
    def blob_of(it):
        best, bd = None, OV_MATCH_MM ** 2
        for bi, p in enumerate(ov):
            d = min((fx - it["x"]) ** 2 + (fy - it["y"]) ** 2 for fx, fy in p.get("footprint") or [(p["x"], p["y"])])
            if d < bd:
                best, bd = bi, d
        return best
    rank = {"depth": 2, "2d": 1, "ov": 0}
    keep, by_blob, dropped = [], {}, 0
    for it in items:
        b = blob_of(it)
        src = it.get("src", "depth")
        if b is None:
            if src == "2d":
                dropped += 1                      # nothing there in the overview: ghost
                continue
            keep.append(it)
            continue
        cur = by_blob.get(b)
        score = (rank.get(src, 2), it.get("points", 0), it.get("length", 0))
        if cur is None or score > cur[0]:
            if cur is not None:
                dropped += 1
            by_blob[b] = (score, it)
        else:
            dropped += 1
    keep += [v[1] for v in by_blob.values()]
    if dropped:
        print(f"Overview de-dup: dropped {dropped} duplicate/ghost detection(s); {len(keep)} part(s) left")
    return keep


_SCREENED = {}               # spot key -> agent verdict on "is this a fastener"


def _decode_depth(data):
    """Viam raw depth (image/vnd.viam.dep): 'DEPTHMAP' + uint64 width + uint64 height (big endian),
    then uint16 mm per pixel, big endian. Returns a float32 array in mm (0 = no reading) or None."""
    import numpy as np
    if data[:8] != b"DEPTHMAP":
        return None
    w = int.from_bytes(data[8:16], "big")
    h = int.from_bytes(data[16:24], "big")
    arr = np.frombuffer(data[24:24 + 2 * w * h], dtype=">u2").reshape(h, w)
    return arr.astype(np.float32)


def _height_map(dcrop):
    """Height above the local paper (mm) as a colour image: black = no depth reading,
    dark blue = flat paper, yellow/red = 10+ mm tall."""
    import cv2
    import numpy as np
    valid = dcrop > 0
    out = np.zeros(dcrop.shape + (3,), np.uint8)
    if valid.sum() < 20:
        return out
    paper = np.percentile(dcrop[valid], 80)          # most of the crop is paper (the farthest surface)
    hgt = np.clip(paper - dcrop, 0, 15)
    col = cv2.applyColorMap((hgt / 15 * 255).astype(np.uint8), cv2.COLORMAP_JET)
    out[valid] = col[valid]
    return out


async def _crop_candidates(machine, items):
    """Crop each not-yet-cropped candidate out of the current wrist colour frame (for the agent screen)."""
    import cv2
    import numpy as np
    try:
        cam = await _cam(machine)
        ip = (await cam.get_properties()).intrinsic_parameters
        fx, fy, cx, cy = ip.focal_x_px, ip.focal_y_px, ip.center_x_px, ip.center_y_px
        imgs, _ = await steady_frame(cam)
        img = next((i for i in imgs if "jpeg" in str(i.mime_type) or "png" in str(i.mime_type)), imgs[0])
        bgr = cv2.imdecode(np.frombuffer(img.data, np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            return
        h, w = bgr.shape[:2]
        depth = None                              # aligned depth (mm), for a height map beside each crop
        try:
            dimg = next((i for i in imgs if "dep" in str(i.mime_type)), None)
            if dimg is not None:
                depth = _decode_depth(dimg.data)
                if depth is not None and depth.shape != (h, w):
                    depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)
        except Exception as e:
            print(f"  depth for screen skipped: {e}")
        for it in items:
            if it.get("_crop") is not None or _SCREENED.get(_key(it)):
                continue
            p = await machine.transform_pose(PoseInFrame(reference_frame="world", pose=Pose(
                x=it["x"], y=it["y"], z=it.get("top", TABLE_Z + 5), o_z=1)), "cam")
            X, Y, Z = p.pose.x, p.pose.y, p.pose.z
            if Z <= 50:
                continue
            u, v = fx * X / Z + cx, fy * Y / Z + cy
            half = int(max(45, (max(it.get("length", 20), 30) * 0.75) * fx / Z))   # part plus some paper
            if half < u < w - half and half < v < h - half:
                it["_crop"] = bgr[int(v) - half:int(v) + half, int(u) - half:int(u) + half].copy()
                it["_dcrop"] = _height_map(depth[int(v) - half:int(v) + half, int(u) - half:int(u) + half]) \
                    if depth is not None else None
    except Exception as e:
        print(f"  candidate crops skipped: {e}")


async def _screen_candidates(items):
    """Ask the agent which candidates are real fasteners; drop confident non-fasteners.
    Verdicts are remembered per spot, so later re-detections cost nothing."""
    if not agent_review.screen_enabled():
        return items
    todo = [it for it in items if _key(it) not in _SCREENED and it.get("_crop") is not None]
    if todo:
        out = await agent_review.screen([it["_crop"] for it in todo],
                                        [it.get("_dcrop") for it in todo],
                                        [f"src={it.get('src', 'depth')}, height {max(0.0, it.get('height', 0)):.0f} mm, "
                                         f"{it.get('points', 0)} depth pts" for it in todo])
        for i, it in enumerate(todo):
            if i in out:
                _SCREENED[_key(it)] = out[i]
    keep = []
    for it in items:
        v = _SCREENED.get(_key(it))
        if v and not v.get("fastener", True) and float(v.get("confidence", 0)) >= agent_review.SCREEN_DROP_CONF:
            print(f"  agent: ({it['x']:.0f},{it['y']:.0f}) is not a fastener ({v.get('what', '')}), skipping")
            continue
        keep.append(it)
    return keep


async def detect(finder, motion=None, machine=None):
    """Find items in the neutral strip. With `motion`, scan from the closer SCAN_POSES and
    merge; without it, detect from wherever the camera is."""
    poses = SCAN_POSES if motion else [None]
    items = []
    ov_img = None
    if machine is not None and USE_OVERVIEW:   # overview frame first, while the arm is still out of its view
        try:
            import overview
            ov_img = await overview.grab(machine)
        except Exception as e:
            print(f"Overview grab failed: {e}")
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
        if machine is not None:                         # small screws the depth pass missed
            for d in await detect_2d(machine):
                # skip if depth already found this part (or a piece of it) anywhere along its length
                reach = max(25.0, d["length"] / 2 + 10)
                if all((e["x"] - d["x"]) ** 2 + (e["y"] - d["y"]) ** 2 >= reach ** 2 for e in items) \
                        and all((e["x"] - d["x"]) ** 2 + (e["y"] - d["y"]) ** 2 >= SAME_ITEM_MM ** 2
                                for e in items if e.get("src") == "2d"):
                    items.append(d)
            if sp and agent_review.screen_enabled():
                await _crop_candidates(machine, items)
    if ov_img is not None:
        try:
            import overview
            ov = overview.find_parts(ov_img)
            added = 0
            def _near_blob(e, p):        # tall parts project away from their base: compare to the whole blob
                fp = p.get("footprint") or [(p["x"], p["y"])]
                return min((fx - e["x"]) ** 2 + (fy - e["y"]) ** 2 for fx, fy in fp) < OV_MATCH_MM ** 2 \
                    or (e["x"] - p["x"]) ** 2 + (e["y"] - p["y"]) ** 2 < 45 ** 2
            for p in ov:
                if not any(_near_blob(e, p) for e in items):
                    items.append({"x": p["x"], "y": p["y"], "top": TABLE_Z + SMALL_EST_TOP_MM,
                                  "height": SMALL_EST_TOP_MM, "zspan": 0.0, "bright": 0.0, "length": 20.0,
                                  "width": 8.0, "long_deg": 0.0, "points": 0, "box": (0, 0), "src": "ov"})
                    added += 1
            print(f"Overview camera sees {len(ov)} part(s); wrist scan found {len(items) - added}"
                  + (f"; added {added} it missed" if added else ""))
            items = _dedupe_with_overview(items, ov)
            try:                          # what the overview camera saw, for checking runs afterwards
                import cv2
                cv2.imwrite(os.path.join(os.path.dirname(os.path.abspath(__file__)), "overview_last.jpg"),
                            overview.annotate(ov_img, ov, [(f"{i}", (e["x"], e["y"])) for i, e in enumerate(items)]))
            except Exception as e:
                print(f"Could not save overview_last.jpg: {e}")
        except Exception as e:
            print(f"Overview check skipped: {e}")
    items = await _screen_candidates(items)
    # far end of the strip first, so the arm works toward the drop zones
    return sorted(items, key=lambda d: -d["x"])



MAX_INSPECT_Z = 520          # world z cap for the camera when backing off to fit a long part
REFRAME_TRIES = 2


async def frame_part(machine, motion, cam, it, cam_z):
    """If the centred part runs off the cam-inspect frame (depth often sees only a piece of a
    thin screw), shift toward the visible part and raise the camera, up to REFRAME_TRIES times.
    Returns (cam_z, times_reframed)."""
    import cv2
    import numpy as np
    from prep.preprocess import preprocess
    n = 0
    for _ in range(REFRAME_TRIES):
        imgs, _ = await steady_frame(cam)
        bgr = cv2.imdecode(np.frombuffer(imgs[0].data, np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            break
        h, w = bgr.shape[:2]
        _, dbg = preprocess(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), out_size=256)
        if not dbg.get("touches_border") or not dbg.get("bbox"):
            break
        new_z = min(cam_z + 0.5 * (cam_z - it["top"]), MAX_INSPECT_Z)
        if new_z <= cam_z + 1:
            break
        bx, by, bw, bh = dbg["bbox"]
        dist = cam_z - it["top"]
        dx = (bx + bw / 2 - w / 2) * dist / 611.2       # wrist cam focal length, px
        dy = (by + bh / 2 - h / 2) * dist / 610.8
        c = await machine.transform_pose(PoseInFrame(reference_frame="cam", pose=Pose(
            x=dx, y=dy, z=0, o_z=1)), "world")
        print(f"  part cut off at the frame edge: moving {math.hypot(dx, dy):.0f} mm, "
              f"camera z {cam_z:.0f} -> {new_z:.0f}")
        await motion.move("cam", pose(c.pose.x, c.pose.y, new_z, CAM_THETA))
        await asyncio.sleep(SETTLE_S)
        cam_z = new_z
        n += 1
    return cam_z, n


async def inspect(machine, motion, finder, anomaly, item):
    await motion.move("cam", pose(item["x"], item["y"], inspect_z(item), CAM_THETA))
    await asyncio.sleep(SETTLE_S)
    near = await center_on(machine, motion, finder, item)
    # Grasp from the close-range measurement, not the far scan (errors grow with distance).
    for k in ("x", "y", "top", "height"):
        if k in near:
            item[k] = near[k]
    try:                                        # orientation/length from the colour blob up close
        blobs = await detect_2d(machine)
        b = min(blobs, key=lambda d: (d["x"] - item["x"]) ** 2 + (d["y"] - item["y"]) ** 2) if blobs else None
        if b and (b["x"] - item["x"]) ** 2 + (b["y"] - item["y"]) ** 2 < 30 ** 2:
            item["long_deg"] = b["long_deg"]
            item["length"] = max(item.get("length", 0), b["length"])
            if item.get("src") in ("2d", "ov"):
                item["x"], item["y"] = b["x"], b["y"]
    except Exception as e:
        print(f"  colour refine skipped: {e}")
    print(f"  grasp target ({item['x']:.1f}, {item['y']:.1f}), long axis {item.get('long_deg', 0):.0f} deg")
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
    await frame_part(machine, motion, cam, item, inspect_z(item))
    item["_review_imgs"] = []
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
        if not item["_review_imgs"]:              # first shot goes to the agent review
            item["_review_imgs"] = [(raw, "image/jpeg"), (buf.tobytes(), "image/png")]
        image = ViamImage(buf.tobytes(), CameraMimeType.PNG)
        _t = time.monotonic()
        cls = await anomaly.get_classifications(image, 1)
        INFER_MS.append((time.monotonic() - _t) * 1000)
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
    items = await detect(finder, motion, machine)
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


async def _wait_jaws_closed(gripper, timeout_s=3.0):
    """grab() can return while the jaws are still travelling. Wait until they stop moving
    (or the reported position stops changing), then a short extra squeeze."""
    t0 = time.monotonic()
    await asyncio.sleep(0.4)
    last = None
    while time.monotonic() - t0 < timeout_s:
        try:
            if await gripper.is_moving():
                await asyncio.sleep(0.1)
                continue
            hs = await gripper.is_holding_something()
            pos = (hs.meta or {}).get("position")
            if pos is None or (last is not None and abs(pos - last) < 0.5):
                break
            last = pos
        except Exception:
            break
        await asyncio.sleep(0.15)
    await asyncio.sleep(0.3)


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
        anomaly = VisionClient.from_robot(machine, os.environ.get("ANOMALY_SVC", "anomaly"))
        motion = MotionClient.from_robot(machine, "builtin")
        # Measured with kashyap-fastener-closeup-v2 at the inspection height: good socket bolt
        # 0.26-0.27, tape-wrapped bolt 0.58-0.62. Cut-off halfway between (also set on the service).
        threshold = THRESHOLD

        if CAPTURE:
            await capture(machine, motion, finder, home, CAPTURE)
            return
        M = run_metrics.Run(dry=not EXECUTE)
        M.infer_ms = INFER_MS
        await M.probe_device(machine)

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
        attempts = {}                             # spot -> grasp attempts; gives up after 2
        n = 0                                     # item number in the log
        last_xy = None                            # position of an item we just missed
        for attempt in range(1, MAX_ITEMS * 2 + 1):
            t_cycle = time.monotonic()
            items = await detect(finder, motion if EXECUTE else None, machine)
            M.detect_s.append(time.monotonic() - t_cycle)
            items = [d for d in items if attempts.get(_key(d), 0) < 2]
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

            t_insp = time.monotonic()
            n_inf0 = len(INFER_MS)
            score = await inspect(machine, motion, finder, anomaly, it)
            rec = {"detect_s": round(t_insp - t_cycle, 1), "inspect_s": round(time.monotonic() - t_insp, 1),
                   "infer_ms_mean": round(sum(INFER_MS[n_inf0:]) / max(1, len(INFER_MS) - n_inf0)),
                   "x": round(it["x"]), "y": round(it["y"]), "score": round(score, 3), "src": it.get("src", "depth")}
            decision = "defective" if score >= threshold else "good"
            conf = score if decision == "defective" else 1 - score
            model_decision = decision
            review_task = agent_review.start(it.get("_review_imgs"), score, threshold, decision)
            review_done = False
            t_dec = now_iso()
            t_dec_mono = time.monotonic()

            theta = jaw_theta(it["long_deg"])
            tip_z = max(TABLE_Z + it["height"] * GRASP_FRACTION,
                        TABLE_Z + MIN_TIP_ABOVE_TABLE) - EXTRA_DEPTH_MM
            tip_z = max(tip_z, TABLE_Z + MIN_TIP_ABOVE_TABLE)     # EXTRA_DEPTH can't break the floor
            if it.get("src") in ("2d", "ov") or it["height"] <= SMALL_PART_HEIGHT_MM:
                tip_z = SMALL_GRASP_TIP_Z                       # teleop-recorded small-screw height
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
            rec.update({"number": n, "retry": retry, "model_decision": decision, "decision_to_motion_ms": lat_ms})
            print(f"{M.clock()} MODEL {n}: {decision} (score {score:.3f}, motion +{lat_ms} ms)"
                  + ("  [retry, not re-logged]" if retry else "")
                  + ("  [agent review running]" if review_task else ""))

            async def settle_and_log():
                """Wait for the agent review, fix the decision, write the log row once."""
                nonlocal decision, conf, review_done
                if review_done:
                    return
                review_done = True
                rv = await agent_review.result(review_task)
                decision, rconf, note = agent_review.final(model_decision, score, rv)
                if rconf is not None:
                    conf = rconf
                t_log, t_mot, lat_log = t_dec, t_motion, lat_ms
                if note == "agent_override":      # final decision made now; next motion is the move to the box
                    t_log = t_mot = now_iso()
                    lat_log = 0
                rec.update({"decision": decision, "review_note": note,
                            "review_s": round(rv["seconds"], 1) if rv else None,
                            "agent_verdict": rv["verdict"] if rv else None})
                if rv:
                    print(f"  AGENT: {rv['verdict']} ({rv['confidence']:.2f}, {rv['seconds']:.1f}s) "
                          f"\"{rv['reason']}\" -> {note}, final {decision}")
                    with open(REVIEW_LOG, "a", newline="") as rf:
                        csv.writer(rf).writerow([n, now_iso(), f"{score:.3f}", model_decision, rv["verdict"],
                                                 f"{rv['confidence']:.2f}", rv["reason"], f"{rv['seconds']:.1f}",
                                                 note, decision])
                if not retry:
                    log.writerow([n, decision, f"{conf:.3f}", t_log, t_mot, lat_log,
                                  f"{it['x']:.0f}", f"{it['y']:.0f}", f"{score:.3f}"])
                    logf.flush()
                    M.judge_row(n, decision, conf, t_log, t_mot, lat_log, score, note)
                print(f"{M.clock()} LOG {n}, {decision}, {conf:.3f}, {t_log}")
            await move
            try:
                if not USE_OVERVIEW:
                    raise RuntimeError("overview off")
                import overview, cv2
                snap = await overview.grab(machine)
                os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "grasp_checks"), exist_ok=True)
                cv2.imwrite(os.path.join(os.path.dirname(os.path.abspath(__file__)), "grasp_checks",
                                         f"{datetime.now():%H%M%S}_item{n}.jpg"),
                            overview.annotate(snap, [], [("target", (it["x"], it["y"]))]))
            except Exception as e:
                print(f"  overview snapshot skipped: {e}")

            await motion.move(GRIPPER, pose(it["x"], it["y"], grasp_z, theta))
            await gripper.grab()
            await _wait_jaws_closed(gripper)      # don't lift until the jaws have actually closed
            await motion.move(GRIPPER, pose(it["x"], it["y"], TRAVEL_Z, theta))
            hs = await gripper.is_holding_something()
            pos = (hs.meta or {}).get("position")
            holding = hs.is_holding_something and (pos is None or pos > 3)   # closed on nothing reads -2..1
            print(f"  grip check: holding={hs.is_holding_something} position={pos}")
            attempts[_key(it)] = attempts.get(_key(it), 0) + 1
            if not holding:
                await settle_and_log()
                rec.update({"outcome": "missed", "cycle_s": round(time.monotonic() - t_cycle, 1)})
                M.add(rec)
                print("  missed the grasp, re-detecting")
                last_xy = (it["x"], it["y"])
                await gripper.open()
                await home.set_position(2)
                continue
            last_xy = None
            await settle_and_log()                # agent review decides the box

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
            rec.update({"outcome": "sorted", "cycle_s": round(time.monotonic() - t_cycle, 1)})
            M.add(rec)
            print(f"{M.clock()} item {n} done in {rec['cycle_s']} s")

        logf.close()
        print(f"Log: {LOG_PATH}")
        await M.finish()


if __name__ == "__main__":
    asyncio.run(main())
