"""Inspection test run for armfarm14 (Track 3). NO picking: the gripper never moves.

For every part in the neutral strip the wrist camera:
  1. flies to the inspection height used by track3_sort.py and centres on the part,
  2. takes N_SHOTS steady frames from cam-inspect,
  3. scores each frame twice with the `anomaly` vision service:
       raw   = the camera frame as is            (matches models trained on raw frames, e.g. v2)
       clean = prep.preprocess() crop of it      (matches the prepared_v3 models)
  4. saves both images, the scores and the preprocess QC numbers.

Output: test_runs/<YYYYmmdd_HHMMSS>/
  item<k>_s<j>_raw.jpg, item<k>_s<j>_clean.png
  results.csv   one row per shot
  items.csv     one row per item (mean scores, decisions, QC flags)
  contact.jpg   one row per item: raw | clean, with scores

Usage (from the launcher or the venv):
  python test_inspect.py                 # visit every detected part
  python test_inspect.py --max 3         # first 3 parts only
  python test_inspect.py --plan          # detect and print the visit plan, then go home (no inspection)
"""
import asyncio
import csv
import os
import sys
import time
from datetime import datetime

import cv2
import numpy as np
from viam.components.camera import Camera
from viam.components.switch import Switch
from viam.media.video import CameraMimeType, ViamImage
from viam.proto.common import Pose, PoseInFrame
from viam.services.motion import MotionClient
from viam.services.vision import VisionClient

import track3_sort as t3
from prep.preprocess import preprocess

N_SHOTS = 3
MAX_ITEMS = int(sys.argv[sys.argv.index("--max") + 1]) if "--max" in sys.argv else t3.MAX_ITEMS
PLAN_ONLY = "--plan" in sys.argv
THRESHOLD = t3.THRESHOLD
HERE = os.path.dirname(os.path.abspath(__file__))

# QC limits for flagging a shot as a bad view (same spirit as prep/build_dataset.py)
QC_MAX_CENTRE_ERR_MM = 6
QC_MIN_FG_FRAC = 0.002


def anomaly_score(cls):
    c = cls[0]
    return c.confidence if c.class_name == "anomalous" else 1 - c.confidence


def decide(score):
    return "defective" if score >= THRESHOLD else "good"


async def get_cam(machine):
    try:
        return Camera.from_robot(machine, t3.INSPECT_CAM)
    except Exception:
        from viam.components.camera.client import CameraClient
        return CameraClient(t3.INSPECT_CAM, machine._channel)


async def centre_error(machine, it):
    """Distance (mm) of the part's top centre from the camera axis, after centring."""
    p = await machine.transform_pose(
        PoseInFrame(reference_frame="world", pose=Pose(x=it["x"], y=it["y"], z=it["top"], o_z=1)), "cam")
    return float(np.hypot(p.pose.x, p.pose.y))


def label_tile(img_bgr, lines, size=220):
    tile = cv2.resize(img_bgr, (size, size), interpolation=cv2.INTER_AREA)
    for i, txt in enumerate(lines):
        y = 16 + i * 16
        cv2.putText(tile, txt, (4, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(tile, txt, (4, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
    return tile


async def main():
    out = os.path.join(HERE, "test_runs", datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out, exist_ok=True)
    async with await t3.connect() as machine:
        home = Switch.from_robot(machine, "home-pose")
        finder = VisionClient.from_robot(machine, "object-cluster")
        anomaly = VisionClient.from_robot(machine, os.environ.get("ANOMALY_SVC", "anomaly"))
        motion = MotionClient.from_robot(machine, "builtin")
        cam = await get_cam(machine)

        await home.set_position(2)
        await asyncio.sleep(0.8)
        items = await t3.detect(finder, motion, machine)
        items = items[:MAX_ITEMS]
        print(f"\n{len(items)} part(s) found. Visit order (far end first):")
        for k, it in enumerate(items):
            print(f"  item{k} [{it.get('src', '3d')}]: x={it['x']:.0f} y={it['y']:.0f} len={it['length']:.0f} "
                  f"w={it['width']:.0f} top={it['top']:.1f} cam_z={t3.inspect_z(it):.0f}")
        if PLAN_ONLY or not items:
            await home.set_position(2)
            print("Plan only, back home." if PLAN_ONLY else "Neutral zone empty.")
            return

        shot_rows, item_rows, sheet = [], [], []
        try:
          for k, it0 in enumerate(items):
              t0 = time.time()
              print(f"\n[item{k}] flying to x={it0['x']:.0f} y={it0['y']:.0f} cam_z={t3.inspect_z(it0):.0f}")
              await motion.move("cam", t3.pose(it0["x"], it0["y"], t3.inspect_z(it0), t3.CAM_THETA))
              await asyncio.sleep(t3.SETTLE_S)
              it = await t3.center_on(machine, motion, finder, it0)
              err = await centre_error(machine, it)
              cam_z, n_reframe = await t3.frame_part(machine, motion, cam, it, t3.inspect_z(it))
              raw_scores, clean_scores, flags = [], [], set()
              if n_reframe:
                  flags.add(f"reframed{n_reframe}_z{cam_z:.0f}")
              if err > QC_MAX_CENTRE_ERR_MM:
                  flags.add(f"offcentre_{err:.0f}mm")
              first_raw = first_clean = None
              for j in range(N_SHOTS):
                  imgs, _ = await t3.steady_frame(cam)
                  raw = imgs[0].data
                  bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
                  if bgr is None:
                      print(f"  s{j}: could not decode frame ({imgs[0].mime_type}), skipped")
                      flags.add("decode_fail")
                      continue
                  cv2.imwrite(os.path.join(out, f"item{k}_s{j}_raw.jpg"), bgr)
                  try:
                      s_raw = anomaly_score(await anomaly.get_classifications(ViamImage(raw, CameraMimeType.JPEG), 1))
                  except Exception as e:
                      print(f"  s{j}: image scoring failed ({e}); using the camera path for raw")
                      s_raw = anomaly_score(await anomaly.get_classifications_from_camera(t3.INSPECT_CAM, 1))

                  rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                  clean, dbg = preprocess(rgb, out_size=256)
                  clean_bgr = cv2.cvtColor(clean, cv2.COLOR_RGB2BGR)
                  cv2.imwrite(os.path.join(out, f"item{k}_s{j}_clean.png"), clean_bgr)
                  ok, buf = cv2.imencode(".png", clean_bgr)
                  try:
                      s_clean = anomaly_score(await anomaly.get_classifications(
                          ViamImage(buf.tobytes(), CameraMimeType.PNG), 1))
                  except Exception as e:
                      print(f"  s{j}: clean scoring failed ({e})")
                      s_clean = float("nan")

                  raw_scores.append(s_raw)
                  clean_scores.append(s_clean)
                  shot_flags = []
                  if dbg.get("fg_frac", 1) < QC_MIN_FG_FRAC:
                      shot_flags.append("empty")
                  if dbg.get("n_objects", 0) >= 1:
                      shot_flags.append("neighbour")
                  if dbg.get("touches_border"):
                      shot_flags.append("cut_off")
                  flags.update(shot_flags)
                  shot_rows.append({
                      "item": k, "shot": j, "x": round(it["x"], 1), "y": round(it["y"], 1),
                      "length_mm": round(it["length"], 1), "width_mm": round(it["width"], 1),
                      "top_mm": round(it["top"], 1), "cam_z": round(cam_z, 1),
                      "centre_err_mm": round(err, 1),
                      "score_raw": round(s_raw, 4), "score_clean": round(s_clean, 4),
                      "tape_frac": round(float(dbg.get("tape_frac", 0)), 4),
                      "n_objects": dbg.get("n_objects"), "touches_border": dbg.get("touches_border"),
                      "blur_var": round(float(dbg.get("blur_var", 0)), 1),
                      "fg_frac": round(float(dbg.get("fg_frac", 0)), 4),
                      "flags": "|".join(shot_flags),
                  })
                  print(f"  s{j}: raw={s_raw:.3f} clean={s_clean:.3f} "
                        f"tape={dbg.get('tape_frac', 0):.3f} objs={dbg.get('n_objects')} "
                        f"cut={dbg.get('touches_border')} blur={dbg.get('blur_var', 0):.0f} {' '.join(shot_flags)}")
                  if first_raw is None:
                      first_raw, first_clean = bgr, clean_bgr

              mr = float(np.mean(raw_scores)) if raw_scores else float("nan")
              mc = float(np.mean(clean_scores)) if clean_scores else float("nan")
              spread = float(np.ptp(clean_scores)) if clean_scores else float("nan")
              if spread > 0.1:
                  flags.add(f"unstable_{spread:.2f}")
              row = {"item": k, "x": round(it["x"], 1), "y": round(it["y"], 1),
                     "length_mm": round(it["length"], 1), "centre_err_mm": round(err, 1),
                     "mean_raw": round(mr, 4), "decision_raw": decide(mr),
                     "mean_clean": round(mc, 4), "decision_clean": decide(mc),
                     "clean_spread": round(spread, 4), "flags": "|".join(sorted(flags)),
                     "seconds": round(time.time() - t0, 1), "truth": ""}
              item_rows.append(row)
              print(f"[item{k}] raw {mr:.3f} -> {decide(mr)} | clean {mc:.3f} -> {decide(mc)} "
                    f"| centre err {err:.1f} mm | {row['flags'] or 'ok'}")
              if first_raw is not None:
                  a = label_tile(first_raw, [f"item{k} RAW", f"{mr:.3f} {decide(mr)}"])
                  b = label_tile(first_clean, [f"CLEAN", f"{mc:.3f} {decide(mc)}", row["flags"][:28]])
                  sheet.append(np.hstack([a, b]))

        finally:
            print("\nReturning home.")
            await home.set_position(2)

    with open(os.path.join(out, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(shot_rows[0].keys()) if shot_rows else ["item"])
        w.writeheader()
        w.writerows(shot_rows)
    with open(os.path.join(out, "items.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(item_rows[0].keys()) if item_rows else ["item"])
        w.writeheader()
        w.writerows(item_rows)
    if sheet:
        cv2.imwrite(os.path.join(out, "contact.jpg"), np.vstack(sheet), [cv2.IMWRITE_JPEG_QUALITY, 88])

    print(f"\nThreshold {THRESHOLD}. Saved to {out}")
    print(f"{'item':>4} {'raw':>6} {'dec_raw':>9} {'clean':>6} {'dec_clean':>9}  flags")
    for r in item_rows:
        print(f"{r['item']:>4} {r['mean_raw']:>6.3f} {r['decision_raw']:>9} "
              f"{r['mean_clean']:>6.3f} {r['decision_clean']:>9}  {r['flags']}")
    print("Fill the 'truth' column in items.csv (good/bad) to score the run.")


if __name__ == "__main__":
    asyncio.run(main())
