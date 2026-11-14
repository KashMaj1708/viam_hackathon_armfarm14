"""Score prepared_v3/val through the same inference path; suggest THRESHOLD."""
from __future__ import annotations

import argparse
import asyncio
import csv
import os
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def auroc(scores: list[float], labels: list[int]) -> float:
    """labels: 1=anomalous, 0=normal. Simple Mann-Whitney AUROC."""
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return float("nan")
    correct = 0.0
    for p in pos:
        for n in neg:
            if p > n:
                correct += 1
            elif p == n:
                correct += 0.5
    return correct / (len(pos) * len(neg))


def best_threshold(scores, labels):
    best_t, best_ba = 0.5, -1.0
    for t in sorted(set(scores)):
        tp = fp = tn = fn = 0
        for s, y in zip(scores, labels):
            pred = 1 if s >= t else 0
            if pred == 1 and y == 1:
                tp += 1
            elif pred == 1 and y == 0:
                fp += 1
            elif pred == 0 and y == 0:
                tn += 1
            else:
                fn += 1
        tpr = tp / max(1, tp + fn)
        tnr = tn / max(1, tn + fp)
        ba = 0.5 * (tpr + tnr)
        if ba > best_ba:
            best_ba, best_t = ba, t
    return best_t, best_ba


def zero_fn_threshold(scores, labels):
    """Lowest threshold with 0 false negatives (catch all anomalous)."""
    anom = [s for s, y in zip(scores, labels) if y == 1]
    if not anom:
        return float("nan")
    return min(anom)


async def score_image(anomaly, png_bytes: bytes) -> float:
    from viam.media.video import CameraMimeType, ViamImage
    image = ViamImage(png_bytes, CameraMimeType.PNG)
    cls = await anomaly.get_classifications(image, 1)
    c = cls[0]
    return c.confidence if c.class_name == "anomalous" else 1 - c.confidence


async def run(val_root: Path, out_csv: Path):
    from viam.robot.client import RobotClient
    from viam.services.vision import VisionClient

    opts = RobotClient.Options.with_api_key(
        api_key=os.environ["VIAM_API_KEY"], api_key_id=os.environ["VIAM_API_KEY_ID"])
    machine = await RobotClient.at_address(os.environ["VIAM_MACHINE_ADDRESS"], opts)
    anomaly = VisionClient.from_robot(machine, "anomaly")

    rows = []
    try:
        for label, sub in (("normal", "normal"), ("anomalous", "anomalous")):
            folder = val_root / sub
            if not folder.is_dir():
                continue
            for path in sorted(folder.glob("*.png")):
                data = path.read_bytes()
                # images already preprocessed in prepared set
                try:
                    score = await score_image(anomaly, data)
                except Exception as e:
                    print(f"fail {path.name}: {e}")
                    continue
                rows.append({"path": str(path), "label": label, "score": score})
                print(f"  {label} {path.name}: {score:.4f}")
    finally:
        await machine.close()

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "label", "score"])
        w.writeheader()
        w.writerows(rows)

    scores = [r["score"] for r in rows]
    labels = [1 if r["label"] == "anomalous" else 0 for r in rows]
    normals = [r["score"] for r in rows if r["label"] == "normal"]
    anoms = [r["score"] for r in rows if r["label"] == "anomalous"]
    print("\n=== calibration ===")
    print(f"n normal={len(normals)} anomalous={len(anoms)}")
    if normals:
        print(f"normal scores: min={min(normals):.4f} max={max(normals):.4f} mean={np.mean(normals):.4f}")
    if anoms:
        print(f"anomalous scores: min={min(anoms):.4f} max={max(anoms):.4f} mean={np.mean(anoms):.4f}")
    auc = auroc(scores, labels)
    t_ba, ba = best_threshold(scores, labels)
    t_fn = zero_fn_threshold(scores, labels)
    print(f"AUROC: {auc:.4f}")
    print(f"best balanced-accuracy threshold: {t_ba:.4f} (BA={ba:.4f})")
    print(f"0-FN threshold (min anomalous): {t_fn:.4f}")
    print(f"SUGGEST THRESHOLD = {t_ba:.4f}  (do not auto-edit track3_sort.py)")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", default="prepared_v3/val")
    args = ap.parse_args()
    val = Path(args.val)
    if not val.is_absolute():
        val = ROOT / val
    out_csv = val.parent / "val_scores.csv"
    asyncio.run(run(val, out_csv))


if __name__ == "__main__":
    main()
