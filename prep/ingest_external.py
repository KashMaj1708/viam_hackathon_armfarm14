"""Convert exported external datasets through preprocess() into prepared_ext/."""
from __future__ import annotations

import csv
import json
import os
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from prep.build_dataset import contact_sheet, dhash, hamming
from prep.preprocess import PAPER_LAB_TARGET, preprocess, _lab_to_rgb

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "dataset_raw"
OUT = ROOT / "prepared_ext"
SEED = 14
PAPER_RGB = tuple(int(x) for x in _lab_to_rgb(PAPER_LAB_TARGET))


def _load_meta(name: str) -> list[dict]:
    path = RAW / name / "meta.jsonl"
    if not path.exists():
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _read_rgb(path: Path) -> np.ndarray | None:
    bgr = cv2.imread(str(path))
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _save(img: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def _qc(dbg, min_bbox_long: int | None = None) -> str:
    if dbg["fg_frac"] < 0.002:
        return "empty"
    if dbg["touches_border"] and dbg["bbox"] is not None:
        if max(dbg["bbox"][2], dbg["bbox"][3]) > 0.8 * 400:
            return "partial"
    if dbg["n_objects"] >= 1:
        return "multi"
    if min_bbox_long is not None and dbg["bbox"] is not None:
        if max(dbg["bbox"][2], dbg["bbox"][3]) < min_bbox_long:
            return "too_small"
    return ""


def process_whole_frame(dataset: str, source: str, label_fn, min_bbox_long=None):
    meta = _load_meta(dataset)
    folder = RAW / dataset
    results = []
    files = meta if meta else [{"file": p.name, "tags": [], "bboxes": []}
                               for p in sorted(list(folder.glob("*.jpg")) + list(folder.glob("*.png")))]
    for row in files:
        path = folder / row["file"]
        if not path.exists():
            continue
        rgb = _read_rgb(path)
        if rgb is None:
            continue
        label = label_fn(row)
        if label is None:
            continue
        out, dbg = preprocess(rgb, out_size=256)
        reason = _qc(dbg, min_bbox_long=min_bbox_long)
        # Use the unique trailing id (filenames share a long org/location prefix)
        stem = path.stem
        uniq = stem.split("_")[-1] if "_" in stem else stem[-24:]
        results.append({
            "src_path": str(path), "label": label, "source": source,
            "ext_dataset": dataset, "group": uniq,
            "reject_reason": reason, "img": out, **{k: dbg[k] for k in
                ("tape_frac", "n_objects", "blur_var", "bbox", "fg_frac", "touches_border")},
        })
    return results


def process_fastener_anomaly():
    dataset = "fastener-anomaly"
    meta = _load_meta(dataset)
    folder = RAW / dataset
    results = []
    for row in meta:
        path = folder / row["file"]
        if not path.exists():
            continue
        rgb = _read_rgb(path)
        if rgb is None:
            continue
        h, w = rgb.shape[:2]
        for i, bb in enumerate(row.get("bboxes") or []):
            lab = (bb.get("label") or "").lower()
            if lab not in ("normal", "normal-bolt"):
                continue
            x0 = int(float(bb["x_min_normalized"]) * w)
            y0 = int(float(bb["y_min_normalized"]) * h)
            x1 = int(float(bb["x_max_normalized"]) * w)
            y1 = int(float(bb["y_max_normalized"]) * h)
            bw, bh = max(1, x1 - x0), max(1, y1 - y0)
            mx, my = int(0.35 * bw), int(0.35 * bh)
            # pad with paper colour
            canvas = np.full((h + 2 * my + 2, w + 2 * mx + 2, 3), PAPER_RGB, dtype=np.uint8)
            canvas[my + 1:my + 1 + h, mx + 1:mx + 1 + w] = rgb
            cx0 = mx + 1 + x0 - mx
            cy0 = my + 1 + y0 - my
            cx1 = mx + 1 + x1 + mx
            cy1 = my + 1 + y1 + my
            crop = canvas[cy0:cy1, cx0:cx1]
            out, dbg = preprocess(crop, out_size=256)
            reason = _qc(dbg)
            results.append({
                "src_path": f"{path}#bb{i}", "label": "normal", "source": "ext_fa",
                "ext_dataset": dataset, "group": f"{path.stem}_bb{i}",
                "reject_reason": reason, "img": out, **{k: dbg[k] for k in
                    ("tape_frac", "n_objects", "blur_var", "bbox", "fg_frac", "touches_border")},
            })
    return results


def write_results(results: list[dict]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    man_path = OUT / "manifest.csv"
    fields = ["src_path", "label", "source", "ext_dataset", "group", "reject_reason",
              "tape_frac", "n_objects", "blur_var", "bbox", "out_path"]
    by_key = defaultdict(list)
    rows_out = []
    for r in results:
        label = r["label"]
        source = r["source"]
        if r["reject_reason"]:
            sub = OUT / source / f"_rejected_{r['reject_reason']}"
        else:
            sub = OUT / source / label
        name = f"{r['group']}.png".replace("/", "_")
        path = sub / name
        if not r["reject_reason"]:
            _save(r["img"], path)
            by_key[(source, label)].append(path)
        else:
            _save(r["img"], path)
        rows_out.append({
            "src_path": r["src_path"], "label": label, "source": source,
            "ext_dataset": r["ext_dataset"], "group": r["group"],
            "reject_reason": r["reject_reason"], "tape_frac": r.get("tape_frac"),
            "n_objects": r.get("n_objects"), "blur_var": r.get("blur_var"),
            "bbox": str(r.get("bbox")), "out_path": str(path),
        })
    with open(man_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows_out)
    for (source, label), paths in by_key.items():
        contact_sheet(paths[:96], OUT / f"contact_{source}_{label}.jpg")
    # also contact for anomalous_candidate
    for (source, label), paths in list(by_key.items()):
        print(f"  {source}/{label}: {len(paths)}")
    print(f"manifest -> {man_path}")


def main():
    print("== ingest_external ==")
    all_rows = []
    if (RAW / "an-14-one").exists():
        print("an-14-one...")
        all_rows += process_whole_frame(
            "an-14-one", "ext_an14", lambda r: "normal")
    if (RAW / "an-14-two").exists():
        print("an-14-two...")
        all_rows += process_whole_frame(
            "an-14-two", "ext_an14", lambda r: "normal")
    if (RAW / "armfarm12-fasteners-412mm").exists():
        print("armfarm12-fasteners-412mm...")
        def lab_af12(r):
            tags = [t.lower() for t in (r.get("tags") or [])]
            if "anomalous" in tags or "anomaly" in tags:
                return "anomalous_candidate"
            if "normal" in tags or not tags:
                return "normal"
            return None
        all_rows += process_whole_frame(
            "armfarm12-fasteners-412mm", "ext_af12", lab_af12, min_bbox_long=40)
    if (RAW / "fastener-anomaly").exists():
        print("fastener-anomaly...")
        all_rows += process_fastener_anomaly()
    if (RAW / "sakshi_am14").exists():
        print("sakshi_am14...")
        all_rows += process_whole_frame(
            "sakshi_am14", "ext_sakshi", lambda r: "anomalous_candidate")

    write_results(all_rows)
    print("\n=== HUMAN GATE ===")
    print("Review these contact sheets, then tell me which to keep/reject:")
    for p in sorted(OUT.glob("contact_*.jpg")):
        print(f"  {p}")
    print("Especially:")
    print("  - sakshi_am14 / anomalous_candidate: which are real black-tape anomalies")
    print("  - ext_af12 anomalous_candidate: tape-wrapped only")
    print("  - any dirty/odd external normals")
    print("STOPPING for your review (prompt 2 step 3).")


if __name__ == "__main__":
    main()
