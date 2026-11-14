"""Build prepared_v3 from train_images + unmatched viam export captures."""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from prep.preprocess import PAPER_LAB_TARGET, preprocess, _lab_to_rgb

ROOT = Path(__file__).resolve().parent.parent
TRAIN = ROOT / "train_images"
RAW_CLEAN = ROOT / "dataset_raw" / "kashyap-armfarm14-clean"
MOVES_LOG = Path(__file__).resolve().parent / "moves.log"
SEED = 14
PAPER_RGB = tuple(int(x) for x in _lab_to_rgb(PAPER_LAB_TARGET))

# Manual exclusions: (glob relative to ROOT or under train_images, dest folder name)
MANUAL_MOVES = [
    ("train_images/good/good_183255_*", "_excluded_unsure"),
    ("train_images/good/good_165858_1_*", "_rejected_offcentre"),
    ("train_images/good/good_173817_0_*", "_rejected_offcentre"),
    ("train_images/good/good_174526_0_*", "_rejected_offcentre"),
    ("train_images/good/good_201319_0_*", "_rejected_offcentre"),
    ("train_images/good/good_200301_1_*", "_rejected_offcentre"),
    ("train_images/good/good_203208_1_*", "_rejected_offcentre"),
    ("train_images/bad/bad_175110_0_*", "_rejected_offcentre"),
]

VIAM_EXCLUDE_SUBSTR = ("T23_33_19", "T23_33_20")


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _log_move(src: Path, dst: Path, reason: str) -> None:
    MOVES_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(MOVES_LOG, "a", encoding="utf-8") as f:
        f.write(f"{src} -> {dst}  # {reason}\n")


def apply_manual_exclusions(dry_run: bool) -> int:
    n = 0
    for pattern, dest_name in MANUAL_MOVES:
        dest_dir = TRAIN / dest_name
        if not dry_run:
            dest_dir.mkdir(parents=True, exist_ok=True)
        for src in sorted(ROOT.glob(pattern)):
            if not src.is_file():
                continue
            # already moved?
            if dest_name in src.parts:
                continue
            dst = dest_dir / src.name
            if dst.exists() and not src.exists():
                continue
            if dst.exists() and src.resolve() == dst.resolve():
                continue
            if src.parent.resolve() == dest_dir.resolve():
                continue
            print(f"  move {src.relative_to(ROOT)} -> train_images/{dest_name}/")
            if not dry_run:
                if dst.exists():
                    # already have a copy; remove source only if different path
                    if src.resolve() != dst.resolve():
                        src.unlink()
                        _log_move(src, dst, f"manual {dest_name} (dst existed)")
                else:
                    shutil.move(str(src), str(dst))
                    _log_move(src, dst, f"manual {dest_name}")
            n += 1
    return n


def index_train_md5s() -> set[str]:
    digests = set()
    for p in TRAIN.rglob("*"):
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"} and p.is_file():
            digests.add(_md5(p))
    return digests


def gather_candidates(dry_run: bool = False) -> list[dict]:
    rows = []
    for label, folder in (("good", TRAIN / "good"), ("bad", TRAIN / "bad")):
        if not folder.is_dir():
            continue
        for p in sorted(folder.glob("*.jpg")):
            m = re.match(r"^(good|bad)_(\d{6})_(\d+)_(\d+)\.jpg$", p.name)
            group = f"{m.group(2)}_{m.group(3)}" if m else p.stem
            rows.append({
                "src_path": str(p),
                "label": "normal" if label == "good" else "anomalous",
                "source": "train_images",
                "group": group,
            })

    digests = index_train_md5s()
    if RAW_CLEAN.is_dir():
        for p in sorted(RAW_CLEAN.glob("*.jpg")):
            if any(s in p.name for s in VIAM_EXCLUDE_SUBSTR):
                dest = TRAIN / "_excluded_unsure" / p.name
                if not dry_run:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    if not dest.exists():
                        shutil.copy2(str(p), str(dest))
                        _log_move(p, dest, "viam_export cable/dark object")
                continue
            digest = _md5(p)
            if digest in digests:
                continue
            mm = re.search(r"T(\d{2})_(\d{2})_", p.name)
            group = f"viam_{mm.group(1)}{mm.group(2)}" if mm else f"viam_{p.stem[:20]}"
            rows.append({
                "src_path": str(p),
                "label": "normal",
                "source": "viam_export",
                "group": group,
            })
    return rows


def dhash(img_rgb: np.ndarray, hash_size: int = 8) -> int:
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    small = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    diff = small[:, 1:] > small[:, :-1]
    bits = 0
    for i, v in enumerate(diff.flatten()):
        if v:
            bits |= 1 << i
    return bits


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def contact_sheet(paths: list[Path], out_path: Path, cols: int = 12, thumb: int = 96) -> None:
    if not paths:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), np.full((thumb, thumb, 3), 200, dtype=np.uint8))
        return
    cells = []
    for p in paths:
        im = cv2.imread(str(p))
        if im is None:
            im = np.full((thumb, thumb, 3), 128, dtype=np.uint8)
        else:
            im = cv2.resize(im, (thumb, thumb), interpolation=cv2.INTER_AREA)
        stem = p.stem[:18]
        cv2.putText(im, stem, (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1, cv2.LINE_AA)
        cells.append(im)
    rows = (len(cells) + cols - 1) // cols
    canvas = np.full((rows * thumb, cols * thumb, 3), 255, dtype=np.uint8)
    for i, cell in enumerate(cells):
        r, c = divmod(i, cols)
        canvas[r * thumb:(r + 1) * thumb, c * thumb:(c + 1) * thumb] = cell
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)


def cluster_type(bbox, img_side: int = 400) -> str:
    if bbox is None:
        return "unk_0"
    _, _, bw, bh = bbox
    long_side = max(bw, bh)
    aspect = max(bw, bh) / max(1, min(bw, bh))
    # 5 bins on long side + aspect
    ls_bin = min(4, int(long_side / img_side * 5))
    return f"bin_{ls_bin}_a{int(aspect)}"


def stratified_group_split(groups: dict[str, list[dict]], seed: int = SEED):
    """80/20 by group, stratified on type_key."""
    rng = random.Random(seed)
    by_type: dict[str, list[str]] = defaultdict(list)
    for g, rows in groups.items():
        # use first row's type
        t = rows[0].get("type_key", "unk")
        by_type[t].append(g)
    train_g, val_g = set(), set()
    for t, glist in by_type.items():
        glist = list(glist)
        rng.shuffle(glist)
        if len(glist) == 1:
            train_g.add(glist[0])
            continue
        n_val = max(1, int(round(len(glist) * 0.2)))
        val_g.update(glist[:n_val])
        train_g.update(glist[n_val:])
    return train_g, val_g


def augment_good(img: np.ndarray, paper_rgb: tuple[int, int, int], rng: random.Random) -> list[tuple[str, np.ndarray]]:
    outs = [("orig", img)]
    for k, name in ((1, "rot90"), (2, "rot180"), (3, "rot270")):
        outs.append((name, np.ascontiguousarray(np.rot90(img, k))))
    outs.append(("hflip", np.ascontiguousarray(np.fliplr(img))))
    outs.append(("vflip", np.ascontiguousarray(np.flipud(img))))
    h, w = img.shape[:2]
    center = (w / 2.0, h / 2.0)
    for i in range(2):
        ang = rng.uniform(-20, 20)
        M = cv2.getRotationMatrix2D(center, ang, 1.0)
        rot = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=paper_rgb)
        outs.append((f"rrot{i}", rot))
    # photometric
    bright = rng.uniform(0.92, 1.08)
    gamma = rng.uniform(0.9, 1.1)
    photo = np.clip(img.astype(np.float64) * bright, 0, 255)
    photo = np.clip(255.0 * ((photo / 255.0) ** gamma), 0, 255).astype(np.uint8)
    outs.append(("photo", photo))
    return outs


def run(out_name: str, dry_run: bool) -> None:
    out_root = ROOT / out_name
    print("== build_dataset ==")
    print(f"manual exclusions ({'dry-run' if dry_run else 'apply'}):")
    n_moved = apply_manual_exclusions(dry_run=dry_run)
    print(f"  {n_moved} file(s)")

    # For dry-run of gather, still need current good/bad (after conceptual moves).
    # If dry-run, simulate exclusions by skipping those globs.
    skip_names = set()
    if dry_run:
        for pattern, _ in MANUAL_MOVES:
            for p in ROOT.glob(pattern):
                skip_names.add(p.name)

    candidates = gather_candidates(dry_run=dry_run)
    if dry_run:
        candidates = [c for c in candidates if Path(c["src_path"]).name not in skip_names]

    print(f"candidates: {len(candidates)}")

    processed = []
    rejects = defaultdict(int)
    blur_by_type: dict[str, list[float]] = defaultdict(list)

    # first pass: preprocess
    for c in candidates:
        bgr = cv2.imread(c["src_path"])
        if bgr is None:
            rejects["read_fail"] += 1
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        out, dbg = preprocess(rgb, out_size=256)
        tkey = cluster_type(dbg["bbox"])
        row = {**c, **dbg, "md5": _md5(Path(c["src_path"])), "reject_reason": "",
               "img": out, "type_key": tkey}
        processed.append(row)
        blur_by_type[f"{c['label']}:{tkey}"].append(dbg["blur_var"])

    # QC
    kept = []
    for row in processed:
        reason = ""
        if row["fg_frac"] < 0.002:
            reason = "empty"
        elif row.get("touches_border") and row["bbox"] is not None:
            bw, bh = row["bbox"][2], row["bbox"][3]
            # Only strong truncations: lots of part pixels on the border.
            # Mild edge contact is normal for long bolts in a 400x400 crop.
            if max(bw, bh) > 0.9 * 400 and row.get("border_px", 0) >= 600:
                reason = "partial"
        if not reason and row["n_objects"] >= 2:
            reason = "multi"
        if not reason:
            # Absolute floor only. Relative-to-median rejects small smooth screws
            # when the label median is dominated by textured long bolts.
            if row["blur_var"] < 3.0:
                reason = "blurry"
        if reason:
            row["reject_reason"] = reason
            rejects[reason] += 1
        kept.append(row)

    accepted = [r for r in kept if not r["reject_reason"]]

    # near-dup dHash within round_item: keep first 4
    by_group: dict[str, list] = defaultdict(list)
    for r in accepted:
        by_group[r["group"]].append(r)
    deduped = []
    for g, rows in by_group.items():
        kept_hashes = []
        kept_rows = []
        for r in rows:
            h = dhash(r["img"])
            if any(hamming(h, kh) <= 3 for kh in kept_hashes):
                r["reject_reason"] = "near_dup"
                rejects["near_dup"] += 1
                continue
            if len(kept_rows) >= 5:
                r["reject_reason"] = "near_dup_cap"
                rejects["near_dup_cap"] += 1
                continue
            kept_hashes.append(h)
            kept_rows.append(r)
        deduped.extend(kept_rows)
        # also keep rejected rows in kept for manifest
    # merge reject updates into kept
    accepted = deduped
    print(f"accepted after QC: {len(accepted)}  rejects: {dict(rejects)}")

    # split by group (normals for train/val; anomalous all go val)
    normal_groups: dict[str, list] = defaultdict(list)
    anomalous = []
    for r in accepted:
        if r["label"] == "anomalous":
            anomalous.append(r)
        else:
            normal_groups[r["group"]].append(r)

    train_g, val_g = stratified_group_split(normal_groups, SEED)
    for r in accepted:
        if r["label"] == "anomalous":
            r["split"] = "val"
        elif r["group"] in val_g:
            r["split"] = "val"
        else:
            r["split"] = "train"

    train_normals = [r for r in accepted if r["label"] == "normal" and r["split"] == "train"]
    val_normals = [r for r in accepted if r["label"] == "normal" and r["split"] == "val"]
    print(f"unique train normal sources: {len(train_normals)}")
    print(f"val normal: {len(val_normals)}  val anomalous: {len(anomalous)}")
    print(f"groups train/val: {len(train_g)}/{len(val_g)}")
    type_counts = defaultdict(int)
    for r in train_normals:
        type_counts[r["type_key"]] += 1
    print(f"train type bins: {dict(type_counts)}")

    # tape_frac assert on accepted (source images after preprocess should be cleaned)
    bad_tape = [r for r in accepted if r["tape_frac"] >= 0.002 and r["label"] == "normal"]
    # Note: tape_frac is measured BEFORE fill; after preprocess the strip is gone.
    # Assert by re-detecting tape on the OUTPUT image.
    from prep.preprocess import _lab, _tape_mask
    high_after = 0
    for r in train_normals[:]:
        t_after = float(np.count_nonzero(_tape_mask(_lab(r["img"])))) / (256 * 256)
        r["tape_frac_after"] = t_after
        if t_after >= 0.002:
            high_after += 1
    print(f"train normals with tape_frac_after >= 0.002: {high_after}")

    if dry_run:
        print("\n--- DRY RUN summary (no files written) ---")
        _print_summary(accepted, kept, rejects)
        return

    # write outputs
    for sub in ("train/normal", "val/normal", "val/anomalous"):
        (out_root / sub).mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    rng = random.Random(SEED)
    written_train = []

    # val normals + anomalous (no aug)
    for r in val_normals + anomalous:
        sub = "val/normal" if r["label"] == "normal" else "val/anomalous"
        name = f"{r['source']}_{r['group']}_{Path(r['src_path']).stem}.png"
        path = out_root / sub / name
        cv2.imwrite(str(path), cv2.cvtColor(r["img"], cv2.COLOR_RGB2BGR))
        manifest_rows.append(_manifest(r, path, split=r["split"], aug="none"))

    # train normals + aug
    for r in train_normals:
        variants = augment_good(r["img"], PAPER_RGB, rng)
        t_after = r.get("tape_frac_after", 0)
        if t_after >= 0.002:
            print(f"  WARN tape_frac_after={t_after:.4f} on {r['src_path']}")
        for aug_name, im in variants:
            name = f"{r['source']}_{r['group']}_{Path(r['src_path']).stem}_{aug_name}.png"
            path = out_root / "train" / "normal" / name
            cv2.imwrite(str(path), cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
            written_train.append(path)
            manifest_rows.append(_manifest(r, path, split="train", aug=aug_name))

    # cap train at 900
    if len(written_train) > 900:
        keep = set(rng.sample(written_train, 900))
        for p in written_train:
            if p not in keep:
                p.unlink(missing_ok=True)
        manifest_rows = [m for m in manifest_rows
                         if m["split"] != "train" or Path(m["out_path"]).exists()]
        print(f"capped train normals to 900 (from {len(written_train)})")

    # rejected contact + manifest rejects
    for r in kept:
        if r["reject_reason"]:
            manifest_rows.append(_manifest(r, Path(""), split="", aug="", reject=r["reject_reason"]))

    man_path = out_root / "manifest.csv"
    fields = ["src_path", "md5", "label", "source", "group", "split", "aug",
              "reject_reason", "tape_frac", "n_objects", "blur_var", "bbox", "out_path", "type_key"]
    with open(man_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for m in manifest_rows:
            m = dict(m)
            if m.get("bbox") is not None:
                m["bbox"] = str(m["bbox"])
            w.writerow(m)

    # contact sheets
    train_paths = sorted((out_root / "train" / "normal").glob("*.png"))[:96]
    val_n = sorted((out_root / "val" / "normal").glob("*.png"))
    val_a = sorted((out_root / "val" / "anomalous").glob("*.png"))
    contact_sheet(train_paths + val_n[:24], out_root / "contact_train.jpg")
    contact_sheet(val_n + val_a, out_root / "contact_val.jpg")
    # rejected: save small preprocessed thumbs to temp list from kept rejects
    rej_dir = out_root / "_rejected_thumbs"
    rej_dir.mkdir(exist_ok=True)
    rej_paths = []
    for i, r in enumerate(kept):
        if not r["reject_reason"]:
            continue
        p = rej_dir / f"{r['reject_reason']}_{i}_{Path(r['src_path']).stem}.png"
        cv2.imwrite(str(p), cv2.cvtColor(r["img"], cv2.COLOR_RGB2BGR))
        rej_paths.append(p)
    contact_sheet(rej_paths[:96], out_root / "contact_rejected.jpg")

    _print_summary(accepted, kept, rejects)
    print(f"wrote {out_root}")


def _manifest(r, path: Path, split: str, aug: str, reject: str = "") -> dict:
    return {
        "src_path": r["src_path"],
        "md5": r.get("md5", ""),
        "label": r["label"],
        "source": r["source"],
        "group": r["group"],
        "split": split,
        "aug": aug,
        "reject_reason": reject or r.get("reject_reason", ""),
        "tape_frac": r.get("tape_frac", ""),
        "n_objects": r.get("n_objects", ""),
        "blur_var": r.get("blur_var", ""),
        "bbox": r.get("bbox"),
        "out_path": str(path) if path else "",
        "type_key": r.get("type_key", ""),
    }


def _print_summary(accepted, kept, rejects) -> None:
    from collections import Counter
    c = Counter((r["split"], r["label"], r["source"]) for r in accepted)
    print("\ncounts per split/label/source:")
    for k, v in sorted(c.items()):
        print(f"  {k}: {v}")
    print("rejects per reason:")
    for k, v in sorted(rejects.items()):
        print(f"  {k}: {v}")
    print(f"total kept rows (incl rejects tracked): {len(kept)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="prepared_v3")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    run(args.out, args.dry_run)


if __name__ == "__main__":
    main()
