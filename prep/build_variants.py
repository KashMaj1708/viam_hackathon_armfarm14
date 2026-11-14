"""Build prepared_v3a / v3b / v3c after human gate on prepared_ext.

Usage (after reviewing contact sheets):
  python -m prep.build_variants --sakshi-keep stem1,stem2 --af12-anom-keep stem3
"""
from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

from prep.build_dataset import PAPER_RGB, SEED, augment_good, contact_sheet
from prep.preprocess import _lab, _tape_mask

ROOT = Path(__file__).resolve().parent.parent


def _copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def _add_normals(src_dir: Path, dst_train: Path, cap_extra: int | None, rng: random.Random) -> int:
    paths = sorted(src_dir.glob("*.png"))
    if cap_extra is not None and len(paths) > cap_extra:
        paths = rng.sample(paths, cap_extra)
    n = 0
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        for aug_name, im in augment_good(rgb, PAPER_RGB, rng):
            out = dst_train / f"ext_{p.stem}_{aug_name}.png"
            cv2.imwrite(str(out), cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="prepared_v3")
    ap.add_argument("--sakshi-keep", default="", help="comma stems to promote to val/anomalous")
    ap.add_argument("--af12-anom-keep", default="", help="comma stems of tape anomalies")
    ap.add_argument("--skip-ext", action="store_true", help="only clone v3a from base")
    args = ap.parse_args()

    base = ROOT / args.base
    if not base.exists():
        raise SystemExit(f"missing {base}; run build_dataset first")

    v3a = ROOT / "prepared_v3a"
    v3b = ROOT / "prepared_v3b"
    v3c = ROOT / "prepared_v3c"
    print("building v3a (own only)...")
    _copy_tree(base, v3a)

    # promote approved sakshi anomalies into all variants' val/anomalous
    sakshi_keeps = {s.strip() for s in args.sakshi_keep.split(",") if s.strip()}
    af12_keeps = {s.strip() for s in args.af12_anom_keep.split(",") if s.strip()}
    ext = ROOT / "prepared_ext"
    for stem in sakshi_keeps:
        hits = list((ext / "ext_sakshi" / "anomalous_candidate").glob(f"*{stem}*.png"))
        for h in hits:
            for variant in (v3a,):
                shutil.copy2(h, variant / "val" / "anomalous" / f"sakshi_{h.name}")

    if args.skip_ext:
        print("skip-ext: only v3a ready")
        return

    print("building v3b (own + an14)...")
    _copy_tree(v3a, v3b)
    rng = random.Random(SEED)
    an14 = ext / "ext_an14" / "normal"
    if an14.is_dir():
        n = _add_normals(an14, v3b / "train" / "normal", None, rng)
        print(f"  added {n} aug images from ext_an14")

    print("building v3c (v3b + af12 + fa, external <=35%)...")
    _copy_tree(v3b, v3c)
    own_train = list((v3a / "train" / "normal").glob("*.png"))
    max_ext = int(0.35 / 0.65 * len(own_train)) if own_train else 0
    # count current ext in v3b roughly by name prefix
    already = len([p for p in (v3c / "train" / "normal").glob("ext_*.png")])
    budget = max(0, max_ext - already)
    # collect candidate external normals not yet in (af12 + fa)
    candidates = []
    for src in (ext / "ext_af12" / "normal", ext / "ext_fa" / "normal"):
        if src.is_dir():
            candidates.extend(sorted(src.glob("*.png")))
    rng.shuffle(candidates)
    # add with aug but respect budget on SOURCE count roughly
    added_src = 0
    for p in candidates:
        if added_src >= budget:
            break
        img = cv2.imread(str(p))
        if img is None:
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        for aug_name, im in augment_good(rgb, PAPER_RGB, rng):
            out = v3c / "train" / "normal" / f"ext_{p.stem}_{aug_name}.png"
            cv2.imwrite(str(out), cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
        added_src += 1
    print(f"  added {added_src} external source normals (budget {budget})")

    for stem in af12_keeps:
        hits = list((ext / "ext_af12" / "anomalous_candidate").glob(f"*{stem}*.png"))
        for h in hits:
            for variant in (v3a, v3b, v3c):
                shutil.copy2(h, variant / "val" / "anomalous" / f"af12_{h.name}")

    for name, folder in (("v3a", v3a), ("v3b", v3b), ("v3c", v3c)):
        tn = len(list((folder / "train" / "normal").glob("*.png")))
        vn = len(list((folder / "val" / "normal").glob("*.png")))
        va = len(list((folder / "val" / "anomalous").glob("*.png")))
        print(f"{name}: train_normal={tn} val_normal={vn} val_anomalous={va}")
    print("Upload with:")
    print("  python -m prep.upload_prepared --folder prepared_v3a --dataset-name kashyap-armfarm14-v3a")
    print("  python -m prep.upload_prepared --folder prepared_v3b --dataset-name kashyap-armfarm14-v3b")
    print("  python -m prep.upload_prepared --folder prepared_v3c --dataset-name kashyap-armfarm14-v3c")


if __name__ == "__main__":
    main()
