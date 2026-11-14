"""Unit-style checks on 4 real files; writes before/after panels to prep/_debug/."""
from __future__ import annotations

import glob
import os
import sys

import cv2
import numpy as np

from prep.preprocess import preprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEBUG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_debug")


def _first(patterns: list[str]) -> str | None:
    for p in patterns:
        hits = sorted(glob.glob(os.path.join(ROOT, p)))
        if hits:
            return hits[0]
    return None


def _pick_cases() -> dict[str, str]:
    cases = {
        "good_wide_tape": _first([
            "train_images/good/good_200456_*.jpg",
            "train_images/good/good_201042_*.jpg",
            "train_images/good/good_170234_*.jpg",
        ]),
        "bad_tape_on_part": _first([
            "train_images/bad/bad_175002_*.jpg",
            "train_images/bad/bad_175419_*.jpg",
            "train_images/bad/*.jpg",
        ]),
        "small_screw": _first([
            "train_images/good/good_174722_*.jpg",
            "train_images/good/good_174621_*.jpg",
            "train_images/good/good_173544_*.jpg",
        ]),
        "long_bolt": _first([
            "train_images/good/good_201319_1_*.jpg",
            "train_images/good/good_200301_0_*.jpg",
            "train_images/good/good_203208_0_*.jpg",
        ]),
    }
    missing = [k for k, v in cases.items() if not v]
    if missing:
        raise SystemExit(f"Could not find test images for: {missing}")
    return cases


def _panel(before_rgb: np.ndarray, after_rgb: np.ndarray, title: str) -> np.ndarray:
    b = cv2.cvtColor(before_rgb, cv2.COLOR_RGB2BGR)
    a = cv2.cvtColor(after_rgb, cv2.COLOR_RGB2BGR)
    # match heights
    h = max(b.shape[0], a.shape[0])
    def pad(im):
        if im.shape[0] == h:
            return im
        out = np.full((h, im.shape[1], 3), 255, dtype=np.uint8)
        out[:im.shape[0]] = im
        return out
    b, a = pad(b), pad(a)
    gap = np.full((h, 8, 3), 255, dtype=np.uint8)
    row = np.hstack([b, gap, a])
    banner = np.full((28, row.shape[1], 3), 240, dtype=np.uint8)
    cv2.putText(banner, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 200), 1, cv2.LINE_AA)
    return np.vstack([banner, row])


def main() -> None:
    os.makedirs(DEBUG, exist_ok=True)
    cases = _pick_cases()
    print("test_preprocess cases:")
    for name, path in cases.items():
        bgr = cv2.imread(path)
        if bgr is None:
            raise SystemExit(f"failed to read {path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        out, dbg = preprocess(rgb, out_size=256)
        print(f"  {name}: {os.path.basename(path)}")
        print(f"    {dbg}")
        panel = _panel(rgb, out, f"{name}  tape={dbg['tape_frac']:.3f} fg={dbg['fg_frac']:.3f}")
        out_path = os.path.join(DEBUG, f"{name}.jpg")
        cv2.imwrite(out_path, panel)
        print(f"    -> {out_path}")
        if name == "bad_tape_on_part":
            # crude check: after preprocess, some dark pixels should remain near centre
            gray = cv2.cvtColor(out, cv2.COLOR_RGB2GRAY)
            cy, cx = gray.shape[0] // 2, gray.shape[1] // 2
            patch = gray[cy - 40:cy + 40, cx - 40:cx + 40]
            if float(np.mean(patch < 80)) < 0.01:
                print("    WARN: little dark content near centre; tape-on-part may be over-masked")
        if name == "good_wide_tape":
            # re-run tape detector on original to confirm strip existed
            from prep.preprocess import _lab, _tape_mask
            t = _tape_mask(_lab(rgb))
            if np.count_nonzero(t) == 0:
                print("    WARN: no side tape detected on this good sample; pick another?")
            elif dbg["tape_frac"] < 0.001:
                print("    WARN: tape_frac near zero despite mask; check fill")
    print("done.")


if __name__ == "__main__":
    main()


# allow: python -m prep.test_preprocess
def __getattr__(name):  # pragma: no cover
    raise AttributeError(name)
