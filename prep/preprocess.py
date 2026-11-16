"""Shared preprocess for training AND inference (object-centred, tape-stripped)."""
from __future__ import annotations

import cv2
import numpy as np

# Target paper colour in OpenCV LAB (L,a,b on 0..255)
PAPER_LAB_TARGET = np.array([200.0, 128.0, 128.0], dtype=np.float64)


def _lab(img_rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)


def _paper_lab(lab: np.ndarray) -> np.ndarray:
    L = lab[:, :, 0].astype(np.float64)
    thr = np.percentile(L, 60)
    mask = L >= thr
    if not np.any(mask):
        return np.median(lab.reshape(-1, 3), axis=0).astype(np.float64)
    return np.median(lab[mask].reshape(-1, 3), axis=0).astype(np.float64)


def _tape_mask(lab: np.ndarray) -> np.ndarray:
    """Dark side-border strip only; leave on-part black tape alone.

    Workspace tape is often inset behind a bright rim, so side membership is
    based on bbox position / centroid in the left or right third — not only col 0.
    On-part tape sits near the image centre and is left alone.
    """
    h, w = lab.shape[:2]
    dark = (lab[:, :, 0] < 70).astype(np.uint8) * 255
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(dark, connectivity=8)
    mask = np.zeros((h, w), dtype=np.uint8)
    min_h = 0.40 * h
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if bh < min_h or bw > 0.50 * w:
            continue
        cx = float(centroids[i][0])
        on_left = cx < 0.35 * w and (x + bw) < 0.55 * w
        on_right = cx > 0.65 * w and x > 0.45 * w
        if on_left or on_right:
            mask[labels == i] = 255
    if np.any(mask):
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        mask = cv2.dilate(mask, k, iterations=1)  # ~5 px
    return mask


def _fill_tape(img_rgb: np.ndarray, tape: np.ndarray, paper_rgb: np.ndarray) -> np.ndarray:
    if not np.any(tape):
        return img_rgb.copy()
    inpainted = cv2.inpaint(img_rgb, tape, 7, cv2.INPAINT_TELEA)
    out = inpainted.copy()
    ys, xs = np.where(tape > 0)
    noise = np.random.normal(0, 2.0, size=(len(ys), 3))
    filled = np.clip(paper_rgb.astype(np.float64) + noise, 0, 255).astype(np.uint8)
    out[ys, xs] = filled
    return out


def _lab_to_rgb(lab_vec: np.ndarray) -> np.ndarray:
    patch = np.zeros((1, 1, 3), dtype=np.uint8)
    patch[0, 0] = np.clip(lab_vec, 0, 255).astype(np.uint8)
    return cv2.cvtColor(patch, cv2.COLOR_LAB2RGB)[0, 0].astype(np.float64)


def _white_balance(img_rgb: np.ndarray, paper_lab: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Scale channels so paper maps to fixed LAB target; return (img, target_paper_rgb)."""
    paper_rgb = _lab_to_rgb(paper_lab)
    target_rgb = _lab_to_rgb(PAPER_LAB_TARGET)
    scale = np.ones(3, dtype=np.float64)
    for c in range(3):
        if paper_rgb[c] > 1e-3:
            scale[c] = target_rgb[c] / paper_rgb[c]
    out = np.clip(img_rgb.astype(np.float64) * scale, 0, 255).astype(np.uint8)
    return out, target_rgb.astype(np.uint8)


def _delta_e76(lab: np.ndarray, paper_lab: np.ndarray) -> np.ndarray:
    d = lab.astype(np.float64) - paper_lab.reshape(1, 1, 3)
    return np.sqrt(np.sum(d * d, axis=2))


def _foreground(lab: np.ndarray, paper_lab: np.ndarray) -> np.ndarray:
    de = _delta_e76(lab, paper_lab)
    # Slightly stricter than the raw prompt defaults so JPEG/shadow noise does not
    # inflate the part bbox to the full frame (which falsely triggers partial QC).
    fg = ((de > 22) | (lab[:, :, 0] < 100)).astype(np.uint8) * 255
    open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, open_k)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, close_k)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    cleaned = np.zeros_like(fg)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= 150:
            cleaned[labels == i] = 255
    return cleaned


def _components(fg: np.ndarray):
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(fg, connectivity=8)
    comps = []
    for i in range(1, n):
        comps.append({
            "id": i,
            "area": int(stats[i, cv2.CC_STAT_AREA]),
            "x": int(stats[i, cv2.CC_STAT_LEFT]),
            "y": int(stats[i, cv2.CC_STAT_TOP]),
            "w": int(stats[i, cv2.CC_STAT_WIDTH]),
            "h": int(stats[i, cv2.CC_STAT_HEIGHT]),
            "cx": float(centroids[i][0]),
            "cy": float(centroids[i][1]),
        })
    comps.sort(key=lambda c: -c["area"])
    return labels, comps


def preprocess(img_rgb: np.ndarray, out_size: int = 256) -> tuple[np.ndarray, dict]:
    """Returns cleaned, object-centred square image and a debug dict."""
    if img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
        raise ValueError("img_rgb must be HxWx3 RGB")
    h0, w0 = img_rgb.shape[:2]
    lab0 = _lab(img_rgb)
    paper0 = _paper_lab(lab0)
    tape = _tape_mask(lab0)
    tape_frac = float(np.count_nonzero(tape)) / float(h0 * w0)

    paper_rgb0 = _lab_to_rgb(paper0).astype(np.uint8)
    filled = _fill_tape(img_rgb, tape, paper_rgb0)

    lab1 = _lab(filled)
    paper1 = _paper_lab(lab1)
    balanced, paper_rgb = _white_balance(filled, paper1)

    lab2 = _lab(balanced)
    paper2 = _paper_lab(lab2)
    # After WB, paper should be near target; use measured paper2 for FG
    fg = _foreground(lab2, paper2)
    # Never let residual side-tape pixels become the "part"
    if np.any(tape):
        fg[tape > 0] = 0
    labels, comps = _components(fg)
    # Drop tall border strips (failed tape leftovers), keep centred fasteners
    comps = [
        c for c in comps
        if not (
            c["h"] >= 0.85 * h0
            and c["w"] <= 0.45 * w0
            and (c["x"] <= 0.08 * w0 or c["x"] + c["w"] >= 0.92 * w0)
        )
    ]
    # Prefer the component whose centroid is closest to image centre when close in area
    # The arm centres the camera on the target part, so the part to score is the one at (or
    # nearest) the image centre, not simply the biggest blob (a long neighbour can be bigger).
    if len(comps) >= 2:
        cx0, cy0 = w0 / 2.0, h0 / 2.0
        big = comps[0]["area"]

        def _d(c):
            dx = max(c["x"] - cx0, 0.0, cx0 - (c["x"] + c["w"]))
            dy = max(c["y"] - cy0, 0.0, cy0 - (c["y"] + c["h"]))
            return float(np.hypot(dx, dy))
        cand = [c for c in comps if c["area"] >= max(80, 0.03 * big)]
        if cand:
            pick = min(cand, key=lambda c: (_d(c), -c["area"]))
            comps = [pick] + [c for c in comps if c is not pick]

    debug = {
        "tape_frac": tape_frac,
        "n_objects": 0,
        "bbox": None,
        "touches_border": False,
        "blur_var": 0.0,
        "fg_frac": float(np.count_nonzero(fg)) / float(h0 * w0),
    }

    if not comps:
        # empty: still return WB square of whole frame
        side = min(h0, w0)
        y0 = (h0 - side) // 2
        x0 = (w0 - side) // 2
        crop = balanced[y0:y0 + side, x0:x0 + side]
        out = cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(out, cv2.COLOR_RGB2GRAY)
        debug["blur_var"] = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        return out, debug

    main = comps[0]
    others = [c for c in comps[1:] if c["area"] >= 0.15 * main["area"]]
    debug["n_objects"] = len(others)

    cleaned = balanced.copy()
    if others:
        dil_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))  # ~6 px
        suppress = np.zeros((h0, w0), dtype=np.uint8)
        for c in others:
            suppress[labels == c["id"]] = 255
        suppress = cv2.dilate(suppress, dil_k)
        ys, xs = np.where(suppress > 0)
        cleaned[ys, xs] = paper_rgb

    bx, by, bw, bh = main["x"], main["y"], main["w"], main["h"]
    debug["bbox"] = (bx, by, bw, bh)
    main_mask = labels == main["id"]
    # True truncation: enough part pixels sit on the image border (not just a loose bbox)
    border_hit = int(
        np.count_nonzero(main_mask[:3, :])
        + np.count_nonzero(main_mask[-3:, :])
        + np.count_nonzero(main_mask[:, :3])
        + np.count_nonzero(main_mask[:, -3:])
    )
    debug["border_px"] = border_hit
    debug["touches_border"] = border_hit >= 40

    cx = bx + bw / 2.0
    cy = by + bh / 2.0
    side = max(bw, bh) * 1.35
    side = max(side, 160.0)
    half = side / 2.0

    # Pad canvas with paper colour so crop never reflects
    pad = int(np.ceil(half)) + 2
    canvas = np.full((h0 + 2 * pad, w0 + 2 * pad, 3), paper_rgb, dtype=np.uint8)
    canvas[pad:pad + h0, pad:pad + w0] = cleaned
    x1 = int(round(cx + pad - half))
    y1 = int(round(cy + pad - half))
    x2 = x1 + int(round(side))
    y2 = y1 + int(round(side))
    crop = canvas[y1:y2, x1:x2]
    if crop.shape[0] != crop.shape[1]:
        s = min(crop.shape[0], crop.shape[1])
        crop = crop[:s, :s]
    out = cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(out, cv2.COLOR_RGB2GRAY)
    debug["blur_var"] = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    # Recompute fg_frac on cleaned original footprint
    main_mask = (labels == main["id"])
    debug["fg_frac"] = float(np.count_nonzero(main_mask)) / float(h0 * w0)
    return out, debug
