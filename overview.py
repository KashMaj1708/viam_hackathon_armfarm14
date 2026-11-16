"""Overview webcam (overview-cam): count parts in the neutral strip and give rough world positions.

Uses overview_calib.json (pixel -> table-plane homography, about +/-7 mm). Good enough to count parts
and to catch parts the wrist-camera scan missed; NOT accurate enough to aim a grasp by itself
(the wrist camera still centres on each part before picking).
"""
import json
import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_CAL = json.load(open(os.path.join(HERE, "overview_calib.json")))
H = np.array(_CAL["H_pix_to_world"], dtype=np.float64)
H_INV = np.linalg.inv(H)
_REF = cv2.imread(os.path.join(HERE, _CAL.get("reference_image", "overview_ref.jpg")), cv2.IMREAD_GRAYSCALE)
_ORB = cv2.ORB_create(3000)
_REF_KP, _REF_DES = _ORB.detectAndCompute(_REF, None) if _REF is not None else (None, None)


def register(bgr):
    """Warp a live frame onto the calibration reference frame (cancels small camera bumps/drift).
    Returns (aligned_bgr, n_inliers); falls back to the frame itself if matching fails."""
    if _REF_DES is None:
        return bgr, 0
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    kp, des = _ORB.detectAndCompute(g, None)
    if des is None:
        return bgr, 0
    m = sorted(cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(des, _REF_DES),
               key=lambda x: x.distance)[:600]
    if len(m) < 30:
        return bgr, 0
    src = np.float32([kp[x.queryIdx].pt for x in m])
    dst = np.float32([_REF_KP[x.trainIdx].pt for x in m])
    Hr, inl = cv2.findHomography(src, dst, cv2.RANSAC, 2.0)
    if Hr is None or inl.sum() < 60:
        return bgr, int(0 if inl is None else inl.sum())
    return cv2.warpPerspective(bgr, Hr, (bgr.shape[1], bgr.shape[0])), int(inl.sum())
STRIP = [(125, -225), (600, -225), (600, -90), (125, -90)]     # world corners of the neutral strip
MIN_AREA_PX = 40
WS_X, WS_Y = (125, 590), (-218, -94)   # same limits as the wrist cam-workspace crop, a little slack


def pix_to_world(u, v):
    p = cv2.perspectiveTransform(np.array([[[u, v]]], np.float64), H)[0, 0]
    return float(p[0]), float(p[1])


def world_to_pix(x, y):
    p = cv2.perspectiveTransform(np.array([[[x, y]]], np.float64), H_INV)[0, 0]
    return float(p[0]), float(p[1])


def find_parts(bgr):
    """Parts on the strip in an overview frame -> list of dicts (x, y world mm, bbox px, area)."""
    bgr, _ = register(bgr)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.int16)
    L, B = lab[..., 0], lab[..., 2]
    poly = np.array([world_to_pix(x, y) for x, y in STRIP], np.int32)
    mask = np.zeros(L.shape, np.uint8)
    cv2.fillPoly(mask, [poly], 1)
    inside = mask > 0
    paper = inside & (L > np.percentile(L[inside], 50))
    pl, pb = np.median(L[paper]), np.median(B[paper])
    fg = (((pl - L) > 30) | (np.abs(B - pb) > 12)) & inside
    fg = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, lbl, st, _ = cv2.connectedComponentsWithStats(fg)
    parts = []
    for k in range(1, n):
        if st[k, cv2.CC_STAT_AREA] < MIN_AREA_PX:
            continue
        ys, xs = np.nonzero(lbl == k)
        Lb, Bb = L[ys, xs], B[ys, xs]
        metal = (Lb.max() >= pl - 5) or (np.abs(Bb - pb).mean() > 10) or (Lb.min() < 100)  # far parts have no highlight but dark edges
        if not metal:                                 # paper fold / shadow
            continue
        # the lower edge of a blob is closest to where the part touches the table
        x, y = pix_to_world(xs.mean(), np.percentile(ys, 85))
        if not (WS_X[0] <= x <= WS_X[1] and WS_Y[0] <= y <= WS_Y[1]):
            continue                                  # centre outside the neutral strip (tape edge, box)
        step = max(1, len(xs) // 300)                 # footprint on the table, for de-duplication
        fp = [pix_to_world(float(u), float(v)) for u, v in zip(xs[::step], ys[::step])]
        parts.append({"x": x, "y": y, "area": int(st[k, cv2.CC_STAT_AREA]),
                      "bbox": tuple(int(v) for v in st[k, :4]), "footprint": fp})
    return parts


async def grab(machine, name="overview-cam"):
    from viam.components.camera import Camera
    try:
        cam = Camera.from_robot(machine, name)
    except Exception:
        from viam.components.camera.client import CameraClient
        cam = CameraClient(name, machine._channel)
    imgs, _ = await cam.get_images()
    return cv2.imdecode(np.frombuffer(imgs[0].data, np.uint8), cv2.IMREAD_COLOR)


def annotate(bgr, parts, extra=()):
    vis, _ = register(bgr)
    poly = np.array([world_to_pix(x, y) for x, y in STRIP], np.int32)
    cv2.polylines(vis, [poly], True, (0, 200, 0), 1)
    for i, p in enumerate(parts):
        x, y, w, h = p["bbox"]
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 1)
        cv2.putText(vis, f"{i}: {p['x']:.0f},{p['y']:.0f}", (x, max(10, y - 3)), 0, 0.4, (0, 0, 255), 1)
    for label, (wx, wy) in extra:
        u, v = world_to_pix(wx, wy)
        cv2.drawMarker(vis, (int(u), int(v)), (255, 0, 0), cv2.MARKER_CROSS, 12, 2)
        cv2.putText(vis, label, (int(u) + 4, int(v) - 4), 0, 0.4, (255, 0, 0), 1)
    return vis
