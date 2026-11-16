"""Second opinion from Claude (Opus 5.5) on each anomaly decision.

Runs in a background thread while the arm picks the part, so it adds little or no time.
Any failure (no key, network, timeout, bad reply) returns None and the v3b decision stands.
"""
import asyncio
import base64
import json
import os
import time
import urllib.request

MODEL = os.environ.get("REVIEW_MODEL", "claude-opus-5-5")
MODE = os.environ.get("AGENT_REVIEW", "off")          # off | all | band  (default off = fully autonomous)
BAND = (0.25, 0.65)                                     # "band" mode: only review scores in here
MIN_CONF = 0.7                                          # agent must be at least this sure to override
TIMEOUT_S = 20

PROMPT = """You are the quality inspector on a robot cell that sorts metal fasteners (screws, bolts).
Each part is either GOOD (a clean, normal, undamaged fastener of any size or length) or ANOMALOUS
(damaged or altered: wrapped in tape, bent, broken, crushed, painted/marked, rusted, missing head, etc.).
Size alone is not an anomaly; short, long, thin and small screws are all normal.

Image 1: wrist-camera view looking straight down on the part (paper background; black tape strips
and neighbouring parts may be visible at the edges, ignore those).
Image 2: the same part cropped and centred (what the anomaly model scored).

The PatchCore anomaly model scored this part {score:.3f} (threshold {thr:.2f}; at or above = anomalous),
so its decision is {decision}.

Judge the part in the centre yourself. Reply with JSON only, no other text:
{{"verdict": "good" or "anomalous", "confidence": 0.0-1.0, "reason": "<= 12 words"}}"""


def wanted(score):
    if MODE == "off" or not os.environ.get("ANTHROPIC_API_KEY"):
        return False
    return MODE == "all" or BAND[0] <= score <= BAND[1]


def _call(images, score, thr, decision):
    content = [{"type": "image", "source": {"type": "base64", "media_type": mt,
                                            "data": base64.b64encode(b).decode()}} for b, mt in images]
    content.append({"type": "text", "text": PROMPT.format(score=score, thr=thr, decision=decision)})
    body = json.dumps({"model": MODEL, "max_tokens": 2000,
                       "messages": [{"role": "user", "content": content}]}).encode()
    req = urllib.request.Request("https://api.anthropic.com/v1/messages", data=body, headers={
        "x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01",
        "content-type": "application/json"})
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
        reply = json.loads(r.read())
    text = "".join(c.get("text", "") for c in reply.get("content", []) if c.get("type") == "text").strip()
    text = text[text.find("{"): text.rfind("}") + 1]
    out = json.loads(text)
    verdict = str(out.get("verdict", "")).lower()
    if verdict not in ("good", "anomalous"):
        raise ValueError(f"bad verdict {out!r}")
    return {"verdict": verdict, "confidence": float(out.get("confidence", 0)),
            "reason": str(out.get("reason", ""))[:80], "seconds": time.monotonic() - t0}


def start(images, score, thr, decision):
    """Kick off the review in a thread; returns an asyncio Task (or None if not wanted)."""
    if not images or not wanted(score):
        return None
    return asyncio.ensure_future(asyncio.to_thread(_call, images, score, thr, decision))


async def result(task, wait_s=10):
    """Wait (at most wait_s more) for the review. None on any problem."""
    if task is None:
        return None
    try:
        return await asyncio.wait_for(task, wait_s)
    except Exception as e:
        print(f"  agent review unavailable ({type(e).__name__}: {e}); keeping the model decision")
        return None


def final(decision, score, review):
    """Final decision, confidence and a note for the log."""
    model_dec = "defective" if decision == "defective" else "good"
    if review is None:
        return model_dec, None, "model"
    agent_dec = "defective" if review["verdict"] == "anomalous" else "good"
    if agent_dec != model_dec and review["confidence"] >= MIN_CONF:
        return agent_dec, review["confidence"], "agent_override"
    return model_dec, None, "agent_agrees" if agent_dec == model_dec else "agent_unsure"


# ---- detection screening: is each candidate actually a fastener? ----------------------
SCREEN = os.environ.get("AGENT_SCREEN", "off") == "on"   # default off; set AGENT_SCREEN=on to enable
SCREEN_DROP_CONF = 0.6                                  # only drop a candidate the agent is this sure about

SCREEN_PROMPT = """These are numbered candidates from a wrist camera looking down at a white paper strip where a
robot picks metal fasteners. Each candidate is a PAIR: left = colour crop centred on what a detector flagged
(yellow cross = centre); right = depth height map of the same area (black = no depth reading,
dark blue = flat paper, cyan/yellow/red = something standing 3 to 15 mm above the paper).
Measured per candidate: {meta}
Use both: a real fastener is visible in colour AND usually stands up in depth. Small shiny screws can give
no depth (black or blue), so a clearly visible screw in colour still counts. Plain white paper in colour with
no height in depth is NOT an object.
For each crop, say whether a real fastener (screw, bolt, nut, washer; possibly wrapped in tape or
painted) is at or near the CENTRE of the crop. Paper folds, creases, pencil marks, shadows, tape
edges, glare, the gripper, empty paper and other non-fastener things are NOT objects.
Reply with JSON only:
{{"crops": [{{"id": 0, "fastener": true or false, "confidence": 0.0-1.0, "what": "<= 6 words"}}, ...]}}
There are {n} crops, ids 0 to {last}."""


def screen_enabled():
    return SCREEN and bool(os.environ.get("ANTHROPIC_API_KEY"))


def _screen_call(jpeg_bytes, n, meta=""):
    content = [{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                            "data": base64.b64encode(jpeg_bytes).decode()}},
               {"type": "text", "text": SCREEN_PROMPT.format(n=n, last=n - 1, meta=meta)}]
    body = json.dumps({"model": MODEL, "max_tokens": 3000,
                       "messages": [{"role": "user", "content": content}]}).encode()
    req = urllib.request.Request("https://api.anthropic.com/v1/messages", data=body, headers={
        "x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01",
        "content-type": "application/json"})
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
        reply = json.loads(r.read())
    text = "".join(c.get("text", "") for c in reply.get("content", []) if c.get("type") == "text")
    out = json.loads(text[text.find("{"): text.rfind("}") + 1])
    return {int(c["id"]): c for c in out.get("crops", [])}, time.monotonic() - t0


async def screen(crops, depths=None, metas=None, wait_s=15):
    """crops: list of BGR images (+ optional height-map images and text per crop).
    Returns {index: verdict dict} or {} on any failure."""
    import cv2
    import numpy as np
    if not crops or not screen_enabled():
        return {}
    tiles = []
    for i, c in enumerate(crops):
        t = cv2.resize(c, (200, 200))
        cv2.rectangle(t, (0, 0), (34, 24), (255, 255, 255), -1)
        cv2.putText(t, str(i), (4, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
        cv2.drawMarker(t, (100, 100), (0, 200, 255), cv2.MARKER_CROSS, 14, 1)
        d = depths[i] if depths and i < len(depths) and depths[i] is not None else np.zeros((200, 200, 3), np.uint8)
        d = cv2.resize(d, (200, 200), interpolation=cv2.INTER_NEAREST)
        cv2.putText(d, "depth", (4, 192), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        tiles.append(np.hstack([t, np.full((200, 4, 3), 255, np.uint8), d]))
    cols = min(2, len(tiles))
    while len(tiles) % cols:
        tiles.append(np.full((200, 404, 3), 255, np.uint8))
    rows = [np.hstack(tiles[r:r + cols]) for r in range(0, len(tiles), cols)]
    sheet = np.vstack([np.pad(r, ((3, 3), (3, 3), (0, 0)), constant_values=40) for r in rows])
    ok, buf = cv2.imencode(".jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    try:
        meta = "; ".join(f"{i}: {m}" for i, m in enumerate(metas or []))
        out, secs = await asyncio.wait_for(asyncio.to_thread(_screen_call, buf.tobytes(), len(crops), meta), wait_s)
        print(f"Agent screen: {len(crops)} candidate(s) checked in {secs:.1f}s")
        try:
            cv2.imwrite(os.path.join(os.path.dirname(os.path.abspath(__file__)), "screen_last.jpg"), sheet)
        except Exception:
            pass
        return out
    except Exception as e:
        print(f"Agent screen unavailable ({type(e).__name__}: {e}); keeping all candidates")
        return {}
