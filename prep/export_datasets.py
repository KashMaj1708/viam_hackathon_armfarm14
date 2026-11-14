"""Export selected hackathons org datasets with tags/bboxes via Viam SDK."""
from __future__ import annotations

import asyncio
import json
import os
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "dataset_raw"

DATASETS = [
    ("an-14-one", "6af77d7170d1366b452134f7"),
    ("an-14-two", "6af7b85bf28d8d238da10f0e"),
    ("sakshi_am14", "6af79b33f64790cdeab3f009"),
    ("armfarm12-fasteners-412mm", "6af7798e974a8b30767eb0f1"),
    ("fastener-anomaly", "6af5edb82deac4224da1b4cd"),
]


def _safe_id(bid: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", bid)[:80]


def _ext_from_mime(meta) -> str:
    mime = ""
    try:
        mime = meta.capture_metadata.mime_type or ""
    except Exception:
        pass
    if "png" in mime:
        return ".png"
    return ".jpg"


async def export_one(dc, name: str, ds_id: str) -> None:
    from viam.proto.app.data import Filter

    out = RAW / name
    out.mkdir(parents=True, exist_ok=True)
    meta_path = out / "meta.jsonl"
    existing = {p.name for p in out.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".png", ".jpeg"}}

    last = None
    tag_counts = Counter()
    bbox_counts = Counter()
    size_hist = Counter()
    n = 0
    mode = "a" if meta_path.exists() else "w"
    with open(meta_path, mode, encoding="utf-8") as mf:
        while True:
            # Viam requires limit=1 when include_binary_data=True
            try:
                data, count, last = await dc.binary_data_by_filter(
                    filter=Filter(dataset_id=ds_id),
                    limit=1,
                    last=last,
                    include_binary_data=True,
                )
            except TypeError:
                data, count, last = await dc.binary_data_by_filter(
                    Filter(dataset_id=ds_id), 1, last, include_binary_data=True
                )
            if not data:
                break
            for item in data:
                bid = getattr(item.metadata, "binary_data_id", None) or getattr(item.metadata, "id", None) or ""
                bid = str(bid)
                meta = item.metadata
                ext = _ext_from_mime(meta)
                fname = f"{_safe_id(bid) or f'img_{n}'}{ext}"
                fpath = out / fname
                tags = []
                try:
                    tags = list(meta.capture_metadata.tags or [])
                except Exception:
                    tags = []
                for t in tags:
                    tag_counts[t] += 1
                bboxes = []
                try:
                    anns = meta.annotations
                    for bb in (anns.bboxes if anns else []) or []:
                        lab = getattr(bb, "label", "") or ""
                        bbox_counts[lab] += 1
                        bboxes.append({
                            "label": lab,
                            "x_min_normalized": getattr(bb, "x_min_normalized", None),
                            "y_min_normalized": getattr(bb, "y_min_normalized", None),
                            "x_max_normalized": getattr(bb, "x_max_normalized", None),
                            "y_max_normalized": getattr(bb, "y_max_normalized", None),
                        })
                except Exception:
                    pass
                w = h = None
                try:
                    w = meta.capture_metadata.width_px
                    h = meta.capture_metadata.height_px
                except Exception:
                    pass
                if w and h:
                    size_hist[f"{w}x{h}"] += 1
                if fname not in existing:
                    raw = item.binary or b""
                    if raw:
                        fpath.write_bytes(raw)
                        existing.add(fname)
                row = {
                    "file": fname,
                    "binary_id": bid,
                    "tags": tags,
                    "bboxes": bboxes,
                    "component_name": getattr(getattr(meta, "capture_metadata", None), "component_name", None),
                    "time_requested": str(getattr(getattr(meta, "capture_metadata", None), "time_requested", None)),
                    "width": w,
                    "height": h,
                }
                mf.write(json.dumps(row) + "\n")
                n += 1
                if n % 25 == 0:
                    print(f"  ... {name}: {n}")
            if last is None:
                break
    print(f"\n{name} ({ds_id}): wrote/seen {n} rows in {out}")
    print(f"  files on disk: {len(list(out.glob('*.jpg')) + list(out.glob('*.png')))}")
    print(f"  tags: {dict(tag_counts)}")
    print(f"  bbox labels: {dict(bbox_counts)}")
    print(f"  sizes: {dict(size_hist)}")


async def main():
    from viam.app.viam_client import ViamClient
    from viam.rpc.dial import DialOptions

    vc = await ViamClient.create_from_dial_options(DialOptions.with_api_key(
        os.environ["VIAM_API_KEY"], os.environ["VIAM_API_KEY_ID"]))
    dc = vc.data_client
    try:
        for name, ds_id in DATASETS:
            try:
                await export_one(dc, name, ds_id)
            except Exception as e:
                print(f"FAILED {name}: {e}")
                print("  Tip: install Viam CLI and run: viam dataset export --destination=dataset_raw/"
                      f"{name} --dataset-id={ds_id}")
    finally:
        vc.close()


if __name__ == "__main__":
    asyncio.run(main())
