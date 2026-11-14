"""Upload a prepared_* folder to a Viam dataset (create if missing)."""
from __future__ import annotations

import argparse
import asyncio
import os
from datetime import datetime, timezone
from pathlib import Path

PART_ID = "49d63d4f-4191-4d94-9e43-e379f846dd0e"
ORG_ID = "6d6c7293-bc67-43e4-9c8b-31c76aac27d3"
INSPECT_CAM = "cam-inspect"


async def ensure_dataset(dc, name: str) -> str:
    existing = await dc.list_datasets_by_organization_id(ORG_ID)
    for ds in existing:
        if getattr(ds, "name", None) == name:
            return ds.id
    ds_id = await dc.create_dataset(name, ORG_ID)
    print(f"created dataset {name} -> {ds_id}")
    return ds_id if isinstance(ds_id, str) else getattr(ds_id, "id", ds_id)


async def upload(folder: Path, dataset_name: str, include_val_normal: bool) -> None:
    from viam.app.viam_client import ViamClient
    from viam.rpc.dial import DialOptions

    vc = await ViamClient.create_from_dial_options(DialOptions.with_api_key(
        os.environ["VIAM_API_KEY"], os.environ["VIAM_API_KEY_ID"]))
    dc = vc.data_client
    ds_id = await ensure_dataset(dc, dataset_name)
    print(f"dataset {dataset_name} id={ds_id}")

    ledger = folder / "uploaded.txt"
    done = set(ledger.read_text().split()) if ledger.exists() else set()
    jobs = [
        (folder / "train" / "normal", ["normal"]),
        (folder / "val" / "anomalous", ["anomalous"]),
    ]
    if include_val_normal:
        jobs.append((folder / "val" / "normal", ["val_normal"]))

    n = 0
    with open(ledger, "a", encoding="utf-8") as lf:
        for directory, tags in jobs:
            if not directory.is_dir():
                continue
            for path in sorted(directory.glob("*.png")):
                key = f"{tags[0]}/{path.name}"
                if key in done or path.name in done:
                    continue
                data = path.read_bytes()
                t = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
                await dc.binary_data_capture_upload(
                    binary_data=data, part_id=PART_ID,
                    component_type="rdk:component:camera", component_name=INSPECT_CAM,
                    method_name="GetImages", file_extension=".png",
                    tags=tags, dataset_ids=[ds_id], data_request_times=(t, t))
                lf.write(key + "\n")
                lf.flush()
                n += 1
                if n % 25 == 0:
                    print(f"  uploaded {n}...")
    vc.close()
    print(f"Uploaded {n} new image(s) to {dataset_name} ({ds_id})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", default="prepared_v3")
    ap.add_argument("--dataset-name", default="kashyap-armfarm14-v3")
    ap.add_argument("--include-val-normal", action="store_true")
    args = ap.parse_args()
    folder = Path(args.folder)
    if not folder.is_absolute():
        folder = Path(__file__).resolve().parent.parent / folder
    asyncio.run(upload(folder, args.dataset_name, args.include_val_normal))


if __name__ == "__main__":
    main()
