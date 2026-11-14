# Cursor prompt 2: pull other teams' fastener datasets, filter, and fold them into prepared_v3

Run this AFTER prompt 1 (it reuses `prep/preprocess.py` and the `prepared_v3` layout). Same folder, venv and keys (`. .\viam_keys.ps1`). Org: hackathons, id `6d6c7293-bc67-43e4-9c8b-31c76aac27d3`.

## Why only these datasets

Surveyed every dataset in the hackathons org. PatchCore stores normal patches in a memory bank, so normals from a different background (orange, blue) dilute it and make dark tape look less unusual. Only light paper / light grey backgrounds with a top down camera are worth pulling.

| dataset | id | files / tags | background, view | use |
|---|---|---|---|---|
| an-14-one | `6af77d7170d1366b452134f7` | 106, all `normal` | SAME station (armfarm14), white paper, tape strip, single small bolt | train normal (primary) |
| an-14-two | `6af7b85bf28d8d238da10f0e` | 3, `normal` | same station, close ups (long bolt, blue socket bolt) | train normal |
| sakshi_am14 | `6af79b33f64790cdeab3f009` | 106, untagged | light paper, black tape wrapped bolts (matches our anomaly) | val anomalous only, after visual check |
| armfarm12-fasteners-412mm | `6af7798e974a8b30767eb0f1` | 102 `normal`, 101 `anomalous` | light grey, single part, part small in frame | candidate: normals for train B, anomalous for val only if they are tape type |
| fastener-anomaly | `6af5edb82deac4224da1b4cd` | 252; bbox labels `normal`, `anomaly`, `normal-bolt` | light grey, several parts per frame | candidate: crop `normal` / `normal-bolt` bboxes into single part images |

Skip: armfarm10-bolts / -v3 / -aligned and armfarm12-anomalydetection-dataset (orange background), fasteners-farm09 and -merged (blue background sessions), armfarm02-* (dark grey small crops, two of them empty), sortit-armfarm01-v1 (yellow blocks), an-14 (overview camera, many parts per frame), everything Sequence type.

Note: browser downloads of an-14-one and sakshi_am14 are already in `C:\Users\kashy\Downloads` as `.tar.gz`, but browser exports drop tags and bboxes. Use the SDK export below for everything; the tarballs are only a fallback.

## Step 1: `prep/export_datasets.py`

For each dataset id above, export with metadata using the Viam Python SDK:

```python
from viam.app.viam_client import ViamClient
from viam.rpc.dial import DialOptions
from viam.proto.app.data import Filter
# ViamClient.create_from_dial_options(DialOptions.with_api_key(api_key=..., api_key_id=...))
data, count, last = await dc.binary_data_by_filter(
    filter=Filter(dataset_id=DS_ID), limit=50, last=last, include_binary_data=True)
```

Page with `last` until empty. For each item save
`dataset_raw/<dataset_name>/<binary_id_sanitised>.<ext>` and one row in `dataset_raw/<dataset_name>/meta.jsonl` with: file, binary id, `metadata.capture_metadata.tags`, `metadata.annotations.bboxes` (label, x_min_normalized, y_min_normalized, x_max_normalized, y_max_normalized), component name, robot/location name, time_requested, width/height. Resume safely (skip files that exist). If `binary_data_by_filter` with `dataset_id` is not supported in the installed SDK version, use `viam dataset export --destination=dataset_raw/<name> --dataset-id=<id>` from the Viam CLI (install it per docs.viam.com CLI page) which writes images plus label JSON.

Print per dataset: file count, tag counts, bbox label counts, image size histogram.

## Step 2: `prep/ingest_external.py`

Convert each external source into single part candidates, then run the SAME `preprocess()` and QC from prompt 1:

- **an-14-one, an-14-two**: whole frame → `preprocess()` → label `normal`, source `ext_an14`.
- **armfarm12-fasteners-412mm**: whole frame → `preprocess()` (the object centred crop handles the small part). Upscaling is large here, so reject crops whose part bbox long side < 40 px (too little detail). Label from tag. Source `ext_af12`.
- **fastener-anomaly**: for every bbox labelled `normal` or `normal-bolt`, crop the bbox with 35% margin from the full frame (pad with paper colour), then `preprocess()`. Skip bboxes labelled `anomaly`, `1`, `2` for training. Source `ext_fa`.
- **sakshi_am14**: whole frame → `preprocess()` → label `anomalous_candidate`. Do not auto label.

Write `prepared_ext/<source>/<label>/*.png` and `prepared_ext/manifest.csv` (same columns as prompt 1 plus `ext_dataset`). Make one contact sheet per source and label (`prepared_ext/contact_<source>_<label>.jpg`, 12 columns, 96 px thumbs).

## Step 3: human gate (stop and ask me)

Print the contact sheet paths and STOP. I will check:
- sakshi_am14: which images really show black tape on the part. I will list rejects; move the rest into `val/anomalous`.
- armfarm12 `anomalous`: what their anomaly is. Only tape wrapped ones go to val; anything else (bent, missing thread, wrong part) is ignored because it is not our defect.
- Any external normal that shows dirt, odd background or two parts.

## Step 4: build two training variants

- **v3a (own only)**: exactly `prepared_v3` from prompt 1.
- **v3b (own + same station)**: v3a train normals + `ext_an14` normals (augmented the same way as prompt 1, step 5).
- **v3c (own + same station + other light background)**: v3b + `ext_af12` and `ext_fa` normals, capped so external images are at most 35% of the train normals.

Validation set is shared and fixed for all three: own `val/normal`, own `val/anomalous`, approved sakshi anomalies. Never put an external image of a group in both train and val.

Upload each variant to its own dataset with `prep/upload_prepared.py` (add a `--dataset-name` argument): `kashyap-armfarm14-v3a`, `kashyap-armfarm14-v3b`, `kashyap-armfarm14-v3c`. Print the dataset ids.

## Step 5: compare (after I train the three models in the Viam UI)

Training script `hackathons:anomalib-train` (PatchCore), models named `kashyap-fastener-closeup-v3a` / `v3b` / `v3c`. For each, deploy into `ov-anomaly` one at a time (or as three mlmodel services if GPU memory allows), then run `prep/calibrate.py` from prompt 1 on the shared val set. Produce `prep/compare.md` with: AUROC, normal score range, anomalous score range, best balanced accuracy threshold, false negatives at that threshold, per screw type normal score max. Pick the variant with the highest AUROC; tie break on the widest gap between max normal and min anomalous.

## Rules

- Never delete files; move to `_excluded_*`.
- Do not modify the Viam machine config from these scripts; I switch models myself.
- Fixed seeds (14) everywhere.
- Keep everything idempotent so reruns resume.
