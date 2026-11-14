# Cursor prompt 1: clean, standardise and augment the armfarm14 anomaly training set

You are working in `C:\Users\kashy\Desktop\viam_hackathon` (Windows, PowerShell, Python venv in `.venv`, Viam SDK installed there, API keys loaded by dot-sourcing `viam_keys.ps1`). Do not touch `track3_sort.py` motion code. Do not delete any image: anything rejected is MOVED into a `_excluded_*` folder so it can be restored.

## Context

- Task: Viam hackathon Track 3. A wrist RealSense camera (`cam-inspect`, a 400x400 centre crop of `cam`) looks down at ONE fastener on white/light grey paper. A PatchCore model (anomalib, trained by the `hackathons:anomalib-train` script on a Viam dataset tagged `normal` / `anomalous`) scores the frame. Normal = clean bolt/screw. Anomalous = part wrapped in BLACK TAPE.
- Inference today: `track3_sort.py` → `inspect()` calls `anomaly.get_classifications_from_camera("cam-inspect", 1)` and compares the score to `THRESHOLD = 0.42`.
- Image sources (all 400x400 RGB JPEG):
  - `train_images/good/*.jpg` (248): name pattern `good_<HHMMSS round>_<item>_<shot>.jpg`, 6 shots per item.
  - `train_images/bad/*.jpg` (60): same pattern, black tape wrapped parts.
  - `dataset_raw/kashyap-armfarm14-clean/*.jpg` (132, exported from Viam, no labels). 107 are byte-identical copies of files in `train_images` (match by MD5). The other 25 are manual captures that only exist here: treat them as `good` candidates EXCEPT the two from `2026-11-13T23_33_19` and `23_33_20` (cable + dark object in frame, move to `_excluded_unsure`).
  - Already quarantined, leave alone: `_excluded_empty`, `_excluded_notapart`, `_excluded_unsure`, `_rejected_offcentre`, `_rejected_tooclose`.

## Known problems in the current data (found by visual review)

1. A dark tape strip (the taped workspace border) sits along the left or right edge of most frames, 0 to 30% of the width. Its width varies frame to frame. This is the single biggest nuisance: the model learns "dark bar" as normal, and black tape IS the anomaly.
2. Partial parts: long bolts (about 140 mm) cut off by the frame edge or showing only a corner.
3. Neighbouring parts poking into the frame.
4. A black USB cable lying across the paper in round 183255.
5. Lighting and white balance drift between rounds (paper ranges from warm grey to cool white).
6. 6 shots per item are near duplicates, so random image level splits leak.

Manual exclusions (move these exact globs before any automatic QC):

| glob | move to | reason |
|---|---|---|
| `train_images/good/good_183255_*` | `_excluded_unsure` | cable in frame; `183255_3_*` shows a black wrapped object labelled good |
| `train_images/good/good_165858_1_*` | `_rejected_offcentre` | bolt cut at left edge |
| `train_images/good/good_173817_0_*` | `_rejected_offcentre` | bolt only in the corner |
| `train_images/good/good_174526_0_*` | `_rejected_offcentre` | bolt cut at the corner |
| `train_images/good/good_201319_0_*` | `_rejected_offcentre` | head cut at top edge |
| `train_images/good/good_200301_1_*` | `_rejected_offcentre` | two parts in frame |
| `train_images/good/good_203208_1_*` | `_rejected_offcentre` | shank only, no head |
| `train_images/bad/bad_175110_0_*` | `_rejected_offcentre` | part mostly out of frame |

## What to build

Create a package folder `prep/` with:

### `prep/preprocess.py` (shared by training AND inference, this is the important part)

Pure numpy + OpenCV (`pip install opencv-python-headless numpy` into `.venv`). One public function:

```python
def preprocess(img_rgb: np.ndarray, out_size: int = 256) -> tuple[np.ndarray, dict]:
    """Returns the cleaned, object centred square image and a debug dict
    (tape_frac, n_objects, bbox, touches_border, blur_var, fg_frac)."""
```

Steps, in order:

1. **Paper estimate**: convert to LAB. Paper colour = median LAB of pixels with L above the 60th percentile.
2. **Tape mask**: pixels with L < 60 (on 0..255 scale) that belong to a connected component which touches the LEFT or RIGHT image border AND whose height spans at least 40% of the image. Dilate 5 px. Do NOT mask dark blobs that do not touch the side borders, those can be the black tape anomaly on the part itself.
3. **Tape fill**: `cv2.inpaint(img, tape_mask, 7, cv2.INPAINT_TELEA)`, then replace the inpainted area with the paper colour plus Gaussian noise (sigma 2) so no smeared texture remains. Check that the bad images still keep the tape that is wrapped on the part.
4. **White balance / exposure**: scale each channel so the paper colour maps to a fixed target (LAB L=200, a=128, b=128). This kills round to round lighting drift.
5. **Foreground mask**: delta E (CIE76) to paper colour > 18 OR L < 120, then morphological open 3 px, close 7 px. Ignore components smaller than 150 px.
6. **Object pick**: largest component = the part. Record `n_objects` = count of other components with area at least 15% of the largest.
7. **Neighbour suppression**: paint every other component (dilated 6 px) with the paper colour.
8. **Object centred crop**: square crop centred on the part's bbox centre, side = max(bbox_w, bbox_h) * 1.35, min 160 px, clamped to the image (pad with paper colour, never reflect). Resize to `out_size` with `INTER_AREA`.
9. Debug dict: `tape_frac`, `n_objects`, `bbox`, `touches_border` (part bbox within 3 px of any image edge), `blur_var` (variance of Laplacian on the grey crop), `fg_frac`.

Write unit style checks in `prep/test_preprocess.py` with 4 real files: one good with a wide tape strip, one bad (tape on the part must survive), one small screw, one long bolt. Save before/after panels to `prep/_debug/` and print the dict.

### `prep/build_dataset.py`

CLI: `python -m prep.build_dataset --out prepared_v3 [--dry-run]`

1. Apply the manual exclusion table above (moves, not deletes; log every move to `prep/moves.log`; idempotent).
2. Gather candidates: `train_images/good`, `train_images/bad`, plus the 23 unmatched manual captures from `dataset_raw/kashyap-armfarm14-clean` (MD5 against everything under `train_images/**` to find them; label `good`, source `viam_export`).
3. Run `preprocess()` on each. Automatic QC, move failures out of the prepared set (originals stay put; just record in the manifest with a `reject_reason`):
   - `fg_frac < 0.002` → `empty`
   - `touches_border` and the part is long (bbox long side > 0.8 * 400) → `partial`
   - `n_objects >= 1` after suppression still overlaps the crop → `multi`
   - `blur_var` below 0.5 * median blur_var of its label → `blurry`
   - near duplicates: dHash (8x8) Hamming distance <= 3 within the same `round_item` group → keep the first 4 per group only
4. Group key for splits: `round_item` (e.g. `200456_1`); for the viam_export files use the capture minute. Split by group, never by image: 80% train, 20% val, stratified so each of the 5 screw types (long hex bolt, short hex bolt, small hex bolt, thin socket screw, big blue socket bolt) appears in val. If type is unknown, cluster on bbox long side and aspect ratio into 5 bins and stratify on that.
5. **Augmentation (train split, good only)**. Bad images are NEVER augmented and NEVER used for training (PatchCore trains on normals only; bad is validation). For each train good image write:
   - original
   - rot90, rot180, rot270 (`np.rot90`, lossless)
   - horizontal flip, vertical flip
   - 2 random small rotations in ±20° using `cv2.warpAffine` with `borderValue` = target paper colour (never reflect, never black)
   - 1 photometric variant: brightness ×U(0.92, 1.08), gamma U(0.9, 1.1)
   Cap the final normal train set at 900 images (random sample with fixed seed 14) so PatchCore coreset training stays fast. Because tape strips are removed in step 3 of preprocess, rotations no longer move a dark bar into new positions; assert that `tape_frac_after < 0.002` for every augmented source image.
6. Output layout:
   ```
   prepared_v3/
     train/normal/*.png
     val/normal/*.png
     val/anomalous/*.png
     manifest.csv   (src_path, md5, label, source, group, split, aug, reject_reason, tape_frac, n_objects, blur_var, bbox)
     contact_train.jpg, contact_val.jpg, contact_rejected.jpg   (12 columns, 96 px thumbs, filename stem in red)
   ```
   Save PNG to avoid a second JPEG generation.
7. Print a summary table: counts per split/label/source, rejects per reason.

### `prep/upload_prepared.py`

Upload `prepared_v3` to a NEW Viam dataset named `kashyap-armfarm14-v3` in the hackathons org (org id `6d6c7293-bc67-43e4-9c8b-31c76aac27d3`). Reuse the exact upload pattern from `upload_saved()` in `track3_sort.py`: `ViamClient` from env keys, `data_client.binary_data_capture_upload(...)` with `part_id="49d63d4f-4191-4d94-9e43-e379f846dd0e"`, `component_type="camera"`, `component_name="cam-inspect"`, `file_extension=".png"`, `tags=["normal"]` or `["anomalous"]`, `dataset_ids=[<new id>]`, and `data_request_times=(t_req, t_rec)` (required, uploads fail without it). Create the dataset with `data_client.create_dataset(name, org_id)` if it does not exist. Record uploaded md5s in `prepared_v3/uploaded.txt` so reruns skip them. Upload train normals and val anomalous only for training; keep `val/normal` local for threshold calibration (or tag them `val_normal` too if the trainer needs them, check the train script options first).

### Inference parity (do not skip)

The model will be trained on cleaned, object centred, white balanced crops, so inference MUST see the same thing. In `track3_sort.py`:

- Add `from prep.preprocess import preprocess`.
- In `inspect()`, replace `anomaly.get_classifications_from_camera(INSPECT_CAM, 1)` with: grab a frame from the `cam-inspect` camera (`CameraClient(INSPECT_CAM, machine._channel).get_image()`; keep the existing `steady_frame` settle logic), convert to numpy, `preprocess()`, encode PNG, build `ViamImage(data, CameraMimeType.PNG)`, call `anomaly.get_classifications(image, 1)`. Average over `N_FRAMES` like today.
- First verify the `viam-labs:anomalib:vision` service accepts `GetClassifications` with an image (try one call in a tiny script `prep/check_service.py`). If it does not, fall back to `ov-anomaly` mlmodel `infer()` directly with the preprocessed tensor (check `metadata()` for input name, shape, dtype and normalisation) and read the anomaly score from the output.
- Keep a flag `PREPROCESS_AT_INFERENCE = True` so the old path can be restored instantly.

### Threshold calibration script `prep/calibrate.py`

After the new model is deployed: run every image in `prepared_v3/val/*` through the same inference path (service `get_classifications` on the preprocessed image), write `prepared_v3/val_scores.csv`, print AUROC, the score ranges per class, and the threshold that maximises balanced accuracy, plus the threshold at 0 false negatives. Suggest the value for `THRESHOLD` but do not edit it automatically.

## Acceptance checks

- `python -m prep.test_preprocess` writes debug panels; tape strip gone in goods, tape on the part still visible in bads.
- `python -m prep.build_dataset --out prepared_v3 --dry-run` prints the summary without writing.
- Real run produces at least 120 unique train normal sources (before augmentation) covering all 5 screw types; contact sheets look clean (one centred part per image, no dark side bars).
- No file under `train_images` deleted; `prep/moves.log` lists every move.
