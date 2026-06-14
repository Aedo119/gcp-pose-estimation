# Decision Log — Aerial GCP Pose Estimation

This log records key findings from EDA and the corresponding decisions/assumptions
made, in chronological order. Each entry is intended to map directly to the
"Challenges & Mitigations" section of the final README.

---

## 1. Label Quality

**Finding:** 4 entries in `gcp_marks.json` (1000 total entries) were
malformed/missing required keys (`mark` or `verified_shape`).

**Decision:** Dropped these 4 entries from the working dataset.
**1000 → 996 valid labels** used going forward.

**Note:** the labels file is named `gcp_marks.json`, not
`curated_gcp_marks.json` as assumed in the original notebook draft — corrected
in the EDA notebook's `LABELS_PATH`.

---

## 2. Image Resolution Discrepancy

**Finding:** The assignment spec states images are 2048x1365. Actual images on disk
are 4096px wide, at two heights: (4096, 2730) — 639 images, and (4096, 3068) — 357
images. No EXIF orientation tags present (all `None`), so no rotation correction
needed.

**Decision:** Work in native pixel coordinates throughout. The spec's 2048x1365
figure is treated as informational/approximate and not authoritative — likely
describes a downscaled reference resolution. Keypoint coordinates from
`curated_gcp_marks.json` are assumed to already be in native (4096-wide) pixel
space, consistent with the values observed (e.g. x up to ~4000). All
crop-then-resize transforms will be tracked explicitly so predictions can be
mapped back to native coordinates for `predictions.json`.

**Two distinct heights (2730 vs 3068) require no special handling** for crop-based
training — crop generation only depends on local bounds around each keypoint, not
total image dimensions.

**Additional finding:** the two sizes correspond to two distinct aspect ratios —
1.5 (4096x2730, 639 images) and 1.34 (4096x3068, 357 images) — indicating two
different camera/drone sources across the surveys. Both are landscape orientation;
no rotation/transpose handling is needed.

---

## 3. Image File Integrity

**Finding:** 0 unreadable/corrupted images out of 992 valid-label entries.

**Decision:** No corruption handling needed. Noted in README as a check performed,
not as a mitigation required.

---

## 4. Duplicate Detection

**Finding:** 0 byte-identical duplicate images found via MD5 hashing across all
valid-label images.

**Decision:** No deduplication step needed. Near-duplicates (different frames of
the same physical marker from a flight pass) do exist and are handled via the
group-aware split (see #6), not via deduplication.

---

## 5. Coordinate Validity

**Finding:** 0 out-of-bounds keypoints — all annotated `(x, y)` marks fall within
their respective image dimensions.

**Decision:** No filtering needed on this basis. Annotations are trustworthy with
respect to image bounds.

---

## 6. GCP Group Diversity & Train/Val Split Strategy

**Finding:**
- 159 distinct physical GCP groups (grouped by `project/survey/gcp_id`, since
  `gcp_id` names like "GCP12" repeat across different surveys and refer to
  different physical markers).
- Group size distribution: mode = 8 images/group (65 groups), with sizes ranging
  from 1 to 12 images per group.
- 0 groups have inconsistent `verified_shape` across their images — confirms
  `verified_shape` is a stable property of the physical marker.

**Decision:** Split at the **group level**, not image level, to prevent leakage
(multiple views of the same physical marker must not appear in both train and val).
Used a stratified-by-shape greedy split targeting 15% val fraction, seed=42.

**Result:**
- Train: 855 images, 136 groups
  - Class distribution: Cross 159, Square 278, L-Shape 418
- Val: 141 images, 23 groups
  - Class distribution: Cross 18, Square 50, L-Shape 73


## 7. Class Distribution (Shape Classification)

**Finding:** Overall class counts — Cross 177, Square 328, L-Shape 491 (from
original full-dataset count before split). Ratio approximately 1 : 1.85 : 2.77.

**Decision:** Mild imbalance. Plan to use class-weighted cross-entropy loss for the
classification head (weights inverse-proportional to class frequency), and report
macro-F1 (as per evaluation criteria) to ensure minority class (Cross) performance
isn't masked by majority-class accuracy.

---

## 8. Keypoint Spatial Distribution

**Finding:** Keypoints are spread roughly uniformly across the full image frame —
no strong central clustering.

**Decision:** Full-frame center-cropping is not viable. Crops must be generated
per-marker, centered on (or near) the annotated `(x, y)` coordinate for each
training sample.

---

## 9. Marker Scale / Crop Size

**Finding:** Visual inspection across candidate crop half-sizes (150 / 300 / 500px)
performed on sample images across all three shape classes. The physical markers
are **small relative to the crop** — typically only tens of pixels across, even
within a 600-1000px crop window. 

**Decision:** Crop half-size = **300px** (600x600 crop) selected as the working
crop size — large enough to include surrounding context for shape classification.


---


## 10. Edge-Clipping / Crop Padding (resolved)

**Finding:** At a crop half-size of 300px (600x600 crop), **211 / 996 (21.2%)**
of training samples have markers within 300px of at least one image border,
meaning the crop window extends beyond the source image.

**Verification:** Tested `PIL.Image.crop()` directly on an edge-case sample
(`x=171.8`, half_size=300 → left=-128.2). PIL's `.crop()` automatically returns a
600x600 image with the out-of-bounds region filled as a black (zero) box — no
exception, no size mismatch.

**Decision:**
- No custom padding logic is required in the dataset class. `img.crop((x-300,
  y-300, x+300, y+300))` is sufficient and always returns a 600x600 image, with
  the keypoint label correctly fixed at `(300, 300)` in crop-local coordinates.
---

## 11. Val PCK Metric Was Trivially 1.0 (resolved)

**Finding:** Initial training run reported `val_pck@25px=1.0000` every epoch,
which appeared suspiciously perfect and was confirmed to be a metric bug, not
a genuine localization result.

**Root cause:** Val crops were centered exactly on the marker (`train=False`,
no jitter), so every val target keypoint was `(0.5, 0.5)` in normalized
model-input space — i.e., exactly `(112, 112)` in 224x224 pixel space. 

**Decision:** Apply **deterministic jitter** to validation samples (seeded by
sample `idx` via `np.random.default_rng(idx)`) so val keypoint targets vary
per-sample (and are consistent epoch-to-epoch for fair comparison). This makes
`val_pck@Xpx` a genuine localization metric: the model must predict where the
marker is within the crop, not just output "somewhere near center."

---

## 12. Inference Sliding-Window Scoring Improved

**Finding:** Initial inference used max classification confidence alone to
select the "marker-containing" window. On real test images this produced
poor localization — background windows that happened to look like one of the
three classes were selected over windows genuinely containing the marker.

**Root cause:** The model was trained exclusively on marker-centered crops,
so it expects the marker near `(0.5, 0.5)` of the crop. A window containing
the marker near its center will therefore predict both (a) high classification
confidence AND (b) a keypoint near `(0.5, 0.5)`. A background window may
achieve (a) but will predict a keypoint far from center (the model "points
toward" the nearest marker-like feature, which in a background window may be
at an arbitrary location).

**Decision:** Score each window by the product of classification confidence
and keypoint centrality:

```
centrality = 1.0 - clamp(||pred_kp - 0.5||, 0, 1)
score = confidence * centrality
```

The highest-scoring window per image is selected. This penalizes windows that
are confident about shape but predict a marker far from the window center —
consistent with the training distribution.

**Additionally:** default `window_stride` changed from 600 (non-overlapping)
to 300 (50% overlap). Non-overlapping windows at stride=600 give only ~35
windows per 4096x2730 image; with exactly one containing the marker, the
chance of that window having the marker near its center is low. 50% overlap
gives ~140 windows, greatly increasing the probability that at least one
window has the marker close to center (matching training distribution).

**Code change:** `src/inference.py` — scoring loop updated to compute and
use `combined_score`. `configs/config.yaml` — `window_stride` default updated
to 300.

## 13. Train/Inference Distribution Mismatch — Random Crop Placement (resolved)

**Finding:** Inference predictions were visually wrong — the predicted marker
position was far from the actual GCP marker on real test images (confirmed
visually). 

**Root cause:** The model was trained exclusively on crops centered on the
marker (`crop center = marker position ± small jitter`). the model was only ever trained to expect
the marker near `(0.5, 0.5)` of the crop. At inference it sees the marker
anywhere in the window and has no learned behavior for off-center markers.
**Decision:** Change the training crop strategy so the marker appears at a
**random position within the crop** during training (not just near center). This means during training the model sees the marker at every possible
position within the 600x600 crop.

**Secondary change:** the centrality term is removed from inference scoring
(entry 13). 

**Tertiary change:** default `window_stride` reduced from 300 to 150px,
giving ~530 windows per 4096x2730 image. 



## 14. Training Environment Constraint — CPU Only (no GPU available)

**Finding:** Colab free tier GPU quota exhausted mid-project. Training is
running on CPU only.

**Decisions made as a result:**
- `train.num_epochs` reduced from 30 to 15 (early stopping at patience=5
  will likely terminate earlier, around epoch 8-10).
- `train.batch_size` reduced from 32 to 16 to fit CPU memory comfortably.
- `train.num_workers` and `inference.num_workers` set to 0 (no
  multiprocessing — avoids Colab CPU DataLoader issues).
- `inference.window_stride` increased from 150 to 300 for inference (fewer
  windows per image, faster CPU inference — accuracy trade-off accepted
  given the constraint).
- `pin_memory=False` in DataLoaders (pin_memory has no effect without a GPU
  and generates warnings).

**Impact on results:** classification (macro-F1) converges quickly and should
still reach reasonable performance within 10-15 epochs. Keypoint regression
(PCK) may not fully converge but should show clear improvement over random.
Final numbers will reflect a CPU-constrained training run — documented here
and in README as an assumption.

---