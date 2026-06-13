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

**Note on class proportions:** Cross is slightly underrepresented in val (12.8% vs
18.6% in train) relative to train. Accepted as-is given group-level constraints —
re-seeding or adjusting `VAL_FRACTION` could rebalance further if needed, but not
considered necessary at this dataset size.

---

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
within a 600-1000px crop window. One sampled marker was barely visible even at
half_size=500.

**Decision:** Crop half-size = **300px** (600x600 crop) selected as the working
crop size — large enough to include surrounding context for shape classification,
small enough that the marker isn't reduced to a handful of pixels after resize.
Crop is resized to 224x224 for the model input (resize factor ~0.373x). Both the
crop offset `(x-300, y-300)` and the resize factor are tracked so predicted
224x224-space coordinates can be mapped back to native image coordinates for
`predictions.json`.

---

## 10. Edge-Clipping / Crop Padding

See entry #11 — resolved.

---

## 11. Edge-Clipping / Crop Padding (resolved)

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
  the keypoint label correctly fixed at `(300, 300)` in crop-local coordinates
  for every sample, including the 211 edge cases.
- **Augmentation caveat:** if random crop-center jitter is used as an
  augmentation, jitter magnitude must be bounded per-sample
  (`max_jitter = min(x, w-x, y, h-y, base_jitter)`) so the marker never falls
  outside the resulting crop frame. This bound is computed per-sample at dataset
  construction time, not globally.
- README will state: "21.2% of training crops contain black zero-padded borders
  due to marker proximity to the source image edge (confirmed via PIL `.crop()`
  auto-padding behavior); keypoint labels remain correct in all cases."

---



EDA phase complete. Proceeding to dataset class / model architecture / training
pipeline design.