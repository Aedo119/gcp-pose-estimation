"""
Shared constants and coordinate-transform utilities.

Coordinate spaces (see DECISION_LOG.md, entries 2 and 9):
  - "native"   : original image pixel coordinates (e.g. up to ~4096 x ~3068).
  - "crop"     : coordinates within the CROP_HALF*2 square crop centered on
                 (cx, cy), i.e. range [0, CROP_SIZE).
  - "model"    : coordinates within the resized model-input square
                 (range [0, MODEL_INPUT_SIZE)).

Transform chain for training: native -> crop -> model
Transform chain for inference: model prediction -> crop -> native
"""

from typing import Tuple

# --- Crop / model input geometry (DECISION_LOG.md entry 9) ---
CROP_HALF = 300                  # half-width of the native-resolution crop
CROP_SIZE = CROP_HALF * 2        # 600
MODEL_INPUT_SIZE = 224           # resized model input (square)
RESIZE_SCALE = MODEL_INPUT_SIZE / CROP_SIZE  # ~0.3733

SHAPE_CLASSES = ["Cross", "Square", "L-Shape"]
SHAPE_TO_IDX = {s: i for i, s in enumerate(SHAPE_CLASSES)}
IDX_TO_SHAPE = {i: s for s, i in SHAPE_TO_IDX.items()}


def get_crop_box(cx: float, cy: float, crop_half: int = CROP_HALF) -> Tuple[int, int, int, int]:
    """Return the (left, top, right, bottom) crop box for a center point.

    Note: this box may extend outside the source image's bounds. PIL's
    Image.crop() handles this by zero-padding automatically (verified in
    DECISION_LOG.md entry 11) — no special-casing needed here.
    """
    left = int(round(cx - crop_half))
    top = int(round(cy - crop_half))
    right = int(round(cx + crop_half))
    bottom = int(round(cy + crop_half))
    return left, top, right, bottom


def native_to_crop(x: float, y: float, left: int, top: int) -> Tuple[float, float]:
    """Convert native-image coordinates to crop-local coordinates."""
    return x - left, y - top


def crop_to_model(x: float, y: float, resize_scale: float = RESIZE_SCALE) -> Tuple[float, float]:
    """Convert crop-local coordinates (range [0, CROP_SIZE)) to model-input
    coordinates (range [0, MODEL_INPUT_SIZE))."""
    return x * resize_scale, y * resize_scale


def model_to_crop(x: float, y: float, resize_scale: float = RESIZE_SCALE) -> Tuple[float, float]:
    """Inverse of crop_to_model."""
    return x / resize_scale, y / resize_scale


def crop_to_native(x: float, y: float, left: int, top: int) -> Tuple[float, float]:
    """Inverse of native_to_crop."""
    return x + left, y + top


def model_to_native(x: float, y: float, left: int, top: int,
                     resize_scale: float = RESIZE_SCALE) -> Tuple[float, float]:
    """Full inverse chain: model-input prediction -> native image coordinates.

    Used in inference.py to convert the model's predicted (x, y) (in
    [0, MODEL_INPUT_SIZE) space) back to the original image's pixel space for
    predictions.json.
    """
    cx, cy = model_to_crop(x, y, resize_scale)
    return crop_to_native(cx, cy, left, top)


def normalize_target(x: float, y: float,
                      size: int = MODEL_INPUT_SIZE) -> Tuple[float, float]:
    """Normalize model-input coordinates to [0, 1] for regression training
    (stabilizes loss scale across the keypoint head)."""
    return x / size, y / size


def denormalize_target(nx: float, ny: float,
                        size: int = MODEL_INPUT_SIZE) -> Tuple[float, float]:
    """Inverse of normalize_target."""
    return nx * size, ny * size