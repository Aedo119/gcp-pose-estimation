"""
GCPDataset: loads aerial images, crops a fixed-size window around each GCP
marker, and returns (image_tensor, target) for joint keypoint regression +
shape classification.

Implements the decisions from DECISION_LOG.md:
  - Crop half-size = 300px (600x600), resized to 224x224 (entry 9)
  - PIL Image.crop() auto-pads out-of-bounds boxes with black (entry 11)
  - Train: crop center is randomly offset from the marker position so the
    marker can appear ANYWHERE in the crop, not just near center. This
    matches the inference distribution (sliding window), where a window
    containing a marker has the marker at an arbitrary position within the
    window (entry 14).
  - Val: same random placement but deterministic per sample (seeded by idx)
    so val PCK is a meaningful localization metric (entry 12).
  - Targets are normalized to [0, 1] in model-input space.
"""

import json
import os
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

from .utils import (
    CROP_HALF,
    MODEL_INPUT_SIZE,
    SHAPE_TO_IDX,
    get_crop_box,
    native_to_crop,
    crop_to_model,
    normalize_target,
)


def load_clean_labels(labels_path: str) -> Dict[str, dict]:
    """Load gcp_marks.json and drop malformed entries (DECISION_LOG entry 1)."""
    with open(labels_path) as f:
        raw = json.load(f)

    clean = {}
    for path, data in raw.items():
        if not isinstance(data, dict):
            continue
        if "mark" not in data or "verified_shape" not in data:
            continue
        mark = data["mark"]
        if "x" not in mark or "y" not in mark:
            continue
        clean[path] = data
    return clean


class GCPDataset(Dataset):
    """
    Each sample is a crop with the GCP marker at a RANDOM position within
    the crop (not necessarily centered). This matches the sliding-window
    inference distribution where a window containing a marker has the marker
    at an arbitrary location within the window (DECISION_LOG entry 14).

    Returns:
        image:       3 x MODEL_INPUT_SIZE x MODEL_INPUT_SIZE float tensor
        keypoint:    2-vector [norm_x, norm_y] in [0, 1]
        shape_label: int class index (0=Cross, 1=Square, 2=L-Shape)
        meta:        dict with path + crop box for coordinate back-transform
    """

    def __init__(
        self,
        data_root: str,
        labels: Dict[str, dict],
        paths: List[str],
        train: bool = True,
        crop_half: int = CROP_HALF,
        model_input_size: int = MODEL_INPUT_SIZE,
        marker_margin: int = 20,
    ):
        """
        Args:
            data_root:        root directory of the dataset.
            labels:           cleaned labels dict from load_clean_labels.
            paths:            image paths for this split.
            train:            if True, random crop placement each call;
                              if False, deterministic per sample (seeded by idx).
            crop_half:        half-width of the native crop window (px).
            model_input_size: size to resize the crop to for the model.
            marker_margin:    minimum distance (px in crop space) the marker
                              must be from the crop edge. Prevents the marker
                              from being cropped out during random placement.
                              Default 20px keeps the marker clearly visible.
        """
        self.data_root = data_root
        self.labels = labels
        self.paths = paths
        self.train = train
        self.crop_half = crop_half
        self.model_input_size = model_input_size
        self.marker_margin = marker_margin

        # Max offset of crop center from marker so marker stays >= margin
        # pixels from crop edge
        self.max_offset = crop_half - marker_margin

        self._normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        self._photometric = T.ColorJitter(
            brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05
        ) if train else None

    def __len__(self) -> int:
        return len(self.paths)

    def _get_offset(self, idx: int, x: float, y: float, w: int, h: int):
        """Return (offset_x, offset_y) to add to the marker position to get
        the crop center. The offset is bounded so:
          1. The marker stays >= marker_margin px from the crop edge.
          2. The crop center stays within the image (no excessive OOB padding).

        Train: random offset each call.
        Val:   deterministic offset seeded by idx (same every epoch).
        """
        # Bound by marker_margin constraint
        max_ox = self.max_offset
        max_oy = self.max_offset

        # Further bound by image edges so we don't create huge black regions
        # (keep crop center within the image)
        max_ox = min(max_ox, x, w - x)
        max_oy = min(max_oy, y, h - y)
        max_ox = max(0.0, float(max_ox))
        max_oy = max(0.0, float(max_oy))

        if self.train:
            ox = np.random.uniform(-max_ox, max_ox)
            oy = np.random.uniform(-max_oy, max_oy)
        else:
            rng = np.random.default_rng(idx)
            ox = rng.uniform(-max_ox, max_ox)
            oy = rng.uniform(-max_oy, max_oy)

        return ox, oy

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        data = self.labels[path]
        x, y = float(data["mark"]["x"]), float(data["mark"]["y"])
        shape_label = SHAPE_TO_IDX[data["verified_shape"]]

        full_path = os.path.join(self.data_root, path)
        img = Image.open(full_path).convert("RGB")
        w, h = img.size

        # Randomly offset the crop center from the marker (entry 14)
        ox, oy = self._get_offset(idx, x, y, w, h)
        cx, cy = x + ox, y + oy

        left, top, right, bottom = get_crop_box(cx, cy, self.crop_half)
        crop = img.crop((left, top, right, bottom))  # PIL auto-pads OOB

        # Marker position within crop, then in model-input space
        crop_x, crop_y = native_to_crop(x, y, left, top)
        crop = crop.resize((self.model_input_size, self.model_input_size), Image.BILINEAR)
        resize_scale = self.model_input_size / (2 * self.crop_half)
        model_x, model_y = crop_to_model(crop_x, crop_y, resize_scale)

        if self._photometric is not None:
            crop = self._photometric(crop)

        img_tensor = T.functional.to_tensor(crop)
        img_tensor = self._normalize(img_tensor)

        norm_x, norm_y = normalize_target(model_x, model_y, self.model_input_size)
        keypoint = torch.tensor([norm_x, norm_y], dtype=torch.float32)

        meta = {
            "path": path,
            "crop_left": left,
            "crop_top": top,
            "resize_scale": resize_scale,
        }

        return img_tensor, keypoint, torch.tensor(shape_label, dtype=torch.long), meta


def build_train_val_datasets(
    data_root: str,
    labels_path: str,
    val_fraction: float = 0.15,
    seed: int = 42,
    marker_margin: int = 20,
) -> Tuple[GCPDataset, GCPDataset]:
    """Load labels, perform group-aware split, return (train_ds, val_ds)."""
    from .splits import group_aware_split

    labels = load_clean_labels(labels_path)
    train_paths, val_paths = group_aware_split(
        labels, val_fraction=val_fraction, seed=seed
    )

    train_ds = GCPDataset(
        data_root, labels, train_paths,
        train=True, marker_margin=marker_margin,
    )
    val_ds = GCPDataset(
        data_root, labels, val_paths,
        train=False, marker_margin=marker_margin,
    )
    return train_ds, val_ds


if __name__ == "__main__":
    import sys

    data_root   = sys.argv[1] if len(sys.argv) > 1 else "."
    labels_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(data_root, "gcp_marks.json")

    train_ds, val_ds = build_train_val_datasets(data_root, labels_path)
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    img, kp, shape_label, meta = train_ds[0]
    print("image shape:", img.shape)
    print("keypoint (normalized):", kp)
    print("shape label:", shape_label.item())
    print("meta:", meta)