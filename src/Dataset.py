"""
GCPDataset: loads aerial images, crops a fixed-size window around each GCP
marker, and returns (image_tensor, target) for joint keypoint regression +
shape classification.

Implements the decisions from DECISION_LOG.md:
  - Crop half-size = 300px (600x600), resized to 224x224 (entry 9)
  - PIL Image.crop() auto-pads out-of-bounds boxes with black (entry 11);
    no custom padding logic needed.
  - Optional random crop-center jitter for augmentation, with per-sample
    bounds so the marker can never fall outside the resulting crop
    (entry 11, augmentation caveat).
  - Targets are normalized to [0, 1] in model-input space (utils.normalize_target).
"""

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

from .utils import (
    CROP_HALF,
    CROP_SIZE,
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
    Each sample is a crop centered on a GCP marker, with:
      - image: 3x224x224 float tensor, ImageNet-normalized
      - keypoint: 2-vector, normalized to [0, 1] in model-input space
      - shape_label: int class index (0=Cross, 1=Square, 2=L-Shape)
      - meta: dict with original path + crop box, for inference-time
              coordinate inverse-transform (not used during training)
    """

    def __init__(
        self,
        data_root: str,
        labels: Dict[str, dict],
        paths: List[str],
        train: bool = True,
        crop_half: int = CROP_HALF,
        model_input_size: int = MODEL_INPUT_SIZE,
        jitter_frac: float = 0.1,
    ):
        """
        Args:
            data_root: root directory containing the nested image folders.
            labels: cleaned labels dict (output of load_clean_labels).
            paths: list of image paths (subset of labels.keys()) for this split.
            train: if True, applies random crop jitter + photometric
                   augmentation; if False, uses a fixed centered crop with no
                   augmentation (for val/test).
            crop_half: half-width of the native-resolution crop window.
            model_input_size: size to resize the crop to for the model.
            jitter_frac: max jitter as a fraction of crop_half (only used if
                   train=True). e.g. 0.1 -> jitter up to 30px for crop_half=300.
        """
        self.data_root = data_root
        self.labels = labels
        self.paths = paths
        self.train = train
        self.crop_half = crop_half
        self.model_input_size = model_input_size
        self.jitter_frac = jitter_frac

        # ImageNet normalization (standard for pretrained CNN backbones)
        self._normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        if train:
            self._photometric = T.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02
            )
        else:
            self._photometric = None

    def __len__(self) -> int:
        return len(self.paths)

    def _max_jitter(self, cx: float, cy: float, w: int, h: int) -> float:
        """Maximum jitter magnitude (pixels) such that the marker stays
        within the resulting crop, bounded by jitter_frac * crop_half.

        See DECISION_LOG.md entry 11 (augmentation caveat): jitter must not
        push the marker outside the crop frame, especially for the ~21% of
        samples already near an image border.
        """
        base = self.jitter_frac * self.crop_half
        # Distance from marker to each image edge limits how far the crop
        # center can move while keeping the marker inside [0, crop_size).
        margin = min(cx, cy, w - cx, h - cy)
        return float(min(base, max(0.0, margin)))

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        data = self.labels[path]
        mark = data["mark"]
        x, y = float(mark["x"]), float(mark["y"])
        shape_label = SHAPE_TO_IDX[data["verified_shape"]]

        full_path = os.path.join(self.data_root, path)
        img = Image.open(full_path).convert("RGB")
        w, h = img.size

        # Crop center = marker position, optionally jittered (train only)
        cx, cy = x, y
        if self.train and self.jitter_frac > 0:
            max_j = self._max_jitter(x, y, w, h)
            if max_j > 0:
                jx = np.random.uniform(-max_j, max_j)
                jy = np.random.uniform(-max_j, max_j)
                cx, cy = x + jx, y + jy

        left, top, right, bottom = get_crop_box(cx, cy, self.crop_half)
        # PIL auto-pads out-of-bounds boxes with black (DECISION_LOG entry 11)
        crop = img.crop((left, top, right, bottom))

        # Marker position within the (CROP_SIZE x CROP_SIZE) crop
        crop_x, crop_y = native_to_crop(x, y, left, top)

        # Resize crop -> model input size
        crop = crop.resize((self.model_input_size, self.model_input_size), Image.BILINEAR)
        resize_scale = self.model_input_size / (2 * self.crop_half)
        model_x, model_y = crop_to_model(crop_x, crop_y, resize_scale)

        # Photometric augmentation (train only; does not affect coordinates)
        if self._photometric is not None:
            crop = self._photometric(crop)

        # To tensor + normalize
        img_tensor = T.functional.to_tensor(crop)
        img_tensor = self._normalize(img_tensor)

        # Normalize keypoint target to [0, 1]
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
    jitter_frac: float = 0.1,
) -> Tuple[GCPDataset, GCPDataset]:
    """Convenience constructor: load labels, perform the group-aware split,
    and return (train_dataset, val_dataset)."""
    from .splits import group_aware_split  # local import to avoid cycle at module load

    labels = load_clean_labels(labels_path)
    train_paths, val_paths = group_aware_split(labels, val_fraction=val_fraction, seed=seed)

    train_ds = GCPDataset(data_root, labels, train_paths, train=True, jitter_frac=jitter_frac)
    val_ds = GCPDataset(data_root, labels, val_paths, train=False)
    return train_ds, val_ds


if __name__ == "__main__":
    import sys

    data_root = sys.argv[1] if len(sys.argv) > 1 else "."
    labels_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(data_root, "gcp_marks.json")

    train_ds, val_ds = build_train_val_datasets(data_root, labels_path)
    print(f"Train: {len(train_ds)} samples | Val: {len(val_ds)} samples")

    img, kp, shape_label, meta = train_ds[0]
    print("image shape:", img.shape)
    print("keypoint (normalized):", kp)
    print("shape label:", shape_label.item())
    print("meta:", meta)