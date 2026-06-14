"""
GCPDataset: loads aerial images, crops a fixed-size window around each GCP
marker, and returns (image_tensor, target) for joint keypoint regression +
shape classification.

Implements the decisions from DECISION_LOG.md:
  - Crop half-size = 300px (600x600), resized to 224x224 (entry 9)
  - PIL Image.crop() auto-pads out-of-bounds boxes with black (entry 11);
    no custom padding logic needed.
  - Train: random crop-center jitter with per-sample bounds (entry 11).
  - Val: deterministic per-sample jitter (seeded by idx) so val keypoint
    targets are NOT always (0.5, 0.5) — makes val PCK a meaningful
    localization metric (entry 12).
  - Targets are normalized to [0, 1] in model-input space (utils.normalize_target).
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
    Each sample is a crop centered near a GCP marker, with:
      - image: 3x224x224 float tensor, ImageNet-normalized
      - keypoint: 2-vector, normalized to [0, 1] in model-input space
      - shape_label: int class index (0=Cross, 1=Square, 2=L-Shape)
      - meta: dict with original path + crop box for coordinate inverse-transform

    Jitter behaviour:
      - train=True:  random jitter each call (data augmentation)
      - train=False: deterministic jitter seeded by sample idx so val targets
                     vary per-sample and val PCK measures real localization
                     error rather than trivially returning 1.0 (DECISION_LOG
                     entry 12).
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
            train: if True applies random jitter + photometric augmentation;
                   if False applies deterministic jitter (seeded by idx) with
                   no photometric augmentation.
            crop_half: half-width of the native-resolution crop window.
            model_input_size: size to resize the crop to for the model.
            jitter_frac: max jitter as a fraction of crop_half. Applied to
                   both train (randomly) and val (deterministically).
                   Set to 0.0 to disable completely.
        """
        self.data_root = data_root
        self.labels = labels
        self.paths = paths
        self.train = train
        self.crop_half = crop_half
        self.model_input_size = model_input_size
        self.jitter_frac = jitter_frac

        self._normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        self._photometric = T.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02
        ) if train else None

    def __len__(self) -> int:
        return len(self.paths)

    def _max_jitter(self, cx: float, cy: float, w: int, h: int) -> float:
        """Max jitter (px) keeping the marker inside [0, crop_size).

        Bounded by both jitter_frac * crop_half AND the marker's distance
        to the nearest image edge (DECISION_LOG.md entry 11).
        """
        base = self.jitter_frac * self.crop_half
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

        cx, cy = x, y
        if self.jitter_frac > 0:
            max_j = self._max_jitter(x, y, w, h)
            if max_j > 0:
                if self.train:
                    # Random jitter for augmentation
                    jx = np.random.uniform(-max_j, max_j)
                    jy = np.random.uniform(-max_j, max_j)
                else:
                    # Deterministic jitter for val: seeded by sample idx so
                    # the same offset is used every epoch (reproducible) but
                    # varies across samples (non-trivial PCK target).
                    rng = np.random.default_rng(idx)
                    jx = rng.uniform(-max_j, max_j)
                    jy = rng.uniform(-max_j, max_j)
                cx, cy = x + jx, y + jy

        left, top, right, bottom = get_crop_box(cx, cy, self.crop_half)
        crop = img.crop((left, top, right, bottom))  # PIL auto-pads OOB

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
    jitter_frac: float = 0.1,
) -> Tuple[GCPDataset, GCPDataset]:
    """Load labels, perform group-aware split, return (train_ds, val_ds).

    Both train and val use jitter_frac — train randomly, val deterministically
    (see GCPDataset docstring and DECISION_LOG.md entry 12).
    """
    from .splits import group_aware_split

    labels = load_clean_labels(labels_path)
    train_paths, val_paths = group_aware_split(labels, val_fraction=val_fraction, seed=seed)

    train_ds = GCPDataset(data_root, labels, train_paths, train=True,  jitter_frac=jitter_frac)
    val_ds   = GCPDataset(data_root, labels, val_paths,   train=False, jitter_frac=jitter_frac)
    return train_ds, val_ds


if __name__ == "__main__":
    import sys

    data_root  = sys.argv[1] if len(sys.argv) > 1 else "."
    labels_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(data_root, "gcp_marks.json")

    train_ds, val_ds = build_train_val_datasets(data_root, labels_path)
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    img, kp, shape_label, meta = train_ds[0]
    print("image shape:", img.shape)
    print("keypoint (normalized):", kp)
    print("shape label:", shape_label.item())
    print("meta:", meta)