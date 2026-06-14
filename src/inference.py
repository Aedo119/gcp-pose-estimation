"""
Inference script: runs the trained model on the (unlabeled) test_dataset and
produces predictions.json in the same format as the training labels:

    {
        "project1/survey1/2/DJI_0431.JPG": {
            "mark": {"x": 1024.5, "y": 850.2},
            "verified_shape": "L-Shape"
        },
        ...
    }

IMPORTANT — train/inference scale mismatch:
The model is trained ONLY on 600x600 crops (resized to 224x224) centered on
a marker (DECISION_LOG.md entry 9). Resizing a full ~4096x2730 test image
directly to 224x224 would shrink the marker to a few pixels — far outside
the training distribution — and produce unreliable results.

This script uses a sliding-window approach:
  1. Slide a CROP_SIZE x CROP_SIZE (600x600) window across the full test
     image with configurable stride (default 300, i.e. 50% overlap).
  2. Resize each window to 224x224 and run the model in batches.
  3. Score each window by BOTH classification confidence AND keypoint
     centrality (how close the predicted keypoint is to the window center).
     The model was trained on marker-centered crops, so a window genuinely
     containing the marker should predict a keypoint near (0.5, 0.5) AND
     output high classification confidence. Background windows fail at least
     one of these criteria.
  4. The highest-scoring window per image provides the final prediction.

Stride defaults to 300 (50% overlap) for a good accuracy/speed trade-off.
Set `inference.window_stride: 600` in config.yaml for non-overlapping
(faster, less accurate).

Edge windows are auto-padded by PIL (DECISION_LOG.md entry 11).

Usage:
    python -m src.inference --config configs/config.yaml
"""

import argparse
import json
import os

import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

from .model import build_model
from .utils import (
    IDX_TO_SHAPE,
    CROP_SIZE,
    MODEL_INPUT_SIZE,
    SHAPE_LABEL_OUTPUT_OVERRIDE,
    get_crop_box,
    model_to_native,
)


IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


def find_test_images(test_root: str):
    """Recursively find all image files under test_root, returning paths
    relative to test_root (matching gcp_marks.json format)."""
    rel_paths = []
    for dirpath, _dirnames, filenames in os.walk(test_root):
        for fname in filenames:
            if fname.endswith(IMG_EXTENSIONS):
                full_path = os.path.join(dirpath, fname)
                rel_path = os.path.relpath(full_path, test_root)
                rel_paths.append(rel_path.replace(os.sep, "/"))
    return sorted(rel_paths)


def compute_window_centers(w: int, h: int, crop_size: int, stride: int):
    """Return (cx, cy) centers tiling the image at the given stride.

    The last window in each row/column is shifted inward to ensure full
    image coverage without a partial edge strip.
    """
    half = crop_size // 2
    centers_x = list(range(half, max(w - half, half) + 1, stride)) or [w // 2]
    centers_y = list(range(half, max(h - half, half) + 1, stride)) or [h // 2]

    if centers_x[-1] < w - half:
        centers_x.append(w - half)
    if centers_y[-1] < h - half:
        centers_y.append(h - half)

    return [(cx, cy) for cy in centers_y for cx in centers_x]


class SlidingWindowDataset(Dataset):
    """Yields (window_tensor, meta) for every window across every test image."""

    def __init__(self, test_root: str, rel_paths,
                 crop_size: int = CROP_SIZE,
                 stride: int = CROP_SIZE // 2,
                 model_input_size: int = MODEL_INPUT_SIZE):
        self.test_root = test_root
        self.crop_size = crop_size
        self.crop_half = crop_size // 2
        self.stride = stride
        self.model_input_size = model_input_size
        self.resize_scale = model_input_size / crop_size

        self._normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        self.windows = []
        self._image_sizes = {}
        for rel_path in rel_paths:
            full_path = os.path.join(test_root, rel_path)
            with Image.open(full_path) as img:
                w, h = img.size
            self._image_sizes[rel_path] = (w, h)
            for cx, cy in compute_window_centers(w, h, crop_size, stride):
                self.windows.append((rel_path, cx, cy))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        rel_path, cx, cy = self.windows[idx]
        full_path = os.path.join(self.test_root, rel_path)

        img = Image.open(full_path).convert("RGB")
        left, top, right, bottom = get_crop_box(cx, cy, self.crop_half)
        crop = img.crop((left, top, right, bottom))
        crop = crop.resize((self.model_input_size, self.model_input_size), Image.BILINEAR)

        img_tensor = T.functional.to_tensor(crop)
        img_tensor = self._normalize(img_tensor)

        return img_tensor, {"path": rel_path, "crop_left": left, "crop_top": top}


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(config_path: str, test_root: str = None, output_path: str = None,
         checkpoint_path: str = None):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    inf_cfg  = config.get("inference", {})
    data_cfg = config.get("data", {})

    test_root        = test_root        or data_cfg.get("test_data_root")
    output_path      = output_path      or inf_cfg.get("output_path", "predictions.json")
    checkpoint_path  = checkpoint_path  or inf_cfg["checkpoint_path"]
    stride           = inf_cfg.get("window_stride", CROP_SIZE // 2)

    device = get_device()
    print(f"Using device: {device}")

    # --- Load model ---
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_model(checkpoint.get("config", config)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint: epoch {checkpoint.get('epoch', '?')}")

    # --- Build sliding-window dataset ---
    rel_paths = find_test_images(test_root)
    print(f"Found {len(rel_paths)} test images")
    if not rel_paths:
        raise RuntimeError(f"No images found under {test_root}")

    sw_ds = SlidingWindowDataset(test_root, rel_paths, stride=stride)
    print(f"{len(sw_ds)} total windows "
          f"(~{len(sw_ds)/max(len(rel_paths),1):.1f}/image, stride={stride})")

    sw_loader = DataLoader(
        sw_ds,
        batch_size=inf_cfg.get("batch_size", 32),
        shuffle=False,
        num_workers=inf_cfg.get("num_workers", 2),
    )

    remap = inf_cfg.get("remap_lshape_to_lshaped", False)

    # best[path] = (score, native_x, native_y, shape_idx)
    best = {}

    with torch.no_grad():
        for img_batch, meta_batch in sw_loader:
            img_batch = img_batch.to(device)
            pred_kp, pred_logits = model(img_batch)

            probs = F.softmax(pred_logits, dim=1)
            confidence, shape_idx = probs.max(dim=1)

            # Keypoint centrality: model was trained on marker-centered crops,
            # so a window genuinely containing the marker should predict kp
            # near (0.5, 0.5). Distance from center in [0, ~0.7]; invert to
            # [0, 1] score. Combined with confidence, background windows that
            # happen to be confident but off-center get penalized.
            kp_dist = torch.norm(pred_kp - 0.5, dim=1).clamp(0, 1)
            centrality = 1.0 - kp_dist
            combined_score = confidence.cpu() * centrality.cpu()

            pred_kp    = pred_kp.cpu()
            shape_idx  = shape_idx.cpu()

            paths      = meta_batch["path"]
            crop_lefts = meta_batch["crop_left"]
            crop_tops  = meta_batch["crop_top"]

            for i in range(len(paths)):
                path  = paths[i]
                left  = int(crop_lefts[i])
                top   = int(crop_tops[i])

                mx, my   = pred_kp[i].tolist()
                mx_px    = mx * MODEL_INPUT_SIZE
                my_px    = my * MODEL_INPUT_SIZE
                native_x, native_y = model_to_native(
                    mx_px, my_px, left, top, sw_ds.resize_scale
                )

                score = combined_score[i].item()
                idx   = shape_idx[i].item()

                if path not in best or score > best[path][0]:
                    best[path] = (score, native_x, native_y, idx)

    predictions = {}
    for path, (_, native_x, native_y, s_idx) in best.items():
        shape_str = IDX_TO_SHAPE[s_idx]
        if remap:
            shape_str = SHAPE_LABEL_OUTPUT_OVERRIDE.get(shape_str, shape_str)
        predictions[path] = {
            "mark": {"x": native_x, "y": native_y},
            "verified_shape": shape_str,
        }

    with open(output_path, "w") as f:
        json.dump(predictions, f, indent=2)

    print(f"Wrote {len(predictions)} predictions to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     type=str, default="configs/config.yaml")
    parser.add_argument("--test_root",  type=str, default=None)
    parser.add_argument("--output",     type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()
    main(args.config, test_root=args.test_root,
         output_path=args.output, checkpoint_path=args.checkpoint)