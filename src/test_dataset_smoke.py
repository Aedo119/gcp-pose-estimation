"""Synthetic smoke test for splits.py and dataset.py (no real data needed)."""

import json
import os
import random
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.dataset import build_train_val_datasets, load_clean_labels
from src.splits import group_aware_split, split_summary

DATA_ROOT = "/tmp/gcp_synthetic"
SHAPES = ["Cross", "Square", "L-Shape"]


def make_synthetic_dataset(n_groups=30, max_images_per_group=6):
    os.makedirs(DATA_ROOT, exist_ok=True)
    labels = {}
    rng = random.Random(0)

    for g in range(n_groups):
        shape = SHAPES[g % 3]
        group_path = f"ProjA/Survey1/GCP{g}"
        n_images = rng.randint(1, max_images_per_group)
        for i in range(n_images):
            # vary image size like real data: two aspect ratios
            if g % 2 == 0:
                w, h = 1024, 683  # scaled-down analog of 4096x2730
            else:
                w, h = 1024, 768  # scaled-down analog of 4096x3068

            img_dir = os.path.join(DATA_ROOT, group_path)
            os.makedirs(img_dir, exist_ok=True)
            img_path_rel = f"{group_path}/img_{i}.JPG"
            img_path_abs = os.path.join(DATA_ROOT, img_path_rel)

            # random color image
            arr = (np.random.rand(h, w, 3) * 255).astype("uint8")
            Image.fromarray(arr).save(img_path_abs)

            # marker position: sometimes near edge to exercise padding path
            if rng.random() < 0.2:
                x = rng.uniform(0, 50)  # near left edge
            else:
                x = rng.uniform(50, w - 50)
            y = rng.uniform(50, h - 50)

            labels[img_path_rel] = {
                "mark": {"x": x, "y": y},
                "verified_shape": shape,
            }

    # inject a few malformed entries
    labels["bad/entry1.JPG"] = {"verified_shape": "Cross"}  # missing mark
    labels["bad/entry2.JPG"] = "not_a_dict"

    labels_path = os.path.join(DATA_ROOT, "gcp_marks.json")
    with open(labels_path, "w") as f:
        json.dump(labels, f)

    return labels_path


def main():
    labels_path = make_synthetic_dataset()

    # --- Test load_clean_labels ---
    clean = load_clean_labels(labels_path)
    print(f"Clean labels: {len(clean)}")
    assert "bad/entry1.JPG" not in clean
    assert "bad/entry2.JPG" not in clean

    # --- Test group_aware_split ---
    train_paths, val_paths = group_aware_split(clean, val_fraction=0.2, seed=42)
    print(split_summary(clean, train_paths, val_paths))

    # No leakage check
    from src.splits import gcp_group_key
    train_groups = {gcp_group_key(p) for p in train_paths}
    val_groups = {gcp_group_key(p) for p in val_paths}
    overlap = train_groups & val_groups
    assert not overlap, f"Leakage! overlapping groups: {overlap}"
    print("No group leakage: OK")

    # --- Test GCPDataset ---
    train_ds, val_ds = build_train_val_datasets(DATA_ROOT, labels_path, val_fraction=0.2, jitter_frac=0.1)
    print(f"\nTrain dataset size: {len(train_ds)} | Val dataset size: {len(val_ds)}")

    img, kp, shape_label, meta = train_ds[0]
    print("image shape:", tuple(img.shape))
    print("keypoint (normalized):", kp.tolist())
    print("shape label idx:", shape_label.item())
    print("meta:", meta)

    assert img.shape == (3, 224, 224)
    assert (0 <= kp).all() and (kp <= 1).all(), f"keypoint out of [0,1]: {kp}"

    # Test a near-edge sample (find one with crop_left or crop_top < 0)
    found_edge_case = False
    for i in range(len(train_ds)):
        img, kp, shape_label, meta = train_ds[i]
        if meta["crop_left"] < 0 or meta["crop_top"] < 0:
            found_edge_case = True
            print(f"\nEdge case sample {i}: meta={meta}, keypoint={kp.tolist()}")
            assert img.shape == (3, 224, 224)
            assert (0 <= kp).all() and (kp <= 1).all(), f"keypoint out of [0,1]: {kp}"
            break
    print("Edge-case sample found and validated:" , found_edge_case)

    # --- Test val dataset (no jitter, deterministic) ---
    img, kp, shape_label, meta = val_ds[0]
    print("\nVal sample 0:", "shape", tuple(img.shape), "kp", kp.tolist(), "meta", meta)

    # --- Test inverse coordinate transform round-trip ---
    from src.utils import denormalize_target, model_to_native
    nx, ny = kp.tolist()
    mx, my = denormalize_target(nx, ny)
    native_x, native_y = model_to_native(mx, my, meta["crop_left"], meta["crop_top"], meta["resize_scale"])
    print(f"Round-trip native coords: ({native_x:.2f}, {native_y:.2f})")

    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
