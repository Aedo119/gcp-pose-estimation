"""
Group-aware, shape-stratified train/val split for the GCP dataset.

Rationale (see DECISION_LOG.md, entry 6):
- Images are grouped by `project/survey/gcp_id` (the full path minus the
  filename). Multiple images of the same physical marker exist and are highly
  correlated (overlapping flight passes).
- All images belonging to one group MUST go entirely into train OR val, never
  both, to avoid leakage.
- Within that constraint, we stratify by each group's `verified_shape` (every
  group has a single consistent shape — verified in EDA) so train/val class
  balance roughly matches the overall distribution.
"""

import random
from collections import Counter, defaultdict
from typing import Dict, List, Tuple


def gcp_group_key(path: str) -> str:
    """Return the group key for an image path: everything except the filename.

    e.g. "Seashell Ras el Hekma/Survey 3/GCP25/DJI_0001.JPG"
         -> "Seashell Ras el Hekma/Survey 3/GCP25"
    """
    parts = path.split("/")
    return "/".join(parts[:-1])


def build_group_majority_shape(labels: Dict[str, dict]) -> Dict[str, str]:
    """Map each group key -> its (majority) verified_shape.

    In practice every group is shape-consistent (verified in EDA, entry 8.2),
    so "majority" is just "the" shape, but Counter.most_common(1) is used
    defensively in case of future data with mixed-shape groups.
    """
    group_shapes: Dict[str, set] = defaultdict(set)
    for path, data in labels.items():
        group_shapes[gcp_group_key(path)].add(data["verified_shape"])

    group_majority_shape = {}
    for group, shapes in group_shapes.items():
        # Counter over a set just picks one of equal-count items; for the
        # (expected) single-shape case this is simply that shape.
        group_majority_shape[group] = Counter(shapes).most_common(1)[0][0]

    return group_majority_shape


def group_aware_split(
    labels: Dict[str, dict],
    val_fraction: float = 0.15,
    seed: int = 42,
) -> Tuple[List[str], List[str]]:
    """Split image paths into (train_paths, val_paths) with no group leakage.

    Splits at the `project/survey/gcp_id` group level, stratified by each
    group's verified_shape, so per-class proportions in train/val roughly
    match the overall dataset.

    Args:
        labels: dict mapping image_path -> {"mark": {...}, "verified_shape": ...}
                Should already be the *cleaned* label set (malformed entries
                removed) — see DECISION_LOG.md entry 1.
        val_fraction: target fraction of groups (per shape class) to place in val.
        seed: RNG seed for reproducibility.

    Returns:
        (train_paths, val_paths) — lists of image paths.
    """
    group_majority_shape = build_group_majority_shape(labels)
    groups = list(group_majority_shape.keys())

    by_shape: Dict[str, List[str]] = defaultdict(list)
    for g in groups:
        by_shape[group_majority_shape[g]].append(g)

    rng = random.Random(seed)
    train_groups, val_groups = set(), set()

    for shape, gs in by_shape.items():
        gs = list(gs)
        rng.shuffle(gs)
        n_val = max(1, int(len(gs) * val_fraction))
        val_groups.update(gs[:n_val])
        train_groups.update(gs[n_val:])

    train_paths = [p for p in labels if gcp_group_key(p) in train_groups]
    val_paths = [p for p in labels if gcp_group_key(p) in val_groups]

    return train_paths, val_paths


def split_summary(labels: Dict[str, dict], train_paths: List[str], val_paths: List[str]) -> str:
    """Return a human-readable summary string of the split (for logging)."""
    lines = [
        f"Train: {len(train_paths)} images from "
        f"{len({gcp_group_key(p) for p in train_paths})} groups",
        f"Val:   {len(val_paths)} images from "
        f"{len({gcp_group_key(p) for p in val_paths})} groups",
    ]
    for name, paths in [("train", train_paths), ("val", val_paths)]:
        c = Counter(labels[p]["verified_shape"] for p in paths)
        lines.append(f"{name} class distribution: {dict(c)}")
    return "\n".join(lines)


if __name__ == "__main__":
    import json
    import os
    import sys

    # Quick CLI check: python -m src.splits <path_to_gcp_marks.json>
    labels_path = sys.argv[1] if len(sys.argv) > 1 else "gcp_marks.json"
    with open(labels_path) as f:
        raw_labels = json.load(f)

    # Drop malformed entries (see DECISION_LOG.md entry 1)
    clean_labels = {
        p: d for p, d in raw_labels.items()
        if isinstance(d, dict) and "mark" in d and "verified_shape" in d
        and "x" in d["mark"] and "y" in d["mark"]
    }
    print(f"Kept {len(clean_labels)} / {len(raw_labels)} entries")

    train_paths, val_paths = group_aware_split(clean_labels)
    print(split_summary(clean_labels, train_paths, val_paths))
