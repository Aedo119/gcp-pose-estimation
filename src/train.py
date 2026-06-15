"""
Training script for joint GCP keypoint regression + shape classification.

Usage (from repo root):
    python -m src.train --config configs/config.yaml

In Colab:
    !python -m src.train --config configs/config.yaml

Produces:
    - best_model.pt  (checkpoint with lowest val total loss)
    - last_model.pt  (final-epoch checkpoint, for resuming/debugging)
    - training_log.json  (per-epoch metrics)

All in `train.checkpoint_dir` from the config.
"""

import argparse
import json
import os
import time
from collections import defaultdict

import torch
import yaml
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

from .dataset import build_train_val_datasets
from .losses import GCPLoss, compute_class_weights
from .model import build_model
from .utils import IDX_TO_SHAPE, denormalize_target


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def compute_pck(pred_kp_norm: torch.Tensor, target_kp_norm: torch.Tensor,
                 model_input_size: int, thresholds=(10, 25, 50)) -> dict:
    """Percentage of Correct Keypoints at given pixel thresholds.

    Predictions/targets are in normalized [0,1] model-input space; we
    de-normalize to pixel space (model-input-size pixels) before computing
    distances, since the assignment's PCK thresholds (10/25/50px) are
    defined in pixel terms. Note: this is PCK in *model-input* pixel space
    (224x224), not native image space — useful as a relative training
    signal. The final evaluation (on predictions.json) will be in native
    image coordinates, which is a different absolute scale (see
    DECISION_LOG.md entry 9 re: resize factor).
    """
    pred_px = pred_kp_norm * model_input_size
    target_px = target_kp_norm * model_input_size
    dists = torch.norm(pred_px - target_px, dim=1)  # (B,)

    results = {}
    for t in thresholds:
        results[f"pck@{t}px"] = (dists <= t).float().mean().item()
    return results


def run_epoch(model, loader, criterion, device, optimizer=None, model_input_size=224):
    """Run one epoch. If optimizer is provided, trains; otherwise evaluates."""
    train_mode = optimizer is not None
    model.train() if train_mode else model.eval()

    total_losses = defaultdict(float)
    all_preds, all_targets = [], []
    all_pred_kp, all_target_kp = [], []
    n_batches = 0

    context = torch.enable_grad() if train_mode else torch.no_grad()
    with context:
        for img, kp, shape_label, _meta in loader:
            img = img.to(device, non_blocking=True)
            kp = kp.to(device, non_blocking=True)
            shape_label = shape_label.to(device, non_blocking=True)

            pred_kp, pred_logits = model(img)
            loss, loss_dict = criterion(pred_kp, pred_logits, kp, shape_label)

            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            for k, v in loss_dict.items():
                total_losses[k] += v
            n_batches += 1

            all_preds.append(pred_logits.argmax(dim=1).detach().cpu())
            all_targets.append(shape_label.detach().cpu())
            all_pred_kp.append(pred_kp.detach().cpu())
            all_target_kp.append(kp.detach().cpu())

    avg_losses = {k: v / n_batches for k, v in total_losses.items()}

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    macro_f1 = f1_score(all_targets.numpy(), all_preds.numpy(), average="macro")

    all_pred_kp = torch.cat(all_pred_kp)
    all_target_kp = torch.cat(all_target_kp)
    pck = compute_pck(all_pred_kp, all_target_kp, model_input_size)

    metrics = {**avg_losses, "macro_f1": macro_f1, **pck}
    return metrics


def main(config_path: str):
    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_cfg = config["data"]
    train_cfg = config["train"]

    train_ds, val_ds = build_train_val_datasets(
        data_root=data_cfg["data_root"],
        labels_path=data_cfg["labels_path"],
        val_fraction=data_cfg.get("val_fraction", 0.15),
        seed=data_cfg.get("split_seed", 42),
        marker_margin=train_cfg.get("marker_margin", 20),
    )
    print(f"Train: {len(train_ds)} samples | Val: {len(val_ds)} samples")

    use_cuda = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=train_cfg.get("num_workers", 2) if use_cuda else 0,
        pin_memory=use_cuda,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=train_cfg.get("num_workers", 2) if use_cuda else 0,
        pin_memory=use_cuda,
    )

    model = build_model(config).to(device)

    class_weights = None
    if train_cfg.get("use_class_weights", True):
        class_weights = compute_class_weights(train_ds.labels, train_ds.paths)
        class_weights = class_weights.to(device)
        print(f"Class weights ({[IDX_TO_SHAPE[i] for i in range(3)]}): "
              f"{class_weights.tolist()}")

    criterion = GCPLoss(class_weights=class_weights, kp_weight=train_cfg.get("kp_loss_weight", 5.0))
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["lr"],
                                  weight_decay=train_cfg.get("weight_decay", 0.0))

    checkpoint_dir = train_cfg["checkpoint_dir"]
    os.makedirs(checkpoint_dir, exist_ok=True)

    best_val_loss = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0
    history = []

    num_epochs = train_cfg["num_epochs"]
    patience = train_cfg.get("early_stopping_patience", 7)

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()

        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer=optimizer)
        val_metrics = run_epoch(model, val_loader, criterion, device, optimizer=None)

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch}/{num_epochs} ({elapsed:.1f}s) | "
            f"train_loss={train_metrics['total']:.4f} "
            f"val_loss={val_metrics['total']:.4f} | "
            f"val_macro_f1={val_metrics['macro_f1']:.4f} | "
            f"val_pck@25px={val_metrics['pck@25px']:.4f}"
        )

        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})

        # Checkpointing on val total loss
        if val_metrics["total"] < best_val_loss:
            best_val_loss = val_metrics["total"]
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {"model_state_dict": model.state_dict(), "config": config, "epoch": epoch},
                os.path.join(checkpoint_dir, "best_model.pt"),
            )
            print(f"  -> New best model saved (val_loss={best_val_loss:.4f})")
        else:
            epochs_without_improvement += 1

        torch.save(
            {"model_state_dict": model.state_dict(), "config": config, "epoch": epoch},
            os.path.join(checkpoint_dir, "last_model.pt"),
        )

        with open(os.path.join(checkpoint_dir, "training_log.json"), "w") as f:
            json.dump(history, f, indent=2)

        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs). "
                  f"Best epoch: {best_epoch} (val_loss={best_val_loss:.4f})")
            break

    print(f"\nTraining complete. Best model: epoch {best_epoch}, val_loss={best_val_loss:.4f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    main(args.config)