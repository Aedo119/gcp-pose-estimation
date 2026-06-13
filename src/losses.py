"""
Combined loss for joint keypoint regression + shape classification.

Design rationale (DECISION_LOG.md entry 7, "Class Distribution"):
- Classification: class-weighted cross-entropy. Class weights are
  inverse-proportional to training-set frequency (Cross is ~18% of data,
  L-Shape ~49%), so the minority class isn't dominated during training —
  important since the assignment evaluates macro-F1, which weights all
  classes equally regardless of support.
- Regression: Smooth L1 (Huber) loss on normalized (x, y) targets. Smooth L1
  is more robust to occasional large errors (e.g. ambiguous/borderline
  annotations) than plain MSE, while still being well-behaved for small
  errors.
- Combined loss: weighted sum of the two terms. The relative weight
  (`kp_weight`) is a tunable hyperparameter — keypoint regression in
  normalized [0,1] space produces small loss values, while cross-entropy can
  be larger early in training, so kp_weight > 1 is used to balance their
  gradient contributions.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn

from .splits import gcp_group_key  # noqa: F401  (re-exported for convenience)


def compute_class_weights(labels: Dict[str, dict], paths, num_classes: int = 3,
                           shape_to_idx: Optional[Dict[str, int]] = None) -> torch.Tensor:
    """Compute inverse-frequency class weights from a training split.

    weight_c = N_total / (num_classes * count_c)

    This is the standard "balanced" weighting used by sklearn's
    class_weight='balanced' and is appropriate here given the assignment's
    macro-F1 evaluation criterion (DECISION_LOG.md entry 7).
    """
    if shape_to_idx is None:
        from .utils import SHAPE_TO_IDX as shape_to_idx  # local import to avoid cycle

    counts = torch.zeros(num_classes)
    for p in paths:
        idx = shape_to_idx[labels[p]["verified_shape"]]
        counts[idx] += 1

    total = counts.sum()
    weights = total / (num_classes * counts.clamp(min=1))
    return weights


class GCPLoss(nn.Module):
    """Combined loss: Smooth-L1 (keypoint) + weighted CrossEntropy (shape).

    Forward signature:
        loss, loss_dict = criterion(pred_kp, pred_logits, target_kp, target_shape)

    Returns the total scalar loss plus a dict of component losses (for
    logging).
    """

    def __init__(self, class_weights: Optional[torch.Tensor] = None, kp_weight: float = 5.0):
        super().__init__()
        self.kp_loss_fn = nn.SmoothL1Loss()
        self.cls_loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        self.kp_weight = kp_weight

    def forward(
        self,
        pred_kp: torch.Tensor,
        pred_logits: torch.Tensor,
        target_kp: torch.Tensor,
        target_shape: torch.Tensor,
    ):
        kp_loss = self.kp_loss_fn(pred_kp, target_kp)
        cls_loss = self.cls_loss_fn(pred_logits, target_shape)

        total = self.kp_weight * kp_loss + cls_loss

        loss_dict = {
            "total": total.item(),
            "keypoint": kp_loss.item(),
            "classification": cls_loss.item(),
        }
        return total, loss_dict


if __name__ == "__main__":
    # Quick smoke test with dummy tensors
    torch.manual_seed(0)
    pred_kp = torch.rand(8, 2)
    pred_logits = torch.randn(8, 3)
    target_kp = torch.rand(8, 2)
    target_shape = torch.randint(0, 3, (8,))

    class_weights = torch.tensor([1.8, 1.0, 0.7])  # example: Cross underrepresented
    criterion = GCPLoss(class_weights=class_weights, kp_weight=5.0)

    loss, loss_dict = criterion(pred_kp, pred_logits, target_kp, target_shape)
    print("loss:", loss.item())
    print("loss_dict:", loss_dict)