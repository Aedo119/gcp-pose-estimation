"""
Model architecture for joint GCP keypoint regression + shape classification.

Design rationale (for README "Network architecture choice"):
- A single shared CNN backbone (ResNet18/34, ImageNet-pretrained) extracts
  visual features from the 224x224 crop. Both tasks operate on the same crop
  and benefit from shared low/mid-level features (edges, textures, marker
  geometry), so a shared trunk with two small task-specific heads is more
  parameter-efficient and trains faster than two separate networks, while
  being simple to deploy (one forward pass, one checkpoint).
- Regression head: outputs 2 values (x, y) in [0, 1] via a Sigmoid, matching
  the normalized target space from dataset.py (utils.normalize_target).
  Sigmoid keeps predictions bounded and well-behaved for MSE/Smooth-L1 loss.
- Classification head: outputs 3 logits (Cross, Square, L-Shape), trained
  with (optionally class-weighted) cross-entropy.
- ResNet18 chosen as the default for a practical, fast-training,
  easy-to-deploy baseline (per the assignment's emphasis on "practical,
  robust solutions ... valued higher than overly complex architectures").
  ResNet34 is offered as a drop-in option if more capacity is needed.
"""

from typing import Tuple

import torch
import torch.nn as nn
import torchvision.models as tv_models


BACKBONE_FEATURE_DIMS = {
    "resnet18": 512,
    "resnet34": 512,
    "resnet50": 2048,
}


class GCPModel(nn.Module):
    """Shared-backbone, dual-head model for GCP pose estimation.

    Forward pass returns:
        keypoint: (B, 2) tensor in [0, 1] (normalized x, y)
        shape_logits: (B, 3) tensor of class logits (Cross, Square, L-Shape)
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        pretrained: bool = True,
        num_shape_classes: int = 3,
        head_hidden_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()

        if backbone not in BACKBONE_FEATURE_DIMS:
            raise ValueError(
                f"Unsupported backbone '{backbone}'. "
                f"Choose from {list(BACKBONE_FEATURE_DIMS.keys())}."
            )

        weights = "DEFAULT" if pretrained else None
        builder = getattr(tv_models, backbone)
        net = builder(weights=weights)

        # Strip the final classification layer; keep everything up to and
        # including global average pooling (output shape: B x feat_dim x 1 x 1)
        self.backbone = nn.Sequential(*list(net.children())[:-1])
        feat_dim = BACKBONE_FEATURE_DIMS[backbone]

        self.keypoint_head = nn.Sequential(
            nn.Linear(feat_dim, head_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_dim, 2),
            nn.Sigmoid(),  # bounds output to [0, 1], matching normalized targets
        )

        self.shape_head = nn.Sequential(
            nn.Linear(feat_dim, head_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_dim, num_shape_classes),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feats = self.backbone(x)          # (B, feat_dim, 1, 1)
        feats = torch.flatten(feats, 1)   # (B, feat_dim)

        keypoint = self.keypoint_head(feats)      # (B, 2), in [0, 1]
        shape_logits = self.shape_head(feats)     # (B, 3)

        return keypoint, shape_logits


def build_model(config: dict) -> GCPModel:
    """Construct a GCPModel from a config dict (see configs/config.yaml)."""
    model_cfg = config.get("model", {})
    return GCPModel(
        backbone=model_cfg.get("backbone", "resnet18"),
        pretrained=model_cfg.get("pretrained", True),
        num_shape_classes=model_cfg.get("num_shape_classes", 3),
        head_hidden_dim=model_cfg.get("head_hidden_dim", 128),
        dropout=model_cfg.get("dropout", 0.2),
    )


if __name__ == "__main__":
    # Quick shape/sanity check with a dummy batch
    model = GCPModel(backbone="resnet18", pretrained=False)
    dummy = torch.randn(4, 3, 224, 224)
    kp, logits = model(dummy)
    print("keypoint shape:", kp.shape)         # expect (4, 2)
    print("keypoint range:", kp.min().item(), kp.max().item())  # expect within [0,1]
    print("shape_logits shape:", logits.shape)  # expect (4, 3)

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {n_params:,} | Trainable: {n_trainable:,}")