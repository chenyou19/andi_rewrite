"""Model definitions for the domain-classifier audit.

Two intentionally small model families are provided:

* :class:`SmallCNN` uses four convolutional blocks with widths
  ``32, 64, 128, 128`` and GroupNorm; the first three blocks downsample.
* ``resnet18`` is constructed from scratch through torchvision and every
  BatchNorm layer is replaced with GroupNorm before training.

Neither factory path downloads weights.  This matters for the audit because
the classifier is a controlled diagnostic, rather than a transfer-learning
experiment.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import torch
from torch import nn


def _group_count(num_channels: int, requested: int) -> int:
    """Return the largest useful GroupNorm group count dividing the channels."""

    channels = int(num_channels)
    requested = max(1, min(int(requested), channels))
    for candidate in range(requested, 0, -1):
        if channels % candidate == 0:
            return candidate
    return 1


def replace_batchnorm_with_groupnorm(
    module: nn.Module,
    *,
    num_groups: int = 32,
) -> nn.Module:
    """Replace BatchNorm layers in ``module`` in place and return ``module``.

    ResNet18 uses ``BatchNorm2d`` exclusively.  ``BatchNorm1d`` is handled as
    well so the helper remains useful for small custom heads; LayerNorm keeps
    the feature dimension semantics of a one-dimensional batch-normalized
    head.
    """

    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm2d):
            replacement = nn.GroupNorm(
                _group_count(child.num_features, num_groups),
                child.num_features,
                eps=child.eps,
                affine=child.affine,
            )
            if child.affine:
                with torch.no_grad():
                    replacement.weight.copy_(child.weight)
                    replacement.bias.copy_(child.bias)
            setattr(module, name, replacement)
        elif isinstance(child, nn.BatchNorm1d):
            replacement = nn.LayerNorm(
                child.num_features,
                eps=child.eps,
                elementwise_affine=child.affine,
            )
            if child.affine:
                with torch.no_grad():
                    replacement.weight.copy_(child.weight)
                    replacement.bias.copy_(child.bias)
            setattr(module, name, replacement)
        else:
            replace_batchnorm_with_groupnorm(child, num_groups=num_groups)
    return module


class SmallCNN(nn.Module):
    """Four-block GroupNorm CNN for 2-D MRI slices."""

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 1,
        *,
        widths: Sequence[int] = (32, 64, 128, 128),
        groupnorm_groups: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        normalized_widths = tuple(int(value) for value in widths)
        if len(normalized_widths) != 4:
            raise ValueError("SmallCNN expects exactly four convolutional widths.")
        if int(in_channels) <= 0 or int(num_classes) <= 0:
            raise ValueError("in_channels and num_classes must be positive.")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1).")

        blocks: list[nn.Module] = []
        current_channels = int(in_channels)
        for block_index, width in enumerate(normalized_widths):
            pooling: nn.Module = nn.MaxPool2d(kernel_size=2, stride=2) if block_index < 3 else nn.Identity()
            blocks.append(
                nn.Sequential(
                    nn.Conv2d(
                        current_channels,
                        width,
                        kernel_size=3,
                        padding=1,
                        bias=False,
                    ),
                    nn.GroupNorm(
                        _group_count(width, int(groupnorm_groups)),
                        width,
                    ),
                    nn.ReLU(inplace=True),
                    pooling,
                )
            )
            current_channels = width
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(float(dropout))
        self.classifier = nn.Linear(current_channels, int(num_classes))
        self.in_channels = int(in_channels)
        self.num_classes = int(num_classes)
        self.widths = normalized_widths

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim == 3:
            inputs = inputs.unsqueeze(1)
        if inputs.ndim != 4:
            raise ValueError(
                "SmallCNN expects [batch, channels, height, width] or "
                "[batch, height, width] input."
            )
        values = inputs.float()
        values = self.features(values)
        values = self.pool(values).flatten(1)
        values = self.dropout(values)
        return self.classifier(values)


def _build_resnet18(
    *,
    in_channels: int,
    num_classes: int,
    groupnorm_groups: int,
) -> nn.Module:
    try:
        from torchvision.models import resnet18
    except Exception as exc:  # pragma: no cover - exercised only without torchvision
        raise RuntimeError(
            "The resnet18 domain classifier requires torchvision to be installed."
        ) from exc

    model = resnet18(weights=None)
    if int(in_channels) != 3:
        old_conv = model.conv1
        model.conv1 = nn.Conv2d(
            int(in_channels),
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )
    model.fc = nn.Linear(model.fc.in_features, int(num_classes))
    replace_batchnorm_with_groupnorm(model, num_groups=int(groupnorm_groups))
    return model


def build_model(
    kind: str = "small_cnn",
    *,
    in_channels: int = 1,
    num_classes: int = 1,
    widths: Iterable[int] = (32, 64, 128, 128),
    groupnorm_groups: int = 8,
    dropout: float = 0.0,
) -> nn.Module:
    """Build a scratch domain-classification model.

    ``num_classes=1`` is the binary-logit convention used by the runner.
    Passing ``num_classes=2`` remains supported for callers that prefer a
    two-logit head.
    """

    normalized = str(kind).strip().lower().replace("-", "_")
    if normalized in {"small_cnn", "cnn", "small"}:
        return SmallCNN(
            in_channels=int(in_channels),
            num_classes=int(num_classes),
            widths=tuple(int(value) for value in widths),
            groupnorm_groups=int(groupnorm_groups),
            dropout=float(dropout),
        )
    if normalized in {"resnet18", "resnet_18", "resnet"}:
        return _build_resnet18(
            in_channels=int(in_channels),
            num_classes=int(num_classes),
            groupnorm_groups=int(groupnorm_groups),
        )
    raise ValueError(
        f"Unknown domain-classifier model {kind!r}; expected 'small_cnn' or 'resnet18'."
    )


def model_description(model: nn.Module) -> dict[str, object]:
    """Return a small serializable architecture description for audit reports."""

    return {
        "class": model.__class__.__name__,
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "has_batchnorm": any(
            isinstance(layer, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))
            for layer in model.modules()
        ),
    }


__all__ = [
    "SmallCNN",
    "build_model",
    "model_description",
    "replace_batchnorm_with_groupnorm",
]
