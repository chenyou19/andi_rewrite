from __future__ import annotations

import pytest
import torch
from torch import nn

from andi_rewrite.domain_classifier.models import (
    SmallCNN,
    build_model,
    model_description,
    replace_batchnorm_with_groupnorm,
)


def test_small_cnn_contract_and_single_channel_forward() -> None:
    model = SmallCNN(in_channels=1, num_classes=1)
    output = model(torch.randn(4, 1, 64, 64))
    assert output.shape == (4, 1)
    assert not any(isinstance(layer, nn.BatchNorm2d) for layer in model.modules())
    assert sum(isinstance(layer, nn.MaxPool2d) for layer in model.modules()) == 3


def test_small_cnn_accepts_three_modalities() -> None:
    model = build_model("small_cnn", in_channels=3, num_classes=1)
    assert model(torch.randn(2, 3, 80, 80)).shape == (2, 1)


def test_resnet18_is_scratch_and_batchnorm_free() -> None:
    pytest.importorskip("torchvision")
    model = build_model("resnet18", in_channels=1, num_classes=1)
    assert model.conv1.in_channels == 1
    assert model.fc.out_features == 1
    assert not any(
        isinstance(layer, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))
        for layer in model.modules()
    )
    assert model_description(model)["has_batchnorm"] is False


def test_batchnorm_replacement_preserves_affine_parameters() -> None:
    module = nn.Sequential(nn.Conv2d(1, 4, 3, padding=1), nn.BatchNorm2d(4))
    with torch.no_grad():
        module[1].weight.fill_(2.0)
        module[1].bias.fill_(0.25)
    replace_batchnorm_with_groupnorm(module, num_groups=8)
    assert isinstance(module[1], nn.GroupNorm)
    assert torch.allclose(module[1].weight, torch.full((4,), 2.0))
    assert torch.allclose(module[1].bias, torch.full((4,), 0.25))
