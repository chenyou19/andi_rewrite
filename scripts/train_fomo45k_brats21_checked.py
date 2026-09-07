"""Train ANDi on FOMO45K BraTS21 data with explicit channel compatibility checks."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

import torch

try:
    from _bootstrap import bootstrap
except ImportError:
    from andi_rewrite.scripts._bootstrap import bootstrap

bootstrap()

from andi_rewrite.scripts.train_fomo45k_brats21 import register_fomo45k_brats21  # noqa: E402
from andi_rewrite.utils import load_config  # noqa: E402


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "train_fomo45k_brats21.json"
BRATS21_DATASET_TYPES = {
    "fomo45k_brats21",
    "fomo45k_brats21_slices",
    "fomo45k_brats21_volume",
}


def _state_dict(payload: Any) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        return payload
    for key in ("ema_model", "model", "state_dict"):
        value = payload.get(key)
        if isinstance(value, Mapping):
            return value
    return payload


def checkpoint_input_channels(path: str | Path) -> int | None:
    payload = torch.load(Path(path), map_location="cpu")
    state = _state_dict(payload)
    if not isinstance(state, Mapping):
        raise ValueError(f"Checkpoint does not contain a state mapping: {path}")
    preferred_suffixes = (
        "inc.double_conv.0.weight",
        "stem.weight",
        "input_conv.weight",
        "conv_in.weight",
    )
    for suffix in preferred_suffixes:
        for key, value in state.items():
            if str(key).endswith(suffix) and isinstance(value, torch.Tensor) and value.ndim >= 2:
                return int(value.shape[1])
    return None


def validate_fomo45k_brats21_config(config: Mapping[str, Any]) -> None:
    data = config.get("data", {})
    data_type = str(data.get("type", "")) if isinstance(data, Mapping) else ""
    if data_type not in BRATS21_DATASET_TYPES:
        raise ValueError(
            f"Checked BraTS21 entry requires data.type in {sorted(BRATS21_DATASET_TYPES)}, "
            f"found {data_type!r}"
        )
    model = config.get("model", {})
    if not isinstance(model, Mapping):
        raise ValueError("model must be a mapping")
    in_channels = int(model.get("in_channels", model.get("channels", -1)))
    out_channels = int(model.get("out_channels", in_channels))
    if (in_channels, out_channels) != (3, 3):
        raise ValueError(
            "FOMO45K BraTS21 input is exactly FLAIR,T1,T2 (3 channels); "
            f"configured model is {in_channels}->{out_channels}. No weights will be modified or copied."
        )

    checkpoint_paths: list[Path] = []
    if model.get("checkpoint"):
        checkpoint_paths.append(Path(str(model["checkpoint"])))
    training = config.get("training", {})
    checkpoint_config = training.get("checkpoint", {}) if isinstance(training, Mapping) else {}
    if isinstance(checkpoint_config, Mapping) and checkpoint_config.get("resume"):
        checkpoint_paths.append(Path(str(checkpoint_config["resume"])))
    for path in checkpoint_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        channels = checkpoint_input_channels(path)
        if channels is not None and channels != 3:
            raise ValueError(
                f"Checkpoint {path} expects {channels} input channels, but FOMO45K BraTS21 "
                "provides FLAIR,T1,T2 (3). The checkpoint is incompatible; weights were not modified."
            )


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    known, _unknown = parser.parse_known_args()
    validate_fomo45k_brats21_config(load_config(known.config))
    register_fomo45k_brats21()
    from andi_rewrite.scripts.train import main as train_main

    train_main()


if __name__ == "__main__":
    main()
