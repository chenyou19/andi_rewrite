"""Checked training entry point for the FOMO45K SRI24 three-channel LMDB."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

try:
    from _bootstrap import bootstrap
except ImportError:
    from andi_rewrite.scripts._bootstrap import bootstrap

bootstrap()

from andi_rewrite.data.fomo45k.brats21_lmdb import CHANNEL_ORDER, NORMALIZATION  # noqa: E402
from andi_rewrite.utils import load_config  # noqa: E402


DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "train_fomo45k_sri24_flair_t1_t2_gaussian233.yaml"
)


def _lmdb_root(data: Mapping[str, Any], split: str) -> Path:
    if str(data.get("type", "")).lower() != "lmdb":
        raise ValueError(f"{split} data.type must be 'lmdb'.")
    path = Path(str(data.get("path", ""))).resolve()
    if path.name.lower() != split:
        raise ValueError(f"{split} LMDB path must end in /{split}: {path}")
    if not (path / "data.mdb").is_file():
        raise FileNotFoundError(path / "data.mdb")
    return path.parent


def validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    data = config.get("data", {})
    validation = config.get("validation", {})
    if not isinstance(data, Mapping) or not isinstance(validation, Mapping):
        raise ValueError("data and validation must be mappings.")
    validation_data = validation.get("data", {})
    if not bool(validation.get("enabled", False)) or not isinstance(validation_data, Mapping):
        raise ValueError("Participant-level validation must be enabled.")
    train_root = _lmdb_root(data, "train")
    val_root = _lmdb_root(validation_data, "val")
    if train_root != val_root:
        raise ValueError(f"Train and val LMDB roots differ: {train_root} != {val_root}")
    manifest_path = train_root / "build_manifest.json"
    audit_path = train_root / "audit_report.json"
    publication_path = train_root / "publication.json"
    for path in (manifest_path, audit_path, publication_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    publication = json.loads(publication_path.read_text(encoding="utf-8"))
    if manifest.get("channel_order") != list(CHANNEL_ORDER):
        raise ValueError("LMDB channel order must be [FLAIR,T1,T2].")
    processing = manifest.get("processing_specification", {})
    if processing.get("normalization") != NORMALIZATION or processing.get("clip") is not False:
        raise ValueError("LMDB normalization must match BraTS21 p99 without clipping.")
    if audit.get("status") != "PASS" or publication.get("status") != "PASS":
        raise ValueError("LMDB audit/publication status is not PASS.")
    if audit.get("processing_fingerprint") != manifest.get("processing_fingerprint"):
        raise ValueError("LMDB audit fingerprint does not match its build manifest.")

    model = config.get("model", {})
    if not isinstance(model, Mapping):
        raise ValueError("model must be a mapping.")
    in_channels = int(model.get("in_channels", -1))
    out_channels = int(model.get("out_channels", -1))
    if (in_channels, out_channels) != (3, 3):
        raise ValueError(f"Model must be 3->3 for FLAIR,T1,T2; found {in_channels}->{out_channels}.")
    if model.get("checkpoint"):
        raise ValueError("This run must start from fresh model initialization; model.checkpoint is forbidden.")
    training = config.get("training", {})
    checkpoint = training.get("checkpoint", {}) if isinstance(training, Mapping) else {}
    if isinstance(checkpoint, Mapping) and checkpoint.get("resume"):
        raise ValueError("This run must start fresh; training.checkpoint.resume is forbidden.")
    if not bool(training.get("normalize_input", False)):
        raise ValueError("training.normalize_input must be true to match BraTS21 inference.")
    return {
        "status": "PASS",
        "lmdb_root": str(train_root),
        "channel_order": list(CHANNEL_ORDER),
        "normalization": NORMALIZATION,
        "processing_fingerprint": manifest["processing_fingerprint"],
        "split_entry_counts": manifest["split_entry_counts"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    known, _unknown = parser.parse_known_args()
    result = validate_config(load_config(known.config))
    print("FOMO45K SRI24 LMDB contract check:")
    print(json.dumps(result, indent=2, allow_nan=False))
    from andi_rewrite.scripts.train import main as train_main

    train_main()


if __name__ == "__main__":
    main()
