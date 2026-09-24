"""Focused checks for the mixed ANDi histogram matching data contract."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("prepare_mixed_andi_histmatch", ROOT / "scripts/prepare_mixed_andi_histmatch.py")
mixed = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mixed)
spec2 = importlib.util.spec_from_file_location("compute_lmdb_spectrum", ROOT / "scripts/compute_lmdb_spectrum.py")
spectrum = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(spectrum)


def test_simpleitk_mapping_preserves_registered_background() -> None:
    source = np.full((3, 8, 8), -1.0, dtype=np.float32)
    mask = np.zeros_like(source, dtype=bool)
    mask[:, 2:6, 2:6] = True
    source[mask] = np.linspace(-1.8, 2.0, int(mask.sum()), dtype=np.float32)
    reference = mixed.synthetic_references(np.tile(np.linspace(-2, 3, 4097), (3, 1)), np.ones(3) * 5)[0]
    raw_result = sitk.GetArrayFromImage(sitk.HistogramMatching(sitk.GetImageFromArray(source.astype(float)), reference))
    actual = mixed.match_volume(source, mask, reference)
    np.testing.assert_allclose(actual[mask], raw_result[mask].astype(np.float32))
    assert np.all(actual[~mask] == -1.0)
    assert actual.dtype == np.float32


def test_support_mismatch_fails_before_matching() -> None:
    source = np.full((2, 4, 4), -1.0, dtype=np.float32)
    source[0, 0, 0] = 0.5
    mask = np.zeros_like(source, dtype=bool)
    mask[1, 1, 1] = True
    reference = mixed.synthetic_references(np.tile(np.linspace(-2, 3, 4097), (3, 1)), np.ones(3))[0]
    with pytest.raises(ValueError, match="support mask"):
        mixed.match_volume(source, mask, reference)


def test_spectrum_manifest_accepts_mixed_jsonl(tmp_path: Path) -> None:
    manifest = tmp_path / "source_entries.jsonl"
    manifest.write_text(
        "\n".join(json.dumps({"key": f"{i:08d}", "source_split": "train", "source_dataset": "mpi", "source_key": f"{i:08d}"}) for i in range(2)) + "\n",
        encoding="utf-8",
    )
    result = spectrum.source_manifest_provenance(manifest)
    assert result["rows"] == 2
    assert result["split_counts"] == {"train": 2}
    assert result["subject_count"] == 0
