"""Focused checks for the mixed ANDi histogram matching data contract."""
from __future__ import annotations

import importlib.util
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk
import lmdb


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


@pytest.mark.parametrize("kind,background,accepted", [
    ("andi_histmatch_mean888_raw_source", -1.0, True),
    ("andi_histmatch_mean888_robust_iqr", -1.0, True),
    ("p99", -1.0, False),
    ("andi_histmatch_mean888_raw_source", 0.0, False),
])
def test_spectrum_normalization_contract(tmp_path, monkeypatch, kind, background, accepted):
    folder = tmp_path / "train"
    folder.mkdir()
    value = np.full((3, 16, 16), -1.0, dtype=np.float32)
    value[:, 4:12, 4:12] = np.arange(64, dtype=np.float32).reshape(8, 8) / 20
    with lmdb.open(str(folder), map_size=1024**2) as env:
        with env.begin(write=True) as txn:
            txn.put(b"00000000", pickle.dumps(value))
    (folder / "normalization.json").write_text(json.dumps({"type": kind, "version": 1, "background": background}))
    target = tmp_path / "spectrum.npz"
    monkeypatch.setattr(sys, "argv", ["compute_lmdb_spectrum.py", "--lmdb-path", str(folder), "--out", str(target),
                                    "--mask-mode", "robust_iqr_background", "--no-progress"])
    if accepted:
        spectrum.compute(spectrum.parse_args())
        with np.load(target) as arrays:
            assert int(arrays["num_slices_used"]) == 1
            assert np.isfinite(arrays["mean_amplitude"]).all()
            assert np.any(arrays["mean_amplitude"] > 0)
        metadata = json.loads(target.with_suffix(".npz.metadata.json").read_text())
        assert metadata["background_after_centering"] == "zero"
        assert metadata["source_lmdb_entry_count"] == 1
    else:
        with pytest.raises(ValueError, match="compatible v1 LMDB"):
            spectrum.compute(spectrum.parse_args())
        assert not target.exists()
