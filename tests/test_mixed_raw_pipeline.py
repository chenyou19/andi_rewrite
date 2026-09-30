"""Prevent test251 inference from using incomplete or unrelated training outputs."""
import importlib.util
import json
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("raw_pipeline", ROOT / "scripts/run_mixed_raw_histmatch200_pipeline.py")
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)


@pytest.mark.parametrize("epoch,run,weight,valid", [
    (199, pipeline.NAME, 1.0, True),
    (179, pipeline.NAME, 1.0, False),
    (199, "another_run", 1.0, False),
    (199, pipeline.NAME, float("nan"), False),
])
def test_final_checkpoint_gate(tmp_path, epoch, run, weight, valid):
    path = tmp_path / "checkpoint.pt"
    torch.save({"epoch": epoch, "config": {"run_name": run},
                "ema_model": {"weight": torch.tensor([weight])}}, path)
    if valid:
        result = pipeline.verify_checkpoint(path)
        assert result["epoch"] == 199
        assert len(result["sha256"]) == 64
    else:
        with pytest.raises(ValueError):
            pipeline.verify_checkpoint(path)


def test_inference_rejects_wrong_subject_roster(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "EVAL_ROOT", tmp_path)
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "manifest.json").write_text(json.dumps({"collection_complete": True, "subjects": 251,
                                                  "entries": [{"subject_id": "unexpected_case"}]}))
    with pytest.raises(ValueError, match="fixed test251 order"):
        pipeline.verify_evaluation([f"case_{i}" for i in range(251)])


def test_resume_discards_uncommitted_metrics_but_archives_them(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "RUN", tmp_path)
    metrics = tmp_path / "training_metrics.csv"
    metrics.write_text("epoch_index,train_loss\n0,0.5\n1,0.4\n2,0.3\n")
    (tmp_path / "stderr.log").write_text("old OOM failure")
    pipeline.prepare_resume_metrics(1)
    assert metrics.read_text().splitlines() == ["epoch_index,train_loss", "0,0.5", "1,0.4"]
    archived = list((tmp_path / "recovery").glob("*/training_metrics.csv"))
    assert len(archived) == 1 and "2,0.3" in archived[0].read_text()
    assert not (tmp_path / "stderr.log").exists()
