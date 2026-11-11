from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from andi_rewrite.anomaly.postprocess.transforms import median_filter_tensor, median_filter_torch_3d
from andi_rewrite.engine import VolumeEvaluator
from andi_rewrite.scripts import eval as eval_script
from andi_rewrite.utils.reporting import save_inference_report


FAST12 = [194, 184, 173, 163, 153, 142, 132, 121, 111, 101, 90, 80]


class _TinyDenoiser(torch.nn.Module):
    def __init__(self, attention_backend: str):
        super().__init__()
        self.attention_backend = attention_backend

    def forward(self, images: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        return images * 0.5


@pytest.fixture
def tiny_denoiser(monkeypatch):
    monkeypatch.setattr(
        eval_script,
        "build_model",
        lambda config, device: _TinyDenoiser(config.get("attention_backend", "legacy")).to(device),
    )


def _config(profile: str | None = None) -> dict:
    return {
        "runtime": {"device": "cpu"},
        "anomaly": {"t_lower": 3, "t_upper": 15},
        "metrics": {"postprocess_mode": "rewrite", "median_filter": {"kernel_size": 3}},
        "evaluation": {} if profile is None else {"inference_profile": profile},
    }


def test_cli_override_resets_manual_acceleration(monkeypatch, tiny_denoiser):
    config = _config("fast12")
    config["anomaly"].update(num_timesteps=2, streaming=True)
    config["model"] = {"attention_backend": "no_weights"}
    config["runtime"]["amp_dtype"] = "float16"
    built = eval_script.build_detector_from_config
    detectors = []

    def capture_detector(resolved):
        detector, accelerator = built(resolved)
        detectors.append(detector)
        return detector, accelerator

    monkeypatch.setattr(eval_script, "load_config", lambda _path: config)
    monkeypatch.setattr(eval_script, "build_detector_from_config", capture_detector)
    monkeypatch.setattr(sys, "argv", ["eval.py", "--inference-profile", "original"])

    eval_script.main()

    detector = detectors[0]
    assert detector.selected_timesteps() == list(range(14, 2, -1))
    assert not detector.streams_timesteps
    assert detector.amp_dtype is None
    assert detector.attention_backend == "legacy"


@pytest.mark.parametrize(
    ("profile", "expected"),
    [("engineering125", list(range(199, 74, -1))), ("fast12", FAST12)],
)
def test_accelerated_presets_use_verified_timesteps(profile, expected, tiny_denoiser):
    config = _config(profile)
    config["runtime"]["cudnn_benchmark"] = True
    detector, _ = eval_script.build_detector_from_config(config)
    assert detector.selected_timesteps() == expected
    assert detector.streams_timesteps
    assert detector.attention_backend == "no_weights"
    assert detector.amp_dtype is None  # CPU fallback
    assert not detector.cudnn_benchmark
    assert config["anomaly"] == {"t_lower": 3, "t_upper": 15}


def test_profile_validates_before_building_model(monkeypatch):
    def unexpected_model(*args, **kwargs):
        pytest.fail("invalid profile reached the model builder")

    monkeypatch.setattr(eval_script, "build_model", unexpected_model)
    with pytest.raises(ValueError, match="Unknown evaluation.inference_profile"):
        eval_script.build_detector_from_config(_config("fast20"))

    config = _config("fast12")
    config["diffusion"] = {"steps": 199}
    with pytest.raises(ValueError, match="diffusion.steps >= 200"):
        eval_script.build_detector_from_config(config)


@pytest.mark.parametrize(
    "aggregation",
    [
        "geometric_mean",
        "mean",
        "sum",
        "max",
        {"type": "weighted_mean", "weights": list(range(1, 13))},
    ],
)
def test_streamed_score_matches_stack_and_preserves_weighted_aggregation(aggregation, tiny_denoiser):
    config = _config("fast12")
    config["anomaly"]["aggregation"] = aggregation
    detector, _ = eval_script.build_detector_from_config(config)
    images = torch.arange(32, dtype=torch.float32).reshape(2, 1, 4, 4) / 31
    torch.manual_seed(405)
    stack = detector.compute_deviation_stack(images)
    expected = detector.pool_modalities(detector.aggregate_time(stack))

    torch.manual_seed(405)
    actual = detector.compute_scores_streaming(images)

    assert detector.streams_timesteps is isinstance(aggregation, str)
    torch.testing.assert_close(actual, expected, rtol=3e-6, atol=1e-10)


def test_tiled_median_matches_scipy_on_reflect_edges():
    maps = torch.arange(2 * 11 * 3 * 2).reshape(2, 11, 3, 2).float().sin()
    expected = median_filter_tensor(maps, kernel_size=5)
    torch.testing.assert_close(
        median_filter_torch_3d(maps, 5, device="cpu"), expected, rtol=0, atol=0
    )


@pytest.mark.parametrize("memory_mode", ["in_memory", "disk_streaming"])
def test_profile_evaluation_isolates_paths_and_reports_effective_settings(
    memory_mode, monkeypatch, tmp_path, tiny_denoiser
):
    config = _config("fast12")
    detector, _ = eval_script.build_detector_from_config(config)
    baseline, _ = eval_script.build_detector_from_config(_config())
    evaluator_config = {
        **config["metrics"],
        "memory_mode": memory_mode,
        "cache": {"directory": tmp_path / "cache"},
        "output_csv": tmp_path / "ANDi.csv",
        "output_mf_csv": tmp_path / "ANDi_mf.csv",
        "prediction_output": {
            "enabled": True,
            "directory": tmp_path / "predictions",
            "restore_native_grid": False,
        },
        "compute_auprc": False,
        "progress": False,
        "size_splits": 2,
        "_run_config": config,
    }
    evaluator = VolumeEvaluator(detector, evaluator_config)
    baseline_evaluator = VolumeEvaluator(baseline, evaluator_config)
    assert evaluator._cache_fingerprints()[0] != baseline_evaluator._cache_fingerprints()[0]
    assert evaluator.output_csv == tmp_path / "fast12" / "ANDi.csv"
    assert baseline_evaluator.output_csv == tmp_path / "ANDi.csv"
    assert evaluator.cache_directory == tmp_path / "cache_fast12"

    def unexpected_stack(*args, **kwargs):
        pytest.fail("fast12 allocated a full timestep stack during evaluation")

    monkeypatch.setattr(detector, "compute_deviation_stack", unexpected_stack)
    image = torch.arange(48, dtype=torch.float32).reshape(1, 1, 4, 4, 3) / 47
    batches = [(image, image[:, 0] > 0.8, {"subject_id": ["test"]})]
    result = evaluator.evaluate(batches)
    report_dir = save_inference_report(config=config, evaluator=evaluator, result=result)
    assert report_dir == tmp_path / "fast12"
    settings = json.loads((report_dir / "inference_report.json").read_text())["inference_settings"]
    assert settings["profile"] == "fast12"
    assert (settings["t_lower"], settings["t_upper"]) == (75, 200)
    assert settings["timesteps"] == FAST12
    metadata_file = tmp_path / "predictions" / "fast12" / "test" / "prediction_metadata.json"
    metadata = json.loads(metadata_file.read_text())
    assert metadata["inference_options"]["timesteps"] == FAST12


def test_original_cache_cleanup_preserves_both_accelerated_caches(tmp_path, tiny_denoiser):
    config = {
        "postprocess_mode": "rewrite",
        "memory_mode": "disk_streaming",
        "cache": {"directory": tmp_path / "cache", "keep_on_success": False},
    }
    accelerated_caches = []
    for profile in ("fast12", "engineering125"):
        detector, _ = eval_script.build_detector_from_config(_config(profile))
        accelerated_caches.append(VolumeEvaluator(detector, config)._open_disk_cache())

    original, _ = eval_script.build_detector_from_config(_config())
    original_cache = VolumeEvaluator(original, config)._open_disk_cache()
    original_cache.cleanup_after_success()

    assert not original_cache.root.exists()
    assert all(cache.manifest_path.is_file() for cache in accelerated_caches)
