"""Create publication-ready ANDi MRI anomaly-detection comparison figures.

The prediction directories consumed by this script are the per-subject folders
written by :class:`andi_rewrite.engine.evaluator.VolumeEvaluator`.  Scores are
loaded from ``anomaly_score_raw.nii.gz`` and ``anomaly_score_mf.nii.gz``.  The
selected Yen threshold and compatibility mask are loaded verbatim from
``prediction_metadata.json`` and ``lesion_mask_yen.nii.gz`` respectively.
Comparison mode additionally requires the evaluator-exported raw/MF branch
thresholds and ``lesion_mask_yen_raw.nii.gz``/``lesion_mask_yen_mf.nii.gz``.

The script never re-thresholds an exported score or re-runs mask postprocessing.
Consequently, every displayed Yen mask is postprocessed and exported by the
evaluator (including dilation or any configured mask pipeline).
Displayed DiceYen, threshold mean, sensitivity, and precision are read directly
from the corresponding Evaluator CSV instead of being recalculated per slice or
on a restored native grid.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to


MODALITIES = ("flair", "t1", "t1ce", "t2")
MODALITY_TITLES = {
    "flair": "FLAIR",
    "t1": "T1",
    "t1ce": "T1CE",
    "t2": "T2",
}
MODALITY_ALIASES = {
    "flair": {"flair"},
    "t1": {"t1", "t1w"},
    "t1ce": {"t1ce", "t1c", "t1post", "t1gd", "t1contrast", "pd"},
    "t2": {"t2", "t2w"},
}
GT_KEYS = {
    "segmentationpath",
    "gtpath",
    "groundtruthpath",
    "labelpath",
    "maskpath",
    "segmentation",
    "seg",
    "gt",
    "groundtruth",
    "label",
    "goldstandard",
}
HEATMAP_CMAP = "inferno"
REPO_ROOT = Path(__file__).resolve().parents[1]


class FigureInputError(RuntimeError):
    """Raised for an incomplete or incompatible figure input."""


def _format_duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "--:--"
    rounded = int(round(seconds))
    hours, remainder = divmod(rounded, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


class FigureProgress:
    """A tqdm progress bar with an elapsed-time/ETA fallback."""

    def __init__(self, total: int, description: str):
        self.total = int(total)
        self.description = description
        self.current = 0
        self.started_at = time.monotonic()
        self._last_width = 0
        self._tqdm = None
        try:
            from tqdm.auto import tqdm

            self._tqdm = tqdm(
                total=self.total,
                desc=self.description,
                unit="figure",
                leave=True,
                dynamic_ncols=True,
                bar_format=(
                    "{l_bar}{bar}| {n_fmt}/{total_fmt} "
                    "[elapsed {elapsed} | ETA {remaining} | {rate_fmt}{postfix}]"
                ),
            )
        except ImportError:
            self._render(slice_index=None)

    def update(self, slice_index: int) -> None:
        self.current += 1
        if self._tqdm is not None:
            self._tqdm.set_postfix({"slice": slice_index}, refresh=False)
            self._tqdm.update(1)
            return
        self._render(slice_index=slice_index)

    def close(self) -> None:
        if self._tqdm is not None:
            self._tqdm.close()
            return
        sys.stderr.write("\n")
        sys.stderr.flush()

    def _render(self, slice_index: int | None) -> None:
        elapsed = time.monotonic() - self.started_at
        rate = self.current / elapsed if elapsed > 0 else 0.0
        remaining = (self.total - self.current) / rate if rate > 0 else None
        fraction = self.current / self.total if self.total else 1.0
        bar_width = 30
        filled = min(bar_width, int(round(bar_width * fraction)))
        bar = "#" * filled + "-" * (bar_width - filled)
        slice_text = f" slice={slice_index}" if slice_index is not None else ""
        message = (
            f"\r{self.description}: |{bar}| {self.current}/{self.total} "
            f"{fraction * 100:5.1f}% elapsed {_format_duration(elapsed)} "
            f"ETA {_format_duration(remaining)}{slice_text}"
        )
        padding = " " * max(0, self._last_width - len(message))
        sys.stderr.write(message + padding)
        sys.stderr.flush()
        self._last_width = len(message)


@dataclass
class PredictionVolume:
    case_dir: Path
    case_id: str
    metadata: dict[str, Any]
    reference: nib.spatialimages.SpatialImage
    raw_score: np.ndarray
    mf_score: np.ndarray
    yen_mask: np.ndarray
    yen_threshold: float
    yen_source: str
    yen_mask_raw: np.ndarray | None
    yen_mask_mf: np.ndarray | None
    yen_threshold_raw: float | None
    yen_threshold_mf: float | None
    postprocess_mode: str
    normalization_scope: str
    raw_display_limits: tuple[float, float]
    mf_display_limits: tuple[float, float]


@dataclass
class InputVolumes:
    arrays: dict[str, np.ndarray]
    display_limits: dict[str, tuple[float, float]]
    gt: np.ndarray


@dataclass(frozen=True)
class EvaluatorYenMetrics:
    csv_path: Path
    mean_threshold: float | str
    dice: float | str
    sensitivity: float | str
    precision: float | str


def _normalized_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _safe_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return cleaned.strip("._") or "case"


def _read_metadata(case_dir: Path) -> dict[str, Any]:
    path = case_dir / "prediction_metadata.json"
    if not path.is_file():
        raise FigureInputError(
            f"Missing evaluator prediction metadata: {path}. "
            "An exact Yen threshold cannot be recovered from score files alone."
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FigureInputError(f"Could not read prediction metadata: {path} ({exc})") from exc
    if not isinstance(payload, dict):
        raise FigureInputError(f"Prediction metadata must contain a JSON object: {path}")
    return payload


def _is_prediction_case_dir(path: Path) -> bool:
    required = (
        "anomaly_score_raw.nii.gz",
        "anomaly_score_mf.nii.gz",
        "lesion_mask_yen.nii.gz",
        "prediction_metadata.json",
    )
    return all((path / name).is_file() for name in required)


def _resolve_prediction_case_dir(path: Path, case_id: str | None) -> Path:
    path = path.expanduser().resolve()
    if _is_prediction_case_dir(path):
        return path
    if case_id:
        candidate = path / str(case_id)
        if _is_prediction_case_dir(candidate):
            return candidate.resolve()
        raise FigureInputError(
            f"Prediction files for case '{case_id}' were not found under {path}. "
            "Expected the two anomaly-score files, lesion_mask_yen.nii.gz, and "
            "prediction_metadata.json in the case folder."
        )
    if not path.is_dir():
        raise FigureInputError(f"Prediction directory does not exist: {path}")
    candidates = sorted(child for child in path.iterdir() if child.is_dir() and _is_prediction_case_dir(child))
    if len(candidates) == 1:
        return candidates[0].resolve()
    if candidates:
        raise FigureInputError(
            f"{path} contains {len(candidates)} prediction cases. Pass --case-id to select one."
        )
    raise FigureInputError(
        f"No complete evaluator prediction case found in {path}. Pass a per-case "
        "prediction directory containing both scores, the exported Yen mask, and "
        "prediction_metadata.json; or pass a run directory together with --case-id."
    )


def _load_nifti(path: Path, label: str) -> tuple[nib.spatialimages.SpatialImage, np.ndarray]:
    if not path.is_file():
        raise FigureInputError(f"Missing {label}: {path}")
    try:
        image = nib.load(str(path))
        array = np.asarray(image.dataobj, dtype=np.float32)
    except Exception as exc:
        raise FigureInputError(f"Could not load {label} from {path}: {exc}") from exc
    if array.ndim != 3:
        raise FigureInputError(f"{label} must be a 3-D NIfTI volume, got shape {array.shape}: {path}")
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    return image, array


def _same_grid(
    source: nib.spatialimages.SpatialImage,
    target: nib.spatialimages.SpatialImage,
) -> bool:
    return tuple(source.shape[:3]) == tuple(target.shape[:3]) and np.allclose(
        source.affine, target.affine, rtol=0.0, atol=1.0e-4
    )


def _resample_array(
    array: np.ndarray,
    source: nib.spatialimages.SpatialImage,
    target: nib.spatialimages.SpatialImage,
    *,
    order: int,
) -> np.ndarray:
    if _same_grid(source, target):
        return array
    source_image = nib.Nifti1Image(array.astype(np.float32, copy=False), source.affine)
    target_grid = (tuple(int(value) for value in target.shape[:3]), target.affine)
    resampled = resample_from_to(source_image, target_grid, order=order)
    return np.asarray(resampled.dataobj, dtype=np.float32)


def _canonical_yen_source(metadata: dict[str, Any]) -> str:
    prediction_output = metadata.get("prediction_output")
    nested_source = prediction_output.get("yen_source") if isinstance(prediction_output, dict) else None
    source = str(metadata.get("yen_source") or nested_source or "score_mf").strip().lower()
    if source in {"raw", "score_raw", "anomaly_score_raw"}:
        return "score_raw"
    if source in {"mf", "score_mf", "anomaly_score_mf", "median_filtered"}:
        return "score_mf"
    raise FigureInputError(
        f"Unsupported evaluator yen_source {source!r}; expected score_raw or score_mf."
    )


def _metadata_yen_threshold(
    metadata: dict[str, Any],
    case_dir: Path,
    key: str = "yen_threshold",
) -> float:
    value = metadata.get(key)
    if value is None:
        raise FigureInputError(
            f"prediction_metadata.json in {case_dir} has no {key}. "
            "Re-run the evaluator to export the exact threshold; this script will not recompute it."
        )
    try:
        threshold = float(value)
    except (TypeError, ValueError) as exc:
        raise FigureInputError(
            f"Invalid evaluator {key} {value!r} in {case_dir}."
        ) from exc
    if not np.isfinite(threshold):
        raise FigureInputError(
            f"Evaluator {key} must be finite in {case_dir}, got {threshold!r}."
        )
    return threshold


def _optional_metadata_yen_threshold(
    metadata: dict[str, Any],
    case_dir: Path,
    key: str,
) -> float | None:
    if metadata.get(key) is None:
        return None
    return _metadata_yen_threshold(metadata, case_dir, key)


def _metadata_text(metadata: dict[str, Any], key: str, default: str = "unknown") -> str:
    value = metadata.get(key)
    if value not in (None, ""):
        return str(value)
    postprocessing = metadata.get("postprocessing")
    if isinstance(postprocessing, dict) and postprocessing.get(key) not in (None, ""):
        return str(postprocessing[key])
    prediction_output = metadata.get("prediction_output")
    if key == "normalization_scope" and isinstance(prediction_output, dict):
        value = prediction_output.get("normalization_scope")
        if isinstance(value, dict):
            mode = str(metadata.get("postprocess_mode", "")).strip()
            value = value.get(mode)
        if value not in (None, ""):
            return str(value)
    return default


def _score_display_limits(score: np.ndarray) -> tuple[float, float]:
    finite = score[np.isfinite(score)]
    if finite.size == 0:
        return (0.0, 1.0)
    low = float(finite.min())
    high = float(finite.max())
    if low >= -1.0e-6 and high <= 1.0 + 1.0e-6:
        return (0.0, 1.0)
    if high <= low:
        high = low + 1.0
    return low, high


def _load_binary_mask(path: Path, label: str) -> tuple[nib.spatialimages.SpatialImage, np.ndarray]:
    image, array = _load_nifti(path, label)
    if not np.all((array == 0.0) | (array == 1.0)):
        values = np.unique(array)
        preview = ", ".join(f"{float(value):g}" for value in values[:8])
        raise FigureInputError(
            f"{label} must be the evaluator's binary 0/1 mask, got values [{preview}] in {path}."
        )
    return image, array.astype(bool, copy=False)


def _load_prediction(path: Path, requested_case_id: str | None) -> PredictionVolume:
    case_dir = _resolve_prediction_case_dir(path, requested_case_id)
    metadata = _read_metadata(case_dir)
    metadata_case_id = str(metadata.get("subject_id", "")).strip()
    if requested_case_id and metadata_case_id and metadata_case_id != str(requested_case_id):
        raise FigureInputError(
            f"Requested case_id '{requested_case_id}' does not match metadata subject_id "
            f"'{metadata_case_id}' in {case_dir}."
        )
    case_id = str(requested_case_id or metadata_case_id or case_dir.name)

    raw_image, raw_score = _load_nifti(case_dir / "anomaly_score_raw.nii.gz", "raw anomaly score")
    mf_image, mf_score = _load_nifti(case_dir / "anomaly_score_mf.nii.gz", "MF anomaly score")
    yen_image, yen_mask = _load_binary_mask(case_dir / "lesion_mask_yen.nii.gz", "final Yen mask")
    for image, label in ((mf_image, "MF anomaly score"), (yen_image, "final Yen mask")):
        if not _same_grid(image, raw_image):
            raise FigureInputError(
                f"{label} in {case_dir} is not on the same grid as anomaly_score_raw.nii.gz. "
                "Current evaluator exports use one shared grid; refusing to resample an exact result."
            )

    branch_masks: dict[str, np.ndarray | None] = {"raw": None, "mf": None}
    for branch, filename, label in (
        ("raw", "lesion_mask_yen_raw.nii.gz", "raw-branch Yen mask"),
        ("mf", "lesion_mask_yen_mf.nii.gz", "MF-branch Yen mask"),
    ):
        branch_path = case_dir / filename
        if not branch_path.is_file():
            continue
        branch_image, branch_mask = _load_binary_mask(branch_path, label)
        if not _same_grid(branch_image, raw_image):
            raise FigureInputError(
                f"{label} in {case_dir} is not on the same grid as anomaly_score_raw.nii.gz. "
                "Current evaluator exports use one shared grid; refusing to resample an exact result."
            )
        branch_masks[branch] = branch_mask

    yen_threshold = _metadata_yen_threshold(metadata, case_dir)
    yen_source = _canonical_yen_source(metadata)
    return PredictionVolume(
        case_dir=case_dir,
        case_id=case_id,
        metadata=metadata,
        reference=raw_image,
        raw_score=raw_score,
        mf_score=mf_score,
        yen_mask=yen_mask,
        yen_threshold=yen_threshold,
        yen_source=yen_source,
        yen_mask_raw=branch_masks["raw"],
        yen_mask_mf=branch_masks["mf"],
        yen_threshold_raw=_optional_metadata_yen_threshold(
            metadata, case_dir, "yen_threshold_raw"
        ),
        yen_threshold_mf=_optional_metadata_yen_threshold(
            metadata, case_dir, "yen_threshold_mf"
        ),
        postprocess_mode=_metadata_text(metadata, "postprocess_mode"),
        normalization_scope=_metadata_text(metadata, "normalization_scope"),
        raw_display_limits=_score_display_limits(raw_score),
        mf_display_limits=_score_display_limits(mf_score),
    )


def _require_comparison_yen_products(prediction: PredictionVolume) -> None:
    missing: list[str] = []
    if prediction.yen_mask_raw is None:
        missing.append("lesion_mask_yen_raw.nii.gz")
    if prediction.yen_mask_mf is None:
        missing.append("lesion_mask_yen_mf.nii.gz")
    if prediction.yen_threshold_raw is None:
        missing.append("metadata.yen_threshold_raw")
    if prediction.yen_threshold_mf is None:
        missing.append("metadata.yen_threshold_mf")
    if missing:
        raise FigureInputError(
            f"Comparison mode requires exact evaluator raw/MF Yen products for "
            f"{prediction.case_dir}; missing: {', '.join(missing)}. Re-run the Evaluator "
            "with prediction_output.save_yen_mask: true. This script will not create "
            "approximate masks or recompute Yen thresholds from exported scores."
        )

    selected_branch = (
        prediction.yen_mask_raw
        if prediction.yen_source == "score_raw"
        else prediction.yen_mask_mf
    )
    if not np.array_equal(prediction.yen_mask, selected_branch):
        raise FigureInputError(
            f"lesion_mask_yen.nii.gz in {prediction.case_dir} does not equal the "
            f"evaluator branch selected by yen_source={prediction.yen_source}. "
            "Re-run the Evaluator to export a consistent exact result."
        )


def _align_prediction(
    prediction: PredictionVolume,
    target: nib.spatialimages.SpatialImage,
) -> PredictionVolume:
    if _same_grid(prediction.reference, target):
        return prediction
    raise FigureInputError(
        f"Prediction grids differ for case {prediction.case_id}. Evaluator-exact comparison "
        "requires both runs to export on the same grid; re-run them with matching "
        "prediction_output.restore_native_grid settings."
    )


def _resolve_recorded_path(value: Any, metadata_dir: Path) -> Path | None:
    if value is None or not str(value).strip():
        return None
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    local_candidate = metadata_dir / path
    if local_candidate.exists():
        return local_candidate.resolve()
    return path.resolve()


def _metadata_items(metadata: dict[str, Any]) -> Iterable[tuple[str, Any]]:
    for key, value in metadata.items():
        yield str(key), value
    input_paths = metadata.get("input_paths")
    if isinstance(input_paths, dict):
        for key, value in input_paths.items():
            yield str(key), value


def _find_modality_path(
    modality: str,
    metadata_sources: Sequence[tuple[dict[str, Any], Path]],
) -> Path | None:
    aliases = MODALITY_ALIASES[modality]
    for metadata, metadata_dir in metadata_sources:
        input_paths = metadata.get("input_paths")
        if not isinstance(input_paths, dict):
            continue
        for key, value in input_paths.items():
            if _normalized_key(key) in aliases:
                return _resolve_recorded_path(value, metadata_dir)
    return None


def _find_gt_path(
    explicit_path: Path | None,
    metadata_sources: Sequence[tuple[dict[str, Any], Path]],
) -> Path:
    if explicit_path is not None:
        return explicit_path.expanduser().resolve()
    for metadata, metadata_dir in metadata_sources:
        for key, value in _metadata_items(metadata):
            if _normalized_key(key) in GT_KEYS:
                path = _resolve_recorded_path(value, metadata_dir)
                if path is not None:
                    return path
    raise FigureInputError(
        "Ground-truth segmentation was not found in prediction_metadata.json. "
        "Pass it explicitly with --gt-path <path-to-segmentation.nii.gz>."
    )


def _display_limits(volume: np.ndarray) -> tuple[float, float]:
    finite = volume[np.isfinite(volume)]
    foreground = finite[finite > 0]
    values = foreground if foreground.size else finite
    if values.size == 0:
        return (0.0, 1.0)
    low, high = (float(value) for value in np.percentile(values, [1.0, 99.0]))
    if high <= low:
        low = float(values.min())
        high = float(values.max())
    if high <= low:
        high = low + 1.0
    return low, high


def _load_inputs(
    target: nib.spatialimages.SpatialImage,
    metadata_sources: Sequence[tuple[dict[str, Any], Path]],
    explicit_gt_path: Path | None,
) -> InputVolumes:
    arrays: dict[str, np.ndarray] = {}
    display_limits: dict[str, tuple[float, float]] = {}
    missing: list[str] = []
    for modality in MODALITIES:
        path = _find_modality_path(modality, metadata_sources)
        if path is None:
            missing.append(modality)
            continue
        image, array = _load_nifti(path, f"{MODALITY_TITLES[modality]} input")
        aligned = _resample_array(array, image, target, order=1)
        arrays[modality] = aligned
        display_limits[modality] = _display_limits(aligned)
    if missing:
        joined = ", ".join(MODALITY_TITLES[item] for item in missing)
        raise FigureInputError(
            f"prediction_metadata.json is missing usable input_paths for: {joined}. "
            "The comparison figure requires FLAIR/T1/T1CE/T2 paths."
        )

    gt_path = _find_gt_path(explicit_gt_path, metadata_sources)
    gt_image, gt_array = _load_nifti(gt_path, "ground-truth segmentation")
    gt = _resample_array(gt_array, gt_image, target, order=0) > 0.5
    return InputVolumes(arrays=arrays, display_limits=display_limits, gt=gt)


def _project_path(value: Any) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _metadata_metrics_csv(
    prediction: PredictionVolume,
    source: str | None = None,
) -> Path | None:
    source_key = source or prediction.yen_source
    output_key = "output_csv" if source_key == "score_raw" else "output_mf_csv"
    for container_key in ("metrics_output", "evaluator_output", "metrics"):
        container = prediction.metadata.get(container_key)
        if not isinstance(container, dict):
            continue
        value = container.get(source_key) or container.get(output_key)
        if value not in (None, ""):
            return _project_path(value)
    value = prediction.metadata.get(output_key)
    return _project_path(value) if value not in (None, "") else None


def _discover_metrics_csv(
    prediction: PredictionVolume,
    source: str | None = None,
) -> Path | None:
    """Find the evaluator CSV from the config that produced a prediction run."""

    try:
        import yaml
    except ImportError:
        return None

    run_dir = prediction.case_dir.parent.resolve()
    source_key = source or prediction.yen_source
    output_key = "output_csv" if source_key == "score_raw" else "output_mf_csv"
    matches: set[Path] = set()
    config_root = REPO_ROOT / "configs"
    config_paths = [*config_root.rglob("*.yaml"), *config_root.rglob("*.yml")]
    for config_path in config_paths:
        try:
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, yaml.YAMLError):
            continue
        if not isinstance(config, dict):
            continue
        prediction_output = config.get("prediction_output")
        metrics = config.get("metrics")
        if not isinstance(prediction_output, dict) or not isinstance(metrics, dict):
            continue
        configured_run = prediction_output.get("directory")
        configured_csv = metrics.get(output_key)
        if configured_run in (None, "") or configured_csv in (None, ""):
            continue
        if _project_path(configured_run) == run_dir:
            matches.add(_project_path(configured_csv))

    if len(matches) == 1:
        return next(iter(matches))
    if len(matches) > 1:
        joined = ", ".join(str(path) for path in sorted(matches))
        raise FigureInputError(
            f"Multiple evaluator CSV files match prediction run {run_dir}: {joined}. "
            "Pass the intended CSV explicitly."
        )
    return None


def _resolve_metrics_csv(
    prediction: PredictionVolume,
    explicit_path: Path | None,
    argument_name: str,
    source: str | None = None,
) -> Path:
    path = (
        _project_path(explicit_path)
        if explicit_path is not None
        else _metadata_metrics_csv(prediction, source) or _discover_metrics_csv(prediction, source)
    )
    if path is None:
        source_key = source or prediction.yen_source
        raise FigureInputError(
            f"Could not locate the Evaluator CSV for {prediction.case_dir} and "
            f"yen_source={source_key}. Pass it with {argument_name}; "
            "the script will not recalculate evaluator metrics from native-grid exports."
        )
    if not path.is_file():
        raise FigureInputError(f"Evaluator metrics CSV does not exist: {path}")
    return path


def _parse_csv_metric(value: Any, *, name: str, path: Path) -> float | str:
    text = str(value or "").strip()
    if text.upper() == "N/A":
        return "N/A"
    try:
        number = float(text)
    except ValueError as exc:
        raise FigureInputError(f"Invalid {name} value {text!r} in Evaluator CSV {path}.") from exc
    if not np.isfinite(number):
        raise FigureInputError(f"Non-finite {name} value {text!r} in Evaluator CSV {path}.")
    return number


@lru_cache(maxsize=32)
def _read_evaluator_yen_metrics(path: Path) -> EvaluatorYenMetrics:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = {
                _normalized_key(row.get("thr", "")): row
                for row in csv.DictReader(handle)
                if row.get("thr")
            }
    except OSError as exc:
        raise FigureInputError(f"Could not read Evaluator metrics CSV {path}: {exc}") from exc

    required = {"yen", "yenthr", "yensen", "yenpre"}
    missing = sorted(required - rows.keys())
    if missing:
        raise FigureInputError(
            f"Evaluator metrics CSV {path} is missing rows: {', '.join(missing)}."
        )

    def value(row_name: str) -> float | str:
        row = rows[row_name]
        raw_value = row.get("value")
        if raw_value in (None, ""):
            # Accept the equivalent named column if a report-oriented CSV was
            # supplied, but never derive one metric from another.
            raw_value = row.get(row_name)
        return _parse_csv_metric(raw_value, name=row_name, path=path)

    return EvaluatorYenMetrics(
        csv_path=path,
        mean_threshold=value("yenthr"),
        dice=value("yen"),
        sensitivity=value("yensen"),
        precision=value("yenpre"),
    )


def _yen_source_label(source: str) -> str:
    return "Raw" if source == "score_raw" else "MF"


def _format_csv_value(value: float | str) -> str:
    return f"{value:.6f}" if isinstance(value, float) else value


def _format_metric_row(
    label: str,
    prediction: PredictionVolume,
    metrics: EvaluatorYenMetrics,
) -> list[str]:
    return [
        label,
        _yen_source_label(prediction.yen_source),
        f"{prediction.yen_threshold:.6f}",
        _format_csv_value(metrics.mean_threshold),
        _format_csv_value(metrics.dice),
        _format_csv_value(metrics.sensitivity),
        _format_csv_value(metrics.precision),
    ]


def _format_compare_row(
    label: str,
    prediction: PredictionVolume,
    metrics: EvaluatorYenMetrics,
) -> list[str]:
    if prediction.yen_threshold_mf is None:
        raise FigureInputError(
            f"Comparison row for {prediction.case_dir} requires metadata.yen_threshold_mf."
        )
    return [
        label,
        "MF",
        f"{prediction.yen_threshold_mf:.6f}",
        _format_csv_value(metrics.mean_threshold),
        _format_csv_value(metrics.dice),
        _format_csv_value(metrics.sensitivity),
        _format_csv_value(metrics.precision),
    ]


def _draw_table(
    axis: plt.Axes,
    columns: Sequence[str],
    rows: Sequence[Sequence[str]],
    widths: Sequence[float],
    *,
    font_size: float,
    title: str = "Full-volume evaluator metrics",
) -> None:
    axis.set_axis_off()
    normalized_widths = np.asarray(widths, dtype=float)
    normalized_widths /= normalized_widths.sum()
    table = axis.table(
        cellText=rows,
        colLabels=columns,
        colWidths=normalized_widths.tolist(),
        cellLoc="center",
        loc="center",
        bbox=[0.0, 0.08, 1.0, 0.82],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("#FFFFFF")
        cell.set_linewidth(1.1)
        cell.PAD = 0.03
        if row == 0:
            cell.set_facecolor("#26374A")
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor("#EAF0F6" if row % 2 else "#F7F9FB")
            if column == 0:
                cell.get_text().set_weight("bold")
    table.scale(1.0, 1.35)
    axis.set_title(title, fontsize=11, fontweight="bold", pad=2)


def _oriented(slice_array: np.ndarray) -> np.ndarray:
    return np.rot90(np.asarray(slice_array))


def _show_gray(
    axis: plt.Axes,
    volume: np.ndarray,
    z_index: int,
    title: str,
    limits: tuple[float, float] | None = None,
) -> None:
    slice_array = volume[:, :, z_index]
    if limits is not None:
        low, high = limits
        slice_array = np.clip((slice_array - low) / (high - low), 0.0, 1.0)
    axis.imshow(_oriented(slice_array), cmap="gray", vmin=0.0 if limits is not None else None, vmax=1.0 if limits is not None else None)
    axis.set_title(title, fontsize=11, fontweight="semibold", pad=6)
    axis.set_axis_off()


def _show_mask(axis: plt.Axes, mask: np.ndarray, z_index: int, title: str) -> None:
    axis.imshow(_oriented(mask[:, :, z_index]), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    axis.set_title(title, fontsize=11, fontweight="semibold", pad=6)
    axis.set_axis_off()


def _show_mask_overlay(
    axis: plt.Axes,
    base: np.ndarray,
    mask: np.ndarray,
    z_index: int,
    title: str,
    limits: tuple[float, float],
) -> None:
    low, high = limits
    base_slice = np.clip((base[:, :, z_index] - low) / (high - low), 0.0, 1.0)
    oriented_mask = _oriented(mask[:, :, z_index]).astype(bool, copy=False)
    overlay = np.zeros((*oriented_mask.shape, 4), dtype=np.float32)
    overlay[oriented_mask] = (1.0, 0.25, 0.0, 0.60)
    axis.imshow(_oriented(base_slice), cmap="gray", vmin=0.0, vmax=1.0)
    axis.imshow(overlay, interpolation="nearest")
    axis.set_title(title, fontsize=11, fontweight="semibold", pad=6)
    axis.set_axis_off()


def _show_heatmap(
    axis: plt.Axes,
    score: np.ndarray,
    z_index: int,
    title: str,
    display_limits: tuple[float, float] | None = None,
) -> None:
    if display_limits is None:
        display_limits = _score_display_limits(score)
    low, high = display_limits
    score_slice = np.nan_to_num(score[:, :, z_index], nan=low, posinf=high, neginf=low)
    axis.imshow(
        _oriented(np.clip(score_slice, low, high)),
        cmap=HEATMAP_CMAP,
        vmin=low,
        vmax=high,
    )
    axis.set_title(title, fontsize=11, fontweight="semibold", pad=6)
    axis.set_axis_off()


def _error_map(prediction: np.ndarray, ground_truth: np.ndarray) -> np.ndarray:
    prediction = np.asarray(prediction, dtype=bool)
    ground_truth = np.asarray(ground_truth, dtype=bool)
    rgb = np.zeros((*prediction.shape, 3), dtype=np.float32)
    rgb[..., 1] = prediction & ground_truth  # TP: green
    rgb[..., 0] = prediction & ~ground_truth  # FP: red
    rgb[..., 2] = ~prediction & ground_truth  # FN: blue
    return rgb


def _show_error_map(
    axis: plt.Axes,
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    title: str,
) -> None:
    axis.imshow(_oriented(_error_map(prediction, ground_truth)), interpolation="nearest")
    axis.set_title(title, fontsize=11, fontweight="semibold", pad=6)
    axis.set_axis_off()


def _selection_indices(selection: str, slice_index: int | None, gt: np.ndarray) -> list[int]:
    slice_count = int(gt.shape[2])
    if selection == "manual":
        if slice_index is None:
            raise FigureInputError("--slice-index is required when --selection manual is used.")
        if not 0 <= slice_index < slice_count:
            raise FigureInputError(
                f"--slice-index {slice_index} is outside the valid range 0..{slice_count - 1}."
            )
        return [int(slice_index)]
    if slice_index is not None:
        raise FigureInputError("--slice-index may only be used with --selection manual.")
    if selection == "gt_max":
        areas = np.count_nonzero(gt, axis=(0, 1))
        return [int(np.argmax(areas))]
    if selection == "all":
        return list(range(slice_count))
    raise FigureInputError(f"Unsupported selection mode: {selection}")


def _single_figure(
    prediction: PredictionVolume,
    inputs: InputVolumes,
    metrics: EvaluatorYenMetrics,
    model_name: str,
    selection: str,
    z_index: int,
) -> plt.Figure:
    figure = plt.figure(figsize=(18.0, 12.4), facecolor="white")
    grid = figure.add_gridspec(3, 4, height_ratios=[1.0, 1.0, 0.42], hspace=0.20, wspace=0.06)
    for column, modality in enumerate(MODALITIES):
        _show_gray(
            figure.add_subplot(grid[0, column]),
            inputs.arrays[modality],
            z_index,
            MODALITY_TITLES[modality],
            inputs.display_limits[modality],
        )

    _show_mask(figure.add_subplot(grid[1, 0]), inputs.gt, z_index, "Ground Truth")
    _show_heatmap(
        figure.add_subplot(grid[1, 1]),
        prediction.raw_score,
        z_index,
        "Exported Raw Anomaly Score",
        prediction.raw_display_limits,
    )
    _show_heatmap(
        figure.add_subplot(grid[1, 2]),
        prediction.mf_score,
        z_index,
        "Exported MF Anomaly Score",
        prediction.mf_display_limits,
    )
    _show_mask(
        figure.add_subplot(grid[1, 3]),
        prediction.yen_mask,
        z_index,
        f"Evaluator Final Yen Mask ({_yen_source_label(prediction.yen_source)})",
    )

    columns = [
        "Result",
        "Yen source",
        "Case Yen Thr",
        "Eval mean Yen Thr",
        "DiceYen",
        "Sensitivity",
        "Precision",
    ]
    rows = [_format_metric_row("Final Yen", prediction, metrics)]
    widths = [1.20, 0.92, 1.05, 1.28, 0.88, 1.05, 0.92]
    _draw_table(
        figure.add_subplot(grid[2, :]),
        columns,
        rows,
        widths,
        font_size=9.0,
        title="Exact Evaluator dataset metrics (CSV) and selected case threshold (metadata)",
    )

    figure.suptitle(
        f"Case: {prediction.case_id} | Slice: {z_index} | Selection: {selection}\n"
        f"Model: {model_name} | Mode: {prediction.postprocess_mode} | "
        f"Normalization: {prediction.normalization_scope}",
        fontsize=16,
        fontweight="bold",
        y=0.985,
        linespacing=1.35,
    )
    figure.subplots_adjust(top=0.90, bottom=0.035, left=0.035, right=0.985)
    return figure


def _compare_figure(
    prediction_a: PredictionVolume,
    prediction_b: PredictionVolume,
    inputs: InputVolumes,
    metrics_a: EvaluatorYenMetrics,
    metrics_b: EvaluatorYenMetrics,
    model_a_name: str,
    model_b_name: str,
    selection: str,
    z_index: int,
) -> plt.Figure:
    _require_comparison_yen_products(prediction_a)
    _require_comparison_yen_products(prediction_b)
    gt_slice = inputs.gt[:, :, z_index]
    raw_mask_a = prediction_a.yen_mask_raw
    mf_mask_a = prediction_a.yen_mask_mf
    raw_mask_b = prediction_b.yen_mask_raw
    mf_mask_b = prediction_b.yen_mask_mf
    assert raw_mask_a is not None and mf_mask_a is not None
    assert raw_mask_b is not None and mf_mask_b is not None
    assert prediction_a.yen_threshold_raw is not None
    assert prediction_a.yen_threshold_mf is not None
    assert prediction_b.yen_threshold_raw is not None
    assert prediction_b.yen_threshold_mf is not None

    figure = plt.figure(figsize=(31.0, 15.0), facecolor="white")
    grid = figure.add_gridspec(3, 6, height_ratios=[0.84, 1.0, 1.0], hspace=0.20, wspace=0.06)
    _show_gray(
        figure.add_subplot(grid[0, 0]),
        inputs.arrays["flair"],
        z_index,
        "FLAIR",
        inputs.display_limits["flair"],
    )
    _show_mask(figure.add_subplot(grid[0, 1]), inputs.gt, z_index, "Ground Truth")

    columns = [
        "Model",
        "Yen source",
        "Case Yen Thr",
        "Eval mean Yen Thr",
        "DiceYen",
        "Sensitivity",
        "Precision",
    ]
    rows = [
        _format_compare_row("A", prediction_a, metrics_a),
        _format_compare_row("B", prediction_b, metrics_b),
    ]
    widths = [0.83, 0.88, 1.02, 1.20, 0.80, 1.02, 0.90]
    _draw_table(
        figure.add_subplot(grid[0, 2:6]),
        columns,
        rows,
        widths,
        font_size=8.0,
        title="Exact Evaluator MF dataset metrics (CSV) and MF case thresholds (metadata)",
    )

    _show_heatmap(
        figure.add_subplot(grid[1, 0]),
        prediction_a.raw_score,
        z_index,
        "Model A Exported Raw Score",
        prediction_a.raw_display_limits,
    )
    _show_mask(
        figure.add_subplot(grid[1, 1]),
        raw_mask_a,
        z_index,
        f"Model A Yen Mask before MF (thr={prediction_a.yen_threshold_raw:.6f})",
    )
    _show_heatmap(
        figure.add_subplot(grid[1, 2]),
        prediction_a.mf_score,
        z_index,
        "Model A Exported MF Score",
        prediction_a.mf_display_limits,
    )
    _show_mask(
        figure.add_subplot(grid[1, 3]),
        mf_mask_a,
        z_index,
        f"Model A Yen Mask after MF (thr={prediction_a.yen_threshold_mf:.6f})",
    )
    _show_error_map(
        figure.add_subplot(grid[1, 4]),
        mf_mask_a[:, :, z_index],
        gt_slice,
        "Model A MF Error Map",
    )
    _show_mask_overlay(
        figure.add_subplot(grid[1, 5]),
        inputs.arrays["flair"],
        mf_mask_a,
        z_index,
        "Model A MF Mask on FLAIR",
        inputs.display_limits["flair"],
    )

    _show_heatmap(
        figure.add_subplot(grid[2, 0]),
        prediction_b.raw_score,
        z_index,
        "Model B Exported Raw Score",
        prediction_b.raw_display_limits,
    )
    _show_mask(
        figure.add_subplot(grid[2, 1]),
        raw_mask_b,
        z_index,
        f"Model B Yen Mask before MF (thr={prediction_b.yen_threshold_raw:.6f})",
    )
    _show_heatmap(
        figure.add_subplot(grid[2, 2]),
        prediction_b.mf_score,
        z_index,
        "Model B Exported MF Score",
        prediction_b.mf_display_limits,
    )
    _show_mask(
        figure.add_subplot(grid[2, 3]),
        mf_mask_b,
        z_index,
        f"Model B Yen Mask after MF (thr={prediction_b.yen_threshold_mf:.6f})",
    )
    _show_error_map(
        figure.add_subplot(grid[2, 4]),
        mf_mask_b[:, :, z_index],
        gt_slice,
        "Model B MF Error Map",
    )
    _show_mask_overlay(
        figure.add_subplot(grid[2, 5]),
        inputs.arrays["flair"],
        mf_mask_b,
        z_index,
        "Model B MF Mask on FLAIR",
        inputs.display_limits["flair"],
    )

    figure.suptitle(
        f"Case: {prediction_a.case_id} | Slice: {z_index} | Selection: {selection}\n"
        f"Model A: {model_a_name} ({prediction_a.postprocess_mode}/{prediction_a.normalization_scope}) | "
        f"Model B: {model_b_name} ({prediction_b.postprocess_mode}/{prediction_b.normalization_scope})",
        fontsize=16,
        fontweight="bold",
        y=0.985,
        linespacing=1.35,
    )
    figure.text(
        0.5,
        0.012,
        "Error maps: TP = green  |  FP = red  |  FN = blue  |  TN/background = black",
        ha="center",
        va="bottom",
        fontsize=10,
        color="#333333",
    )
    figure.subplots_adjust(top=0.90, bottom=0.047, left=0.03, right=0.985)
    return figure


def _save_figures(
    figures: Iterable[tuple[int, plt.Figure]],
    output_dir: Path,
    case_id: str,
    suffix: str,
    dpi: int,
    total: int,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_case_id = _safe_filename(case_id)
    saved: list[Path] = []
    progress = FigureProgress(total=total, description=f"Generating {suffix} figures")
    try:
        for z_index, figure in figures:
            path = output_dir / f"{safe_case_id}_slice_{z_index:03d}_{suffix}.png"
            try:
                figure.savefig(path, dpi=dpi, facecolor="white", bbox_inches="tight", pad_inches=0.12)
            finally:
                plt.close(figure)
            saved.append(path)
            progress.update(slice_index=z_index)
    finally:
        progress.close()
    print(f"Generated {len(saved)} figure(s) in {output_dir.resolve()}")
    return saved


def _run_single(args: argparse.Namespace) -> list[Path]:
    prediction = _load_prediction(args.prediction_dir, args.case_id)
    metadata_sources = [(prediction.metadata, prediction.case_dir)]
    inputs = _load_inputs(prediction.reference, metadata_sources, args.gt_path)
    metrics_path = _resolve_metrics_csv(prediction, args.metrics_csv, "--metrics-csv")
    metrics = _read_evaluator_yen_metrics(metrics_path)
    indices = _selection_indices(args.selection, args.slice_index, inputs.gt)
    figures = (
        (z_index, _single_figure(prediction, inputs, metrics, args.model_name, args.selection, z_index))
        for z_index in indices
    )
    print(
        "Evaluator-exported final Yen result: "
        f"source={prediction.yen_source}, threshold={prediction.yen_threshold:.6f}, "
        f"dataset DiceYen={_format_csv_value(metrics.dice)}, "
        f"sensitivity={_format_csv_value(metrics.sensitivity)}, "
        f"precision={_format_csv_value(metrics.precision)}, CSV={metrics.csv_path}"
    )
    return _save_figures(figures, args.output_dir, prediction.case_id, "single", args.dpi, len(indices))


def _run_compare(args: argparse.Namespace) -> list[Path]:
    prediction_a = _load_prediction(args.prediction_dir_a, args.case_id)
    prediction_b_native = _load_prediction(args.prediction_dir_b, args.case_id)
    _require_comparison_yen_products(prediction_a)
    _require_comparison_yen_products(prediction_b_native)
    if prediction_a.case_id != prediction_b_native.case_id:
        raise FigureInputError(
            f"Prediction case mismatch: model A is '{prediction_a.case_id}', "
            f"model B is '{prediction_b_native.case_id}'. Use matching cases or pass --case-id."
        )
    prediction_b = _align_prediction(prediction_b_native, prediction_a.reference)
    metadata_sources = [
        (prediction_a.metadata, prediction_a.case_dir),
        (prediction_b_native.metadata, prediction_b_native.case_dir),
    ]
    inputs = _load_inputs(prediction_a.reference, metadata_sources, args.gt_path)
    metrics_path_a = _resolve_metrics_csv(
        prediction_a,
        args.metrics_csv_a,
        "--metrics-csv-a",
        source="score_mf",
    )
    metrics_path_b = _resolve_metrics_csv(
        prediction_b,
        args.metrics_csv_b,
        "--metrics-csv-b",
        source="score_mf",
    )
    metrics_a = _read_evaluator_yen_metrics(metrics_path_a)
    metrics_b = _read_evaluator_yen_metrics(metrics_path_b)
    indices = _selection_indices(args.selection, args.slice_index, inputs.gt)
    figures = (
        (
            z_index,
            _compare_figure(
                prediction_a,
                prediction_b,
                inputs,
                metrics_a,
                metrics_b,
                args.model_a_name,
                args.model_b_name,
                args.selection,
                z_index,
            ),
        )
        for z_index in indices
    )
    print(
        "Evaluator-exported MF Yen results: "
        f"A(source=score_mf, threshold={prediction_a.yen_threshold_mf:.6f}, "
        f"dataset DiceYen={_format_csv_value(metrics_a.dice)}), "
        f"B(source=score_mf, threshold={prediction_b.yen_threshold_mf:.6f}, "
        f"dataset DiceYen={_format_csv_value(metrics_b.dice)})"
    )
    return _save_figures(figures, args.output_dir, prediction_a.case_id, "compare", args.dpi, len(indices))


def _case_ids_from_csv(path: Path, limit: int | None) -> list[str]:
    if not path.is_file():
        raise FigureInputError(f"Case CSV does not exist: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = csv.reader(handle)
        next(rows, None)
        case_ids = [row[0].strip() for row in rows if row and row[0].strip()]
    if limit is not None:
        case_ids = case_ids[:limit]
    if not case_ids:
        raise FigureInputError(f"No case IDs were found in the first column of {path}.")
    return case_ids


def _batch_case_complete(args: argparse.Namespace, case_id: str) -> bool:
    case_output = args.output_dir / case_id
    if not case_output.is_dir():
        return False
    existing = list(case_output.glob(f"{_safe_filename(case_id)}_slice_*_compare.png"))
    if args.selection == "gt_max":
        return len(existing) == 1
    raw_path = args.prediction_dir_a / case_id / "anomaly_score_raw.nii.gz"
    if not raw_path.is_file():
        return False
    expected = int(nib.load(str(raw_path)).shape[2])
    return len(existing) == expected


def _run_compare_batch(args: argparse.Namespace) -> list[Path]:
    case_ids = _case_ids_from_csv(args.case_csv, args.limit)
    saved: list[Path] = []
    completed_cases = 0
    started_at = time.monotonic()
    for case_number, case_id in enumerate(case_ids, start=1):
        if args.skip_existing and _batch_case_complete(args, case_id):
            print(f"Case {case_number}/{len(case_ids)}: {case_id} already complete; skipping.")
            continue
        print(f"Case {case_number}/{len(case_ids)}: generating {args.selection} comparison figures for {case_id}")
        case_args = argparse.Namespace(
            prediction_dir_a=args.prediction_dir_a,
            prediction_dir_b=args.prediction_dir_b,
            metrics_csv_a=args.metrics_csv_a,
            metrics_csv_b=args.metrics_csv_b,
            model_a_name=args.model_a_name,
            model_b_name=args.model_b_name,
            output_dir=args.output_dir / case_id,
            selection=args.selection,
            slice_index=None,
            gt_path=None,
            case_id=case_id,
            dpi=args.dpi,
        )
        saved.extend(_run_compare(case_args))
        completed_cases += 1
        elapsed = time.monotonic() - started_at
        rate = completed_cases / elapsed if elapsed > 0 else 0.0
        remaining_cases = len(case_ids) - case_number
        eta = remaining_cases / rate if rate > 0 else None
        print(
            f"Case progress: {case_number}/{len(case_ids)} | "
            f"elapsed {_format_duration(elapsed)} | ETA {_format_duration(eta)}"
        )
    return saved or [args.output_dir]


def _positive_dpi(value: str) -> int:
    dpi = int(value)
    if dpi <= 0:
        raise argparse.ArgumentTypeError("DPI must be a positive integer.")
    return dpi


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate ANDi MRI anomaly-detection diagnostic and model-comparison figures.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/make_comparison_figures.py single --prediction-dir <case-dir> "
            "--model-name ANDi --output-dir outputs/figures/run --selection gt_max\n"
            "  python scripts/make_comparison_figures.py compare --prediction-dir-a <case-dir-a> "
            "--prediction-dir-b <case-dir-b> --model-a-name A --model-b-name B "
            "--output-dir outputs/figures/compare --selection manual --slice-index 80 --gt-path <gt.nii.gz>"
        ),
    )
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--case-id",
        help="Case ID. Required when a prediction path is a run directory containing multiple cases; otherwise inferred.",
    )
    common.add_argument(
        "--selection",
        choices=("manual", "gt_max", "all"),
        default="gt_max",
        help="Slice selection policy (default: gt_max).",
    )
    common.add_argument("--slice-index", type=int, help="Zero-based axial index; required only for manual selection.")
    common.add_argument(
        "--gt-path",
        type=Path,
        help="GT segmentation NIfTI. Required if prediction_metadata.json has no GT/segmentation path.",
    )
    common.add_argument("--output-dir", type=Path, required=True, help="Directory for generated PNG files.")
    common.add_argument("--dpi", type=_positive_dpi, default=200, help="Output DPI (default: 200).")

    subparsers = parser.add_subparsers(dest="command", required=True)
    single = subparsers.add_parser("single", parents=[common], help="Create a single-model diagnostic figure.")
    single.add_argument(
        "--prediction-dir",
        type=Path,
        required=True,
        help="Per-case prediction directory, or a run directory used with --case-id.",
    )
    single.add_argument(
        "--metrics-csv",
        type=Path,
        help=(
            "Evaluator ANDi.csv/ANDi_mf.csv matching the exported yen_source. "
            "If omitted, metadata and configs/*.yaml are searched."
        ),
    )
    single.add_argument("--model-name", required=True, help="Model label shown in the figure title.")
    single.set_defaults(handler=_run_single)

    compare = subparsers.add_parser("compare", parents=[common], help="Create a two-model comparison figure.")
    compare.add_argument(
        "--prediction-dir-a",
        type=Path,
        required=True,
        help="Model A per-case prediction directory, or run directory used with --case-id.",
    )
    compare.add_argument(
        "--prediction-dir-b",
        type=Path,
        required=True,
        help="Model B per-case prediction directory, or run directory used with --case-id.",
    )
    compare.add_argument(
        "--metrics-csv-a",
        type=Path,
        help="Model A Evaluator CSV matching its exported yen_source; auto-discovered when omitted.",
    )
    compare.add_argument(
        "--metrics-csv-b",
        type=Path,
        help="Model B Evaluator CSV matching its exported yen_source; auto-discovered when omitted.",
    )
    compare.add_argument("--model-a-name", required=True, help="Model A label shown in the figure title.")
    compare.add_argument("--model-b-name", required=True, help="Model B label shown in the figure title.")
    compare.set_defaults(handler=_run_compare)

    batch = subparsers.add_parser(
        "compare-batch",
        help="Create comparison figures for every case listed in a CSV.",
    )
    batch.add_argument("--prediction-dir-a", type=Path, required=True, help="Model A prediction run directory.")
    batch.add_argument("--prediction-dir-b", type=Path, required=True, help="Model B prediction run directory.")
    batch.add_argument(
        "--metrics-csv-a",
        type=Path,
        help="Model A Evaluator CSV matching its exported yen_source; auto-discovered when omitted.",
    )
    batch.add_argument(
        "--metrics-csv-b",
        type=Path,
        help="Model B Evaluator CSV matching its exported yen_source; auto-discovered when omitted.",
    )
    batch.add_argument("--model-a-name", required=True, help="Model A label shown in figure titles.")
    batch.add_argument("--model-b-name", required=True, help="Model B label shown in figure titles.")
    batch.add_argument("--case-csv", type=Path, required=True, help="CSV whose first column contains case IDs.")
    batch.add_argument("--limit", type=int, help="Optional maximum number of CSV cases to process.")
    batch.add_argument(
        "--selection",
        choices=("gt_max", "all"),
        default="all",
        help="Slice selection policy (default: all).",
    )
    batch.add_argument("--output-dir", type=Path, required=True, help="Root output directory; one folder per case.")
    batch.add_argument("--dpi", type=_positive_dpi, default=200, help="Output DPI (default: 200).")
    batch.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip cases whose expected comparison PNG set is already complete.",
    )
    batch.set_defaults(handler=_run_compare_batch)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        saved = args.handler(args)
    except FigureInputError as exc:
        parser.error(str(exc))
    if not saved:
        parser.error("No figures were generated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
