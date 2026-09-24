"""Training and manifest adapters for the controlled domain-classifier audit.

The runner owns optimization and validation only.  Image preparation remains
in ``data.domain_classifier`` so raw, final, and registered stages can be
audited without silently duplicating preprocessing in this package.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import random
import subprocess
import time
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from .metrics import (
    bootstrap_subject_auc,
    compute_dataset_metrics,
    holm_adjust,
    paired_heldout_swap_test,
)
from .models import build_model, model_description


def set_deterministic_seed(seed: int) -> None:
    """Seed Python, NumPy, and torch for one registered training replicate."""

    value = int(seed)
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class TrainConfig:
    """Configuration for one fixed split and one training seed."""

    model: str = "small_cnn"
    in_channels: int = 3
    modalities: tuple[str, ...] = ("flair", "t1", "t2")
    num_classes: int = 1
    widths: tuple[int, ...] = (32, 64, 128, 128)
    groupnorm_groups: int = 8
    dropout: float = 0.0
    learning_rate: float | None = None
    weight_decay: float = 1.0e-4
    max_epochs: int = 40
    patience: int = 8
    batch_size: int = 32
    num_workers: int = 0
    threshold: float = 0.5
    split_seed: int = 73
    seed: int = 73
    stage: str = "final"
    device: str = "auto"
    subject_method: str = "mean"
    no_augmentation: bool = True
    tiny: bool = False
    early_stopping: bool = True
    bootstrap_replicates: int = 2000
    swap_replicates: int = 1000
    permutation_replicates: int = 0
    permutation_mode: str = "none"
    shared_train_scalar: float | None = None

    def __post_init__(self) -> None:
        self.modalities = tuple(str(value).strip().lower() for value in self.modalities)
        valid_modalities = {"flair", "t1", "t2"}
        if not self.modalities or any(value not in valid_modalities for value in self.modalities):
            raise ValueError("modalities must be a non-empty subset of flair, t1, and t2.")
        if len(set(self.modalities)) != len(self.modalities):
            raise ValueError("modalities must not contain duplicates.")
        # The three-channel default follows the FLAIR/T1/T2 primary audit.  A
        # selected one- or two-modality control automatically narrows the head.
        if int(self.in_channels) == 3 and len(self.modalities) != 3:
            self.in_channels = len(self.modalities)

    def resolved_learning_rate(self) -> float:
        if self.learning_rate is not None:
            return float(self.learning_rate)
        return 3.0e-4 if self.model.lower().replace("-", "_") in {"resnet", "resnet18", "resnet_18"} else 1.0e-3

    def resolved_device(self) -> torch.device:
        choice = str(self.device).strip().lower()
        if choice in {"auto", "cuda_if_available"}:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if choice == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("device='cuda' was requested but CUDA is unavailable.")
        return torch.device(choice)


def _record_value(record: Any, key: str, fallback: Any = None) -> Any:
    if isinstance(record, Mapping):
        value = record.get(key, fallback)
    else:
        value = getattr(record, key, fallback)
    return fallback if value is None else value


def _record_mapping(record: Any) -> dict[str, Any]:
    if isinstance(record, Mapping):
        return dict(record)
    converter = getattr(record, "to_dict", None)
    if callable(converter):
        return dict(converter())
    raise TypeError(f"Manifest record must be a mapping or expose to_dict(), got {type(record).__name__}.")


def _to_image_tensor(value: Any) -> Tensor:
    if isinstance(value, Mapping):
        for key in ("image", "array", "tensor", "final", "registered", "raw"):
            if key in value:
                return _to_image_tensor(value[key])
    if isinstance(value, (str, Path)):
        path = Path(value)
        if path.suffix.lower() in {".npy", ".npz"}:
            loaded = np.load(path, allow_pickle=False)
            if isinstance(loaded, np.lib.npyio.NpzFile):
                first_key = loaded.files[0]
                array = loaded[first_key]
            else:
                array = loaded
            return torch.as_tensor(array).float()
        raise TypeError(f"Unsupported image path format: {path}")
    if isinstance(value, Tensor):
        return value.float()
    return torch.as_tensor(np.asarray(value)).float()


class ManifestDataset(Dataset[dict[str, Any]]):
    """Small adapter around manifest records and a stage-aware slice loader."""

    def __init__(
        self,
        records: Sequence[Any],
        *,
        loader: Callable[..., Any] | None = None,
        stage: str = "final",
        modalities: Sequence[str] = ("flair", "t1", "t2"),
        shared_train_scalar: float | None = None,
    ) -> None:
        self.records = list(records)
        self.loader = loader
        self.stage = str(stage)
        normalized_modalities = tuple(str(value).strip().lower() for value in modalities)
        modality_indices = {"flair": 0, "t1": 1, "t2": 2}
        if not normalized_modalities or any(value not in modality_indices for value in normalized_modalities):
            raise ValueError("modalities must be a non-empty subset of flair, t1, and t2.")
        if len(set(normalized_modalities)) != len(normalized_modalities):
            raise ValueError("modalities must not contain duplicates.")
        self.modalities = normalized_modalities
        self.modality_indices = tuple(modality_indices[value] for value in normalized_modalities)
        if shared_train_scalar is not None and (not np.isfinite(float(shared_train_scalar)) or float(shared_train_scalar) <= 0):
            raise ValueError("shared_train_scalar must be a finite positive scalar.")
        self.shared_train_scalar = None if shared_train_scalar is None else float(shared_train_scalar)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[int(index)]
        if self.loader is not None:
            try:
                image = self.loader(record, stage=self.stage)
            except TypeError:
                image = self.loader(record, self.stage)
        else:
            image = _record_value(record, self.stage)
            if image is None:
                image = _record_value(record, "image")
            if image is None:
                image = _record_value(record, "array")
            if image is None:
                image = _record_value(record, "tensor")
            if image is None:
                paths = _record_value(record, "modality_paths")
                if paths is None:
                    paths = _record_value(record, "image_paths")
                if isinstance(paths, Mapping):
                    image = paths.get(self.stage)
                elif isinstance(paths, Sequence) and not isinstance(paths, (str, bytes)):
                    image = paths
        if image is None:
            raise KeyError(
                f"Record {index} has no image for stage={self.stage!r}; "
                "provide a loader or an image/array/tensor field."
            )
        if _record_value(record, "label") is None or _record_value(record, "participant_id") is None:
            raise KeyError(
                "Domain-classifier manifest rows must include label and participant_id; "
                f"row {index} is missing a required field."
            )
        image_tensor = _to_image_tensor(image)
        if image_tensor.ndim == 2:
            image_tensor = image_tensor.unsqueeze(0)
        if image_tensor.ndim != 3:
            raise ValueError(f"Domain classifier image must be [C,H,W], got {tuple(image_tensor.shape)}")
        if image_tensor.shape[0] == 3:
            image_tensor = image_tensor[list(self.modality_indices)]
        elif image_tensor.shape[0] != len(self.modality_indices):
            raise ValueError(
                f"Image has {image_tensor.shape[0]} channels but selected modalities require "
                f"{len(self.modality_indices)}."
            )
        if self.shared_train_scalar is not None:
            image_tensor = image_tensor / self.shared_train_scalar
        return {
            "image": image_tensor,
            "label": int(_record_value(record, "label")),
            "participant_id": _record_value(record, "participant_id"),
            "pair_id": _record_value(record, "pair_id"),
            "case_id": _record_value(record, "case_id", index),
            "record_index": int(index),
        }


def read_jsonl_manifest(path: str | Path) -> list[dict[str, Any]]:
    """Read one manifest without changing or normalizing its provenance fields."""

    manifest_path = Path(path)
    records: list[dict[str, Any]] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {manifest_path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Manifest row {manifest_path}:{line_number} must be an object.")
            records.append(value)
    return records


def dataset_from_manifest(
    manifest_path: str | Path,
    *,
    stage: str = "final",
    dataset_factory: Callable[..., Any] | None = None,
    modalities: Sequence[str] = ("flair", "t1", "t2"),
    shared_train_scalar: float | None = None,
) -> Dataset[dict[str, Any]]:
    """Construct the data-agent dataset when available, with a testable fallback.

    The production data package is expected to expose
    ``DomainClassifierDataset.from_manifest(path)`` and a ``load_slice``
    method.  The fallback is useful for synthetic contract tests and does not
    attempt to infer preprocessing from filenames.
    """

    if dataset_factory is None:
        try:
            module = importlib.import_module("andi_rewrite.data.domain_classifier")
        except ModuleNotFoundError as exc:
            if exc.name != "andi_rewrite.data.domain_classifier":
                raise
            module = None
        if module is not None:
            try:
                dataset_factory = module.DomainClassifierDataset.from_manifest
            except AttributeError as exc:
                raise TypeError(
                    "andi_rewrite.data.domain_classifier must expose "
                    "DomainClassifierDataset.from_manifest."
                ) from exc
    if dataset_factory is None:
        return ManifestDataset(
            read_jsonl_manifest(manifest_path),
            stage=stage,
            modalities=modalities,
            shared_train_scalar=shared_train_scalar,
        )
    source = dataset_factory(manifest_path)
    records = getattr(source, "records", None)
    loader = getattr(source, "load_slice", None)
    if records is None or loader is None:
        raise TypeError(
            "DomainClassifierDataset must expose records and load_slice(record, stage=...)."
        )
    return ManifestDataset(
        records,
        loader=loader,
        stage=stage,
        modalities=modalities,
        shared_train_scalar=shared_train_scalar,
    )


def _subject_counts(dataset: Dataset[Any]) -> Counter[object]:
    records = getattr(dataset, "records", None)
    if records is None:
        return Counter()
    missing = [index for index, record in enumerate(records) if _record_value(record, "participant_id") is None]
    if missing:
        raise KeyError(f"Manifest records missing participant_id at indices {missing[:5]!r}.")
    return Counter(_record_value(record, "participant_id") for record in records)


def materialize_dataset(dataset: Dataset[Any]) -> dict[str, Any]:
    """Read a dataset once for statistical controls or deterministic smoke subsets."""

    images: list[np.ndarray] = []
    labels: list[int] = []
    participants: list[object] = []
    pairs: list[object] = []
    cases: list[object] = []
    for index in range(len(dataset)):
        sample = dataset[index]
        if isinstance(sample, Mapping):
            image = sample["image"]
            label = sample["label"]
            participant = sample.get("participant_id", index)
            pair = sample.get("pair_id")
            case = sample.get("case_id", index)
        elif isinstance(sample, (tuple, list)) and len(sample) >= 2:
            image, label = sample[:2]
            participant, pair, case = index, None, index
            if len(sample) >= 3 and isinstance(sample[2], Mapping):
                metadata = sample[2]
                participant = metadata.get("participant_id", index)
                pair = metadata.get("pair_id")
                case = metadata.get("case_id", index)
        else:
            raise TypeError("Dataset samples must be mappings or (image, label) tuples.")
        images.append(_to_image_tensor(image).detach().cpu().numpy())
        labels.append(int(label))
        participants.append(participant)
        pairs.append(pair)
        cases.append(case)
    if images:
        image_array = np.stack(images, axis=0)
    else:
        image_array = np.empty((0, 0, 0, 0), dtype=np.float32)
    return {
        "images": image_array,
        "labels": np.asarray(labels, dtype=np.int64),
        "participant_ids": np.asarray(participants, dtype=object),
        "pair_ids": np.asarray(pairs, dtype=object),
        "case_ids": np.asarray(cases, dtype=object),
    }


def _tensor_cache_row_key(record: Any) -> tuple[str, str, str, str, str, int]:
    """Return the immutable manifest identity used by tensor caches.

    Control designs move complete matched pairs between newly assigned
    train/validation/test splits.  The target split is therefore not part of
    this key; the original source fields remain the stable identity of the
    image.  Keeping this helper shared by the single- and multi-split
    materializers prevents a moved row from becoming invisible to a target
    split's loader.
    """

    row = _record_mapping(record)
    return (
        str(row.get("source_dataset", "")),
        str(row.get("source_split", "")),
        str(row.get("source_key", "")),
        str(row.get("participant_id", "")),
        str(row.get("case_id", "")),
        int(row.get("z", 0)),
    )


def materialize_tensor_dataset(dataset: Dataset[Any]) -> ManifestDataset:
    """Cache one exact tensor per manifest row for repeated neural epochs.

    This is used by controls whose source reader is intentionally read-only
    but expensive per sample (healthy final LMDB and BraTS NPZ cache reads).
    Each row is loaded once through the existing dataset contract, then the
    resulting model-grid tensor is reused by every epoch and per-seed label
    clone.  No normalization, resize, or label/provenance field is changed.
    """

    records = getattr(dataset, "records", None)
    if records is None:
        raise TypeError("Tensor materialization requires a dataset exposing records.")

    cached: list[Tensor] = []
    for index in range(len(records)):
        sample = dataset[index]
        if not isinstance(sample, Mapping) or "image" not in sample:
            raise TypeError("Tensor materialization requires mapping samples with an image field.")
        value = _to_image_tensor(sample["image"]).detach().cpu().contiguous()
        if value.ndim != 3 or not bool(torch.isfinite(value).all()):
            raise ValueError(f"Materialized image at index {index} is not finite [C,H,W].")
        cached.append(value)
    by_key = {
        _tensor_cache_row_key(record): cached[index]
        for index, record in enumerate(records)
    }

    def loader(record: Any, *, stage: str) -> Tensor:
        del stage
        value = by_key.get(_tensor_cache_row_key(record))
        if value is None:
            raise KeyError("Tensor cache lookup did not find the requested manifest record.")
        return value

    # ``subset_dataset`` normally wraps production readers so mapping rows
    # become SliceRecord instances.  This cache is deliberately mapping based
    # and must keep transformed rows (including labels/metadata) untouched.
    loader._accepts_mapping_records = True  # type: ignore[attr-defined]

    return ManifestDataset(
        records,
        loader=loader,
        stage=getattr(dataset, "stage", "final"),
        modalities=getattr(dataset, "modalities", ("flair", "t1", "t2")),
        # Cached tensors already include any scalar applied by the source
        # dataset.  Avoid applying it a second time in ManifestDataset.
        shared_train_scalar=None,
    )


def materialize_tensor_datasets(
    datasets: Mapping[str, Dataset[Any]],
) -> dict[str, ManifestDataset]:
    """Cache several splits behind one shared immutable tensor lookup.

    The powered shuffle control reassigns complete pairs to a fresh 40/10/50
    split.  A row originally read from ``val`` can consequently be presented
    through the new ``test`` dataset.  Caching each original split separately
    makes that legitimate move fail at lookup time.  This helper reads each
    original row once and gives every returned split the same cache, while
    retaining each split's stage and selected modality metadata.
    """

    required = ("train", "val", "test")
    missing = [split for split in required if split not in datasets]
    if missing:
        raise KeyError(f"Tensor materialization requires splits {missing!r}.")

    by_key: dict[tuple[str, str, str, str, str, int], Tensor] = {}
    records_by_split: dict[str, list[Any]] = {}
    for split in required:
        dataset = datasets[split]
        records = getattr(dataset, "records", None)
        if records is None:
            raise TypeError("Tensor materialization requires every dataset to expose records.")
        records_by_split[split] = list(records)
        for index, record in enumerate(records):
            sample = dataset[index]
            if not isinstance(sample, Mapping) or "image" not in sample:
                raise TypeError(
                    "Tensor materialization requires mapping samples with an image field."
                )
            value = _to_image_tensor(sample["image"]).detach().cpu().contiguous()
            if value.ndim != 3 or not bool(torch.isfinite(value).all()):
                raise ValueError(
                    f"Materialized image at {split}[{index}] is not finite [C,H,W]."
                )
            key = _tensor_cache_row_key(record)
            previous = by_key.get(key)
            if previous is not None:
                # Repeated manifest rows are harmless only when they really
                # refer to the same exact tensor.  Refuse silent collisions.
                if not torch.equal(previous, value):
                    raise ValueError(f"Conflicting tensors share manifest key {key!r}.")
            else:
                by_key[key] = value

    def loader(record: Any, *, stage: str) -> Tensor:
        del stage
        value = by_key.get(_tensor_cache_row_key(record))
        if value is None:
            raise KeyError("Tensor cache lookup did not find the requested manifest record.")
        return value

    loader._accepts_mapping_records = True  # type: ignore[attr-defined]

    output: dict[str, ManifestDataset] = {}
    for split in required:
        source = datasets[split]
        output[split] = ManifestDataset(
            records_by_split[split],
            loader=loader,
            stage=getattr(source, "stage", "final"),
            modalities=getattr(source, "modalities", ("flair", "t1", "t2")),
            # Cached tensors already include source-side scalar application.
            shared_train_scalar=None,
        )
    return output


def fit_shared_train_scalar(images: np.ndarray | Tensor, *, quantile: float = 0.995) -> float:
    """Fit one scalar on all train images for an optional registered control."""

    values = images.detach().cpu().numpy() if isinstance(images, Tensor) else np.asarray(images)
    if values.size == 0:
        raise ValueError("Cannot fit a shared scalar on an empty train set.")
    if not 0.0 < float(quantile) <= 1.0:
        raise ValueError("quantile must be in (0, 1].")
    finite = np.abs(values.astype(np.float64, copy=False).reshape(-1))
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("Train images contain no finite values.")
    scalar = float(np.quantile(finite, float(quantile)))
    return max(scalar, 1.0e-6)


def select_tiny_records(
    records: Sequence[Any],
    *,
    subjects_per_label: int = 10,
    max_slices: int = 300,
    seed: int = 73,
) -> list[Any]:
    """Select a deterministic 10+10-subject smoke set with a slice cap."""

    if int(subjects_per_label) <= 0 or int(max_slices) <= 0:
        raise ValueError("subjects_per_label and max_slices must be positive.")
    by_label: dict[int, dict[object, list[Any]]] = {0: {}, 1: {}}
    for record in records:
        label = _record_value(record, "label")
        participant = _record_value(record, "participant_id")
        if label not in (0, 1) or participant is None:
            raise ValueError("Tiny selection requires binary labels and participant_id on every record.")
        by_label[int(label)].setdefault(participant, []).append(record)
    rng = np.random.default_rng(int(seed))
    selected_participants: list[object] = []
    for label in (0, 1):
        candidates = sorted(by_label[label], key=lambda value: str(value))
        if len(candidates) < int(subjects_per_label):
            raise ValueError(
                f"Tiny selection needs {subjects_per_label} subjects for label {label}, found {len(candidates)}."
            )
        order = rng.permutation(len(candidates))[: int(subjects_per_label)]
        selected_participants.extend(candidates[int(index)] for index in order)
    selected_by_subject: dict[object, list[Any]] = {}
    for label in (0, 1):
        for participant in selected_participants:
            if participant in by_label[label]:
                selected_by_subject[participant] = by_label[label][participant]
    # The comprehensions above preserve the chosen subject order; sort slices
    # by z/case only after the participant sample is frozen.
    ordered: list[Any] = []
    for participant in selected_participants:
        rows = selected_by_subject[participant]
        ordered.extend(
            sorted(rows, key=lambda row: (int(_record_value(row, "z", 0)), str(_record_value(row, "case_id", ""))))
        )
    if len(ordered) <= int(max_slices):
        return ordered
    # Keep at least one slice for every chosen subject, then fill the remaining
    # budget in deterministic subject/z order.
    minimum = len(selected_participants)
    if int(max_slices) < minimum:
        raise ValueError("max_slices is smaller than the number of selected subjects.")
    kept: list[Any] = []
    remaining: list[Any] = []
    for participant in selected_participants:
        rows = selected_by_subject[participant]
        kept.append(rows[0])
        remaining.extend(rows[1:])
    budget = int(max_slices) - len(kept)
    kept.extend(remaining[:budget])
    return kept


def subset_dataset(dataset: Dataset[Any], records: Sequence[Any]) -> ManifestDataset:
    """Preserve a data-agent loader while replacing records for a smoke run."""

    return ManifestDataset(
        records,
        loader=_wrap_slice_loader(getattr(dataset, "loader", None)),
        stage=getattr(dataset, "stage", "final"),
        modalities=getattr(dataset, "modalities", ("flair", "t1", "t2")),
        shared_train_scalar=getattr(dataset, "shared_train_scalar", None),
    )


def dataset_with_shared_train_scalar(
    dataset: Dataset[Any],
    scalar: float | None,
) -> ManifestDataset:
    """Clone a manifest-backed dataset while changing one shared input scalar.

    The registered positive control fits this value from the training split
    once and applies that same value to every split.  Keeping this operation
    explicit prevents accidental per-volume normalization and also preserves
    the data-agent stage-aware loader when records are ``SliceRecord`` objects.
    """

    records = getattr(dataset, "records", None)
    if records is None:
        raise TypeError("A shared train scalar requires a dataset exposing records.")
    return ManifestDataset(
        records,
        loader=_wrap_slice_loader(getattr(dataset, "loader", None)),
        stage=getattr(dataset, "stage", "final"),
        modalities=getattr(dataset, "modalities", ("flair", "t1", "t2")),
        shared_train_scalar=scalar,
    )


def dataset_with_stage(
    dataset: Dataset[Any],
    stage: str,
) -> ManifestDataset:
    """Clone a manifest-backed dataset while selecting an explicit image stage."""

    records = getattr(dataset, "records", None)
    if records is None:
        raise TypeError("An explicit stage requires a dataset exposing records.")
    return ManifestDataset(
        records,
        loader=_wrap_slice_loader(getattr(dataset, "loader", None)),
        stage=str(stage),
        modalities=getattr(dataset, "modalities", ("flair", "t1", "t2")),
        shared_train_scalar=getattr(dataset, "shared_train_scalar", None),
    )


def materialize_registered_dataset(
    dataset: Dataset[Any],
    *,
    image_size: int = 128,
) -> ManifestDataset:
    """Read registered NIfTI volumes once and expose exact cached slices.

    The registered positive control is intentionally pre-IQR and therefore
    cannot use the final LMDB cache.  The generic reader's path-backed branch
    reads every full NIfTI once per slice, which makes a multi-epoch control
    needlessly I/O bound.  This adapter groups manifest rows by their three
    source paths, reads one volume at a time, applies the reader's exact
    resize operation, and retains only the resulting model-grid slices.  The
    resulting in-memory dataset is bounded by the selected split (about 1 GB
    for the FOMO audit) and never keeps full volumes from earlier groups.
    """

    records = getattr(dataset, "records", None)
    if records is None:
        raise TypeError("Registered caching requires a dataset exposing records.")
    if int(image_size) != 128:
        raise ValueError("Registered caching currently follows the fixed 128x128 reader contract.")
    try:
        from andi_rewrite.data.domain_classifier.readers import (
            _as_channel_volume,
            _read_array,
            _resize_slice,
        )
    except ImportError as exc:  # pragma: no cover - production data package contract
        raise ImportError("Registered caching requires the domain-classifier reader package.") from exc

    def path_key(record: Any) -> tuple[str, str, str]:
        row = _record_mapping(record)
        value = row.get("registered_paths") or row.get("image_paths")
        if not isinstance(value, Mapping):
            raise ValueError("Registered records require registered_paths or image_paths.")
        canonical = {
            str(key).strip().lower(): str(path)
            for key, path in value.items()
        }
        paths = tuple(canonical.get(modality, "") for modality in ("flair", "t1", "t2"))
        if any(not path for path in paths):
            raise ValueError(f"Registered record is missing a modality path: {paths!r}")
        return paths

    def row_key(record: Any) -> tuple[str, str, str, str, str, int]:
        row = _record_mapping(record)
        return (
            str(row.get("source_dataset", "")),
            str(row.get("source_split", "")),
            str(row.get("source_key", "")),
            str(row.get("participant_id", "")),
            str(row.get("case_id", "")),
            int(row.get("z", 0)),
        )

    groups: dict[tuple[str, str, str], list[int]] = {}
    for index, record in enumerate(records):
        groups.setdefault(path_key(record), []).append(int(index))
    cached: list[Tensor | None] = [None] * len(records)
    for paths, indices in groups.items():
        channels: list[np.ndarray] = []
        for path_text in paths:
            array = _read_array(Path(path_text))
            channel_volume = _as_channel_volume(array, channels=3)
            if channel_volume.shape[0] != 1:
                raise ValueError(f"Expected one modality per registered path, found {array.shape}.")
            channels.append(np.asarray(channel_volume[0]))
        reference_shape = tuple(channels[0].shape)
        if any(tuple(channel.shape) != reference_shape for channel in channels[1:]):
            raise ValueError("Registered modality geometry mismatch.")
        volume = np.stack(channels, axis=0)
        for index in indices:
            z = int(_record_value(records[index], "z", 0))
            if z < 0 or z >= volume.shape[-1]:
                raise IndexError(f"Registered z={z} outside volume depth={volume.shape[-1]}.")
            tensor = _resize_slice(volume[..., z], image_size=int(image_size))
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"Registered input contains NaN/Inf at record index {index}.")
            cached[index] = tensor.contiguous()
        del volume

    if any(value is None for value in cached):
        raise RuntimeError("Registered cache did not populate every manifest record.")
    by_key = {row_key(record): cached[index] for index, record in enumerate(records)}

    def loader(record: Any, *, stage: str) -> Tensor:
        del stage
        value = by_key.get(row_key(record))
        if value is None:
            raise KeyError("Registered cache lookup did not find the requested manifest record.")
        return value

    return ManifestDataset(
        records,
        loader=loader,
        stage="registered",
        modalities=getattr(dataset, "modalities", ("flair", "t1", "t2")),
        shared_train_scalar=getattr(dataset, "shared_train_scalar", None),
    )


def dataset_with_modalities(
    dataset: Dataset[Any],
    modalities: Sequence[str],
) -> ManifestDataset:
    """Clone a manifest-backed dataset with an explicit channel subset."""

    records = getattr(dataset, "records", None)
    if records is None:
        raise TypeError("An explicit modality subset requires a dataset exposing records.")
    return ManifestDataset(
        records,
        loader=_wrap_slice_loader(getattr(dataset, "loader", None)),
        stage=getattr(dataset, "stage", "final"),
        modalities=modalities,
        shared_train_scalar=getattr(dataset, "shared_train_scalar", None),
    )


def subject_balanced_bce_with_logits(
    logits: Tensor,
    labels: Tensor,
    participant_ids: Sequence[object] | Tensor,
    *,
    subject_slice_counts: Mapping[object, int] | None = None,
    subject_normalizer: float | None = None,
) -> Tensor:
    """BCE with each participant contributing equal total slice weight.

    When ``subject_slice_counts`` is the complete training map, callers may
    provide ``subject_normalizer`` (normally the number of training subjects).
    This keeps the denominator fixed across minibatches.  Dividing by the
    minibatch weight sum would make the effective objective depend on which
    subjects happened to land in that batch and can cancel the intended
    inverse-slice weighting for tiny or uneven batches.
    """

    flattened_logits = logits.reshape(-1)
    flattened_labels = labels.float().reshape(-1)
    if flattened_logits.numel() != flattened_labels.numel():
        raise ValueError("logits and labels must have the same number of values.")
    if isinstance(participant_ids, Tensor):
        identifiers = participant_ids.detach().cpu().reshape(-1).tolist()
    else:
        identifiers = list(participant_ids)
    if len(identifiers) != flattened_logits.numel():
        raise ValueError("participant_ids must match the batch size.")
    counts = Counter(identifiers)
    if subject_slice_counts is not None:
        counts = Counter({key: int(value) for key, value in subject_slice_counts.items()})
    weights = torch.as_tensor(
        [1.0 / max(int(counts.get(identifier, 1)), 1) for identifier in identifiers],
        dtype=flattened_logits.dtype,
        device=flattened_logits.device,
    )
    losses = F.binary_cross_entropy_with_logits(flattened_logits, flattened_labels, reduction="none")
    if subject_normalizer is None:
        denominator = weights.sum()
    else:
        value = float(subject_normalizer)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("subject_normalizer must be finite and positive.")
        denominator = torch.as_tensor(value, dtype=weights.dtype, device=weights.device)
    return torch.sum(losses * weights) / torch.clamp(denominator, min=torch.finfo(weights.dtype).eps)


def _collate_samples(samples: Sequence[Any]) -> dict[str, Any]:
    if not samples:
        raise ValueError("Cannot collate an empty batch.")
    if isinstance(samples[0], Mapping):
        images = [_to_image_tensor(sample["image"]) for sample in samples]
        labels = [int(sample["label"]) for sample in samples]
        return {
            "image": torch.stack(images, dim=0),
            "label": torch.as_tensor(labels, dtype=torch.float32),
            "participant_id": [sample.get("participant_id", index) for index, sample in enumerate(samples)],
            "pair_id": [sample.get("pair_id") for sample in samples],
            "case_id": [sample.get("case_id") for sample in samples],
            "record_index": [sample.get("record_index") for sample in samples],
        }
    if isinstance(samples[0], (tuple, list)) and len(samples[0]) >= 2:
        images = [_to_image_tensor(sample[0]) for sample in samples]
        labels = [int(sample[1]) for sample in samples]
        return {
            "image": torch.stack(images, dim=0),
            "label": torch.as_tensor(labels, dtype=torch.float32),
            "participant_id": list(range(len(samples))),
            "pair_id": [None] * len(samples),
            "case_id": list(range(len(samples))),
            "record_index": list(range(len(samples))),
        }
    raise TypeError("Dataset samples must be mappings or (image, label) tuples.")


def _scores_from_output(output: Tensor) -> Tensor:
    values = output
    if values.ndim == 2 and values.shape[1] == 2:
        return torch.softmax(values, dim=1)[:, 1]
    return torch.sigmoid(values.reshape(-1))


def _evaluate(
    model: nn.Module,
    dataset: Dataset[Any],
    *,
    config: TrainConfig,
    device: torch.device,
) -> dict[str, Any]:
    loader = DataLoader(
        dataset,
        batch_size=max(1, int(config.batch_size)),
        shuffle=False,
        num_workers=max(0, int(config.num_workers)),
        collate_fn=_collate_samples,
    )
    model.eval()
    labels: list[int] = []
    scores: list[float] = []
    participants: list[object] = []
    pairs: list[object] = []
    cases: list[object] = []
    losses: list[float] = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            y = batch["label"].to(device)
            output = model(images)
            probability = _scores_from_output(output)
            logits = output[:, 1] - output[:, 0] if output.ndim == 2 and output.shape[1] == 2 else output.reshape(-1)
            batch_loss = F.binary_cross_entropy_with_logits(logits, y)
            if not torch.isfinite(batch_loss):
                raise FloatingPointError("Non-finite domain-classifier evaluation loss.")
            losses.append(float(batch_loss.item()))
            labels.extend(int(value) for value in batch["label"].tolist())
            scores.extend(float(value) for value in probability.detach().cpu().tolist())
            participants.extend(batch["participant_id"])
            pairs.extend(batch["pair_id"])
            cases.extend(batch["case_id"])
    metrics = compute_dataset_metrics(
        labels,
        scores,
        participant_ids=participants,
        pair_ids=pairs if any(value is not None and str(value).strip() != "" for value in pairs) else None,
        threshold=config.threshold,
        subject_method=config.subject_method,
    )
    metrics["loss"] = float(np.mean(losses)) if losses else float("nan")
    metrics["labels"] = np.asarray(labels, dtype=np.int64)
    metrics["scores"] = np.asarray(scores, dtype=np.float64)
    metrics["participant_ids"] = np.asarray(participants, dtype=object)
    metrics["pair_ids"] = np.asarray(pairs, dtype=object)
    metrics["case_ids"] = np.asarray(cases, dtype=object)
    return metrics


def prediction_rows(metrics: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Convert an internal evaluation result into stable per-slice rows."""

    labels = np.asarray(metrics.get("labels", []), dtype=np.int64).reshape(-1)
    scores = np.asarray(metrics.get("scores", []), dtype=np.float64).reshape(-1)
    participants = np.asarray(metrics.get("participant_ids", []), dtype=object).reshape(-1)
    pairs = np.asarray(metrics.get("pair_ids", [None] * labels.size), dtype=object).reshape(-1)
    cases = np.asarray(metrics.get("case_ids", list(range(labels.size))), dtype=object).reshape(-1)
    sizes = {labels.size, scores.size, participants.size, pairs.size, cases.size}
    if len(sizes) != 1:
        raise ValueError("Evaluation prediction arrays have incompatible lengths.")
    return [
        {
            "record_index": int(index),
            "label": int(labels[index]),
            "probability": float(scores[index]),
            "participant_id": _json_scalar(participants[index]),
            "pair_id": _json_scalar(pairs[index]),
            "case_id": _json_scalar(cases[index]),
        }
        for index in range(labels.size)
    ]


def subject_prediction_rows(metrics: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Convert subject aggregation retained inside an evaluation result."""

    subject = metrics.get("subject_predictions")
    if not isinstance(subject, Mapping):
        return []
    labels = np.asarray(subject.get("labels", []), dtype=np.int64).reshape(-1)
    scores = np.asarray(subject.get("scores", []), dtype=np.float64).reshape(-1)
    participants = np.asarray(subject.get("participant_ids", []), dtype=object).reshape(-1)
    pairs = np.asarray(subject.get("pair_ids", [None] * labels.size), dtype=object).reshape(-1)
    counts = np.asarray(subject.get("slice_counts", [1] * labels.size), dtype=np.int64).reshape(-1)
    if len({labels.size, scores.size, participants.size, pairs.size, counts.size}) != 1:
        raise ValueError("Subject prediction arrays have incompatible lengths.")
    return [
        {
            "subject_index": int(index),
            "label": int(labels[index]),
            "probability": float(scores[index]),
            "participant_id": _json_scalar(participants[index]),
            "pair_id": _json_scalar(pairs[index]),
            "slice_count": int(counts[index]),
        }
        for index in range(labels.size)
    ]


def _json_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def write_prediction_rows(path: str | Path, rows: Iterable[Mapping[str, Any]], *, overwrite: bool = False) -> None:
    """Write prediction rows atomically as JSONL, preserving one row per slice."""

    destination = Path(path)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite prediction artifact: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps({str(key): _json_scalar(value) for key, value in row.items()}, sort_keys=True))
            handle.write("\n")
    temporary.replace(destination)


def atomic_write_json(path: str | Path, value: Mapping[str, Any], *, overwrite: bool = False) -> None:
    """Atomically persist JSON and refuse an existing artifact by default."""

    destination = Path(path)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite JSON artifact: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=True, default=_json_default)
        handle.write("\n")
    temporary.replace(destination)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Tensor):
        return None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (set, tuple)):
        return list(value)
    raise TypeError(f"Value of type {type(value).__name__} is not JSON serializable.")


def sha256_file(path: str | Path) -> str:
    """Hash a manifest/config/code file for resumable-run identity."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None


def run_fingerprint(
    config: TrainConfig,
    manifest_paths: Mapping[str, str | Path],
    *,
    code_paths: Sequence[str | Path] = (),
) -> str:
    """Build a content identity used to guard resumable output directories."""

    payload = {
        "config": asdict(config),
        "manifests": {split: {"path": str(path), "sha256": sha256_file(path)} for split, path in sorted(manifest_paths.items())},
        "code": {str(path): sha256_file(path) for path in sorted((Path(item) for item in code_paths), key=str)},
        "git_revision": git_revision(),
    }
    encoded = json.dumps(payload, sort_keys=True, default=_json_default).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validation_key(metrics: Mapping[str, Any]) -> tuple[float, float]:
    subject = metrics.get("subject")
    if not isinstance(subject, Mapping):
        subject = metrics.get("slice", {})
    auc = float(subject.get("roc_auc", float("nan")))
    bce = float(subject.get("bce", float("nan")))
    return (auc if np.isfinite(auc) else -float("inf"), -(bce if np.isfinite(bce) else float("inf")))


def _is_better(current: Mapping[str, Any], best: Mapping[str, Any] | None) -> bool:
    if best is None:
        return True
    current_key = _validation_key(current)
    best_key = _validation_key(best)
    if current_key[0] > best_key[0] + 1.0e-12:
        return True
    return abs(current_key[0] - best_key[0]) <= 1.0e-12 and current_key[1] > best_key[1] + 1.0e-12


class DomainClassifierRunner:
    """Train one or more fixed-split domain classifiers."""

    def __init__(self, config: TrainConfig | None = None) -> None:
        self.config = config or TrainConfig()

    def train_one_seed(
        self,
        train_dataset: Dataset[Any],
        val_dataset: Dataset[Any],
        test_dataset: Dataset[Any],
        *,
        seed: int | None = None,
        config: TrainConfig | None = None,
    ) -> dict[str, Any]:
        cfg = config or self.config
        if seed is not None:
            cfg = replace(cfg, seed=int(seed))
        set_deterministic_seed(cfg.seed)
        device = cfg.resolved_device()
        model = build_model(
            cfg.model,
            in_channels=cfg.in_channels,
            num_classes=cfg.num_classes,
            widths=cfg.widths,
            groupnorm_groups=cfg.groupnorm_groups,
            dropout=cfg.dropout,
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.resolved_learning_rate(),
            weight_decay=float(cfg.weight_decay),
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=max(1, int(cfg.batch_size)),
            shuffle=True,
            num_workers=max(0, int(cfg.num_workers)),
            collate_fn=_collate_samples,
            generator=torch.Generator().manual_seed(int(cfg.seed)),
        )
        counts = _subject_counts(train_dataset)
        history: list[dict[str, Any]] = []
        best_state: dict[str, Tensor] | None = None
        best_val: dict[str, Any] | None = None
        best_epoch = 0
        epochs_without_improvement = 0
        for epoch in range(1, max(1, int(cfg.max_epochs)) + 1):
            epoch_started = time.perf_counter()
            model.train()
            train_losses: list[float] = []
            for batch in train_loader:
                images = batch["image"].to(device)
                labels = batch["label"].to(device)
                output = model(images)
                logits = output[:, 1] - output[:, 0] if output.ndim == 2 and output.shape[1] == 2 else output.reshape(-1)
                loss = subject_balanced_bce_with_logits(
                    logits,
                    labels,
                    batch["participant_id"],
                    subject_slice_counts=counts,
                    subject_normalizer=float(len(counts)) if counts else None,
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite domain-classifier loss at epoch {epoch}.")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                train_losses.append(float(loss.detach().cpu().item()))
            val_metrics = _evaluate(model, val_dataset, config=cfg, device=device)
            entry = {
                "epoch": epoch,
                "train_loss": float(np.mean(train_losses)) if train_losses else float("nan"),
                "val_loss": val_metrics.get("loss"),
                "val_slice": val_metrics.get("slice"),
                "val_subject": val_metrics.get("subject"),
                "epoch_elapsed_seconds": float(time.perf_counter() - epoch_started),
            }
            history.append(entry)
            if _is_better(val_metrics, best_val):
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                best_val = copy.deepcopy(val_metrics)
                best_epoch = epoch
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if cfg.early_stopping and not cfg.tiny and epochs_without_improvement >= max(1, int(cfg.patience)):
                    break
        final_val_metrics = _evaluate(model, val_dataset, config=cfg, device=device)
        selected_val_metrics = final_val_metrics if cfg.tiny or not cfg.early_stopping else (best_val or final_val_metrics)
        if best_state is not None and not (cfg.tiny or not cfg.early_stopping):
            model.load_state_dict(best_state)
        state_for_result = (
            {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            if cfg.tiny or not cfg.early_stopping
            else best_state
        )
        selected_epoch = int(len(history)) if cfg.tiny or not cfg.early_stopping else int(best_epoch)
        test_metrics = _evaluate(model, test_dataset, config=cfg, device=device)
        test_statistics: dict[str, Any] = {}
        subject_predictions = test_metrics.get("subject_predictions")
        if isinstance(subject_predictions, Mapping) and len(subject_predictions.get("labels", [])):
            subject_pair_ids = subject_predictions.get("pair_ids")
            pair_values = (
                subject_pair_ids
                if subject_pair_ids is not None and all(
                    value is not None and str(value).strip() != ""
                    for value in np.asarray(subject_pair_ids, dtype=object).tolist()
                )
                else None
            )
            # Retrained null replicates may explicitly set both statistics
            # budgets to zero.  They still fit, select, and persist the exact
            # held-out predictions, while avoiding thousands of redundant
            # resamples per replicate.  Ordinary observed fits retain the
            # preregistered positive budgets.
            if int(cfg.bootstrap_replicates) > 0:
                test_statistics["subject_bootstrap"] = bootstrap_subject_auc(
                    subject_predictions["labels"],
                    subject_predictions["scores"],
                    pair_ids=pair_values,
                    n_bootstrap=int(cfg.bootstrap_replicates),
                    seed=int(cfg.seed),
                )
            if pair_values is not None and int(cfg.swap_replicates) > 0:
                test_statistics["heldout_pair_swap"] = paired_heldout_swap_test(
                    subject_predictions["labels"],
                    subject_predictions["scores"],
                    subject_pair_ids,
                    n_swaps=int(cfg.swap_replicates),
                    seed=int(cfg.seed),
                )
        train_final_metrics = None
        # Tiny controls explicitly describe fit quality at the final state.  The
        # normal audit never uses train metrics for checkpoint selection.
        if cfg.tiny:
            train_final_metrics = _evaluate(model, train_dataset, config=cfg, device=device)
        result = {
            "seed": int(cfg.seed),
            "split_seed": int(cfg.split_seed),
            "config": asdict(cfg),
            "device": str(device),
            "best_epoch": selected_epoch,
            "epochs_completed": int(len(history)),
            "history": history,
            "validation": _strip_prediction_arrays(selected_val_metrics),
            "test": _strip_prediction_arrays(test_metrics),
            "test_statistics": _strip_prediction_arrays(test_statistics),
            "train_final": _strip_prediction_arrays(train_final_metrics) if train_final_metrics is not None else None,
            "validation_predictions": prediction_rows(selected_val_metrics),
            "validation_subject_predictions": subject_prediction_rows(selected_val_metrics),
            "test_predictions": prediction_rows(test_metrics),
            "test_subject_predictions": subject_prediction_rows(test_metrics),
            "train_final_predictions": prediction_rows(train_final_metrics) if train_final_metrics is not None else [],
            "train_final_subject_predictions": subject_prediction_rows(train_final_metrics) if train_final_metrics is not None else [],
            "model": model_description(model),
            "state_dict": state_for_result,
        }
        return result

    def train_many_seeds(
        self,
        train_dataset: Dataset[Any],
        val_dataset: Dataset[Any],
        test_dataset: Dataset[Any],
        *,
        seeds: Sequence[int] = (73, 173, 273),
        config: TrainConfig | None = None,
    ) -> list[dict[str, Any]]:
        return [
            self.train_one_seed(
                train_dataset,
                val_dataset,
                test_dataset,
                seed=int(seed),
                config=config,
            )
            for seed in seeds
        ]


def _strip_prediction_arrays(value: Mapping[str, Any]) -> dict[str, Any]:
    """Remove internal NumPy arrays before serializing a run result."""

    output: dict[str, Any] = {}
    for key, item in value.items():
        if isinstance(item, np.ndarray):
            continue
        if isinstance(item, Mapping):
            output[key] = _strip_prediction_arrays(item)
        else:
            output[key] = item
    return output


class TrainOnlyStandardizer:
    """A tiny train-only scaler for statistical feature controls."""

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, features: np.ndarray) -> "TrainOnlyStandardizer":
        values = np.asarray(features, dtype=np.float64)
        if values.ndim != 2 or values.shape[0] == 0:
            raise ValueError("features must be a non-empty [n_samples, n_features] array.")
        self.mean_ = values.mean(axis=0)
        scale = values.std(axis=0)
        self.scale_ = np.where(scale > 1.0e-12, scale, 1.0)
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("TrainOnlyStandardizer.fit must be called first.")
        values = np.asarray(features, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.mean_.shape[0]:
            raise ValueError("features have an incompatible shape.")
        return (values - self.mean_) / self.scale_


def extract_statistical_features(
    images: np.ndarray | Tensor,
    *,
    background_value: float = -1.0,
    background_tolerance: float = 1.0e-6,
) -> np.ndarray:
    """Extract fixed per-slice intensity, foreground, sharpness, and radial features.

    The feature control is intentionally simple and interpretable: intensity
    quantiles are computed over all pixels and over a robust foreground mask,
    while edge energy and coarse radial FFT bins capture sharpness and scanner
    spectrum.  Its scaler is fit on train rows only by
    :func:`fit_statistical_logistic`.  The default background value ``-1`` is
    the robust-IQR model-space sentinel.  Registered/raw controls must pass
    ``background_value=0`` explicitly; inferring a background from each image
    would leak distribution information into the feature definition.
    """

    values = images.detach().cpu().numpy() if isinstance(images, Tensor) else np.asarray(images)
    if values.ndim == 3:
        values = values[:, None, :, :]
    if values.ndim != 4:
        raise ValueError("images must have shape [n, channels, height, width].")
    if not np.isfinite(float(background_value)) or float(background_tolerance) < 0.0:
        raise ValueError("background_value must be finite and tolerance non-negative.")
    values = values.astype(np.float64, copy=False)
    features: list[np.ndarray] = []
    for channel in range(values.shape[1]):
        data = values[:, channel]
        quantiles = np.quantile(data, [0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99], axis=(1, 2))
        mean = data.mean(axis=(1, 2))
        std = data.std(axis=(1, 2))
        minimum = data.min(axis=(1, 2))
        maximum = data.max(axis=(1, 2))
        # Robust-IQR stores background as exactly -1.  Registered/raw arrays
        # generally use zero background and select that explicitly at the call
        # site.  A foreground value can theoretically equal the sentinel, so
        # this predicate is an auditable approximation rather than a lesion
        # mask; its convention is persisted with the control result.
        foreground = np.abs(data - float(background_value)) > float(background_tolerance)
        foreground_area = foreground.mean(axis=(1, 2))
        foreground_count = foreground.sum(axis=(1, 2)).astype(np.float64)
        foreground_values = np.where(foreground, data, 0.0)
        foreground_mean = foreground_values.sum(axis=(1, 2)) / np.maximum(foreground_count, 1.0)
        foreground_second = (foreground_values * foreground_values).sum(axis=(1, 2)) / np.maximum(foreground_count, 1.0)
        foreground_std = np.sqrt(np.maximum(foreground_second - foreground_mean * foreground_mean, 0.0))
        gx = np.diff(data, axis=2)
        gy = np.diff(data, axis=1)
        grad = np.concatenate([gx.reshape(data.shape[0], -1), gy.reshape(data.shape[0], -1)], axis=1)
        gradient_mean = np.mean(np.abs(grad), axis=1)
        gradient_std = np.std(grad, axis=1)
        laplace = (
            -4.0 * data
            + np.roll(data, 1, axis=1)
            + np.roll(data, -1, axis=1)
            + np.roll(data, 1, axis=2)
            + np.roll(data, -1, axis=2)
        )
        sharpness = np.mean(np.abs(laplace), axis=(1, 2))
        radial = _radial_fft_features(data)
        features.extend(
            [
                mean,
                std,
                minimum,
                maximum,
                foreground_area,
                foreground_mean,
                foreground_std,
                gradient_mean,
                gradient_std,
                sharpness,
                *quantiles,
                *[radial[:, index] for index in range(radial.shape[1])],
            ]
        )
    return np.stack(features, axis=1)


def _radial_fft_features(data: np.ndarray, bins: int = 8) -> np.ndarray:
    """Return mean log-power in concentric normalized frequency bins."""

    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(data, axes=(-2, -1)), axes=(-2, -1)))
    power = np.log1p(spectrum)
    height, width = data.shape[-2:]
    y = np.linspace(-1.0, 1.0, height, dtype=np.float64)[:, None]
    x = np.linspace(-1.0, 1.0, width, dtype=np.float64)[None, :]
    radius = np.sqrt(y * y + x * x) / np.sqrt(2.0)
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    output = np.zeros((data.shape[0], int(bins)), dtype=np.float64)
    for index in range(int(bins)):
        mask = (radius >= edges[index]) & (radius < edges[index + 1] if index + 1 < len(edges) else radius <= edges[index + 1])
        if np.any(mask):
            output[:, index] = power[:, mask].mean(axis=1)
    return output


def fit_statistical_logistic(
    train_features: np.ndarray,
    train_labels: Sequence[int] | np.ndarray,
    eval_features: np.ndarray,
    *,
    seed: int = 73,
    c: float = 1.0,
    train_participant_ids: Sequence[object] | np.ndarray | None = None,
) -> dict[str, Any]:
    """Fit a logistic control with train-only scaling and subject weighting."""

    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("The statistical control requires scikit-learn.") from exc
    train_values = np.asarray(train_features)
    eval_values = np.asarray(eval_features)
    train_y = np.asarray(train_labels, dtype=np.int64).reshape(-1)
    if train_values.ndim != 2 or train_values.shape[0] != train_y.size:
        raise ValueError("train_features and train_labels must have matching rows.")
    if not np.isin(train_y, (0, 1)).all():
        raise ValueError("train_labels must contain only 0 and 1.")
    if eval_values.ndim != 2 or eval_values.shape[1] != train_values.shape[1]:
        raise ValueError("eval_features must have the same feature columns as train_features.")
    sample_weight: np.ndarray | None = None
    if train_participant_ids is not None:
        participants = np.asarray(train_participant_ids, dtype=object).reshape(-1)
        if participants.size != train_y.size:
            raise ValueError("train_participant_ids must match train_features rows.")
        if any(value is None or str(value).strip() == "" for value in participants.tolist()):
            raise ValueError("train_participant_ids must be non-empty when supplied.")
        counts = Counter(participants.tolist())
        sample_weight = np.asarray(
            [1.0 / float(max(counts[value], 1)) for value in participants.tolist()],
            dtype=np.float64,
        )
    scaler = TrainOnlyStandardizer().fit(train_values)
    x_train = scaler.transform(train_values)
    x_eval = scaler.transform(eval_values)
    model = LogisticRegression(C=float(c), max_iter=1000, random_state=int(seed), solver="lbfgs")
    if sample_weight is None:
        model.fit(x_train, train_y)
    else:
        model.fit(x_train, train_y, sample_weight=sample_weight)
    scores = model.predict_proba(x_eval)[:, 1]
    return {
        "model": model,
        "scaler": scaler,
        "scores": scores,
        "train_feature_mean": scaler.mean_.copy() if scaler.mean_ is not None else None,
        "train_feature_scale": scaler.scale_.copy() if scaler.scale_ is not None else None,
        "train_only_scaler": True,
        "subject_weighting": sample_weight is not None,
        "train_sample_weight": sample_weight,
    }


def run_statistical_control(
    train_images: np.ndarray | Tensor,
    train_labels: Sequence[int] | np.ndarray,
    eval_images: np.ndarray | Tensor,
    eval_labels: Sequence[int] | np.ndarray,
    *,
    train_participant_ids: Sequence[object] | np.ndarray | None = None,
    eval_participant_ids: Sequence[object] | np.ndarray | None = None,
    eval_pair_ids: Sequence[object] | np.ndarray | None = None,
    eval_case_ids: Sequence[object] | np.ndarray | None = None,
    background_value: float = -1.0,
    threshold: float = 0.5,
    seed: int = 73,
    bootstrap_replicates: int = 2000,
    swap_replicates: int = 1000,
) -> dict[str, Any]:
    """Run the statistical feature/logistic control end to end.

    Feature extraction is independent of labels.  The standardizer and
    logistic coefficients are fit using ``train_images``/``train_labels``;
    evaluation rows are touched only after that fit.
    """

    train_features = extract_statistical_features(train_images, background_value=background_value)
    eval_features = extract_statistical_features(eval_images, background_value=background_value)
    fitted = fit_statistical_logistic(
        train_features,
        train_labels,
        eval_features,
        seed=int(seed),
        train_participant_ids=train_participant_ids,
    )
    eval_pair_values = None
    if eval_pair_ids is not None:
        candidate_pairs = np.asarray(eval_pair_ids, dtype=object).reshape(-1)
        if candidate_pairs.size != np.asarray(eval_labels).reshape(-1).size:
            raise ValueError("eval_pair_ids must match eval_labels.")
        if all(value is not None and str(value).strip() != "" for value in candidate_pairs.tolist()):
            eval_pair_values = candidate_pairs
    metrics = compute_dataset_metrics(
        eval_labels,
        fitted["scores"],
        participant_ids=eval_participant_ids,
        pair_ids=eval_pair_values,
        threshold=float(threshold),
    )
    output: dict[str, Any] = {
        "model_type": "statistical_logistic",
        "seed": int(seed),
        "metrics": metrics,
        "scores": np.asarray(fitted["scores"], dtype=np.float64),
        "train_feature_mean": fitted["train_feature_mean"],
        "train_feature_scale": fitted["train_feature_scale"],
        "train_only_scaler": True,
        "background_value": float(background_value),
        "subject_weighting": bool(fitted.get("subject_weighting", False)),
        "model_coef": np.asarray(fitted["model"].coef_, dtype=np.float64),
        "model_intercept": np.asarray(fitted["model"].intercept_, dtype=np.float64),
        "prediction_rows": [
            {
                "record_index": int(index),
                "label": int(np.asarray(eval_labels).reshape(-1)[index]),
                "probability": float(fitted["scores"][index]),
                "participant_id": _json_scalar(eval_participant_ids[index]) if eval_participant_ids is not None else None,
                "pair_id": _json_scalar(eval_pair_values[index]) if eval_pair_values is not None else None,
                "case_id": _json_scalar(eval_case_ids[index]) if eval_case_ids is not None else int(index),
            }
            for index in range(len(fitted["scores"]))
        ],
    }
    if isinstance(metrics.get("subject"), Mapping):
        subject = metrics["subject_predictions"]
        output["subject_bootstrap"] = bootstrap_subject_auc(
            subject["labels"],
            subject["scores"],
            pair_ids=subject["pair_ids"] if eval_pair_values is not None else None,
            n_bootstrap=int(bootstrap_replicates),
            seed=int(seed),
        )
        if eval_pair_values is not None:
            output["heldout_pair_swap"] = paired_heldout_swap_test(
                subject["labels"],
                subject["scores"],
                subject["pair_ids"],
                n_swaps=int(swap_replicates),
                seed=int(seed),
            )
        output["subject_prediction_rows"] = subject_prediction_rows(
            {"subject_predictions": subject}
        )
    return output


def permute_pair_labels(
    records: Sequence[Mapping[str, Any] | Any],
    *,
    seed: int,
) -> list[dict[str, Any]]:
    """Swap labels once per matched pair, retaining every participant's slices."""

    normalized_records = [_record_mapping(record) for record in records]
    if not normalized_records:
        return []
    grouped: dict[object, list[int]] = {}
    participants: dict[object, set[object]] = {}
    participant_labels: dict[tuple[object, object], int] = {}
    for index, record in enumerate(normalized_records):
        if "pair_id" not in record or record["pair_id"] is None or str(record["pair_id"]).strip() == "":
            raise ValueError("Every record needs a non-empty pair_id for permutation controls.")
        if "participant_id" not in record or "label" not in record:
            raise ValueError("Every record needs participant_id and label for permutation controls.")
        pair = record["pair_id"]
        participant = record["participant_id"]
        label = float(record["label"])
        if not np.isfinite(label) or label not in (0.0, 1.0):
            raise ValueError("Permutation labels must be finite binary values.")
        grouped.setdefault(pair, []).append(index)
        participants.setdefault(pair, set()).add(participant)
        key = (pair, participant)
        old = participant_labels.get(key)
        if old is not None and old != int(label):
            raise ValueError(f"Participant {participant!r} has conflicting labels in pair {pair!r}.")
        participant_labels[key] = int(label)
    rng = np.random.default_rng(int(seed))
    mapping: dict[tuple[object, object], int] = {}
    for pair, members in participants.items():
        if len(members) != 2:
            raise ValueError(f"Pair {pair!r} must contain exactly two participants.")
        labels = [participant_labels[(pair, participant)] for participant in members]
        if sorted(labels) != [0, 1]:
            raise ValueError(f"Pair {pair!r} must contain one class-0 and one class-1 participant.")
        swap = bool(rng.integers(0, 2))
        for participant in members:
            old_label = participant_labels[(pair, participant)]
            mapping[(pair, participant)] = 1 - old_label if swap else old_label
    output: list[dict[str, Any]] = []
    for record in normalized_records:
        updated = dict(record)
        updated["label"] = mapping[(record["pair_id"], record["participant_id"])]
        output.append(updated)
    return output


def shuffle_participant_labels(
    records: Sequence[Mapping[str, Any] | Any],
    *,
    seed: int,
) -> list[dict[str, Any]]:
    """Permute labels once per participant while retaining every slice.

    This is the retrained training-label control.  Labels are shuffled among
    participants as whole units, so slices from one participant can never be
    split between classes.  The caller should apply this only to the training
    records; validation and held-out test labels remain untouched and true.
    """

    normalized = [_record_mapping(record) for record in records]
    if not normalized:
        return []
    participant_rows: dict[object, list[int]] = {}
    participant_labels: dict[object, int] = {}
    for index, record in enumerate(normalized):
        participant = record.get("participant_id")
        if participant is None or str(participant).strip() == "":
            raise ValueError("Every record needs a non-empty participant_id for label shuffling.")
        label = record.get("label")
        try:
            numeric = float(label)
        except (TypeError, ValueError) as exc:
            raise ValueError("Participant labels must be finite binary values.") from exc
        if not np.isfinite(numeric) or numeric not in (0.0, 1.0):
            raise ValueError("Participant labels must be finite binary values.")
        value = int(numeric)
        previous = participant_labels.get(participant)
        if previous is not None and previous != value:
            raise ValueError(f"Participant {participant!r} has conflicting labels.")
        participant_labels[participant] = value
        participant_rows.setdefault(participant, []).append(index)
    participants = sorted(participant_labels, key=lambda value: str(value))
    labels = np.asarray([participant_labels[value] for value in participants], dtype=np.int64)
    shuffled = labels[np.random.default_rng(int(seed)).permutation(labels.size)]
    output = [dict(record) for record in normalized]
    for participant, label in zip(participants, shuffled.tolist()):
        for index in participant_rows[participant]:
            output[index]["label"] = int(label)
    return output


# A descriptive alias keeps call sites readable when the control is referred
# to as a training-label permutation in an experiment plan.
permute_participant_labels = shuffle_participant_labels


def make_same_cohort_negative_records(
    records: Sequence[Mapping[str, Any] | Any],
    *,
    seed: int,
    source_label: int = 0,
) -> list[dict[str, Any]]:
    """Build a same-cohort negative control with matched z-bin histograms.

    Participants carrying ``source_label`` are randomly assigned to two
    pseudo-domains within each split.  Whole participants are then greedily
    matched on normalized z-bin overlap, and each matched pair contributes the
    same number of slices (at most two per common bin).  This preserves the
    random group assignment while controlling the axial-position distribution
    used by the classifier.  An unmatched odd participant or a participant
    without a common z bin is omitted rather than creating an invalid pair.
    """

    normalized = [_record_mapping(record) for record in records]
    source = int(source_label)
    if source not in (0, 1):
        raise ValueError("source_label must be 0 or 1.")
    if not normalized:
        return []
    by_split: dict[str, dict[object, list[tuple[int, dict[str, Any]]]]] = {}
    for index, record in enumerate(normalized):
        try:
            label = int(float(record.get("label")))
        except (TypeError, ValueError) as exc:
            raise ValueError("Negative-control labels must be binary.") from exc
        if label != source:
            continue
        participant = record.get("participant_id")
        if participant is None or str(participant).strip() == "":
            raise ValueError("Negative-control records require participant_id.")
        split = str(record.get("split", record.get("source_split", "")))
        by_split.setdefault(split, {}).setdefault(participant, []).append((index, record))
    rng = np.random.default_rng(int(seed))
    output: list[dict[str, Any]] = []
    for split, participants_map in by_split.items():
        participants = sorted(participants_map, key=lambda value: str(value))
        if len(participants) < 2:
            raise ValueError(f"Negative control split {split!r} needs at least two participants.")
        shuffled = [participants[int(index)] for index in rng.permutation(len(participants)).tolist()]
        midpoint = len(shuffled) // 2
        group_zero = shuffled[:midpoint]
        group_one = shuffled[midpoint : midpoint + midpoint]
        if not group_zero or not group_one:
            raise ValueError(f"Negative control split {split!r} has no complete participant groups.")

        def z_bin(record: Mapping[str, Any]) -> int:
            value = record.get("z_bin")
            if value is not None:
                return int(value)
            z_norm = record.get("z_norm")
            if z_norm is not None:
                return min(19, int(float(z_norm) * 20.0))
            return int(record.get("z", 0))

        def profile(participant: object) -> tuple[float, set[int]]:
            rows = [record for _index, record in participants_map[participant]]
            bins = {z_bin(record) for record in rows}
            mean_bin = float(np.mean([z_bin(record) for record in rows])) if rows else 0.0
            return mean_bin, bins

        remaining_one = list(group_one)
        pair_index = 0
        for participant_zero in group_zero:
            if not remaining_one:
                break
            mean_zero, bins_zero = profile(participant_zero)
            candidate = min(
                remaining_one,
                key=lambda participant_one: (
                    -len(bins_zero.intersection(profile(participant_one)[1])),
                    abs(mean_zero - profile(participant_one)[0]),
                    str(participant_one),
                ),
            )
            remaining_one.remove(candidate)
            _mean_one, bins_one = profile(candidate)
            common_bins = sorted(bins_zero.intersection(bins_one))
            if not common_bins:
                continue
            pair_id = f"negative:{split}:pair_{pair_index:04d}"
            pair_index += 1
            rows_by_participant: dict[object, dict[int, list[tuple[int, dict[str, Any]]]]] = {}
            for participant in (participant_zero, candidate):
                rows_by_bin: dict[int, list[tuple[int, dict[str, Any]]]] = {}
                for row_index, record in participants_map[participant]:
                    rows_by_bin.setdefault(z_bin(record), []).append((row_index, record))
                rows_by_participant[participant] = rows_by_bin
            for bin_value in common_bins:
                selected_by_participant = {
                    participant_zero: sorted(
                        rows_by_participant[participant_zero].get(bin_value, []),
                        key=lambda item: (int(item[1].get("z", 0)), str(item[1].get("case_id", ""))),
                    ),
                    candidate: sorted(
                        rows_by_participant[candidate].get(bin_value, []),
                        key=lambda item: (int(item[1].get("z", 0)), str(item[1].get("case_id", ""))),
                    ),
                }
                keep_count = min(
                    len(selected_by_participant[participant_zero]),
                    len(selected_by_participant[candidate]),
                    2,
                )
                for participant, label in ((participant_zero, 0), (candidate, 1)):
                    for _row_index, record in selected_by_participant[participant][:keep_count]:
                        updated = dict(record)
                        updated["label"] = int(label)
                        updated["pair_id"] = pair_id
                        updated["domain"] = "same_cohort_negative"
                        metadata = dict(updated.get("metadata", {}) or {})
                        metadata["negative_control"] = True
                        metadata["original_label"] = int(source)
                        metadata["negative_seed"] = int(seed)
                        metadata["negative_common_z_bins"] = common_bins
                        updated["metadata"] = metadata
                        output.append(updated)
    return output


def _result_test_metrics(result: Mapping[str, Any]) -> Mapping[str, Any]:
    value = result.get("test")
    return value if isinstance(value, Mapping) else {}


def _result_subject_auc(result: Mapping[str, Any]) -> float:
    test = _result_test_metrics(result)
    subject = test.get("subject") if isinstance(test.get("subject"), Mapping) else {}
    try:
        return float(subject.get("roc_auc", float("nan")))
    except (TypeError, ValueError):
        return float("nan")


def evaluate_tiny_gate(
    results: Sequence[Mapping[str, Any]],
    *,
    accuracy_threshold: float = 0.99,
    bce_threshold: float = 0.02,
) -> dict[str, Any]:
    """Evaluate the preregistered tiny train-set overfit control."""

    rows: list[dict[str, Any]] = []
    for result in results:
        train = result.get("train_final")
        train_metrics = train if isinstance(train, Mapping) else {}
        slice_metrics = train_metrics.get("slice") if isinstance(train_metrics.get("slice"), Mapping) else train_metrics
        try:
            accuracy = float(slice_metrics.get("accuracy", float("nan")))
            bce = float(slice_metrics.get("bce", float("nan")))
        except (TypeError, ValueError):
            accuracy = bce = float("nan")
        rows.append({
            "seed": int(result.get("seed", -1)),
            "accuracy": accuracy,
            "bce": bce,
            "passed": bool(
                np.isfinite(accuracy)
                and np.isfinite(bce)
                and accuracy >= float(accuracy_threshold)
                and bce <= float(bce_threshold)
            ) if np.isfinite(accuracy) and np.isfinite(bce) else None,
        })
    if not rows:
        status = "INCONCLUSIVE"
    elif any(row["passed"] is False for row in rows):
        status = "FAIL"
    elif any(row["passed"] is None for row in rows):
        status = "INCONCLUSIVE"
    else:
        status = "PASS"
    return {
        "name": "tiny_overfit",
        "status": status,
        "passed": status == "PASS",
        "accuracy_threshold": float(accuracy_threshold),
        "bce_threshold": float(bce_threshold),
        "n_seeds": int(len(rows)),
        "rows": rows,
        "reason": (
            "Every requested seed met the train-only accuracy/BCE thresholds."
            if status == "PASS"
            else "At least one tiny run missed a threshold."
            if status == "FAIL"
            else "Tiny train metrics are missing or non-finite."
        ),
    }


def evaluate_positive_gate(
    results: Sequence[Mapping[str, Any]],
    *,
    minimum_auc: float = 0.80,
    minimum_bootstrap_low: float = 0.50,
) -> dict[str, Any]:
    """Evaluate the registered/raw positive-control adequacy gate."""

    rows: list[dict[str, Any]] = []
    for result in results:
        auc = _result_subject_auc(result)
        statistics = result.get("test_statistics")
        statistics = statistics if isinstance(statistics, Mapping) else {}
        bootstrap = statistics.get("subject_bootstrap")
        bootstrap = bootstrap if isinstance(bootstrap, Mapping) else {}
        try:
            ci_low = float(bootstrap.get("ci_low", float("nan")))
        except (TypeError, ValueError):
            ci_low = float("nan")
        adequate = bool(
            np.isfinite(auc)
            and np.isfinite(ci_low)
            and auc >= float(minimum_auc)
            and ci_low > float(minimum_bootstrap_low)
        ) if np.isfinite(auc) and np.isfinite(ci_low) else None
        try:
            ci_high = float(bootstrap.get("ci_high", float("nan")))
        except (TypeError, ValueError):
            ci_high = float("nan")
        rows.append({
            "seed": int(result.get("seed", -1)),
            "subject_auc": auc,
            "bootstrap_ci_low": ci_low,
            "bootstrap_ci_high": ci_high,
            "passed": adequate,
        })
    if not rows:
        status = "INCONCLUSIVE"
    elif any(row["passed"] is False for row in rows):
        status = "FAIL"
    elif any(row["passed"] is None for row in rows):
        status = "INCONCLUSIVE"
    else:
        status = "PASS"
    return {
        "name": "positive_registered",
        "status": status,
        "passed": status == "PASS",
        "minimum_subject_auc": float(minimum_auc),
        "minimum_bootstrap_ci_low": float(minimum_bootstrap_low),
        "n_seeds": int(len(rows)),
        "rows": rows,
        "reason": (
            "Registered/raw paired control met the subject-AUC and lower-CI thresholds."
            if status == "PASS"
            else "Positive control did not meet the preregistered adequacy thresholds."
            if status == "FAIL"
            else "Positive-control statistics are missing or non-finite."
        ),
    }


def evaluate_negative_gate(
    results: Sequence[Mapping[str, Any]],
    *,
    alpha: float = 0.01,
    deviation_threshold: float = 0.15,
    minimum_reproducible_seeds: int = 2,
    ci_band: tuple[float, float] = (0.35, 0.65),
) -> dict[str, Any]:
    """Evaluate negative/shuffle controls with explicit inconclusive status.

    A failure requires a family-adjusted two-sided paired-swap p-value below
    ``alpha`` or a direction-invariant deviation of at least ``0.15`` in the
    requested number of seeds.  A PASS requires every seed's bootstrap CI to
    lie within the configured chance band; a wide or missing CI remains
    INCONCLUSIVE rather than being called chance-level evidence.
    """

    rows: list[dict[str, Any]] = []
    p_values: dict[str, float] = {}
    p_keys: list[str | None] = []
    for index, result in enumerate(results):
        seed = int(result.get("seed", -1))
        auc = _result_subject_auc(result)
        statistics = result.get("test_statistics")
        statistics = statistics if isinstance(statistics, Mapping) else {}
        bootstrap = statistics.get("subject_bootstrap")
        bootstrap = bootstrap if isinstance(bootstrap, Mapping) else {}
        swap = statistics.get("heldout_pair_swap")
        swap = swap if isinstance(swap, Mapping) else {}
        def finite_value(container: Mapping[str, Any], key: str) -> float:
            try:
                value = float(container.get(key, float("nan")))
            except (TypeError, ValueError):
                value = float("nan")
            return value
        ci_low = finite_value(bootstrap, "ci_low")
        ci_high = finite_value(bootstrap, "ci_high")
        p_value = finite_value(swap, "p_value")
        deviation = finite_value(swap, "two_sided_deviation")
        if not np.isfinite(deviation) and np.isfinite(auc):
            deviation = abs(auc - 0.5)
        row = {
            "seed": seed,
            "subject_auc": auc,
            "abs_auc_minus_half": deviation,
            "bootstrap_ci_low": ci_low,
            "bootstrap_ci_high": ci_high,
            "paired_swap_p_value": p_value,
            "paired_swap_p_value_holm": float("nan"),
            "paired_swap_status": str(swap.get("status", "missing")),
        }
        rows.append(row)
        if np.isfinite(p_value):
            p_key = str(seed) if str(seed) not in p_values else f"{seed}:{index}"
            p_values[p_key] = p_value
        else:
            p_key = None
        p_keys.append(p_key)
    adjusted = holm_adjust(p_values)
    for row, p_key in zip(rows, p_keys):
        row["paired_swap_p_value_holm"] = float(adjusted.get(p_key, float("nan"))) if p_key is not None else float("nan")
    p_fail = [row for row in rows if np.isfinite(row["paired_swap_p_value_holm"]) and row["paired_swap_p_value_holm"] < float(alpha)]
    deviation_seeds = [row for row in rows if np.isfinite(row["abs_auc_minus_half"]) and row["abs_auc_minus_half"] >= float(deviation_threshold)]
    deviation_fail = len(deviation_seeds) >= int(minimum_reproducible_seeds)
    ci_ready = bool(rows) and all(
        np.isfinite(row["bootstrap_ci_low"])
        and np.isfinite(row["bootstrap_ci_high"])
        and float(ci_band[0]) <= row["bootstrap_ci_low"]
        and row["bootstrap_ci_high"] <= float(ci_band[1])
        for row in rows
    )
    swap_ready = bool(rows) and all(
        np.isfinite(row["paired_swap_p_value"]) and row["paired_swap_status"] == "complete"
        for row in rows
    )
    if p_fail or deviation_fail:
        status = "FAIL"
    elif ci_ready and swap_ready:
        status = "PASS"
    else:
        status = "INCONCLUSIVE"
    closeness_ready = bool(rows) and all(
        np.isfinite(row["subject_auc"])
        and np.isfinite(row["bootstrap_ci_low"])
        and np.isfinite(row["bootstrap_ci_high"])
        for row in rows
    )
    direction_invariant_upper = [
        max(float(row["bootstrap_ci_high"]), 1.0 - float(row["bootstrap_ci_low"]))
        for row in rows
        if np.isfinite(row["bootstrap_ci_low"]) and np.isfinite(row["bootstrap_ci_high"])
    ]
    closeness_status = (
        "PASS"
        if status == "PASS" and closeness_ready and direction_invariant_upper and max(direction_invariant_upper) < 0.60
        else "INCONCLUSIVE"
    )
    return {
        "name": "negative_or_shuffle",
        "status": status,
        "passed": status == "PASS",
        "alpha": float(alpha),
        "deviation_threshold": float(deviation_threshold),
        "minimum_reproducible_seeds": int(minimum_reproducible_seeds),
        "chance_ci_band": [float(ci_band[0]), float(ci_band[1])],
        "holm_adjusted_p_values": adjusted,
        "rows": rows,
        "failure_p_value_seeds": [int(row["seed"]) for row in p_fail],
        "failure_deviation_seeds": [int(row["seed"]) for row in deviation_seeds] if deviation_fail else [],
        "closeness_status": closeness_status,
        "direction_invariant_bootstrap_upper": direction_invariant_upper,
        "reason": (
            "A paired-swap or reproducible AUC-deviation failure criterion was met."
            if status == "FAIL"
            else "Every seed has a bootstrap CI wholly inside the configured chance band."
            if status == "PASS"
            else "The available controls do not establish either a failure or a narrow chance-band result."
        ),
    }


def formal_expansion_plan(
    *,
    comparisons: Sequence[str] = ("fomo45k", "mpi", "oasis3", "mixed"),
    modality_sets: Sequence[Sequence[str]] = (("flair",), ("t1",), ("t2",), ("flair", "t1", "t2")),
    models: Sequence[str] = ("statistical_logistic", "small_cnn", "resnet18"),
    seeds: Sequence[int] = (73, 173, 273),
) -> list[dict[str, Any]]:
    """Return the formal 112-cell cohort/modality/capacity/seed matrix.

    Each cohort and channel set has one train-only standardized Logistic fit,
    plus three Small-CNN and three ResNet18 seeds: ``4 * 4 * (1 + 3 + 3)``.
    Tiny overfit runs are controls and are intentionally absent from this
    formal expansion plan.
    """

    plan: list[dict[str, Any]] = []
    if not comparisons or not modality_sets or not models or not seeds:
        raise ValueError("Formal expansion dimensions must be non-empty.")
    for comparison in comparisons:
        for modalities in modality_sets:
            normalized = tuple(str(value).strip().lower() for value in modalities)
            if not normalized:
                raise ValueError("Every expansion modality set must be non-empty.")
            for model in models:
                model_name = str(model)
                model_seeds = (int(seeds[0]),) if model_name in {"statistical", "statistical_logistic", "logistic"} else tuple(int(seed) for seed in seeds)
                for seed in model_seeds:
                    plan.append({
                        "comparison": str(comparison),
                        "model": model_name,
                        "classifier": model_name,
                        "modalities": list(normalized),
                        "seed": int(seed),
                    })
    return plan


def seed_matrix_gate(
    results: Sequence[Mapping[str, Any]],
    *,
    expected_plan: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Check that every planned model/capacity/modality/seed cell completed."""

    plan = list(expected_plan) if expected_plan is not None else formal_expansion_plan()
    def key(value: Mapping[str, Any]) -> tuple[Any, ...]:
        return (
            str(value.get("comparison", "")),
            str(value.get("model", "")),
            tuple(value.get("modalities", ())),
            int(value.get("seed", -1)),
        )
    expected = [key(item) for item in plan]
    observed = [key(item) for item in results]
    counts = Counter(observed)
    missing = [item for item in expected if item not in counts]
    duplicate = [item for item, count in counts.items() if count > 1]
    status = "PASS" if not missing and not duplicate and len(observed) == len(expected) else "INCOMPLETE"
    return {
        "name": "seed_matrix",
        "status": status,
        "passed": status == "PASS",
        "expected": int(len(expected)),
        "completed": int(len(observed)),
        "missing": [list(item) for item in missing],
        "duplicate": [list(item) for item in duplicate],
    }


def retrained_pair_permutation_control(
    train_dataset: Dataset[Any],
    val_dataset: Dataset[Any],
    test_dataset: Dataset[Any],
    *,
    config: TrainConfig,
    replicates: int,
    seed: int = 73,
    permute_test_labels: bool = False,
) -> dict[str, Any]:
    """Retrain on whole-pair label swaps and return resumable control summaries.

    By default test labels remain the true held-out labels.  Setting
    ``permute_test_labels=True`` is available for a fully permuted diagnostic,
    but must be reported explicitly because it changes the test estimand.
    """

    train_records = getattr(train_dataset, "records", None)
    val_records = getattr(val_dataset, "records", None)
    test_records = getattr(test_dataset, "records", None)
    if train_records is None or val_records is None or test_records is None:
        raise TypeError("Retrained permutation controls require datasets exposing records.")
    loader_map = {
        "train": _wrap_slice_loader(getattr(train_dataset, "loader", None)),
        "val": _wrap_slice_loader(getattr(val_dataset, "loader", None)),
        "test": _wrap_slice_loader(getattr(test_dataset, "loader", None)),
    }
    modalities_map = {
        "train": getattr(train_dataset, "modalities", ("flair", "t1", "t2")),
        "val": getattr(val_dataset, "modalities", ("flair", "t1", "t2")),
        "test": getattr(test_dataset, "modalities", ("flair", "t1", "t2")),
    }
    scalar_map = {
        "train": getattr(train_dataset, "shared_train_scalar", None),
        "val": getattr(val_dataset, "shared_train_scalar", None),
        "test": getattr(test_dataset, "shared_train_scalar", None),
    }
    stage_map = {
        "train": getattr(train_dataset, "stage", config.stage),
        "val": getattr(val_dataset, "stage", config.stage),
        "test": getattr(test_dataset, "stage", config.stage),
    }
    requested = max(0, int(replicates))
    results: list[dict[str, Any]] = []
    for index in range(requested):
        permutation_seed = int(seed) + index
        permuted_train = ManifestDataset(
            permute_pair_labels(train_records, seed=permutation_seed),
            loader=loader_map["train"],
            stage=stage_map["train"],
            modalities=modalities_map["train"],
            shared_train_scalar=scalar_map["train"],
        )
        permuted_val = ManifestDataset(
            permute_pair_labels(val_records, seed=permutation_seed + 1),
            loader=loader_map["val"],
            stage=stage_map["val"],
            modalities=modalities_map["val"],
            shared_train_scalar=scalar_map["val"],
        )
        permuted_test = test_dataset
        if permute_test_labels:
            permuted_test = ManifestDataset(
                permute_pair_labels(test_records, seed=permutation_seed + 2),
                loader=loader_map["test"],
                stage=stage_map["test"],
                modalities=modalities_map["test"],
                shared_train_scalar=scalar_map["test"],
            )
        result = DomainClassifierRunner(config).train_one_seed(
            permuted_train,
            permuted_val,
            permuted_test,
            seed=permutation_seed,
        )
        result.pop("state_dict", None)
        results.append(result)
    return {
        "status": permutation_control_status(
            requested=requested,
            completed=len(results),
            mode=config.permutation_mode,
        ),
        "permute_test_labels": bool(permute_test_labels),
        "results": results,
    }


def iter_retrained_pair_permutation_control(
    train_dataset: Dataset[Any],
    val_dataset: Dataset[Any],
    test_dataset: Dataset[Any],
    *,
    config: TrainConfig,
    replicates: int,
    seed: int = 73,
    permute_test_labels: bool = False,
    indices: Sequence[int] | None = None,
) -> Iterable[tuple[int, dict[str, Any]]]:
    """Yield retrained controls one at a time for resumable CLI execution."""

    train_records = getattr(train_dataset, "records", None)
    val_records = getattr(val_dataset, "records", None)
    test_records = getattr(test_dataset, "records", None)
    if train_records is None or val_records is None or test_records is None:
        raise TypeError("Retrained permutation controls require datasets exposing records.")
    loader_map = {
        "train": _wrap_slice_loader(getattr(train_dataset, "loader", None)),
        "val": _wrap_slice_loader(getattr(val_dataset, "loader", None)),
        "test": _wrap_slice_loader(getattr(test_dataset, "loader", None)),
    }
    modalities_map = {
        "train": getattr(train_dataset, "modalities", ("flair", "t1", "t2")),
        "val": getattr(val_dataset, "modalities", ("flair", "t1", "t2")),
        "test": getattr(test_dataset, "modalities", ("flair", "t1", "t2")),
    }
    scalar_map = {
        "train": getattr(train_dataset, "shared_train_scalar", None),
        "val": getattr(val_dataset, "shared_train_scalar", None),
        "test": getattr(test_dataset, "shared_train_scalar", None),
    }
    stage_map = {
        "train": getattr(train_dataset, "stage", config.stage),
        "val": getattr(val_dataset, "stage", config.stage),
        "test": getattr(test_dataset, "stage", config.stage),
    }
    work_indices = (
        [int(index) for index in indices]
        if indices is not None
        else list(range(max(0, int(replicates))))
    )
    if any(index < 0 or index >= max(0, int(replicates)) for index in work_indices):
        raise ValueError("Permutation indices must lie in [0, replicates).")
    for index in work_indices:
        permutation_seed = int(seed) + index
        permuted_train = ManifestDataset(
            permute_pair_labels(train_records, seed=permutation_seed),
            loader=loader_map["train"],
            stage=stage_map["train"],
            modalities=modalities_map["train"],
            shared_train_scalar=scalar_map["train"],
        )
        permuted_val = ManifestDataset(
            permute_pair_labels(val_records, seed=permutation_seed + 1),
            loader=loader_map["val"],
            stage=stage_map["val"],
            modalities=modalities_map["val"],
            shared_train_scalar=scalar_map["val"],
        )
        permuted_test = test_dataset
        if permute_test_labels:
            permuted_test = ManifestDataset(
                permute_pair_labels(test_records, seed=permutation_seed + 2),
                loader=loader_map["test"],
                stage=stage_map["test"],
                modalities=modalities_map["test"],
                shared_train_scalar=scalar_map["test"],
            )
        result = DomainClassifierRunner(config).train_one_seed(
            permuted_train,
            permuted_val,
            permuted_test,
            seed=permutation_seed,
        )
        result.pop("state_dict", None)
        yield index, result


def _wrap_slice_loader(loader: Callable[..., Any] | None) -> Callable[..., Any] | None:
    if loader is None:
        return None
    if bool(getattr(loader, "_accepts_mapping_records", False)):
        return loader

    def wrapped(record: Any, *, stage: str) -> Any:
        candidate = record
        if isinstance(record, Mapping):
            try:
                from andi_rewrite.data.domain_classifier.records import SliceRecord  # type: ignore

                candidate = SliceRecord.from_mapping(record)
            except ModuleNotFoundError as exc:
                if exc.name != "andi_rewrite.data.domain_classifier.records":
                    raise
        try:
            return loader(candidate, stage=stage)
        except TypeError as exc:
            # Test doubles often expose the positional stage form.  Retry only
            # the signature variant; errors raised by the loader itself are
            # allowed to propagate.
            if "unexpected keyword argument" not in str(exc) and "positional" not in str(exc):
                raise
            return loader(candidate, stage)

    return wrapped


def permutation_control_status(
    *,
    requested: int,
    completed: int,
    mode: str,
) -> dict[str, Any]:
    """Describe resumability when retrained permutation controls are omitted."""

    requested = max(0, int(requested))
    completed = max(0, min(requested, int(completed)))
    return {
        "mode": str(mode),
        "requested": requested,
        "completed": completed,
        "remaining": requested - completed,
        "status": "complete" if completed == requested else "incomplete",
        "unit": "whole_pair_label_swap",
        "selection_and_scaler_refit": True,
    }


__all__ = [
    "DomainClassifierRunner",
    "ManifestDataset",
    "TrainConfig",
    "TrainOnlyStandardizer",
    "atomic_write_json",
    "dataset_from_manifest",
    "dataset_with_shared_train_scalar",
    "dataset_with_stage",
    "dataset_with_modalities",
    "evaluate_negative_gate",
    "evaluate_positive_gate",
    "evaluate_tiny_gate",
    "extract_statistical_features",
    "formal_expansion_plan",
    "fit_shared_train_scalar",
    "fit_statistical_logistic",
    "iter_retrained_pair_permutation_control",
    "make_same_cohort_negative_records",
    "materialize_dataset",
    "materialize_registered_dataset",
    "materialize_tensor_dataset",
    "materialize_tensor_datasets",
    "prediction_rows",
    "permutation_control_status",
    "permute_pair_labels",
    "retrained_pair_permutation_control",
    "read_jsonl_manifest",
    "run_fingerprint",
    "run_statistical_control",
    "seed_matrix_gate",
    "select_tiny_records",
    "set_deterministic_seed",
    "subset_dataset",
    "subject_balanced_bce_with_logits",
    "subject_prediction_rows",
    "shuffle_participant_labels",
    "write_prediction_rows",
]
