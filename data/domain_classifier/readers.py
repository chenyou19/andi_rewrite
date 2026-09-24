"""Read-only model-input readers for domain-classifier manifests.

The reader intentionally has no default transform or intensity scaling.  The
``final`` stage is the exact robust-IQR model space already used by ANDi; the
``registered`` stage is an explicit, unnormalised pre-IQR view for the positive
control.  New cache files, when used, are expected under the diagnostics output
tree and never beside source datasets.
"""

from __future__ import annotations

import csv
import functools
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .records import MODALITIES, MODEL_SHAPE, SliceRecord


def _resolve(path: str | Path, base_dir: Path | None = None) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute() and base_dir is not None:
        candidate = base_dir / candidate
    return candidate


def _read_array(path: Path) -> np.ndarray:
    """Read NPY/NPZ/NIfTI without modifying or memory-mapping the source."""

    if not path.is_file():
        raise FileNotFoundError(path)
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".npy"):
        return np.asarray(np.load(path, allow_pickle=False))
    if suffixes.endswith(".npz"):
        with np.load(path, allow_pickle=False) as archive:
            for key in ("image", "array", "data", "volume"):
                if key in archive:
                    return np.asarray(archive[key])
            if not archive.files:
                raise ValueError(f"NPZ contains no arrays: {path}")
            return np.asarray(archive[archive.files[0]])
    if suffixes.endswith(".nii") or suffixes.endswith(".nii.gz"):
        try:
            import nibabel as nib
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("NIfTI readers require nibabel.") from exc
        return np.asarray(nib.load(str(path)).dataobj, dtype=np.float32)
    raise ValueError(f"Unsupported image format: {path}")


@functools.lru_cache(maxsize=128)
def _file_sha256_cached(path_text: str, size: int, mtime_ns: int) -> str:
    del size, mtime_ns
    digest = hashlib.sha256()
    with Path(path_text).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_channel_volume(array: np.ndarray, *, channels: int = 3) -> np.ndarray:
    """Convert common channel/volume layouts to ``[C,H,W,Z]``."""

    values = np.asarray(array)
    if values.ndim == 2:
        return values[None, :, :, None]
    if values.ndim == 3:
        # A single modality volume is interpreted as H,W,Z.  A 3-channel
        # 2-D stack is accepted only when the final dimension is absent.
        return values[None, ...]
    if values.ndim != 4:
        raise ValueError(f"Expected a 2-D/3-D/4-D array, found shape={values.shape}")
    if values.shape[0] == channels:
        return values
    if values.shape[1] == channels:
        return np.moveaxis(values, 1, 0)
    if values.shape[-1] == channels:
        return np.moveaxis(values, -1, 0)
    raise ValueError(f"Could not identify {channels} channels in array shape={values.shape}")


def _resize_slice(slice_array: np.ndarray, image_size: int = 128) -> torch.Tensor:
    values = torch.as_tensor(np.asarray(slice_array), dtype=torch.float32)
    if values.ndim != 3 or values.shape[0] != 3:
        raise ValueError(f"Expected [3,H,W] slice, found shape={tuple(values.shape)}")
    if tuple(values.shape[-2:]) == (image_size, image_size):
        output = values
    else:
        # torchvision's antialiased Resize is the operation used by existing
        # model datasets.  Do not silently substitute another interpolation
        # kernel: parity with ANDi is part of the data contract.
        try:
            from torchvision.transforms import Resize
        except ImportError as exc:  # pragma: no cover - dependency contract
            raise ImportError("Domain-classifier readers require torchvision.") from exc
        output = Resize(image_size, antialias=True)(values)
    if tuple(output.shape) != (3, image_size, image_size):
        raise ValueError(f"Resize produced unexpected shape: {tuple(output.shape)}")
    if not bool(torch.isfinite(output).all()):
        raise ValueError("Model input contains NaN/Inf")
    return output.contiguous()


def _normalise_raw_volume(values: np.ndarray) -> torch.Tensor:
    from ..robust_normalization import robust_normalize_volume

    volume = torch.as_tensor(values, dtype=torch.float32)
    if volume.ndim != 4 or volume.shape[0] != 3:
        raise ValueError(f"Expected [3,H,W,Z] raw volume, found {tuple(volume.shape)}")
    result = robust_normalize_volume(volume)
    if not bool(torch.isfinite(result).all()):
        raise ValueError("robust_iqr produced NaN/Inf")
    return result


def _source_lmdb_path(record: SliceRecord, base_dir: Path | None) -> Path | None:
    candidates: list[Any] = []
    for container in (record.metadata, record.provenance):
        if isinstance(container, Mapping):
            candidates.extend(
                container.get(key)
                for key in ("lmdb_path", "source_lmdb", "cache_path")
                if container.get(key)
            )
    for item in candidates:
        path = _resolve(str(item), base_dir)
        if path.is_dir() and (path / "data.mdb").is_file():
            return path
    return None


def _open_lmdb(path_text: str):
    from ..datasets.lmdb import LMDBSliceDataset

    return LMDBSliceDataset(path_text, image_size=None)


def _read_lmdb_slice(record: SliceRecord, base_dir: Path | None) -> torch.Tensor:
    path = _source_lmdb_path(record, base_dir)
    if path is None:
        raise ValueError(f"FOMO/LMDB record has no readable lmdb_path: {record.record_id}")
    if not record.source_key:
        raise ValueError(f"LMDB record has no source_key: {record.record_id}")
    normalization_path = path / "normalization.json"
    if not normalization_path.is_file():
        raise ValueError(f"LMDB has no robust-IQR normalization contract: {normalization_path}")
    from ..robust_normalization import ROBUST_SPEC

    if json.loads(normalization_path.read_text(encoding="utf-8")) != ROBUST_SPEC:
        raise ValueError(f"LMDB normalization contract mismatch: {normalization_path}")
    try:
        index = int(record.source_key)
    except ValueError as exc:
        raise ValueError(f"LMDB source_key is not an integer: {record.source_key!r}") from exc
    dataset = _open_lmdb(str(path))
    try:
        return torch.as_tensor(dataset[index], dtype=torch.float32)
    finally:
        # LMDB keeps a read transaction/environment alive on the adapter.  A
        # per-read close preserves the read-only contract and avoids locking a
        # temporary/cache directory after a dataset fixture is removed.
        transaction = getattr(dataset, "txn", None)
        if transaction is not None:
            transaction.abort()
            dataset.txn = None
        environment = getattr(dataset, "env", None)
        if environment is not None:
            environment.close()
            dataset.env = None


def _read_path_slice(
    paths: Mapping[str, str],
    z: int,
    *,
    stage: str,
    image_size: int,
    base_dir: Path | None,
    precomputed_final: bool = False,
) -> torch.Tensor:
    ordered: list[np.ndarray] = []
    reference_shape: tuple[int, ...] | None = None
    for modality in MODALITIES:
        value = paths.get(modality, "")
        if not value:
            raise ValueError(f"Missing {modality} path")
        array = _read_array(_resolve(value, base_dir))
        channel_volume = _as_channel_volume(array, channels=3)
        if channel_volume.shape[0] != 1:
            raise ValueError(f"Expected one modality per path, found {array.shape}")
        if reference_shape is None:
            reference_shape = tuple(channel_volume.shape[1:])
        elif tuple(channel_volume.shape[1:]) != reference_shape:
            raise ValueError("Modality geometry mismatch")
        if z < 0 or z >= channel_volume.shape[-1]:
            raise IndexError(f"z={z} outside volume depth={channel_volume.shape[-1]}")
        ordered.append(channel_volume[0, ..., z])
    volume = np.stack(ordered, axis=0)
    if stage == "final" and not precomputed_final:
        # This branch is intentionally full-volume only.  Production healthy
        # records use LMDB and BraTS records use MRIDataVolume; it remains here
        # for the explicit synthetic/path-backed reader contract.
        full = np.stack(
            [
                _as_channel_volume(_read_array(_resolve(paths[m], base_dir)), channels=3)[0]
                for m in MODALITIES
            ],
            axis=0,
        )
        if z < 0 or z >= full.shape[-1]:
            raise IndexError(f"z={z} outside volume depth={full.shape[-1]}")
        volume = _normalise_raw_volume(full)[..., z].numpy()
    return _resize_slice(volume, image_size)


def load_slice(
    record: SliceRecord | Mapping[str, Any],
    *,
    stage: str = "final",
    image_size: int = 128,
    base_dir: str | Path | None = None,
) -> torch.Tensor:
    """Load one exact model-input slice.

    Parameters
    ----------
    stage:
        ``"final"`` loads robust-IQR model space.  ``"registered"`` loads
        registered images before robust-IQR and performs no intensity change.
    """

    item = record if isinstance(record, SliceRecord) else SliceRecord.from_mapping(record)
    if stage not in {"final", "registered"}:
        raise ValueError("stage must be 'final' or 'registered'")
    root = Path(base_dir) if base_dir is not None else None

    source = item.source_dataset.lower()
    healthy_sources = {"fomo", "fomo45k", "mpi", "oasis", "oasis3", "mixed", "lmdb"}
    brats_sources = {"brats", "brats21", "brats2021"}
    if stage == "final" and (source in healthy_sources or item.metadata.get("input_kind") == "lmdb"):
        # Healthy final inputs must come from the existing read-only LMDBs.
        # Falling back to NIfTI here would reimplement preprocessing and can
        # silently break parity.
        tensor = _read_lmdb_slice(item, root)
        if tuple(tensor.shape) != (3, 128, 128):
            raise ValueError(f"Healthy LMDB input must be [3,128,128], found {tuple(tensor.shape)}")
        if image_size != 128:
            raise ValueError("The domain-classifier final reader is fixed at 128x128")
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"LMDB input contains NaN/Inf: {item.record_id}")
        return tensor.contiguous()

    if stage == "final" and source in brats_sources:
        cache_text = item.metadata.get("selected_cache_path")
        if cache_text:
            cache_path = _resolve(str(cache_text), root)
            if not cache_path.is_file():
                raise FileNotFoundError(cache_path)
            expected_cache_hash = item.metadata.get("selected_cache_sha256")
            if expected_cache_hash:
                stat = cache_path.stat()
                actual_cache_hash = _file_sha256_cached(
                    str(cache_path.resolve()), int(stat.st_size), int(stat.st_mtime_ns)
                )
                if actual_cache_hash != str(expected_cache_hash):
                    raise ValueError(f"BraTS slice cache file hash mismatch: {cache_path}")
            cache_index = int(item.metadata.get("selected_cache_index", 0))
            with np.load(cache_path, allow_pickle=False) as archive:
                if "images" not in archive:
                    raise ValueError(f"BraTS slice cache has no images array: {cache_path}")
                images = np.asarray(archive["images"], dtype=np.float32)
                if "z" not in archive:
                    raise ValueError(f"BraTS slice cache has no z array: {cache_path}")
                cache_z = np.asarray(archive["z"])
                if images.ndim != 4 or images.shape[1:] != MODEL_SHAPE:
                    raise ValueError(f"Invalid BraTS slice cache shape: {images.shape}")
                if not 0 <= cache_index < images.shape[0]:
                    raise IndexError(f"cache index {cache_index} outside {images.shape[0]}")
                if cache_z.ndim != 1 or cache_z.shape[0] != images.shape[0] or int(cache_z[cache_index]) != int(item.z):
                    raise ValueError(f"BraTS slice cache z mismatch: {cache_path}")
                tensor = torch.from_numpy(images[cache_index]).contiguous()
                if not bool(torch.isfinite(tensor).all()):
                    raise ValueError(f"BraTS slice cache contains NaN/Inf: {cache_path}")
                expected_digest = item.provenance.get("model_input_sha256")
                if expected_digest:
                    digest = hashlib.sha256(tensor.numpy().tobytes()).hexdigest()
                    if digest != str(expected_digest):
                        raise ValueError(f"BraTS slice cache input hash mismatch: {cache_path}")
                return tensor
        # Reuse the actual MRIDataVolume implementation so full-volume robust
        # IQR statistics, model-grid resize, and nearest mask interpolation are
        # exactly the ANDi inference path.
        root_hint = item.metadata.get("dataset_path") or item.provenance.get("dataset_path")
        if root_hint:
            dataset_root = _resolve(str(root_hint), root)
        elif item.image_paths.get("flair"):
            dataset_root = _resolve(item.image_paths["flair"], root).parent.parent
        else:
            raise ValueError(f"BraTS record has no dataset_path or modality paths: {item.record_id}")
        from ..datasets.brats import MRIDataVolume

        dataset = _open_brats_dataset(str(dataset_root))
        subject_ids = [str(value) for value in dataset.df.iloc[:, 0].tolist()]
        subject = (
            item.metadata.get("source_participant_id")
            or item.metadata.get("subject_id")
            or item.participant_id
            or item.case_id
        )
        # Production manifests namespace participant IDs for cross-cohort
        # split safety (``brats21:BraTS2021_...``).  MRIDataVolume indexes the
        # on-disk subject directory and therefore needs the source component.
        if ":" in str(subject):
            subject = str(subject).split(":", 1)[1]
        if subject not in subject_ids:
            raise ValueError(f"BraTS subject {subject!r} is absent from {dataset_root}")
        volume, model_mask, _metadata = dataset[subject_ids.index(subject)]
        if image_size != 128:
            raise ValueError("The domain-classifier final reader is fixed at 128x128")
        if item.z < 0 or item.z >= volume.shape[-1]:
            raise IndexError(f"z={item.z} outside BraTS model depth={volume.shape[-1]}")
        if not bool(torch.isfinite(model_mask).all()):
            raise ValueError(f"BraTS model mask contains NaN/Inf: {item.record_id}")
        if bool(model_mask[..., item.z].bool().any()):
            raise ValueError(f"BraTS model mask contains lesion voxels: {item.record_id}")
        tensor = volume[..., item.z].float().contiguous()
        if tuple(tensor.shape) != MODEL_SHAPE or not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"Invalid BraTS model input for {item.record_id}: {tuple(tensor.shape)}")
        return tensor

    if stage == "registered":
        paths = item.registered_paths or item.image_paths
        precomputed = False
    else:
        paths = item.image_paths
        precomputed = bool(item.metadata.get("precomputed_final", False))
    return _read_path_slice(
        paths,
        int(item.z),
        stage=stage,
        image_size=image_size,
        base_dir=root,
        precomputed_final=precomputed,
    )


def load_mask_slice(
    record: SliceRecord | Mapping[str, Any],
    *,
    stage: str = "native",
    base_dir: str | Path | None = None,
) -> np.ndarray:
    """Read a native or model-grid binary mask for eligibility audits."""

    item = record if isinstance(record, SliceRecord) else SliceRecord.from_mapping(record)
    path_text = item.seg_path if stage == "native" else item.model_mask_path
    if not path_text:
        raise ValueError(f"Missing {stage} mask path for {item.record_id}")
    array = _read_array(_resolve(path_text, Path(base_dir) if base_dir else None))
    if not bool(np.isfinite(array).all()):
        raise ValueError(f"{stage} mask contains NaN/Inf: {path_text}")
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if array.ndim == 3:
        if item.z < 0 or item.z >= array.shape[-1]:
            raise IndexError(f"z={item.z} outside mask depth={array.shape[-1]}")
        array = array[..., item.z]
    elif array.ndim != 2:
        raise ValueError(f"Expected 2-D/3-D mask, found shape={array.shape}")
    return np.asarray(array != 0, dtype=bool)


@functools.lru_cache(maxsize=8)
def _open_brats_dataset(dataset_root_text: str):
    from ..datasets.brats import MRIDataVolume

    root = Path(dataset_root_text)
    return MRIDataVolume(
        csv_path=None,
        dataset_path=root,
        image_size=128,
        modalities=["flair", "t1", "t2"],
        segmentation_suffix="seg",
        filename_separator="_",
        return_metadata=True,
        intensity_normalization="robust_iqr",
    )


def validate_tumor_free(
    record: SliceRecord | Mapping[str, Any],
    *,
    base_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Fail closed unless native and model-grid lesion masks are empty."""

    item = record if isinstance(record, SliceRecord) else SliceRecord.from_mapping(record)
    native = load_mask_slice(item, stage="native", base_dir=base_dir)
    if native.size == 0:
        raise ValueError(f"Native segmentation mask is empty/unreadable: {item.record_id}")
    if bool(np.any(native != 0)):
        raise ValueError(f"Native segmentation contains lesion voxels: {item.record_id}")
    model = None
    if item.model_mask_path:
        model = load_mask_slice(item, stage="model", base_dir=base_dir)
        if bool(np.any(model != 0)):
            raise ValueError(f"Model-grid segmentation contains lesion voxels: {item.record_id}")
    elif item.metadata.get("model_mask_checked") is True and item.model_mask_voxels == 0:
        # Prepared manifests may carry a trusted, explicit zero result from
        # the actual MRIDataVolume mask check.  Every other record fails
        # closed because a native-only assertion cannot prove model-space
        # tumor freedom.
        model = np.zeros(tuple(item.model_shape[-2:]), dtype=bool)
    else:
        raise ValueError(
            f"Missing model-grid mask or trusted model_mask_checked provenance: {item.record_id}"
        )
    return {
        "native_seg_voxels": 0,
        "model_mask_voxels": 0 if model is not None else None,
        "native_shape": list(native.shape),
        "model_shape": list(model.shape) if model is not None else None,
    }


def load_records(path: str | Path) -> list[SliceRecord]:
    """Load JSONL (or CSV) records while preserving deterministic ordering."""

    manifest = Path(path)
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    if manifest.suffix.lower() == ".csv":
        with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
            return [SliceRecord.from_mapping(row) for row in csv.DictReader(handle)]
    records: list[SliceRecord] = []
    with manifest.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {manifest}:{line_number}") from exc
            if not isinstance(row, Mapping):
                raise ValueError(f"Manifest row is not an object at {manifest}:{line_number}")
            records.append(SliceRecord.from_mapping(row))
    return records


def write_records(path: str | Path, records: Iterable[SliceRecord | Mapping[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            item = record if isinstance(record, SliceRecord) else SliceRecord.from_mapping(record)
            handle.write(json.dumps(item.to_dict(), sort_keys=True, allow_nan=False) + "\n")


class DomainClassifierDataset(Dataset):
    """PyTorch dataset backed by one comparison/split JSONL manifest."""

    def __init__(
        self,
        manifest: str | Path | Sequence[SliceRecord | Mapping[str, Any]],
        *,
        stage: str = "final",
        image_size: int = 128,
        return_metadata: bool = False,
    ) -> None:
        self.manifest_path = Path(manifest) if isinstance(manifest, (str, Path)) else None
        self.base_dir = self.manifest_path.parent if self.manifest_path is not None else None
        self.records = (
            load_records(self.manifest_path)
            if self.manifest_path is not None
            else [r if isinstance(r, SliceRecord) else SliceRecord.from_mapping(r) for r in manifest]
        )
        self.stage = stage
        self.image_size = int(image_size)
        self.return_metadata = bool(return_metadata)
        if self.stage not in {"final", "registered"}:
            raise ValueError("stage must be 'final' or 'registered'")

    @classmethod
    def from_manifest(cls, path: str | Path, **kwargs: Any) -> "DomainClassifierDataset":
        return cls(path, **kwargs)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        image = load_slice(record, stage=self.stage, image_size=self.image_size, base_dir=self.base_dir)
        if self.return_metadata:
            return image, int(record.label), record.to_dict()
        return image, int(record.label)

    def load_slice(self, record_or_index: int | SliceRecord, *, stage: str | None = None) -> torch.Tensor:
        record = self.records[record_or_index] if isinstance(record_or_index, int) else record_or_index
        return load_slice(
            record,
            stage=stage or self.stage,
            image_size=self.image_size,
            base_dir=self.base_dir,
        )

    def iter_records(self) -> Iterator[SliceRecord]:
        return iter(self.records)

    def load_pair(self, pair_id: str, *, stage: str | None = None) -> list[tuple[torch.Tensor, int, SliceRecord]]:
        rows = [record for record in self.records if record.pair_id == pair_id]
        return [(self.load_slice(record, stage=stage), int(record.label), record) for record in rows]


def file_fingerprint(path: str | Path, *, content: bool = False) -> dict[str, Any]:
    """Return a stable source identity for manifests and audits."""

    target = Path(path)
    stat = target.stat()
    result: dict[str, Any] = {
        "path": str(target.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if content:
        digest = hashlib.sha256()
        with target.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        result["sha256"] = digest.hexdigest()
    return result


__all__ = [
    "DomainClassifierDataset",
    "file_fingerprint",
    "load_mask_slice",
    "load_records",
    "load_slice",
    "write_records",
]
