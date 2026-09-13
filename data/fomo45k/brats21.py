"""BraTS21-style SRI24 preprocessing for the FOMO45K PT007_NIMH cohort.

The module deliberately orchestrates the ANTs command-line tools and
FreeSurfer SynthStrip instead of silently substituting a different registration
implementation.  Source images are never modified.  N4-corrected images are
temporary registration inputs; the estimated transforms are applied once to
the original intensities.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import nibabel as nib
import numpy as np


BRATS_SHAPE = (240, 240, 155)
BRATS_SPACING_MM = (1.0, 1.0, 1.0)
MODEL_CHANNEL_ORDER = ("flair", "t1", "t2")
NIFTI_SUFFIXES = (".nii", ".nii.gz")
PIPELINE_SCHEMA_VERSION = 1


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _json_dump(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_nifti(path: Path) -> bool:
    lowered = path.name.lower()
    return any(lowered.endswith(suffix) for suffix in NIFTI_SUFFIXES)


def _strip_nifti_suffix(name: str) -> str:
    lowered = name.lower()
    for suffix in NIFTI_SUFFIXES:
        if lowered.endswith(suffix):
            return name[: -len(suffix)]
    return name


def classify_modality(path: str | Path) -> str | None:
    """Classify a source filename without treating T1ce as native T1."""

    stem = _strip_nifti_suffix(Path(path).name).lower()
    excluded = ("mask", "seg", "label", "n4", "mni", "sri24", "registered", "warped")
    if any(token in stem for token in excluded):
        return None
    tokens = [token for token in re.split(r"[^a-z0-9]+", stem) if token]
    if "flair" in stem or "t2f" in tokens:
        return "flair"
    if any(token in tokens for token in ("t1ce", "t1c", "t1gd", "t1post")):
        return None
    if any(token in tokens for token in ("post", "gd", "contrast")) and "t1" in stem:
        return None
    if any(token in tokens for token in ("t1", "t1w", "t1n")) or re.fullmatch(r"t1w?", stem):
        return "t1"
    if any(token in tokens for token in ("t2", "t2w")) or re.fullmatch(r"t2w?", stem):
        return "t2"
    return None


@dataclass(frozen=True)
class SessionRecord:
    participant_id: str
    session_id: str
    relative_dir: str
    t1_path: str | None
    t2_path: str | None
    flair_path: str | None
    status: str
    reasons: tuple[str, ...] = ()

    @property
    def case_id(self) -> str:
        return f"{self.participant_id}/{self.session_id}"

    @property
    def file_stem(self) -> str:
        value = f"{self.participant_id}_{self.session_id}"
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)

    @property
    def paths(self) -> dict[str, Path | None]:
        return {
            "t1": None if self.t1_path is None else Path(self.t1_path),
            "t2": None if self.t2_path is None else Path(self.t2_path),
            "flair": None if self.flair_path is None else Path(self.flair_path),
        }

    def as_row(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "participant_id": self.participant_id,
            "session_id": self.session_id,
            "relative_dir": self.relative_dir,
            "status": self.status,
            "reasons": ";".join(self.reasons),
            "t1_input": self.t1_path or "",
            "t2_input": self.t2_path or "",
            "flair_input": self.flair_path or "",
        }


def _record_from_candidates(
    source_root: Path,
    session_dir: Path,
    candidates: Mapping[str, Sequence[Path]],
) -> SessionRecord:
    reasons: list[str] = []
    resolved: dict[str, str | None] = {}
    for modality in MODEL_CHANNEL_ORDER:
        values = sorted({path.resolve() for path in candidates.get(modality, ())})
        if len(values) > 1:
            reasons.append(f"ambiguous_{modality}:{len(values)}")
            resolved[modality] = None
        elif not values:
            reasons.append(f"missing_{modality}")
            resolved[modality] = None
        else:
            resolved[modality] = str(values[0])
    relative = session_dir.resolve().relative_to(source_root.resolve())
    parts = relative.parts
    participant = parts[0] if parts else session_dir.name
    session = parts[1] if len(parts) > 1 else session_dir.name
    if resolved["t1"] is None:
        status = "EXCLUDED"
    elif any(reason.startswith("ambiguous_") for reason in reasons):
        status = "EXCLUDED"
    elif resolved["t2"] is None or resolved["flair"] is None:
        status = "PARTIAL"
    else:
        status = "READY"
    return SessionRecord(
        participant_id=participant,
        session_id=session,
        relative_dir=relative.as_posix(),
        t1_path=resolved["t1"],
        t2_path=resolved["t2"],
        flair_path=resolved["flair"],
        status=status,
        reasons=tuple(reasons),
    )


def discover_sessions(
    source_root: str | Path,
    metadata_tsv: str | Path | None = None,
) -> list[SessionRecord]:
    """Discover sessions, preferring the authoritative FOMO metadata TSV."""

    source_root = Path(source_root).resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"FOMO source root does not exist: {source_root}")
    records: list[SessionRecord] = []
    if metadata_tsv is not None:
        metadata_tsv = Path(metadata_tsv).resolve()
        if not metadata_tsv.is_file():
            raise FileNotFoundError(f"FOMO metadata TSV does not exist: {metadata_tsv}")
        with metadata_tsv.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            required = {
                "participant_id",
                "session_id",
                "T1_filename",
                "T2_filename",
                "FLAIR_filename",
            }
            missing = required.difference(reader.fieldnames or ())
            if missing:
                raise ValueError(f"FOMO metadata TSV is missing columns: {sorted(missing)}")
            for row in reader:
                participant = str(row["participant_id"])
                session = str(row["session_id"])
                session_dir = source_root / participant / session
                candidates = {
                    "t1": [session_dir / str(row["T1_filename"])],
                    "t2": [session_dir / str(row["T2_filename"])],
                    "flair": [session_dir / str(row["FLAIR_filename"])],
                }
                existing = {
                    modality: [path for path in paths if path.is_file()]
                    for modality, paths in candidates.items()
                }
                records.append(_record_from_candidates(source_root, session_dir, existing))
    else:
        grouped: dict[Path, dict[str, list[Path]]] = {}
        for path in source_root.rglob("*"):
            if not path.is_file() or not _is_nifti(path):
                continue
            modality = classify_modality(path)
            if modality is None:
                continue
            grouped.setdefault(path.parent, {}).setdefault(modality, []).append(path)
        records = [
            _record_from_candidates(source_root, directory, candidates)
            for directory, candidates in grouped.items()
        ]
    records.sort(key=lambda record: (record.participant_id, record.session_id))
    identifiers = [record.case_id for record in records]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Duplicate participant/session identifiers were discovered.")
    return records


MANIFEST_FIELDS = (
    "case_id",
    "participant_id",
    "session_id",
    "relative_dir",
    "status",
    "reasons",
    "t1_input",
    "t2_input",
    "flair_input",
)


def write_manifest(path: str | Path, records: Sequence[SessionRecord]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(record.as_row() for record in records)


@dataclass(frozen=True)
class Toolchain:
    ants_registration: str
    ants_apply_transforms: str
    n4_bias_field_correction: str
    synthstrip: str
    atlas_t1: str
    atlas_brain: str
    atlas_mask: str


@dataclass(frozen=True)
class RegistrationSettings:
    random_seed: int = 73
    dice_min: float = 0.70
    atlas_com_distance_max_mm: float = 15.0
    modality_dice_min: float = 0.75
    modality_com_distance_max_mm: float = 10.0
    determinant_min: float = 0.5
    determinant_max: float = 2.0
    geometry_atol: float = 1.0e-5


DEFAULT_ATLAS_ROOT = Path(r"C:\ML\data\atlases\brats_sri24")
DEFAULT_TOOLCHAIN = Toolchain(
    ants_registration="antsRegistration",
    ants_apply_transforms="antsApplyTransforms",
    n4_bias_field_correction="N4BiasFieldCorrection",
    synthstrip="mri_synthstrip",
    atlas_t1=str(DEFAULT_ATLAS_ROOT / "brats_sri24.nii"),
    atlas_brain=str(DEFAULT_ATLAS_ROOT / "brats_sri24_skullstripped.nii"),
    atlas_mask=str(DEFAULT_ATLAS_ROOT / "brats_sri24_mask.nii.gz"),
)


def _resolve_executable(value: str) -> Path | None:
    explicit = Path(value)
    if explicit.is_file():
        return explicit.resolve()
    found = shutil.which(value)
    return None if found is None else Path(found).resolve()


def _validate_nifti(path: Path, *, expected_shape: tuple[int, int, int] | None = None) -> nib.Nifti1Image:
    image = nib.load(str(path))
    if len(image.shape) != 3:
        raise ValueError(f"Expected a 3-D NIfTI: {path}, found shape={image.shape}")
    if expected_shape is not None and tuple(int(value) for value in image.shape) != expected_shape:
        raise ValueError(f"Unexpected atlas shape for {path}: {image.shape} != {expected_shape}")
    affine = np.asarray(image.affine, dtype=np.float64)
    if not np.all(np.isfinite(affine)) or abs(float(np.linalg.det(affine[:3, :3]))) < 1.0e-12:
        raise ValueError(f"Invalid NIfTI affine: {path}")
    return image


def preflight_report(toolchain: Toolchain = DEFAULT_TOOLCHAIN) -> dict[str, Any]:
    tools = {
        "antsRegistration": _resolve_executable(toolchain.ants_registration),
        "antsApplyTransforms": _resolve_executable(toolchain.ants_apply_transforms),
        "N4BiasFieldCorrection": _resolve_executable(toolchain.n4_bias_field_correction),
        "mri_synthstrip": _resolve_executable(toolchain.synthstrip),
    }
    atlas_paths = {
        "atlas_t1": Path(toolchain.atlas_t1),
        "atlas_brain": Path(toolchain.atlas_brain),
        "atlas_mask": Path(toolchain.atlas_mask),
    }
    errors: list[str] = []
    for name, path in tools.items():
        if path is None:
            errors.append(f"missing executable: {name}")
    atlas_metadata: dict[str, Any] = {}
    loaded: dict[str, nib.Nifti1Image] = {}
    for name, path in atlas_paths.items():
        if not path.is_file():
            errors.append(f"missing atlas resource: {path}")
            continue
        try:
            image = _validate_nifti(path, expected_shape=BRATS_SHAPE)
            loaded[name] = image
            atlas_metadata[name] = {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "shape": list(image.shape),
                "spacing_mm": [float(value) for value in image.header.get_zooms()[:3]],
                "axcodes": "".join(str(value) for value in nib.aff2axcodes(image.affine)),
            }
        except Exception as exc:  # noqa: BLE001 - report every preflight problem together
            errors.append(f"invalid atlas resource {path}: {exc}")
    if len(loaded) == 3:
        reference = loaded["atlas_t1"]
        for name in ("atlas_brain", "atlas_mask"):
            image = loaded[name]
            if not np.allclose(image.affine, reference.affine, rtol=0.0, atol=1.0e-6):
                errors.append(f"atlas geometry mismatch: {name} affine != atlas_t1 affine")
        spacing = tuple(float(value) for value in reference.header.get_zooms()[:3])
        if not np.allclose(spacing, BRATS_SPACING_MM, rtol=0.0, atol=1.0e-6):
            errors.append(f"atlas spacing is not 1 mm isotropic: {spacing}")
    return {
        "status": "PASS" if not errors else "BLOCKED",
        "checked_at": _now(),
        "tools": {name: None if path is None else str(path) for name, path in tools.items()},
        "atlas": atlas_metadata,
        "errors": errors,
    }


def require_preflight(toolchain: Toolchain = DEFAULT_TOOLCHAIN) -> dict[str, Any]:
    report = preflight_report(toolchain)
    if report["status"] != "PASS":
        raise RuntimeError("BraTS21 preprocessing preflight failed:\n- " + "\n- ".join(report["errors"]))
    return report


def geometry_matches(t1_path: str | Path, modality_path: str | Path, atol: float = 1.0e-5) -> bool:
    t1 = _validate_nifti(Path(t1_path))
    modality = _validate_nifti(Path(modality_path))
    return tuple(t1.shape) == tuple(modality.shape) and np.allclose(
        t1.affine,
        modality.affine,
        rtol=0.0,
        atol=float(atol),
    )


def build_registration_command(
    executable: str,
    fixed: str | Path,
    moving: str | Path,
    fixed_mask: str | Path,
    moving_mask: str | Path,
    output_prefix: str | Path,
    *,
    affine: bool,
    random_seed: int,
) -> list[str]:
    prefix = str(output_prefix)
    warped = f"{prefix}Warped.nii.gz"
    command = [
        executable,
        "--dimensionality",
        "3",
        "--float",
        "0",
        "--collapse-output-transforms",
        "1",
        "--output",
        f"[{prefix},{warped}]",
        "--interpolation",
        "Linear",
        "--winsorize-image-intensities",
        "[0.005,0.995]",
        "--use-histogram-matching",
        "0",
        "--initial-moving-transform",
        f"[{fixed},{moving},1]",
        "--transform",
        "Rigid[0.1]",
        "--metric",
        f"MI[{fixed},{moving},1,32,Regular,0.25]",
        "--convergence",
        "[1000x500x250x100,1e-6,10]",
        "--shrink-factors",
        "8x4x2x1",
        "--smoothing-sigmas",
        "3x2x1x0vox",
    ]
    if affine:
        command.extend(
            [
                "--transform",
                "Affine[0.1]",
                "--metric",
                f"MI[{fixed},{moving},1,32,Regular,0.25]",
                "--convergence",
                "[1000x500x250x100,1e-6,10]",
                "--shrink-factors",
                "8x4x2x1",
                "--smoothing-sigmas",
                "3x2x1x0vox",
            ]
        )
    command.extend(["--masks", f"[{fixed_mask},{moving_mask}]", "--random-seed", str(random_seed)])
    return command


def build_apply_command(
    executable: str,
    moving: str | Path,
    reference: str | Path,
    output: str | Path,
    transforms: Sequence[str | Path],
    *,
    nearest: bool = False,
) -> list[str]:
    command = [
        executable,
        "-d",
        "3",
        "-i",
        str(moving),
        "-r",
        str(reference),
        "-o",
        str(output),
        "-n",
        "NearestNeighbor" if nearest else "Linear",
        "--float",
        "0",
        "--default-value",
        "0",
    ]
    for transform in transforms:
        command.extend(["-t", str(transform)])
    return command


def build_n4_command(
    executable: str,
    moving: str | Path,
    mask: str | Path,
    output: str | Path,
) -> list[str]:
    return [
        executable,
        "-d",
        "3",
        "-i",
        str(moving),
        "-x",
        str(mask),
        "-o",
        str(output),
        "-s",
        "4",
        "-b",
        "[200]",
        "-c",
        "[50x50x50x50,1e-7]",
    ]


def build_synthstrip_command(
    executable: str,
    moving: str | Path,
    brain_output: str | Path,
    mask_output: str | Path,
) -> list[str]:
    return [
        executable,
        "-i",
        str(moving),
        "-o",
        str(brain_output),
        "-m",
        str(mask_output),
    ]


def _run_command(command: Sequence[str], log_path: Path, env: Mapping[str, str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\n$ " + subprocess.list2cmdline([str(value) for value in command]) + "\n")
        log.flush()
        completed = subprocess.run(
            [str(value) for value in command],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
            env=dict(env),
            text=True,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {completed.returncode}: "
            f"{subprocess.list2cmdline([str(value) for value in command])}; log={log_path}"
        )


def _load_float(path: str | Path) -> tuple[nib.Nifti1Image, np.ndarray]:
    image = _validate_nifti(Path(path))
    data = np.asarray(image.dataobj, dtype=np.float32)
    if not np.all(np.isfinite(data)):
        raise ValueError(f"NaN/Inf found in {path}")
    return image, data


def _save_like_atlas(path: Path, data: np.ndarray, atlas: nib.Nifti1Image, *, mask: bool = False) -> None:
    header = atlas.header.copy()
    dtype = np.uint8 if mask else np.float32
    header.set_data_dtype(dtype)
    output = nib.Nifti1Image(np.asarray(data, dtype=dtype), atlas.affine, header=header)
    output.set_qform(atlas.get_qform(), int(atlas.header["qform_code"]))
    output.set_sform(atlas.get_sform(), int(atlas.header["sform_code"]))
    nib.save(output, str(path))


def _mask_image(image_path: Path, mask_path: Path, output_path: Path) -> None:
    image, data = _load_float(image_path)
    mask_image, mask = _load_float(mask_path)
    if tuple(image.shape) != tuple(mask_image.shape) or not np.allclose(
        image.affine, mask_image.affine, rtol=0.0, atol=1.0e-5
    ):
        raise ValueError(f"Mask geometry mismatch: {image_path} vs {mask_path}")
    header = image.header.copy()
    header.set_data_dtype(np.float32)
    nib.save(
        nib.Nifti1Image((data * (mask > 0.5)).astype(np.float32), image.affine, header=header),
        str(output_path),
    )


def _read_itk_affine_linear(path: Path) -> np.ndarray:
    try:
        from scipy.io import loadmat

        payload = loadmat(str(path))
    except Exception as exc:  # noqa: BLE001 - include scipy/format errors in QC
        raise ValueError(f"Cannot read ANTs affine transform {path}: {exc}") from exc
    candidates = [
        np.asarray(value).reshape(-1)
        for key, value in payload.items()
        if not key.startswith("__") and np.asarray(value).size >= 9
    ]
    if not candidates:
        raise ValueError(f"No affine parameters found in ANTs transform: {path}")
    return np.asarray(candidates[0][:9], dtype=np.float64).reshape(3, 3)


def _dice(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=bool)
    right = np.asarray(right, dtype=bool)
    denominator = int(left.sum()) + int(right.sum())
    if denominator == 0:
        return 1.0
    return float(2.0 * np.count_nonzero(left & right) / denominator)


def _center_of_mass_world(mask: np.ndarray, affine: np.ndarray) -> np.ndarray:
    points = np.argwhere(np.asarray(mask, dtype=bool))
    if points.size == 0:
        raise ValueError("Cannot compute center of mass of an empty mask.")
    return nib.affines.apply_affine(np.asarray(affine, dtype=np.float64), points.mean(axis=0))


def _center_distance_mm(
    left: np.ndarray,
    left_affine: np.ndarray,
    right: np.ndarray,
    right_affine: np.ndarray,
) -> float:
    return float(
        np.linalg.norm(
            _center_of_mass_world(left, left_affine)
            - _center_of_mass_world(right, right_affine)
        )
    )


def _write_montage(
    path: Path,
    volumes: Mapping[str, np.ndarray],
    brain_mask: np.ndarray,
    atlas_mask: np.ndarray,
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    center = np.rint(np.argwhere(brain_mask).mean(axis=0)).astype(int)
    modalities = [name for name in MODEL_CHANNEL_ORDER if name in volumes]
    figure, axes = plt.subplots(3, len(modalities), figsize=(4 * len(modalities), 11))
    axes = np.asarray(axes).reshape(3, len(modalities))
    for row, axis in enumerate((0, 1, 2)):
        index = int(np.clip(center[axis], 0, brain_mask.shape[axis] - 1))
        mask_slice = np.take(atlas_mask, index, axis=axis).T
        for column, modality in enumerate(modalities):
            image_slice = np.take(volumes[modality], index, axis=axis).T
            valid = image_slice[brain_mask.take(index, axis=axis).T]
            if valid.size:
                low, high = np.percentile(valid, (1.0, 99.0))
            else:
                low, high = float(image_slice.min()), float(image_slice.max())
            axes[row, column].imshow(image_slice, cmap="gray", origin="lower", vmin=low, vmax=high)
            if mask_slice.any() and not mask_slice.all():
                axes[row, column].contour(mask_slice, levels=[0.5], colors="lime", linewidths=0.5)
            axes[row, column].set_title(f"{modality} axis={axis} index={index}")
            axes[row, column].axis("off")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=120)
    plt.close(figure)


def _source_state(record: SessionRecord) -> dict[str, dict[str, Any]]:
    state: dict[str, dict[str, Any]] = {}
    for modality, path in record.paths.items():
        if path is None:
            continue
        stat = path.stat()
        state[modality] = {
            "path": str(path.resolve()),
            "size": int(stat.st_size),
            "sha256": sha256_file(path),
        }
    return state


def _signature(
    source_state: Mapping[str, Any],
    preflight: Mapping[str, Any],
    settings: RegistrationSettings,
) -> str:
    payload = {
        "schema_version": PIPELINE_SCHEMA_VERSION,
        "source": source_state,
        "atlas": preflight.get("atlas", {}),
        "settings": asdict(settings),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _prepare_registration_input(
    modality: str,
    source: Path,
    work: Path,
    toolchain: Toolchain,
    env: Mapping[str, str],
    log_path: Path,
) -> tuple[Path, Path]:
    brain = work / f"{modality}_synthstrip_brain.nii.gz"
    mask = work / f"{modality}_native_mask.nii.gz"
    n4 = work / f"{modality}_n4.nii.gz"
    n4_brain = work / f"{modality}_n4_brain.nii.gz"
    _run_command(build_synthstrip_command(toolchain.synthstrip, source, brain, mask), log_path, env)
    _run_command(build_n4_command(toolchain.n4_bias_field_correction, source, mask, n4), log_path, env)
    _mask_image(n4, mask, n4_brain)
    return n4_brain, mask


def process_session(
    record: SessionRecord,
    output_root: str | Path,
    toolchain: Toolchain,
    settings: RegistrationSettings,
    preflight: Mapping[str, Any],
    *,
    resume: bool = True,
    force: bool = False,
) -> dict[str, Any]:
    if record.status == "EXCLUDED" or record.t1_path is None:
        return {**record.as_row(), "processing_status": "EXCLUDED"}
    output_root = Path(output_root).resolve()
    target = output_root / Path(record.relative_dir)
    source_before = _source_state(record)
    signature = _signature(source_before, preflight, settings)
    status_path = target / "status.json"
    if status_path.is_file() and resume and not force:
        existing = json.loads(status_path.read_text(encoding="utf-8"))
        if existing.get("signature") == signature and existing.get("status") in {"PASS", "PARTIAL"}:
            return {**existing, "resumed": True}
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{record.file_stem}.staging-", dir=target.parent))
    work = stage / "work"
    transforms_dir = stage / "transforms"
    qc_dir = stage / "qc"
    work.mkdir()
    transforms_dir.mkdir()
    qc_dir.mkdir()
    log_path = stage / "processing.log"
    env = os.environ.copy()
    env["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "1"
    env["ANTS_RANDOM_SEED"] = str(settings.random_seed)
    try:
        t1_source = Path(record.t1_path)
        t1_n4_brain, t1_native_mask = _prepare_registration_input(
            "t1", t1_source, work, toolchain, env, log_path
        )
        t1_prefix = transforms_dir / "t1_to_sri24_"
        _run_command(
            build_registration_command(
                toolchain.ants_registration,
                toolchain.atlas_brain,
                t1_n4_brain,
                toolchain.atlas_mask,
                t1_native_mask,
                t1_prefix,
                affine=True,
                random_seed=settings.random_seed,
            ),
            log_path,
            env,
        )
        t1_to_sri24 = Path(f"{t1_prefix}0GenericAffine.mat")
        if not t1_to_sri24.is_file():
            raise FileNotFoundError(f"ANTs did not produce the expected affine: {t1_to_sri24}")

        atlas = _validate_nifti(Path(toolchain.atlas_t1), expected_shape=BRATS_SHAPE)
        atlas_mask_image, atlas_mask_data = _load_float(toolchain.atlas_mask)
        atlas_mask = atlas_mask_data > 0.5
        warped_mask_path = work / "t1_mask_sri24.nii.gz"
        _run_command(
            build_apply_command(
                toolchain.ants_apply_transforms,
                t1_native_mask,
                toolchain.atlas_t1,
                warped_mask_path,
                [t1_to_sri24],
                nearest=True,
            ),
            log_path,
            env,
        )
        _, warped_mask_data = _load_float(warped_mask_path)
        common_mask = warped_mask_data > 0.5
        mask_output = stage / f"{record.file_stem}_brainmask.nii.gz"
        _save_like_atlas(mask_output, common_mask.astype(np.uint8), atlas, mask=True)

        outputs: dict[str, str] = {}
        geometry_actions: dict[str, str] = {"t1": "t1_to_sri24_rigid_affine"}
        modality_qc: dict[str, Any] = {}
        transformed_volumes: dict[str, np.ndarray] = {}
        for modality in MODEL_CHANNEL_ORDER:
            source = record.paths[modality]
            if source is None:
                continue
            transform_chain: list[Path] = [t1_to_sri24]
            native_modality_mask: Path | None = None
            if modality != "t1" and not geometry_matches(
                t1_source, source, atol=settings.geometry_atol
            ):
                modality_n4_brain, native_modality_mask = _prepare_registration_input(
                    modality, source, work, toolchain, env, log_path
                )
                rigid_prefix = transforms_dir / f"{modality}_to_t1_"
                _run_command(
                    build_registration_command(
                        toolchain.ants_registration,
                        t1_n4_brain,
                        modality_n4_brain,
                        t1_native_mask,
                        native_modality_mask,
                        rigid_prefix,
                        affine=False,
                        random_seed=settings.random_seed,
                    ),
                    log_path,
                    env,
                )
                modality_to_t1 = Path(f"{rigid_prefix}0GenericAffine.mat")
                if not modality_to_t1.is_file():
                    raise FileNotFoundError(
                        f"ANTs did not produce the expected rigid transform: {modality_to_t1}"
                    )
                # ANTs image order: outer T1->SRI24 transform, then modality->T1.
                transform_chain.append(modality_to_t1)
                geometry_actions[modality] = "modality_to_t1_rigid_then_t1_to_sri24_affine"
            elif modality != "t1":
                geometry_actions[modality] = "shared_t1_to_sri24_affine"

            temporary_output = work / f"{modality}_sri24_unmasked.nii.gz"
            _run_command(
                build_apply_command(
                    toolchain.ants_apply_transforms,
                    source,
                    toolchain.atlas_t1,
                    temporary_output,
                    transform_chain,
                ),
                log_path,
                env,
            )
            warped_image, warped = _load_float(temporary_output)
            if tuple(warped_image.shape) != BRATS_SHAPE or not np.allclose(
                warped_image.affine, atlas.affine, rtol=0.0, atol=1.0e-6
            ):
                raise ValueError(f"ANTs output grid mismatch for {record.case_id}/{modality}")
            masked = np.where(common_mask, warped, 0.0).astype(np.float32, copy=False)
            output_path = stage / f"{record.file_stem}_{modality}.nii.gz"
            _save_like_atlas(output_path, masked, atlas)
            outputs[modality] = str(output_path.name)
            transformed_volumes[modality] = masked

            if native_modality_mask is not None:
                warped_modality_mask_path = work / f"{modality}_mask_sri24.nii.gz"
                _run_command(
                    build_apply_command(
                        toolchain.ants_apply_transforms,
                        native_modality_mask,
                        toolchain.atlas_t1,
                        warped_modality_mask_path,
                        transform_chain,
                        nearest=True,
                    ),
                    log_path,
                    env,
                )
                warped_modality_mask_image, warped_modality_mask_data = _load_float(
                    warped_modality_mask_path
                )
                warped_modality_mask = warped_modality_mask_data > 0.5
                modality_qc[modality] = {
                    "mask_dice_to_t1": _dice(warped_modality_mask, common_mask),
                    "mask_com_distance_to_t1_mm": _center_distance_mm(
                        warped_modality_mask,
                        warped_modality_mask_image.affine,
                        common_mask,
                        atlas.affine,
                    ),
                }

        affine_linear = _read_itk_affine_linear(t1_to_sri24)
        determinant = float(np.linalg.det(affine_linear))
        atlas_dice = _dice(common_mask, atlas_mask)
        atlas_distance = _center_distance_mm(
            common_mask, atlas.affine, atlas_mask, atlas_mask_image.affine
        )
        failures: list[str] = []
        if not (settings.determinant_min <= abs(determinant) <= settings.determinant_max):
            failures.append("affine_determinant")
        if atlas_dice < settings.dice_min:
            failures.append("low_sri24_mask_dice")
        if atlas_distance > settings.atlas_com_distance_max_mm:
            failures.append("sri24_mask_center_distance")
        for modality, metrics in modality_qc.items():
            if metrics["mask_dice_to_t1"] < settings.modality_dice_min:
                failures.append(f"low_{modality}_to_t1_mask_dice")
            if metrics["mask_com_distance_to_t1_mm"] > settings.modality_com_distance_max_mm:
                failures.append(f"high_{modality}_to_t1_center_distance")

        qc = {
            "status": "PASS" if not failures else "QC_FAILED",
            "failures": failures,
            "atlas_mask_dice": atlas_dice,
            "atlas_mask_com_distance_mm": atlas_distance,
            "affine_determinant": determinant,
            "modality_registration": modality_qc,
            "output_shape": list(BRATS_SHAPE),
            "output_spacing_mm": list(BRATS_SPACING_MM),
            "geometry_actions": geometry_actions,
        }
        _json_dump(qc_dir / "metrics.json", qc)
        _write_montage(
            qc_dir / "montage.png", transformed_volumes, common_mask, atlas_mask
        )

        source_after = _source_state(record)
        if source_after != source_before:
            raise RuntimeError(f"Source files changed while processing {record.case_id}")
        if failures:
            status = "QC_FAILED"
        elif len(outputs) < len(MODEL_CHANNEL_ORDER):
            status = "PARTIAL"
        else:
            status = "PASS"
        result = {
            "schema_version": PIPELINE_SCHEMA_VERSION,
            "status": status,
            "case_id": record.case_id,
            "participant_id": record.participant_id,
            "session_id": record.session_id,
            "relative_dir": record.relative_dir,
            "completed_at": _now(),
            "signature": signature,
            "channel_order": list(MODEL_CHANNEL_ORDER),
            "outputs": outputs,
            "brain_mask": mask_output.name,
            "source_state": source_after,
            "qc": qc,
            "resumed": False,
        }
        _json_dump(stage / "status.json", result)
        if target.exists():
            if not force:
                raise FileExistsError(
                    f"Output exists with a different signature; use --force: {target}"
                )
            resolved_target = target.resolve()
            if output_root not in resolved_target.parents:
                raise RuntimeError(f"Refusing to replace output outside root: {resolved_target}")
            shutil.rmtree(resolved_target)
        stage.rename(target)
        return result
    except Exception:
        failure_root = output_root / "_failed" / Path(record.relative_dir)
        failure_root.parent.mkdir(parents=True, exist_ok=True)
        if failure_root.exists():
            shutil.rmtree(failure_root)
        stage.rename(failure_root)
        raise


SUMMARY_FIELDS = (
    "case_id",
    "participant_id",
    "session_id",
    "relative_dir",
    "processing_status",
    "reason",
    "t1_output",
    "t2_output",
    "flair_output",
    "brain_mask_output",
)


def _summary_row(result: Mapping[str, Any], output_root: Path) -> dict[str, Any]:
    relative = str(result.get("relative_dir", ""))
    outputs = result.get("outputs", {})
    status = str(result.get("status", result.get("processing_status", "FAILED")))
    base = output_root / Path(relative)
    qc = result.get("qc", {})
    reason = ";".join(str(value) for value in qc.get("failures", ()))
    return {
        "case_id": result.get("case_id", ""),
        "participant_id": result.get("participant_id", ""),
        "session_id": result.get("session_id", ""),
        "relative_dir": relative,
        "processing_status": status,
        "reason": reason,
        "t1_output": str(base / outputs["t1"]) if "t1" in outputs else "",
        "t2_output": str(base / outputs["t2"]) if "t2" in outputs else "",
        "flair_output": str(base / outputs["flair"]) if "flair" in outputs else "",
        "brain_mask_output": str(base / result["brain_mask"]) if result.get("brain_mask") else "",
    }


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_dataset_manifests(output_root: Path, summary_rows: Sequence[Mapping[str, Any]]) -> None:
    complete = [
        row
        for row in summary_rows
        if row["processing_status"] == "PASS"
        and all(row[f"{modality}_output"] for modality in MODEL_CHANNEL_ORDER)
    ]
    _write_csv(output_root / "dataset_manifest.csv", complete, SUMMARY_FIELDS)
    slice_rows: list[dict[str, Any]] = []
    for row in complete:
        mask = np.asarray(nib.load(str(row["brain_mask_output"])).dataobj) > 0
        for index in np.flatnonzero(np.any(mask, axis=(0, 1))):
            slice_rows.append({"case_id": row["case_id"], "slice": int(index)})
    _write_csv(output_root / "slice_manifest.csv", slice_rows, ("case_id", "slice"))


def run_pipeline(
    source_root: str | Path,
    metadata_tsv: str | Path | None,
    output_root: str | Path,
    toolchain: Toolchain = DEFAULT_TOOLCHAIN,
    settings: RegistrationSettings = RegistrationSettings(),
    *,
    case_id: str | None = None,
    workers: int = 1,
    dry_run: bool = False,
    resume: bool = True,
    force: bool = False,
) -> dict[str, Any]:
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    records = discover_sessions(source_root, metadata_tsv)
    if case_id:
        records = [record for record in records if record.case_id == case_id]
        if not records:
            raise ValueError(f"Unknown --case-id {case_id!r}")
    write_manifest(output_root / "manifest.csv", records)
    preflight = preflight_report(toolchain)
    configuration = {
        "schema_version": PIPELINE_SCHEMA_VERSION,
        "created_at": _now(),
        "source_root": str(Path(source_root).resolve()),
        "metadata_tsv": None if metadata_tsv is None else str(Path(metadata_tsv).resolve()),
        "output_root": str(output_root),
        "channel_order": list(MODEL_CHANNEL_ORDER),
        "toolchain": asdict(toolchain),
        "settings": asdict(settings),
        "preflight": preflight,
    }
    _json_dump(output_root / "configuration.json", configuration)
    if dry_run:
        return {
            "status": "DRY_RUN",
            "record_count": len(records),
            "ready_count": sum(record.status == "READY" for record in records),
            "partial_count": sum(record.status == "PARTIAL" for record in records),
            "excluded_count": sum(record.status == "EXCLUDED" for record in records),
            "preflight": preflight,
            "manifest": str(output_root / "manifest.csv"),
        }
    require_preflight(toolchain)
    eligible = [record for record in records if record.status != "EXCLUDED"]
    results: list[dict[str, Any]] = [
        {**record.as_row(), "processing_status": "EXCLUDED"}
        for record in records
        if record.status == "EXCLUDED"
    ]
    errors: list[dict[str, str]] = []
    workers = max(int(workers), 1)
    if workers == 1:
        for record in eligible:
            try:
                results.append(
                    process_session(
                        record,
                        output_root,
                        toolchain,
                        settings,
                        preflight,
                        resume=resume,
                        force=force,
                    )
                )
            except Exception as exc:  # noqa: BLE001 - batch must continue and report
                errors.append({"case_id": record.case_id, "error": str(exc)})
                results.append(
                    {
                        **record.as_row(),
                        "processing_status": "FAILED",
                        "qc": {"failures": [str(exc)]},
                    }
                )
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    process_session,
                    record,
                    output_root,
                    toolchain,
                    settings,
                    preflight,
                    resume=resume,
                    force=force,
                ): record
                for record in eligible
            }
            for future in as_completed(futures):
                record = futures[future]
                try:
                    results.append(future.result())
                except Exception as exc:  # noqa: BLE001
                    errors.append({"case_id": record.case_id, "error": str(exc)})
                    results.append(
                        {
                            **record.as_row(),
                            "processing_status": "FAILED",
                            "qc": {"failures": [str(exc)]},
                        }
                    )
    summary_rows = sorted(
        (_summary_row(result, output_root) for result in results),
        key=lambda row: str(row["case_id"]),
    )
    _write_csv(output_root / "summary.csv", summary_rows, SUMMARY_FIELDS)
    _write_dataset_manifests(output_root, summary_rows)
    final = {
        "status": "PASS" if not errors else "COMPLETED_WITH_FAILURES",
        "completed_at": _now(),
        "counts": {
            status: sum(row["processing_status"] == status for row in summary_rows)
            for status in ("PASS", "PARTIAL", "QC_FAILED", "EXCLUDED", "FAILED")
        },
        "errors": errors,
        "summary": str(output_root / "summary.csv"),
        "dataset_manifest": str(output_root / "dataset_manifest.csv"),
        "slice_manifest": str(output_root / "slice_manifest.csv"),
    }
    _json_dump(output_root / "run_summary.json", final)
    return final


__all__ = [
    "BRATS_SHAPE",
    "BRATS_SPACING_MM",
    "DEFAULT_TOOLCHAIN",
    "MODEL_CHANNEL_ORDER",
    "RegistrationSettings",
    "SessionRecord",
    "Toolchain",
    "build_apply_command",
    "build_registration_command",
    "classify_modality",
    "discover_sessions",
    "geometry_matches",
    "preflight_report",
    "process_session",
    "require_preflight",
    "run_pipeline",
    "sha256_file",
    "write_manifest",
]
