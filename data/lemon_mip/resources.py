"""MNI template and isolated HD-BET runtime integration."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np

from .manifest import SessionRecord


MNI_FILES = {
    "T1": "mni_icbm152_t1_tal_nlin_asym_09c.nii",
    "mask": "mni_icbm152_t1_tal_nlin_asym_09c_mask.nii",
}
MNI_OFFICIAL_MD5 = {
    "T1": "3d5dd9b0cd727a17ceec610b782f66c1",
    "mask": "a243e249cd01a23dc30f033b9656a786",
}
_DIPY_MNI_FETCH_FILES = (
    ("5572676", "mni_icbm152_t2_tal_nlin_asym_09a.nii", "f41f2e1516d880547fbf7d6a83884f0d"),
    ("5572673", "mni_icbm152_t1_tal_nlin_asym_09a.nii", "1ea8f4f1e41bc17a94602e48141fdbc8"),
    ("5572670", "mni_icbm152_t1_tal_nlin_asym_09c_mask.nii", "a243e249cd01a23dc30f033b9656a786"),
    ("5572661", "mni_icbm152_t1_tal_nlin_asym_09c.nii", "3d5dd9b0cd727a17ceec610b782f66c1"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _md5_file(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - compatibility checksum supplied by DIPY
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fetch_mni_via_figshare_api(fetch_folder: Path) -> list[str]:
    """Fallback for DIPY 1.7's retired/private-link Figshare URLs.

    The file IDs, destination filenames, and MD5 values are exactly those in
    DIPY 1.7.0's ``fetch_mni_template`` definition. The documented public
    Figshare file-download API is used instead of changing the data source.
    """

    fetch_folder.mkdir(parents=True, exist_ok=True)
    urls: list[str] = []
    for file_id, filename, expected_md5 in _DIPY_MNI_FETCH_FILES:
        url = f"https://api.figshare.com/v2/file/download/{file_id}"
        urls.append(url)
        destination = fetch_folder / filename
        if destination.is_file() and _md5_file(destination) == expected_md5:
            continue
        temporary = destination.with_suffix(destination.suffix + ".download")
        if temporary.exists():
            temporary.unlink()
        request = Request(url, headers={"User-Agent": "andi-rewrite-lemon/1.0"})
        try:
            with urlopen(request, timeout=120) as response, temporary.open("wb") as output:
                shutil.copyfileobj(response, output)
            actual_md5 = _md5_file(temporary)
            if actual_md5 != expected_md5:
                raise ValueError(
                    f"Figshare fallback MD5 mismatch for {filename}: "
                    f"actual={actual_md5}, expected={expected_md5}."
                )
            temporary.replace(destination)
        finally:
            if temporary.exists():
                temporary.unlink()
    return urls


def disk_free_gib(path: str | Path) -> float:
    return float(shutil.disk_usage(Path(path).anchor or str(path)).free / (1024**3))


def fetch_mni2009c_template(output_root: str | Path, *, overwrite: bool = False) -> dict:
    output_root = Path(output_root).resolve()
    template_dir = output_root / "templates" / "mni2009c"
    if template_dir.exists() and any(template_dir.iterdir()):
        if not overwrite:
            metadata_path = template_dir / "template_metadata.json"
            if metadata_path.is_file() and all((template_dir / name).is_file() for name in MNI_FILES.values()):
                return json.loads(metadata_path.read_text(encoding="utf-8"))
            raise FileExistsError(f"Template directory is non-empty: {template_dir}")
        shutil.rmtree(template_dir)
    template_dir.mkdir(parents=True, exist_ok=True)

    fetch_home = output_root / "templates" / "dipy_fetch"
    previous = os.environ.get("DIPY_HOME")
    os.environ["DIPY_HOME"] = str(fetch_home)
    fetch_method = "DIPY 1.7.0 fetch_mni_template"
    fallback_urls: list[str] = []
    try:
        from dipy import __version__ as dipy_version
        from dipy.data import fetch_mni_template, read_mni_template

        try:
            fetch_mni_template()
        except (HTTPError, URLError) as exc:
            if not isinstance(exc, HTTPError) or exc.code not in {401, 403, 404}:
                raise
            fallback_urls = _fetch_mni_via_figshare_api(fetch_home / "mni_template")
            fetch_method = (
                "DIPY 1.7.0 fetch_mni_template returned HTTP "
                f"{getattr(exc, 'code', 'error')}; same DIPY file IDs fetched through "
                "documented Figshare public file API; read_mni_template used afterward"
            )
        t1_image, mask_image = read_mni_template(version="c", contrast=["T1", "mask"])
    finally:
        if previous is None:
            os.environ.pop("DIPY_HOME", None)
        else:
            os.environ["DIPY_HOME"] = previous

    import nibabel as nib

    images = {"T1": t1_image, "mask": mask_image}
    files: dict[str, dict] = {}
    for key, image in images.items():
        target = template_dir / MNI_FILES[key]
        nib.save(image, str(target))
        files[key] = {
            "path": str(target),
            "sha256": sha256_file(target),
            "official_md5": MNI_OFFICIAL_MD5[key],
            "shape": [int(value) for value in image.shape],
            "voxel_spacing_mm": [float(value) for value in nib.affines.voxel_sizes(image.affine)],
            "affine": np.asarray(image.affine, dtype=float).tolist(),
        }
    metadata = {
        "template": "MNI152 nonlinear 2009c asymmetric",
        "source": "DIPY fetch_mni_template; original MNI files mirrored by Figshare",
        "source_base_url": "https://ndownloader.figshare.com/files/",
        "fetch_method": fetch_method,
        "fallback_urls": fallback_urls,
        "dipy_version": dipy_version,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "files": files,
    }
    (template_dir / "template_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return metadata


def load_mni2009c_template(output_root: str | Path):
    import nibabel as nib

    template_dir = Path(output_root).resolve() / "templates" / "mni2009c"
    t1 = nib.as_closest_canonical(nib.load(str(template_dir / MNI_FILES["T1"])), enforce_diag=False)
    mask = nib.as_closest_canonical(nib.load(str(template_dir / MNI_FILES["mask"])), enforce_diag=False)
    if t1.shape != mask.shape or not np.allclose(t1.affine, mask.affine):
        raise ValueError("MNI T1 and mask geometry mismatch.")
    return (
        t1.get_fdata(dtype=np.float32, caching="unchanged"),
        (mask.get_fdata(dtype=np.float32, caching="unchanged") > 0.5).astype(np.uint8),
        np.asarray(t1.affine, dtype=np.float64),
    )


def brain_mask_path(output_root: str | Path, record: SessionRecord) -> Path:
    return Path(output_root) / "brain_masks" / record.case_id / record.session / "T1_bet_mask.nii.gz"


def inspect_hdbet_runtime(hdbet_executable: str | Path) -> dict:
    executable = Path(hdbet_executable).resolve()
    python = executable.parent.parent / "python.exe"
    if not executable.is_file() or not python.is_file():
        raise FileNotFoundError(f"Incomplete HD-BET environment near {executable}.")
    script = (
        "import importlib.metadata as m,json,torch; "
        "print(json.dumps({'hd_bet':m.version('HD-BET'),'torch':torch.__version__,"
        "'torch_cuda':torch.version.cuda,'cuda_available':torch.cuda.is_available(),"
        "'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,"
        "'nnunetv2':m.version('nnunetv2')}))"
    )
    probe = subprocess.run(
        [str(python), "-c", script], check=True, capture_output=True, text=True
    )
    runtime = json.loads(probe.stdout.strip().splitlines()[-1])
    if runtime["hd_bet"] != "2.0.1":
        raise ValueError(f"HD-BET version is not locked to 2.0.1: {runtime['hd_bet']}")
    if "+cu" not in runtime["torch"] or not runtime["cuda_available"]:
        raise RuntimeError(f"HD-BET PyTorch is not a working locked CUDA build: {runtime}")
    help_result = subprocess.run(
        [str(executable), "-h"], check=True, capture_output=True, text=True
    )
    for flag in ("--save_bet_mask", "--no_bet_image", "--disable_tta"):
        if flag not in help_result.stdout:
            raise RuntimeError(f"HD-BET CLI does not expose required flag {flag}.")
    freeze = subprocess.run(
        [str(python), "-m", "pip", "freeze"], check=True, capture_output=True, text=True
    ).stdout.splitlines()
    return {
        **runtime,
        "python_executable": str(python),
        "hdbet_executable": str(executable),
        "pip_freeze": freeze,
        "required_flags_present": ["--save_bet_mask", "--no_bet_image"],
        "default_device": "cuda",
        "default_tta": "enabled because --disable_tta is not supplied",
    }


def run_hdbet(
    records: Iterable[SessionRecord],
    output_root: str | Path,
    hdbet_executable: str | Path,
    *,
    scope: str,
) -> list[Path]:
    """Run HD-BET once for all masks missing from the requested record set."""

    import nibabel as nib

    output_root = Path(output_root).resolve()
    executable = Path(hdbet_executable).resolve()
    if not executable.is_file():
        raise FileNotFoundError(f"HD-BET executable does not exist: {executable}")
    records = list(records)
    pending = [record for record in records if not brain_mask_path(output_root, record).is_file()]
    if not pending:
        return [brain_mask_path(output_root, record) for record in records]

    staging = output_root / "hdbet_inputs" / scope
    raw_outputs = output_root / "hdbet_outputs" / scope
    for directory in (staging, raw_outputs):
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError(
                f"HD-BET staging/output directory is non-empty: {directory}. "
                "Remove the incomplete stage explicitly before retrying."
            )
        directory.mkdir(parents=True, exist_ok=True)

    mapping: dict[str, SessionRecord] = {}
    staging_method: dict[str, str] = {}
    for record in pending:
        name = f"{record.case_id}__{record.session}.nii.gz"
        target = staging / name
        try:
            os.link(record.t1_path, target)
            method = "hardlink"
        except OSError:
            shutil.copy2(record.t1_path, target)
            method = "copy"
        mapping[name] = record
        staging_method[name] = method

    logs = output_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    stdout_path = logs / f"hdbet_{scope}_stdout.log"
    stderr_path = logs / f"hdbet_{scope}_stderr.log"
    command = [
        str(executable),
        "-i",
        str(staging),
        "-o",
        str(raw_outputs),
        "-device",
        "cuda",
        "--save_bet_mask",
        "--no_bet_image",
    ]
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        completed = subprocess.run(command, stdout=stdout, stderr=stderr, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"HD-BET failed with exit code {completed.returncode}; see {stdout_path} and {stderr_path}."
        )

    manifest_rows: list[dict[str, object]] = []
    for input_name, record in mapping.items():
        source_mask = raw_outputs / f"{input_name[:-7]}_bet.nii.gz"
        if not source_mask.is_file():
            raise FileNotFoundError(f"HD-BET did not produce expected mask: {source_mask}")
        source_image = nib.load(str(record.t1_path))
        mask_image = nib.load(str(source_mask))
        mask_data = mask_image.get_fdata(dtype=np.float32, caching="unchanged")
        if mask_data.shape != source_image.shape or not np.all(np.isfinite(mask_data)) or not np.any(mask_data > 0.5):
            raise ValueError(f"Invalid HD-BET mask for {record.session_id}: {source_mask}")
        destination = brain_mask_path(output_root, record)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source_mask), destination)
        manifest_rows.append(
            {
                "case_id": record.case_id,
                "session": record.session,
                "source_t1": str(record.t1_path),
                "mask_path": str(destination),
                "staging_method": staging_method[input_name],
                "command": json.dumps(command),
                "environment": str(executable.parent.parent),
                "hd_bet_version": "2.0.1",
                "sha256": sha256_file(destination),
            }
        )

    manifest_path = output_root / "brain_masks" / "manifest.csv"
    existing: list[dict[str, str]] = []
    if manifest_path.is_file():
        with manifest_path.open(newline="", encoding="utf-8") as handle:
            existing = list(csv.DictReader(handle))
    combined = {(row["case_id"], row["session"]): row for row in existing}
    combined.update({(str(row["case_id"]), str(row["session"])): row for row in manifest_rows})
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(manifest_rows[0].keys())
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(combined[key] for key in sorted(combined))
    return [brain_mask_path(output_root, record) for record in records]
