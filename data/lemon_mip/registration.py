"""DIPY registration with explicit pull/forward world-transform conventions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

import numpy as np


@dataclass(frozen=True)
class RegistrationSettings:
    nbins: int = 32
    sampling_proportion: float | None = None
    level_iters: tuple[int, ...] = (1000, 100, 10)
    sigmas: tuple[float, ...] = (3.0, 1.0, 0.0)
    factors: tuple[int, ...] = (4, 2, 1)

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class RegistrationResult:
    pull_static_to_moving_world: np.ndarray
    forward_moving_to_static_world: np.ndarray
    transformed: np.ndarray
    mi_before: float
    mi_after: float
    stage_objectives: dict[str, float | None]

    @property
    def mi_improvement(self) -> float:
        return float(self.mi_after - self.mi_before)


def load_canonical_nifti(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    try:
        import nibabel as nib
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Registration requires nibabel.") from exc
    image = nib.as_closest_canonical(nib.load(str(path)), enforce_diag=False)
    data = image.get_fdata(dtype=np.float32, caching="unchanged")
    affine = np.asarray(image.affine, dtype=np.float64)
    if data.ndim != 3 or not np.all(np.isfinite(data)):
        raise ValueError(f"Canonical NIfTI must be finite 3D: {path}, shape={data.shape}.")
    if not np.all(np.isfinite(affine)) or np.linalg.det(affine[:3, :3]) == 0:
        raise ValueError(f"Canonical NIfTI has invalid affine: {path}.")
    return data.astype(np.float32, copy=False), affine


def dipy_pull_to_forward(pull_static_to_moving_world: np.ndarray) -> np.ndarray:
    """Convert DIPY's static-world→moving-world sampling map to moving→static."""

    pull = np.asarray(pull_static_to_moving_world, dtype=np.float64)
    if pull.shape != (4, 4) or not np.all(np.isfinite(pull)):
        raise ValueError("DIPY pull matrix must be finite 4x4.")
    return np.linalg.inv(pull)


def compose_forward_transforms(*moving_to_static_world: np.ndarray) -> np.ndarray:
    """Compose forward transforms supplied in traversal order.

    ``compose_forward_transforms(flair_to_t1, t1_to_mni)`` returns
    ``t1_to_mni @ flair_to_t1``.
    """

    result = np.eye(4, dtype=np.float64)
    for transform in moving_to_static_world:
        value = np.asarray(transform, dtype=np.float64)
        if value.shape != (4, 4) or not np.all(np.isfinite(value)):
            raise ValueError("Every forward transform must be finite 4x4.")
        result = value @ result
    return result


def resample_with_forward_transform(
    moving: np.ndarray,
    moving_affine: np.ndarray,
    static_shape: Iterable[int],
    static_affine: np.ndarray,
    forward_moving_to_static_world: np.ndarray,
    *,
    interpolation: Literal["linear", "nearest"] = "linear",
) -> np.ndarray:
    """Pull a moving image once onto a static grid using a forward world map."""

    try:
        from dipy.align.imaffine import AffineMap
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Registration requires DIPY.") from exc
    moving = np.asarray(moving)
    static_shape = tuple(int(value) for value in static_shape)
    pull = np.linalg.inv(np.asarray(forward_moving_to_static_world, dtype=np.float64))
    mapping = AffineMap(
        pull,
        domain_grid_shape=static_shape,
        domain_grid2world=np.asarray(static_affine, dtype=np.float64),
        codomain_grid_shape=moving.shape,
        codomain_grid2world=np.asarray(moving_affine, dtype=np.float64),
    )
    result = mapping.transform(moving, interpolation=interpolation)
    if interpolation == "nearest":
        return (result > 0.5).astype(np.uint8)
    return np.asarray(result, dtype=np.float32)


def mutual_information(
    static: np.ndarray,
    moving_on_static: np.ndarray,
    *,
    bins: int = 32,
    mask: np.ndarray | None = None,
    max_samples: int = 500_000,
) -> float:
    """Deterministic Shannon mutual information in nats for QC."""

    first = np.asarray(static, dtype=np.float32)
    second = np.asarray(moving_on_static, dtype=np.float32)
    if first.shape != second.shape:
        raise ValueError(f"MI arrays must have the same shape, got {first.shape} and {second.shape}.")
    valid = np.isfinite(first) & np.isfinite(second)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    else:
        valid &= (np.abs(first) > 1.0e-8) | (np.abs(second) > 1.0e-8)
    x = first[valid]
    y = second[valid]
    if x.size < max(64, bins * 2):
        return 0.0
    if x.size > max_samples:
        step = int(np.ceil(x.size / max_samples))
        x, y = x[::step], y[::step]
    x_low, x_high = np.percentile(x, [0.5, 99.5])
    y_low, y_high = np.percentile(y, [0.5, 99.5])
    if x_high <= x_low or y_high <= y_low:
        return 0.0
    x = np.clip(x, x_low, x_high)
    y = np.clip(y, y_low, y_high)
    joint, _, _ = np.histogram2d(x, y, bins=int(bins), range=((x_low, x_high), (y_low, y_high)))
    probability = joint / max(float(joint.sum()), 1.0)
    px = probability.sum(axis=1, keepdims=True)
    py = probability.sum(axis=0, keepdims=True)
    expected = px @ py
    nonzero = probability > 0
    return float(np.sum(probability[nonzero] * np.log(probability[nonzero] / expected[nonzero])))


def _identity_resample(
    static: np.ndarray,
    static_affine: np.ndarray,
    moving: np.ndarray,
    moving_affine: np.ndarray,
) -> np.ndarray:
    try:
        from dipy.align.imaffine import AffineMap
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Registration requires DIPY.") from exc
    mapping = AffineMap(
        np.eye(4),
        domain_grid_shape=static.shape,
        domain_grid2world=static_affine,
        codomain_grid_shape=moving.shape,
        codomain_grid2world=moving_affine,
    )
    return np.asarray(mapping.transform(moving, interpolation="linear"), dtype=np.float32)


def register_affine_multistage(
    static: np.ndarray,
    static_affine: np.ndarray,
    moving: np.ndarray,
    moving_affine: np.ndarray,
    *,
    final_transform: Literal["rigid", "affine"],
    settings: RegistrationSettings,
    static_mask: np.ndarray | None = None,
    moving_mask: np.ndarray | None = None,
) -> RegistrationResult:
    try:
        from dipy.align.imaffine import (
            AffineRegistration,
            MutualInformationMetric,
            transform_centers_of_mass,
        )
        from dipy.align.transforms import AffineTransform3D, RigidTransform3D, TranslationTransform3D
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Registration requires DIPY.") from exc

    static = np.asarray(static, dtype=np.float32)
    moving = np.asarray(moving, dtype=np.float32)
    static_affine = np.asarray(static_affine, dtype=np.float64)
    moving_affine = np.asarray(moving_affine, dtype=np.float64)
    optimization_static_mask = (
        np.asarray(static_mask, dtype=np.float32) if static_mask is not None else None
    )
    optimization_moving_mask = (
        np.asarray(moving_mask, dtype=np.float32) if moving_mask is not None else None
    )
    if optimization_static_mask is not None and optimization_static_mask.shape != static.shape:
        raise ValueError("static_mask shape must match static image shape.")
    if optimization_moving_mask is not None and optimization_moving_mask.shape != moving.shape:
        raise ValueError("moving_mask shape must match moving image shape.")
    before = _identity_resample(static, static_affine, moving, moving_affine)
    metric_mask = (
        np.asarray(optimization_static_mask, dtype=bool)
        if optimization_static_mask is not None
        else None
    )
    mi_before = mutual_information(static, before, bins=settings.nbins, mask=metric_mask)

    center = transform_centers_of_mass(static, static_affine, moving, moving_affine)
    metric = MutualInformationMetric(settings.nbins, settings.sampling_proportion)
    registration = AffineRegistration(
        metric=metric,
        level_iters=list(settings.level_iters),
        sigmas=list(settings.sigmas),
        factors=list(settings.factors),
        verbosity=0,
    )
    stage_objectives: dict[str, float | None] = {"center_of_mass": None}

    def optimize(transform: Any, starting: np.ndarray, name: str):
        result, _, objective = registration.optimize(
            static,
            moving,
            transform,
            None,
            static_affine,
            moving_affine,
            starting_affine=starting,
            ret_metric=True,
            static_mask=optimization_static_mask,
            moving_mask=optimization_moving_mask,
        )
        stage_objectives[name] = float(objective)
        return result

    translation = optimize(TranslationTransform3D(), center.affine, "translation")
    rigid = optimize(RigidTransform3D(), translation.affine, "rigid")
    final_map = rigid
    if final_transform == "affine":
        final_map = optimize(AffineTransform3D(), rigid.affine, "affine")
    elif final_transform != "rigid":
        raise ValueError("final_transform must be 'rigid' or 'affine'.")

    transformed = np.asarray(final_map.transform(moving, interpolation="linear"), dtype=np.float32)
    mi_after = mutual_information(static, transformed, bins=settings.nbins, mask=metric_mask)
    pull = np.asarray(final_map.affine, dtype=np.float64)
    return RegistrationResult(
        pull_static_to_moving_world=pull,
        forward_moving_to_static_world=dipy_pull_to_forward(pull),
        transformed=transformed,
        mi_before=mi_before,
        mi_after=mi_after,
        stage_objectives=stage_objectives,
    )
