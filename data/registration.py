"""ANDi data preparation 使用的 registration backend。

registration 依賴較重且執行較慢，因此獨立在這裡。data preparation 可以選擇
backend，而 dataset/evaluator 不需要知道 DIPY 或 SimpleITK 的細節。
"""

from __future__ import annotations

import multiprocessing
import os
from pathlib import Path
from time import time
from typing import Any

import numpy as np


class MRIRegistrator:
    """使用 DIPY 做 affine MRI registration，流程相容於原版 ANDi。"""

    def __init__(
        self,
        template_path: str | Path,
        brain_mask_path: str | Path | None = None,
        nbins: int = 32,
        sampling_proportion: int | None = None,
        level_iters: list[int] | None = None,
        sigmas: list[float] | None = None,
        factors: list[int] | None = None,
        verbose: bool = False,
        rotate: bool = True,
    ):
        try:
            import nibabel as nib
        except ImportError as exc:
            raise ImportError("MRIRegistrator requires the optional 'nibabel' package.") from exc

        template_path = Path(template_path)
        if not template_path.exists():
            raise RuntimeError("Template path does not exist. Download/configure the SRI atlas first.")

        template_data = nib.load(str(template_path))
        self.template = template_data.get_fdata()
        self.template_affine = template_data.affine
        if self.template.ndim == 4:
            self.template = self.template.squeeze(-1)

        if rotate:
            # 原版前處理會旋轉 SRI template，使其符合 Shifts data 預期的方向。
            self.template = np.rot90(self.template, 2, axes=(0, 1))
            center = np.array(self.template.shape) / 2.0
            translation_matrix = np.eye(4)
            translation_matrix[:3, 3] = center
            rotation_matrix = np.diag([-1, -1, 1, 1])
            self.template_affine = self.template_affine @ translation_matrix @ rotation_matrix

        if brain_mask_path is not None:
            mask = nib.load(str(brain_mask_path)).get_fdata()
            self.template = self.template * mask

        self.nbins = nbins
        self.sampling_proportion = sampling_proportion
        self.level_iters = level_iters or [100, 10, 5]
        self.sigmas = sigmas or [3.0, 1.0, 1.0]
        self.factors = factors or [4, 2, 1]
        self.verbose = verbose

    def _print(self, message: str) -> None:
        if self.verbose:
            print(message)

    @staticmethod
    def save_nii(path: str | Path, image: np.ndarray, affine: np.ndarray, dtype: str | np.dtype) -> None:
        import nibabel as nib

        nib.save(nib.Nifti1Image(image.astype(dtype), affine), str(path))

    @staticmethod
    def load_nii(path: str | Path, dtype: str = "short") -> tuple[np.ndarray, np.ndarray]:
        import nibabel as nib

        data = nib.load(str(path), keep_file_open=False)
        volume = data.get_fdata(caching="unchanged", dtype=np.float32).astype(np.dtype(dtype))
        return volume, data.affine

    def transform(
        self,
        img: str | Path | np.ndarray,
        save_path: str | Path,
        transformation: Any,
        affine: np.ndarray,
        dtype: str | np.dtype = "float32",
    ) -> None:
        if isinstance(img, (str, Path)):
            img, _ = self.load_nii(img, dtype="float32")
        transformed = transformation.transform(img)
        out_dtype = np.short if (transformed - transformed.astype(np.short)).sum() == 0.0 else np.dtype("<f4")
        self.save_nii(save_path, transformed, affine, out_dtype if dtype is None else dtype)

    def register_batch(self, moving_list: list[str | Path], num_cpus: int | None = None) -> dict[str, Any]:
        """平行 registration 多個 T1 image，並回傳 path -> transform。"""

        num_cpus = os.cpu_count() if num_cpus is None else num_cpus
        moving_batches = [list(batch) for batch in np.array_split(moving_list, num_cpus) if len(batch) > 0]
        with multiprocessing.Pool(processes=len(moving_batches)) as pool:
            results = pool.starmap(self._register_batch, zip(moving_batches, range(len(moving_batches))))
        transformations = {}
        for result in results:
            transformations.update(result)
        return transformations

    def _register_batch(self, moving_list: list[str | Path], process_index: int) -> dict[str, Any]:
        transformations = {}
        start = time()
        for index, path in enumerate(moving_list):
            path = str(path)
            save_path = "_".join(path.split("_")[0:-2]) + "_t1.nii"
            if path.endswith(".gz"):
                save_path += ".gz"
            _, transformation = self(path, save_path=save_path)
            transformations[path] = transformation
            print(f"Process {process_index} finished {index + 1} of {len(moving_list)} in {time() - start:.2f}s")
        return transformations

    def __call__(
        self,
        moving: str | Path | np.ndarray,
        moving_affine: np.ndarray | None = None,
        save_path: str | Path | None = None,
    ) -> tuple[np.ndarray, Any]:
        try:
            from dipy.align.imaffine import AffineMap, AffineRegistration, MutualInformationMetric, transform_centers_of_mass
            from dipy.align.transforms import AffineTransform3D, RigidTransform3D, TranslationTransform3D
        except ImportError as exc:
            raise ImportError("MRIRegistrator requires the optional 'dipy' package.") from exc

        if isinstance(moving, (str, Path)):
            moving, moving_affine = self.load_nii(moving, dtype="<f4")
        if moving_affine is None:
            raise ValueError("moving_affine is required when moving is an array.")

        del AffineMap
        # registration 由粗到細分階段進行；比直接 fitting affine 更穩定。
        c_of_mass = transform_centers_of_mass(self.template, self.template_affine, moving, moving_affine)
        metric = MutualInformationMetric(self.nbins, self.sampling_proportion)
        affreg = AffineRegistration(
            metric=metric,
            level_iters=self.level_iters,
            sigmas=self.sigmas,
            factors=self.factors,
            verbosity=1 if self.verbose else 0,
        )
        translation = affreg.optimize(
            self.template,
            moving,
            TranslationTransform3D(),
            None,
            self.template_affine,
            moving_affine,
            starting_affine=c_of_mass.affine,
        )
        rigid = affreg.optimize(
            self.template,
            moving,
            RigidTransform3D(),
            None,
            self.template_affine,
            moving_affine,
            starting_affine=translation.affine,
        )
        affine = affreg.optimize(
            self.template,
            moving,
            AffineTransform3D(),
            None,
            self.template_affine,
            moving_affine,
            starting_affine=rigid.affine,
        )
        registered = affine.transform(moving)
        if save_path is not None:
            dtype = "short" if np.abs(registered - registered.astype(np.short)).sum() == 0 else "<f4"
            self.save_nii(save_path, registered, self.template_affine, dtype)
        return registered, affine


class SitkRegistrator:
    """SimpleITK registration backend，供偏好 elastix transform map 的專案使用。"""

    def __init__(self, template_path: str | Path):
        try:
            import SimpleITK as sitk
        except ImportError as exc:
            raise ImportError("SitkRegistrator requires the optional 'SimpleITK' package.") from exc

        self.sitk = sitk
        template_path = Path(template_path)
        if not template_path.exists():
            raise RuntimeError("Template path does not exist.")
        self.fixed_image = self.sitk.ReadImage(str(template_path))

    def register_batch(self, moving_list: list[str | Path]) -> dict[str, Any]:
        """使用 SimpleITK/elastix registration 多個檔案，並收集 transforms。"""

        transformations = {}
        for path in moving_list:
            save_path = str(path).split("nii")[0][:-1] + "_registered.nii"
            if str(path).endswith(".gz"):
                save_path += ".gz"
            _, transformation = self(path, save_path=save_path)
            transformations[str(path)] = transformation
        return transformations

    def transform(self, img: str | Path | Any, transform_parameter_map: Any, save_path: str | Path | None = None) -> Any:
        if isinstance(img, (str, Path)):
            img = self.sitk.ReadImage(str(img))
        transformix = self.sitk.TransformixImageFilter()
        transformix.SetTransformParameterMap(transform_parameter_map)
        transformix.LogToConsoleOff()
        transformix.SetMovingImage(img)
        transformix.Execute()
        result = transformix.GetResultImage()
        if save_path is not None:
            self.sitk.WriteImage(result, str(save_path))
        return result

    def __call__(self, img: str | Path | Any, save_path: str | Path | None = None, transform: str = "affine") -> tuple[Any, Any]:
        elastix = self.sitk.ElastixImageFilter()
        elastix.LogToConsoleOff()
        elastix.SetFixedImage(self.fixed_image)
        if isinstance(img, (str, Path)):
            img = self.sitk.ReadImage(str(img))
        elastix.SetMovingImage(img)
        elastix.SetParameterMap(self.sitk.GetDefaultParameterMap(transform))
        elastix.Execute()
        result = elastix.GetResultImage()
        if save_path is not None:
            self.sitk.WriteImage(result, str(save_path))
        return result, elastix.GetTransformParameterMap()
