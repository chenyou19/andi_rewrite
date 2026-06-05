"""外部 MRI dataset 的整理、registration 與 normalization 流程。

這些 routine 對應原版 prepare_data.py 的責任，但拆成可由 Python 重用的模組；
CLI 只是一層薄 wrapper。
"""

from __future__ import annotations

import shutil
from glob import glob
from pathlib import Path

import numpy as np

from andi_rewrite.data.registration import MRIRegistrator


class ShiftsDataPreparer:
    """將 Shifts/MSSEG 風格資料夾整理成 ANDi evaluation 可讀格式。"""

    def __init__(self, dataset_path: str | Path):
        self.dataset_path = Path(dataset_path)

    def prepare_patient_folders(self, output_dir: str | Path | None = None) -> Path:
        """把分散的 challenge 檔案收斂成每個 patient 一個資料夾。"""

        parent = self.dataset_path.parent
        dataset_name = self.dataset_path.name
        target = Path(output_dir) if output_dir is not None else parent / "patients"
        target.mkdir(parents=True, exist_ok=True)

        for folder in self.dataset_path.glob("*"):
            if "unsu" in folder.name.lower() or not folder.is_dir():
                continue
            sub_folders = [path for path in folder.glob("*") if path.is_dir()]
            if not sub_folders:
                continue
            patient_files = list(sub_folders[0].glob("*"))
            patient_ids = [path.name.split("_")[0] for path in patient_files]
            for sub_folder in sub_folders:
                if "indiv" in str(sub_folder).lower():
                    continue
                folder_name = sub_folder.parent.name
                for patient_id in patient_ids:
                    matches = list(sub_folder.glob(f"{patient_id}_*"))
                    if not matches:
                        continue
                    final_folder = target / f"{dataset_name}_{folder_name}_{patient_id}"
                    final_folder.mkdir(parents=True, exist_ok=True)
                    # 使用 copy 而不是 move，避免改動原始 challenge 下載檔。
                    shutil.copy(matches[0], final_folder / matches[0].name)
        return target

    def register(
        self,
        template_path: str | Path,
        backend: str = "dipy",
        t1_glob: str = "*/*_T1_*",
    ) -> dict:
        """對 T1 做 registration，並將同一個 transform 套到其他 modalities。"""

        if backend != "dipy":
            raise ValueError("Currently ShiftsDataPreparer.register uses backend='dipy'.")
        files = glob(str(self.dataset_path / t1_glob))
        if not files:
            raise RuntimeError(f"Found 0 files to register under {self.dataset_path}")
        registrator = MRIRegistrator(template_path)
        transformations = registrator.register_batch(files)
        for path, transformation in transformations.items():
            base = path[: path.rfind("T1")]
            # 原版 Shifts conversion 會把 registered output 命名成小寫 modality suffix；
            # volume loader 也預期這種命名格式。
            for source_suffix, output_suffix, dtype in [
                ("T2_isovox.nii.gz", "t2.nii.gz", "float32"),
                ("FLAIR_isovox.nii.gz", "flair.nii.gz", "float32"),
                ("T1CE_isovox.nii.gz", "t1ce.nii.gz", "float32"),
                ("gt_isovox.nii.gz", "seg.nii.gz", "short"),
            ]:
                source = base + source_suffix
                if Path(source).exists():
                    registrator.transform(
                        img=source,
                        save_path=base + output_suffix,
                        transformation=transformation,
                        affine=registrator.template_affine,
                        dtype=dtype,
                    )
        return transformations

    def histogram_matching(self, source_volume: str | Path) -> None:
        """將每個 patient modality 的 histogram 對齊到指定 BraTS reference subject。"""

        try:
            import nibabel as nib
            import SimpleITK as sitk
        except ImportError as exc:
            raise ImportError("Histogram matching requires 'nibabel' and 'SimpleITK'.") from exc

        source_volume = Path(source_volume)
        source_id = source_volume.name
        modalities = ["flair", "t1", "t1ce", "t2"]
        references = []
        for modality in modalities:
            image_path = source_volume / f"{source_id}_{modality}.nii.gz"
            references.append(sitk.GetImageFromArray(np.asarray(nib.load(str(image_path)).dataobj, dtype=float)))

        for patient_dir in self.dataset_path.glob("*"):
            if not patient_dir.is_dir():
                continue
            for file_path in patient_dir.glob("*.nii.gz"):
                for index, modality in enumerate(modalities):
                    if file_path.name.endswith(f"_{modality}.nii.gz"):
                        volume = nib.load(str(file_path))
                        image = sitk.GetImageFromArray(np.asarray(volume.dataobj, dtype=float))
                        transformed = sitk.GetArrayFromImage(sitk.HistogramMatching(image, references[index]))
                        nib.save(nib.Nifti1Image(transformed.astype("float32"), volume.affine), str(file_path))
                        break
