import tempfile
import unittest
from pathlib import Path
import nibabel as nib
import numpy as np
from andi_rewrite.scripts.prepare_healthy_sri24 import discover, split_subjects


class HealthyInventoryTest(unittest.TestCase):
    def image(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        image = nib.Nifti1Image(np.ones((4,4,4),dtype=np.float32),np.eye(4))
        image.set_qform(np.eye(4),1)
        image.set_sform(np.eye(4),1)
        nib.save(image,path)

    def test_oasis_duplicates_exclude_whole_visit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for mod in ('T1w','T2w','FLAIR'):
                self.image(root/'OAS1_MR_d0001'/'anat1'/f'sub-OAS1_{mod}.nii.gz')
            records,_=discover('oasis3',root)
            self.assertEqual(records[0].status,'READY')
            self.image(root/'OAS1_MR_d0001'/'anat2'/'sub-OAS1_run-02_T2w.nii.gz')
            records,_=discover('oasis3',root)
            self.assertEqual(records[0].status,'EXCLUDED')
            self.assertIn('t2_candidates:2',records[0].reasons)

    def test_mpi_highres_and_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); anat=root/'sub-1'/'ses-01'/'anat'
            for name in ('T1w','T2w','acq-lowres_FLAIR'):
                self.image(anat/f'sub-1_{name}.nii.gz')
            records,_=discover('mpi',root)
            self.assertEqual(records[0].status,'EXCLUDED')
            self.image(anat/'sub-1_acq-highres_FLAIR.nii.gz')
            records,_=discover('mpi',root)
            self.assertEqual(records[0].status,'READY')
            self.assertIn('highres',records[0].flair_path)

    def test_split_deduplicates_subjects_and_is_stable(self):
        ids=[f'p{i}' for i in range(21)]
        split=split_subjects(ids+ids)
        self.assertEqual(split,split_subjects(reversed(ids)))
        self.assertEqual(sum(v=='val' for v in split.values()),3)
