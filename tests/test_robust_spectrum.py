import unittest
import numpy as np
from andi_rewrite.scripts.compute_lmdb_spectrum import spectrum_foreground, center_spectrum_image

class RobustSpectrumTest(unittest.TestCase):
    def test_negative_and_zero_foreground_retained(self):
        x=np.full((3,4,4),-1.,dtype=np.float32)
        x[:,1,1]=[-2.,0.,1.]
        mask=spectrum_foreground(x,'robust_iqr_background',1e-6)
        self.assertEqual(mask.sum(),1)
        self.assertTrue(mask[1,1])
        self.assertFalse(spectrum_foreground(np.full_like(x,-1),'robust_iqr_background',1e-6).any())

    def test_centering_zeros_background(self):
        x=np.array([[-1,-1],[-3,3]],dtype=np.float32);mask=np.array([[False,False],[True,True]])
        out=center_spectrum_image(x,mask,'robust_iqr_background')
        np.testing.assert_array_equal(out,[[0,0],[-3,3]])
        np.testing.assert_array_equal(center_spectrum_image(x,mask,'union_nonzero'),x)

    def test_legacy_mask_unchanged(self):
        x=np.random.default_rng(1).normal(size=(3,4,4)).astype(np.float32)
        np.testing.assert_array_equal(spectrum_foreground(x,'union_nonzero',1e-6),np.abs(x).sum(axis=0)>1e-6)
