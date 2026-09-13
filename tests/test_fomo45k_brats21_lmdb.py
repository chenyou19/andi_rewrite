from __future__ import annotations

import unittest

import numpy as np
import torch

from andi_rewrite.data.datasets.imaging import normalize_volume
from andi_rewrite.data.fomo45k.brats21_lmdb import (
    normalize_like_brats21,
    split_participants,
)


class FOMO45KBraTS21LMDBTest(unittest.TestCase):
    def test_split_is_deterministic_and_has_no_participant_leakage(self) -> None:
        participants = [f"sub_{index:03d}" for index in range(240)]
        train_a, val_a = split_participants(participants, seed=73)
        train_b, val_b = split_participants(participants, seed=73)
        self.assertEqual(train_a, train_b)
        self.assertEqual(val_a, val_b)
        self.assertEqual(len(train_a), 216)
        self.assertEqual(len(val_a), 24)
        self.assertFalse(train_a.intersection(val_a))

    def test_normalization_is_exact_brats21_p99_without_clipping(self) -> None:
        values = torch.zeros(3, 2, 2, 3, dtype=torch.float32)
        values[0].reshape(-1)[:] = torch.arange(12, dtype=torch.float32)
        values[1].reshape(-1)[:] = torch.arange(12, dtype=torch.float32) * 2
        values[2].reshape(-1)[:] = torch.arange(12, dtype=torch.float32) * 3
        expected = normalize_volume(values.clone())
        actual = normalize_like_brats21(values.clone())
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
        self.assertGreater(float(actual.max()), 1.0)
        np.testing.assert_array_equal(actual[:, 0, 0, 0].numpy(), 0.0)


if __name__ == "__main__":
    unittest.main()
