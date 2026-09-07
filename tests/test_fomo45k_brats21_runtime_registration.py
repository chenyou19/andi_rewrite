from __future__ import annotations

import unittest

from andi_rewrite.data.datasets.factory import DATASET_BUILDERS
from andi_rewrite.scripts.train_fomo45k_brats21 import register_fomo45k_brats21


class FOMO45KBraTS21RuntimeRegistrationTest(unittest.TestCase):
    def test_optional_builders_register_without_replacing_existing_aliases(self) -> None:
        before = dict(DATASET_BUILDERS)
        register_fomo45k_brats21()
        try:
            self.assertIn("fomo45k_brats21", DATASET_BUILDERS)
            self.assertIn("fomo45k_brats21_slices", DATASET_BUILDERS)
            self.assertIn("fomo45k_brats21_volume", DATASET_BUILDERS)
            for name, builder in before.items():
                self.assertIs(DATASET_BUILDERS[name], builder)
        finally:
            DATASET_BUILDERS.clear()
            DATASET_BUILDERS.update(before)


if __name__ == "__main__":
    unittest.main()
