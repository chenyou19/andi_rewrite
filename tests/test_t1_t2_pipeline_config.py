from __future__ import annotations

import unittest
from pathlib import Path

import yaml


class T1T2PipelineConfigTest(unittest.TestCase):
    def test_train_and_eval_configs_are_matched(self) -> None:
        root = Path(__file__).resolve().parents[1]
        train_path = root / "configs" / "train_brats_t1_t2_empirical_spectrum20_eval50.yaml"
        eval_path = root / "configs" / "eval_brats21_50_t1_t2_empirical_spectrum20.yaml"
        with train_path.open(encoding="utf-8") as handle:
            train = yaml.safe_load(handle)
        with eval_path.open(encoding="utf-8") as handle:
            evaluation = yaml.safe_load(handle)

        self.assertEqual(train["data"]["path"], "C:/ML/data/BraTS_2021_healthy_lmdb")
        self.assertEqual(train["data"]["channel_indices"], [1, 3])
        self.assertEqual(train["data"]["channels"], 2)
        self.assertEqual(train["model"]["in_channels"], 2)
        self.assertEqual(train["model"]["out_channels"], 2)
        self.assertEqual(train["training"]["epochs"], 20)
        self.assertFalse(train["validation"]["enabled"])
        self.assertFalse(train["training"]["samples"]["enabled"])
        self.assertTrue(train["training"]["checkpoint"]["save_last"])
        self.assertEqual(train["training"]["checkpoint"]["start_epoch"], 19)
        self.assertEqual(train["training"]["checkpoint"]["save_every_epochs"], 1)
        self.assertTrue(train["training"]["eval_after_fit"]["enabled"])
        self.assertEqual(
            train["training"]["eval_after_fit"]["config"],
            str(eval_path.relative_to(root)).replace("\\", "/"),
        )

        self.assertEqual(
            evaluation["data"]["path_to_csv"], "splits/BraTS21/scans_test_50.csv"
        )
        self.assertEqual(evaluation["data"]["modalities"], ["t1", "t2"])
        self.assertEqual(evaluation["data"]["channels"], 2)
        self.assertEqual(evaluation["model"]["in_channels"], 2)
        self.assertEqual(evaluation["model"]["out_channels"], 2)
        self.assertTrue(evaluation["model"]["use_ema"])
        self.assertEqual(
            evaluation["model"]["checkpoint"],
            "outputs/runs/brats_t1_t2_empirical_spectrum20_eval50/epoch_0019.pt",
        )
        self.assertEqual(
            train["noise"]["schedule"]["sampler"]["stats_path"],
            evaluation["noise"]["schedule"]["sampler"]["stats_path"],
        )
        self.assertEqual(train["noise"]["schedule"]["sampler"]["channel_indices"], [1, 3])
        self.assertEqual(
            evaluation["noise"]["schedule"]["sampler"]["channel_indices"], [1, 3]
        )
        self.assertFalse(evaluation["prediction_output"]["enabled"])


if __name__ == "__main__":
    unittest.main()
