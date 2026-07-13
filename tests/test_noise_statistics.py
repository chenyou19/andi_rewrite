from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from andi_rewrite.utils.noise_statistics import (  # noqa: E402
    RawMomentAccumulator,
    RunningMeanVariance,
    build_radial_bin_map,
    channel_correlation_matrices,
    coefficient_of_variation,
    distribution_summary,
    ensure_finite,
    expand_normalized_radial_profile,
    fft_magnitude,
    fixed_magnitude_from_white,
    legacy_filtered_gaussian_from_white,
    normalize_per_sample_channel,
    pairwise_cosine_values,
    radial_psd_from_magnitude,
    valid_frequency_mask,
)


class NoiseStatisticsTest(unittest.TestCase):
    """Synthetic, CPU-only coverage for the standalone diagnostics helpers."""

    num_samples = 64
    channels = 2
    height = 16
    width = 16
    seed = 73
    eps = 1.0e-8

    @classmethod
    def setUpClass(cls) -> None:
        radius = np.arange(12, dtype=np.float64)
        radial_profile = np.stack(
            [
                1.0 / (1.0 + 0.30 * radius),
                1.0 / (1.0 + 0.55 * radius),
            ],
            axis=0,
        )
        target = expand_normalized_radial_profile(
            radial_profile,
            cls.height,
            cls.width,
        )
        target /= target.mean(axis=(-2, -1), keepdims=True)
        cls.target_amplitude = torch.from_numpy(target).to(torch.float32)

        generator = torch.Generator(device="cpu")
        generator.manual_seed(cls.seed)
        cls.white = torch.randn(
            (cls.num_samples, cls.channels, cls.height, cls.width),
            generator=generator,
            dtype=torch.float32,
            device="cpu",
        )
        cls.legacy = legacy_filtered_gaussian_from_white(
            cls.white,
            cls.target_amplitude,
            eps=cls.eps,
        )
        cls.fixed = fixed_magnitude_from_white(
            cls.white,
            cls.target_amplitude,
            eps=cls.eps,
        )
        cls.legacy_magnitude = fft_magnitude(cls.legacy)
        cls.fixed_magnitude = fft_magnitude(cls.fixed)

        cls.legacy_mean, cls.legacy_std = cls._streaming_mean_std(
            cls.legacy_magnitude
        )
        cls.fixed_mean, cls.fixed_std = cls._streaming_mean_std(
            cls.fixed_magnitude
        )
        cls.legacy_cv = coefficient_of_variation(
            cls.legacy_std,
            cls.legacy_mean,
            eps=cls.eps,
        )
        cls.fixed_cv = coefficient_of_variation(
            cls.fixed_std,
            cls.fixed_mean,
            eps=cls.eps,
        )
        cls.legacy_valid = valid_frequency_mask(cls.legacy_mean)
        cls.fixed_valid = valid_frequency_mask(cls.fixed_mean)

        cls.radial_map = build_radial_bin_map(cls.height, cls.width)
        cls.fixed_radial_psd = radial_psd_from_magnitude(
            cls.fixed_magnitude,
            cls.radial_map,
        )

    @staticmethod
    def _streaming_mean_std(values: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        accumulator = RunningMeanVariance()
        for chunk in values.split(13, dim=0):
            accumulator.update(chunk)
        result = accumulator.finalize()
        return np.asarray(result["mean"]), np.asarray(result["std"])

    def test_cv_is_population_std_divided_by_mean(self) -> None:
        std = torch.tensor([2.0, 3.0, 5.0], dtype=torch.float64)
        mean = torch.tensor([4.0, 12.0, 20.0], dtype=torch.float64)

        result = coefficient_of_variation(std, mean, eps=0.0)

        torch.testing.assert_close(
            result,
            torch.tensor([0.5, 0.25, 0.25], dtype=torch.float64),
            rtol=0.0,
            atol=0.0,
        )

    def test_fixed_magnitude_random_phase_has_near_zero_valid_bin_cv(self) -> None:
        self.assertFalse(bool(np.any(self.fixed_valid[:, 0, 0])))
        valid_cv = np.asarray(self.fixed_cv)[np.asarray(self.fixed_valid)]

        self.assertGreater(valid_cv.size, 0)
        self.assertLess(float(np.median(valid_cv)), 1.0e-5)
        self.assertLess(float(np.percentile(valid_cv, 99)), 1.0e-4)

    def test_filtered_gaussian_cv_is_clearly_larger(self) -> None:
        legacy_cv = np.asarray(self.legacy_cv)[np.asarray(self.legacy_valid)]
        fixed_cv = np.asarray(self.fixed_cv)[np.asarray(self.fixed_valid)]
        legacy_median = float(np.median(legacy_cv))
        fixed_median = float(np.median(fixed_cv))

        self.assertGreater(legacy_median, 0.1)
        self.assertGreater(legacy_median, fixed_median * 10.0)

    def test_gaussian_pearson_kurtosis_is_near_three(self) -> None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(1907)
        samples = torch.randn(
            (50_000, 2, 2),
            generator=generator,
            dtype=torch.float64,
        )
        accumulator = RawMomentAccumulator()
        for chunk in samples.split(4096, dim=0):
            accumulator.update(chunk)

        result = accumulator.finalize()
        kurtosis = np.asarray(result["kurtosis"])

        self.assertEqual(int(result["count"]), 50_000)
        self.assertTrue(np.all(np.abs(kurtosis - 3.0) < 0.12), kurtosis)

    def test_pairwise_cosine_excludes_diagonal_and_has_expected_count(self) -> None:
        samples = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]],
            dtype=torch.float64,
        )

        values = pairwise_cosine_values(samples)

        self.assertEqual(values.numel(), 4 * 3 // 2)
        torch.testing.assert_close(
            torch.sort(values).values,
            torch.tensor([-1.0, -1.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64),
            rtol=0.0,
            atol=0.0,
        )
        self.assertFalse(bool(torch.any(values == 1.0)))

    def test_radial_psd_shape_and_values_are_finite(self) -> None:
        expected_shape = (
            self.num_samples,
            self.channels,
            self.radial_map.bins,
        )

        self.assertEqual(tuple(self.fixed_radial_psd.shape), expected_shape)
        self.assertTrue(bool(torch.all(self.fixed_radial_psd >= 0.0)))
        ensure_finite(self.fixed_radial_psd, name="synthetic radial PSD")

    def test_all_relevant_statistics_are_finite(self) -> None:
        moments = RawMomentAccumulator()
        moments.update(self.fixed[:31])
        moments.update(self.fixed[31:])
        moment_result = moments.finalize()
        spatial_correlations = channel_correlation_matrices(self.fixed)
        fft_correlations = channel_correlation_matrices(self.fixed_magnitude)
        cosine_values = pairwise_cosine_values(self.fixed[:16])
        summary = distribution_summary(
            np.asarray(self.fixed_cv)[np.asarray(self.fixed_valid)]
        )

        for name, value in {
            "fixed noise": self.fixed,
            "legacy noise": self.legacy,
            "fixed FFT magnitude": self.fixed_magnitude,
            "legacy FFT magnitude": self.legacy_magnitude,
            "fixed FFT CV": self.fixed_cv,
            "legacy FFT CV": self.legacy_cv,
            "spatial correlations": spatial_correlations,
            "FFT correlations": fft_correlations,
            "sample cosine": cosine_values,
            "radial PSD": self.fixed_radial_psd,
        }.items():
            ensure_finite(value, name=name)
        for key in ("mean", "std", "skewness", "kurtosis"):
            ensure_finite(moment_result[key], name=f"pixel {key}")
        ensure_finite(np.asarray(list(summary.values())), name="distribution summary")

        with self.assertRaisesRegex(ValueError, "contains NaN or Inf"):
            ensure_finite(np.asarray([0.0, np.nan]), name="deliberately invalid")

    def test_same_seed_repeats_exactly(self) -> None:
        def generate(seed: int) -> torch.Tensor:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(seed)
            white = torch.randn(
                (8, self.channels, self.height, self.width),
                generator=generator,
                dtype=torch.float32,
            )
            return fixed_magnitude_from_white(
                white,
                self.target_amplitude,
                eps=self.eps,
            )

        first = generate(self.seed)
        second = generate(self.seed)

        torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)

    def test_cpu_mode_executes_channel_statistics(self) -> None:
        self.assertEqual(self.white.device.type, "cpu")
        self.assertEqual(self.legacy.device.type, "cpu")
        self.assertEqual(self.fixed.device.type, "cpu")

        correlations = channel_correlation_matrices(self.fixed)
        self.assertEqual(
            tuple(correlations.shape),
            (self.num_samples, self.channels, self.channels),
        )
        diagonal = torch.diagonal(correlations, dim1=-2, dim2=-1)
        torch.testing.assert_close(
            diagonal,
            torch.ones_like(diagonal),
            rtol=1.0e-5,
            atol=1.0e-5,
        )

    def test_small_synthetic_smoke_without_mri_stats(self) -> None:
        # This fixture is deliberately the requested [64, 2, 16, 16] shape and
        # is constructed entirely from an artificial radial profile above.
        self.assertEqual(
            tuple(self.fixed.shape),
            (64, 2, 16, 16),
        )
        normalized = normalize_per_sample_channel(self.fixed, eps=self.eps)
        means = normalized.mean(dim=(-2, -1))
        standard_deviations = normalized.std(
            dim=(-2, -1),
            unbiased=False,
        )
        torch.testing.assert_close(
            means,
            torch.zeros_like(means),
            rtol=0.0,
            atol=1.0e-6,
        )
        torch.testing.assert_close(
            standard_deviations,
            torch.ones_like(standard_deviations),
            rtol=1.0e-5,
            atol=1.0e-5,
        )

        fixed_fft_cosine = pairwise_cosine_values(self.fixed_magnitude[:12])
        legacy_fft_cosine = pairwise_cosine_values(self.legacy_magnitude[:12])
        self.assertGreater(float(torch.median(fixed_fft_cosine)), 0.9999)
        self.assertLess(
            float(torch.median(legacy_fft_cosine)),
            float(torch.median(fixed_fft_cosine)),
        )


if __name__ == "__main__":
    unittest.main()
