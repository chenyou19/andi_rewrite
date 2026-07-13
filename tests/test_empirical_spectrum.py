from __future__ import annotations

import sys
import tempfile
import unittest
import warnings
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from andi_rewrite.noise.empirical_spectrum import EmpiricalSpectrumNoise  # noqa: E402
from andi_rewrite.noise.factory import build_noise_sampler  # noqa: E402


class EmpiricalSpectrumNoiseTest(unittest.TestCase):
    channels = 2
    height = 16
    width = 16
    eps = 1.0e-8

    @classmethod
    def setUpClass(cls) -> None:
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.stats_path = Path(cls.temp_dir.name) / "synthetic_spectrum.npz"
        cls.amplitude_only_path = Path(cls.temp_dir.name) / "synthetic_amplitude_only.npz"

        bins = 12
        coordinate = np.linspace(0.0, 1.0, bins, dtype=np.float32)
        radial_amplitude = np.stack(
            [
                1.0 / np.power(1.0 + 8.0 * coordinate, 1.3),
                1.0 / np.power(1.0 + 13.0 * coordinate, 1.1),
            ]
        ).astype(np.float32)
        radial_power = np.square(radial_amplitude, dtype=np.float32)

        unshifted_amplitude = cls._expand_profile(radial_amplitude)
        mean_amplitude = np.fft.fftshift(unshifted_amplitude, axes=(-2, -1)).copy()
        mean_power = np.square(mean_amplitude, dtype=np.float32)
        payload = {
            "radial_amplitude": radial_amplitude,
            "radial_power": radial_power,
            "mean_amplitude": mean_amplitude,
            "mean_power": mean_power,
            "channels": np.asarray(cls.channels, dtype=np.int64),
            "height": np.asarray(cls.height, dtype=np.int64),
            "width": np.asarray(cls.width, dtype=np.int64),
        }
        np.savez_compressed(cls.stats_path, **payload)
        np.savez_compressed(
            cls.amplitude_only_path,
            **{key: value for key, value in payload.items() if key not in {"radial_power", "mean_power"}},
        )
        cls.radial_power = radial_power

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp_dir.cleanup()

    @classmethod
    def _expand_profile(cls, profile: np.ndarray) -> np.ndarray:
        fy = np.fft.fftfreq(cls.height).astype(np.float32)
        fx = np.fft.fftfreq(cls.width).astype(np.float32)
        radius = np.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
        positions = radius / float(radius.max()) * float(profile.shape[1] - 1)
        source = np.arange(profile.shape[1], dtype=np.float32)
        result = np.empty((profile.shape[0], cls.height, cls.width), dtype=np.float32)
        for channel in range(profile.shape[0]):
            result[channel] = np.interp(positions.ravel(), source, profile[channel]).reshape(
                cls.height, cls.width
            )
        return result

    def _sampler(self, **overrides: object) -> EmpiricalSpectrumNoise:
        config: dict[str, object] = {
            "stats_path": self.stats_path,
            "mode": "radial",
            "generation_method": "fixed_magnitude",
            "strength": 1.0,
            "normalize": True,
            "per_channel": True,
            "eps": self.eps,
        }
        config.update(overrides)
        return EmpiricalSpectrumNoise(**config)

    @staticmethod
    def _fft_cv(samples: torch.Tensor) -> float:
        magnitude = torch.abs(torch.fft.fft2(samples, dim=(-2, -1)))
        mean = magnitude.mean(dim=0)
        cv = magnitude.std(dim=0, unbiased=False) / mean.clamp_min(1.0e-8)
        valid = mean > mean.amax(dim=(-2, -1), keepdim=True) * 1.0e-5
        return float(torch.median(cv[valid]))

    @staticmethod
    def _fft_cosine(samples: torch.Tensor) -> float:
        magnitude = torch.abs(torch.fft.fft2(samples, dim=(-2, -1))).flatten(1)
        normalized = torch.nn.functional.normalize(magnitude, dim=1)
        return float(torch.median(torch.sum(normalized[0::2] * normalized[1::2], dim=1)))

    @classmethod
    def _radial_psd(cls, samples: torch.Tensor, bins: int = 8) -> torch.Tensor:
        power = torch.abs(torch.fft.fft2(samples, dim=(-2, -1))).square()
        fy = torch.fft.fftfreq(cls.height)
        fx = torch.fft.fftfreq(cls.width)
        radius = torch.sqrt(fy[:, None].square() + fx[None, :].square())
        indices = torch.clamp((radius / radius.max() * bins).long(), max=bins - 1)
        curves = []
        for index in range(bins):
            curves.append(power[..., indices == index].mean(dim=-1))
        return torch.stack(curves, dim=-1)

    def test_factory_default_is_backward_compatible_fixed_magnitude(self) -> None:
        sampler = build_noise_sampler({"type": "empirical_spectrum", "stats_path": str(self.stats_path)})
        self.assertIsInstance(sampler, EmpiricalSpectrumNoise)
        self.assertEqual(sampler.generation_method, "fixed_magnitude")
        description = sampler.describe()
        self.assertEqual(description["loaded_statistic_type"], "amplitude")
        self.assertEqual(description["loaded_statistic_key"], "radial_amplitude")
        self.assertEqual(description["filter_normalization"], "not_applicable")

    def test_generation_method_aliases_are_canonicalized(self) -> None:
        for alias in ("fixed_magnitude", "fixed", "phase_randomized"):
            self.assertEqual(self._sampler(generation_method=alias).describe()["generation_method"], "fixed_magnitude")
        for alias in ("filtered_gaussian", "gaussian_filter", "legacy_filter"):
            self.assertEqual(
                self._sampler(generation_method=alias).describe()["generation_method"],
                "filtered_gaussian",
            )

    def test_fixed_magnitude_matches_pre_change_formula(self) -> None:
        sampler = self._sampler()
        shape = (5, self.channels, self.height, self.width)
        torch.manual_seed(73)
        actual = sampler.sample(shape, device="cpu", dtype=torch.float32)

        torch.manual_seed(73)
        white = torch.randn(shape, dtype=torch.float32)
        spectrum = torch.fft.fft2(white, dim=(-2, -1))
        white_amp = torch.abs(spectrum)
        phase = spectrum / (white_amp + self.eps)
        target = sampler.target_amp.unsqueeze(0)
        shaped_amp = (white_amp + self.eps).pow(0.0) * (target + self.eps).pow(1.0)
        expected = torch.fft.ifft2(phase * shaped_amp, dim=(-2, -1)).real
        mean = expected.mean(dim=(-2, -1), keepdim=True)
        std = expected.std(dim=(-2, -1), keepdim=True, unbiased=False)
        expected = (expected - mean) / (std + self.eps)

        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    def test_filtered_strength_zero_is_normalized_white_gaussian(self) -> None:
        sampler = self._sampler(generation_method="filtered_gaussian", strength=0.0)
        shape = (7, self.channels, self.height, self.width)
        torch.manual_seed(101)
        actual = sampler.sample(shape, device="cpu", dtype=torch.float32)
        torch.manual_seed(101)
        white = torch.randn(shape, dtype=torch.float32)
        expected = (white - white.mean(dim=(-2, -1), keepdim=True)) / (
            white.std(dim=(-2, -1), keepdim=True, unbiased=False) + self.eps
        )
        torch.testing.assert_close(actual, expected, rtol=1.0e-5, atol=1.0e-5)

    def test_filtered_strength_half_and_one_use_filter_exponent(self) -> None:
        shape = (3, self.channels, self.height, self.width)
        for strength in (0.5, 1.0):
            sampler = self._sampler(generation_method="filtered_gaussian", strength=strength)
            torch.manual_seed(151)
            actual = sampler.sample(shape, "cpu", torch.float32)

            torch.manual_seed(151)
            white = torch.randn(shape, dtype=torch.float32)
            white_fft = torch.fft.rfft2(white, dim=(-2, -1))
            effective_filter = sampler.filter_amp_rfft.clamp_min(self.eps).pow(strength)
            expected = torch.fft.irfft2(
                white_fft * effective_filter.unsqueeze(0),
                s=(self.height, self.width),
                dim=(-2, -1),
            )
            expected = (expected - expected.mean(dim=(-2, -1), keepdim=True)) / (
                expected.std(dim=(-2, -1), keepdim=True, unbiased=False) + self.eps
            )
            torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    def test_filtered_has_sample_varying_fft_magnitude_and_radial_psd(self) -> None:
        shape = (256, self.channels, self.height, self.width)
        torch.manual_seed(207)
        fixed = self._sampler().sample(shape, device="cpu", dtype=torch.float32)
        torch.manual_seed(207)
        filtered = self._sampler(generation_method="filtered_gaussian").sample(
            shape, device="cpu", dtype=torch.float32
        )

        fixed_cv = self._fft_cv(fixed)
        filtered_cv = self._fft_cv(filtered)
        self.assertLess(fixed_cv, 1.0e-4)
        self.assertGreater(filtered_cv, 1.0e-2)
        self.assertGreater(filtered_cv, fixed_cv * 100.0)

        fixed_cosine = self._fft_cosine(fixed[:64])
        filtered_cosine = self._fft_cosine(filtered[:64])
        self.assertGreater(fixed_cosine, 0.999)
        self.assertLess(filtered_cosine, fixed_cosine)

        fixed_radial = self._radial_psd(fixed)
        filtered_radial = self._radial_psd(filtered)
        fixed_radial_cv = torch.median(
            fixed_radial.std(dim=0, unbiased=False) / fixed_radial.mean(dim=0).clamp_min(self.eps)
        )
        filtered_radial_cv = torch.median(
            filtered_radial.std(dim=0, unbiased=False)
            / filtered_radial.mean(dim=0).clamp_min(self.eps)
        )
        self.assertGreater(float(filtered_radial_cv), float(fixed_radial_cv))

    def test_filtered_mean_power_tracks_target_shape(self) -> None:
        sampler = self._sampler(generation_method="filtered_gaussian")
        torch.manual_seed(308)
        samples = sampler.sample((512, self.channels, self.height, self.width), "cpu", torch.float32)
        measured = torch.abs(torch.fft.rfft2(samples, dim=(-2, -1))).square().mean(dim=0)
        target = sampler.filter_amp_rfft.square()
        # Spatial mean removal forces DC to zero; compare all remaining bins.
        mask = torch.ones_like(target, dtype=torch.bool)
        mask[:, 0, 0] = False
        correlation = np.corrcoef(measured[mask].numpy(), target[mask].numpy())[0, 1]
        self.assertGreater(float(correlation), 0.98)

    def test_normalize_true_is_per_sample_per_channel(self) -> None:
        values = self._sampler(generation_method="filtered_gaussian").sample(
            (11, self.channels, self.height, self.width), "cpu", torch.float32
        )
        torch.testing.assert_close(
            values.mean(dim=(-2, -1)), torch.zeros((11, self.channels)), rtol=0.0, atol=1.0e-6
        )
        torch.testing.assert_close(
            values.std(dim=(-2, -1), unbiased=False),
            torch.ones((11, self.channels)),
            rtol=1.0e-5,
            atol=1.0e-5,
        )

    def test_normalize_false_does_not_force_unit_std(self) -> None:
        values = self._sampler(generation_method="filtered_gaussian", normalize=False).sample(
            (6, self.channels, self.height, self.width), "cpu", torch.float32
        )
        standard_deviations = values.std(dim=(-2, -1), unbiased=False)
        self.assertFalse(torch.allclose(standard_deviations, torch.ones_like(standard_deviations), atol=0.02))

    def test_radial_and_full2d_modes(self) -> None:
        for mode in ("radial", "2d", "full2d"):
            sampler = self._sampler(generation_method="filtered_gaussian", mode=mode)
            values = sampler.sample((2, self.channels, self.height, self.width), "cpu", torch.float32)
            self.assertEqual(tuple(values.shape), (2, self.channels, self.height, self.width))
            self.assertTrue(bool(torch.isfinite(values).all()))

    def test_per_channel_and_shared_power_filters(self) -> None:
        per_channel = self._sampler(generation_method="filtered_gaussian", per_channel=True)
        shared = self._sampler(generation_method="filtered_gaussian", per_channel=False)
        self.assertEqual(per_channel.filter_amp_rfft.shape[0], self.channels)
        self.assertEqual(shared.filter_amp_rfft.shape[0], 1)
        values = shared.sample((3, 4, self.height, self.width), "cpu", torch.float32)
        self.assertEqual(tuple(values.shape), (3, 4, self.height, self.width))

        expanded_power = shared._expand_radial_profile(self.radial_power, self.height, self.width)
        expected = np.sqrt(np.maximum(expanded_power.mean(axis=0, keepdims=True), self.eps))
        expected /= np.sqrt(np.mean(expected**2, axis=(-2, -1), keepdims=True))
        torch.testing.assert_close(
            shared.filter_amp_rfft,
            torch.from_numpy(expected[..., : self.width // 2 + 1]).to(torch.float32),
            rtol=1.0e-6,
            atol=1.0e-6,
        )

    def test_amplitude_fallback_warns_and_is_described(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            sampler = self._sampler(
                stats_path=self.amplitude_only_path,
                generation_method="filtered_gaussian",
            )
        self.assertTrue(any("falling back" in str(item.message) for item in caught))
        description = sampler.describe()
        self.assertEqual(description["loaded_statistic_type"], "amplitude")
        self.assertEqual(description["loaded_statistic_key"], "radial_amplitude")
        self.assertTrue(description["used_statistic_fallback"])
        self.assertEqual(description["filter_normalization"], "rms")

    def test_invalid_method_and_shape_mismatches_are_clear(self) -> None:
        with self.assertRaisesRegex(ValueError, "fixed_magnitude, filtered_gaussian"):
            self._sampler(generation_method="not-a-method")
        sampler = self._sampler()
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            sampler.sample((1, self.channels, 8, 8), "cpu", torch.float32)
        with self.assertRaisesRegex(ValueError, "channel mismatch"):
            sampler.sample((1, 3, self.height, self.width), "cpu", torch.float32)

    def test_float32_half_and_bfloat16_are_finite_and_return_requested_dtype(self) -> None:
        sampler = self._sampler(generation_method="filtered_gaussian")
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            values = sampler.sample((2, self.channels, self.height, self.width), "cpu", dtype)
            self.assertEqual(values.dtype, dtype)
            self.assertTrue(bool(torch.isfinite(values).all()))

    def test_seed_is_deterministic(self) -> None:
        sampler = self._sampler(generation_method="filtered_gaussian", strength=0.5)
        torch.manual_seed(911)
        first = sampler.sample((4, self.channels, self.height, self.width), "cpu", torch.float32)
        torch.manual_seed(911)
        second = sampler.sample((4, self.channels, self.height, self.width), "cpu", torch.float32)
        torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)

    def test_ab_configs_only_differ_in_method_identity(self) -> None:
        root = Path(__file__).resolve().parents[1]
        with (root / "configs" / "train_empirical_filtered_gaussian40_lmdb_zbalanced.yaml").open(
            encoding="utf-8"
        ) as handle:
            filtered = yaml.safe_load(handle)
        with (root / "configs" / "train_empirical_fixed_magnitude40_lmdb_zbalanced.yaml").open(
            encoding="utf-8"
        ) as handle:
            fixed = yaml.safe_load(handle)

        self.assertEqual(
            filtered["noise"]["schedule"]["sampler"]["generation_method"],
            "filtered_gaussian",
        )
        self.assertEqual(
            fixed["noise"]["schedule"]["sampler"]["generation_method"],
            "fixed_magnitude",
        )
        self.assertEqual(filtered["training"]["epochs"], 40)
        self.assertFalse(filtered["training"]["eval_after_fit"]["enabled"])
        self.assertEqual(filtered["training"]["checkpoint"]["start_epoch"], 1)
        self.assertEqual(filtered["training"]["checkpoint"]["save_every_epochs"], 2)

        filtered_comparable = deepcopy(filtered)
        fixed_comparable = deepcopy(fixed)
        del filtered_comparable["noise"]["schedule"]["sampler"]["generation_method"]
        del fixed_comparable["noise"]["schedule"]["sampler"]["generation_method"]
        # Identifiers differ so A/B outputs cannot overwrite each other.
        del filtered_comparable["experiment"]["name"]
        del fixed_comparable["experiment"]["name"]
        del filtered_comparable["training"]["run_name"]
        del fixed_comparable["training"]["run_name"]
        self.assertEqual(filtered_comparable, fixed_comparable)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_cuda_smoke(self) -> None:
        sampler = self._sampler(generation_method="filtered_gaussian")
        values = sampler.sample((2, self.channels, self.height, self.width), "cuda", torch.float32)
        self.assertEqual(values.device.type, "cuda")
        self.assertTrue(bool(torch.isfinite(values).all()))


if __name__ == "__main__":
    unittest.main()
