from __future__ import annotations

from copy import deepcopy
from io import BytesIO
from pathlib import Path
import sys
import unittest

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from andi_rewrite.diffusion import DDPMDiffusion, DDPMScheduler
from andi_rewrite.engine import EMA, Trainer
from andi_rewrite.models import build_model
from andi_rewrite.models.aer import AERConfig, AERFeatureRefiner2D, LatentState, load_pretrained_backbone
from andi_rewrite.models.aer_energy import downsample, downsample_adjoint
from andi_rewrite.noise import build_noise_plan


def setUpModule() -> None:
    global original_threads
    original_threads = torch.get_num_threads()
    torch.set_num_threads(2)


def tearDownModule() -> None:
    torch.set_num_threads(original_threads)


def config(kind: str, **aer) -> dict:
    result = {
        "type": kind, "in_channels": 1, "out_channels": 1, "image_size": 16,
        "time_dim": 16, "base_channels": 8, "channel_mults": [1, 2], "num_blocks": 1,
    }
    if aer:
        result["aer"] = {"latent_channels": 8, **aer}
    return result


def sample(kind: str) -> tuple[torch.Tensor, torch.Tensor]:
    shape = (16, 16) if kind == "andi_unet" else (13, 17)
    return torch.randn((2, 1, *shape)), torch.tensor([2, 11])


class ScriptedProposer(nn.Module):
    def __init__(self, *multipliers: float):
        super().__init__()
        self.multipliers = nn.Parameter(torch.tensor(multipliers, dtype=torch.float32))

    def forward(self, state, context, reference_delta):
        return reference_delta.scale(self.multipliers)


class Wrapper(nn.Module):
    """Observe the forward kwargs accepted by a module wrapper."""

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        self.forward_kwargs = []

    def forward(self, *args, **kwargs):
        self.forward_kwargs.append(kwargs)
        return self.module(*args, **kwargs)


class AERTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(73)

    def test_disabled_rng_keys_output_and_k0(self) -> None:
        for kind in ("andi_unet", "convnext_unet"):
            with self.subTest(kind=kind):
                torch.manual_seed(17)
                baseline = build_model(config(kind), "cpu").eval()
                baseline_rng = torch.get_rng_state()
                torch.manual_seed(17)
                disabled = build_model(config(kind, enabled=False), "cpu").eval()
                self.assertTrue(torch.equal(torch.get_rng_state(), baseline_rng))
                self.assertIsNone(disabled.aer)
                self.assertEqual(list(disabled.state_dict()), list(baseline.state_dict()))
                for key, value in baseline.state_dict().items():
                    self.assertTrue(torch.equal(value, disabled.state_dict()[key]), key)
                disabled.load_state_dict(baseline.state_dict(), strict=True)
                enabled_zero = build_model(config(kind, enabled=True, steps=0), "cpu").eval()
                load_pretrained_backbone(enabled_zero, baseline.state_dict())
                x, t = sample(kind)
                with torch.inference_mode():
                    expected = baseline(x, t)
                    self.assertTrue(torch.equal(disabled(x, t), expected))
                    self.assertTrue(torch.equal(enabled_zero(x, t), expected))

    def test_factory_validates_optional_config(self) -> None:
        for aer, pattern in (
            (False, "mapping"),
            ({"enabled": "false"}, "boolean"),
            ({"steps": -1}, "nonnegative integer"),
            ({"latent_channels": 5, "attention_heads": 2}, "divisible"),
            ({"enable": True}, "unknown"),
            ({"aux_loss_weight": float("inf")}, "finite"),
        ):
            with self.subTest(aer=aer):
                with self.assertRaisesRegex(ValueError, pattern):
                    build_model({**config("convnext_unet"), "aer": aer}, "cpu")

    def test_both_denoisers_forward_and_backward(self) -> None:
        for kind in ("andi_unet", "convnext_unet"):
            with self.subTest(kind=kind):
                model = build_model(config(kind, enabled=True, steps=2), "cpu")
                x, t = sample(kind)
                prediction, aux = model(x, t, return_aer_aux=True)
                self.assertEqual(prediction.shape, x.shape)
                self.assertTrue(torch.isfinite(prediction).all())
                self.assertIsNotNone(aux)
                (prediction.square().mean() + 0.1 * aux).backward()
                for name, parameter in model.named_parameters():
                    if parameter.grad is not None:
                        self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                self.assertGreater(model.aer.output_adapter.weight.grad.norm().item(), 0)
                self.assertGreater(model.aer.proposer.coarse_reads_tokens.in_proj_weight.grad.norm().item(), 0)
                with torch.inference_mode():
                    result = model.eval()(x, t)
                    self.assertIsInstance(result, torch.Tensor)
                    self.assertTrue(torch.isfinite(result).all())

    def test_odd_features_and_per_sample_gate(self) -> None:
        refiner = AERFeatureRefiner2D(3).eval()
        refiner.proposer = ScriptedProposer(1, -1, float("nan"))
        feature = torch.randn((3, 3, 3, 5))
        with torch.inference_mode():
            ctx = refiner.encode(feature)
            self.assertEqual(ctx.energy.anchor.fine.shape[-2:], (4, 6))
            self.assertEqual(ctx.energy.anchor.coarse.shape[-2:], (2, 3))
            self.assertTrue(torch.equal(ctx.coarse, downsample(ctx.energy.anchor.fine)))
            result = refiner(feature, steps=1)
            reference = refiner(feature, mode="reference", steps=1)
            bypass = refiner(feature, bypass=True)
            zero = refiner(feature, steps=0)
        self.assertEqual(result.feature.shape, feature.shape)
        self.assertEqual(result.feature.dtype, feature.dtype)
        self.assertTrue(torch.isfinite(result.feature).all())
        self.assertEqual(result.trace.accepted.tolist(), [[True, False, False]])
        self.assertTrue(torch.isnan(result.trace.candidate_energy[0, 2]))
        self.assertFalse(result.trace.energy.requires_grad)
        self.assertTrue(torch.equal(result.feature, reference.feature))
        self.assertIs(bypass.feature, feature)
        self.assertIsNone(bypass.trace)
        self.assertIs(zero.feature, feature)
        with self.assertRaises(FloatingPointError):
            refiner.train()(feature, steps=1)

    def test_energy_adjoint_gradient_and_hessian(self) -> None:
        refiner = AERFeatureRefiner2D(2, AERConfig(latent_channels=2, num_tokens=2)).double()
        with torch.no_grad():
            energy = refiner.encode(torch.randn((1, 2, 3, 1), dtype=torch.float64)).energy
        state = LatentState(*(torch.randn_like(part).requires_grad_() for part in energy.anchor))
        fine, coarse, _ = state
        torch.testing.assert_close((downsample(fine) * coarse).sum(), (fine * downsample_adjoint(coarse)).sum())
        expected = torch.autograd.grad(energy.energy(state).sum(), state)
        for actual, target in zip(energy.gradient(state), expected):
            torch.testing.assert_close(actual, target, atol=1e-12, rtol=1e-12)

        def pack(value):
            return torch.cat([part.flatten() for part in value])

        def unpack(vector):
            pieces = vector.split([part.numel() for part in state])
            return LatentState(*(piece.reshape_as(part) for piece, part in zip(pieces, state)))

        vector = pack(state)
        hessian = torch.autograd.functional.hessian(lambda v: energy.energy(unpack(v)).sum(), vector)
        eigenvalues = torch.linalg.eigvalsh(hessian)
        self.assertGreaterEqual(eigenvalues[0].item(), energy.strong_convexity - 1e-12)
        self.assertLessEqual(eigenvalues[-1].item(), energy.lipschitz.item() + 1e-12)
        rhs = pack(energy.precision) * pack(energy.anchor)
        pcg = list(energy.pcg_iterates(energy.anchor, vector.numel()))[-1]
        torch.testing.assert_close(pack(pcg), torch.linalg.solve(hessian, rhs), atol=1e-10, rtol=1e-10)

    def test_frozen_backbone_still_passes_task_gradient(self) -> None:
        model = build_model(config("convnext_unet", enabled=True, freeze_backbone=True, mode="reference"), "cpu")
        x, t = sample("convnext_unet")
        model.train()
        self.assertTrue(model.aer.training)
        self.assertTrue(all(not child.training for child in model.children() if child is not model.aer))
        frozen = {name: p.detach().clone() for name, p in model.named_parameters() if not name.startswith("aer.")}
        optimizer = torch.optim.AdamW(model.aer.parameters(), lr=1e-3)
        model(x, t).square().mean().backward()
        self.assertGreater(model.aer.input_adapter.weight.grad.norm().item(), 0)
        self.assertGreater(model.aer.output_adapter.weight.grad.norm().item(), 0)
        optimizer.step()
        for name, parameter in model.named_parameters():
            if name in frozen:
                self.assertFalse(parameter.requires_grad)
                self.assertIsNone(parameter.grad)
                self.assertTrue(torch.equal(parameter, frozen[name]))

    def test_rejected_proposal_and_wrapped_trainer_auxiliary(self) -> None:
        model = build_model(config("convnext_unet", enabled=True, steps=2, aux_loss_weight=0.2), "cpu")
        model.aer.proposer = ScriptedProposer(-1, -1)
        wrapped = Wrapper(model)
        trainer = Trainer(
            model=wrapped,
            diffusion=DDPMDiffusion(DDPMScheduler(steps=16)),
            noise_plan=build_noise_plan({"schedule": {"type": "static", "sampler": {"type": "gaussian"}}}),
            config={"ema": False, "scheduler": {"type": "none"}, "normalize_input": False},
            device="cpu",
        )
        x, t = sample("convnext_unet")
        prediction, aux = wrapped(x, t, return_aer_aux=True)
        task_grad, = torch.autograd.grad(prediction.square().mean(), model.aer.proposer.multipliers, retain_graph=True)
        self.assertTrue(torch.equal(task_grad, torch.zeros_like(task_grad)))
        aux_grad, = torch.autograd.grad(aux, model.aer.proposer.multipliers)
        self.assertTrue((aux_grad.abs() > 0).all())

        torch.manual_seed(18)
        total, train = trainer._forward_loss(x, epoch=0)
        self.assertEqual(wrapped.forward_kwargs[-1], {"return_aer_aux": True})
        self.assertGreater(train["auxiliary_loss"], 0)
        self.assertAlmostEqual(total.item(), train["noise_loss"] + 0.2 * train["auxiliary_loss"], places=6)
        torch.manual_seed(18)
        with torch.no_grad():
            val_loss, validation = trainer._forward_loss(x, epoch=0, model=wrapped.eval())
        self.assertEqual(wrapped.forward_kwargs[-1], {})
        self.assertEqual(val_loss.item(), validation["noise_loss"])
        self.assertEqual(validation["auxiliary_loss"], 0)
        self.assertAlmostEqual(validation["noise_loss"], train["noise_loss"], places=6)

    def test_checkpoint_ema_and_strict_warm_start(self) -> None:
        enabled = config("convnext_unet", enabled=True, steps=2)
        model = build_model(enabled, "cpu")
        baseline = build_model(config("convnext_unet"), "cpu")
        with self.assertRaises(RuntimeError):
            model.load_state_dict(baseline.state_dict(), strict=True)
        with self.assertRaisesRegex(RuntimeError, "unexpected"):
            load_pretrained_backbone(model, {**baseline.state_dict(), "bad": torch.zeros(1)})
        bad = dict(baseline.state_dict())
        bad["stem.weight"] = torch.zeros(1)
        with self.assertRaisesRegex(RuntimeError, "bad_shapes"):
            load_pretrained_backbone(model, bad)
        adapter_before = model.aer.input_adapter.weight.detach().clone()
        load_pretrained_backbone(model, baseline.state_dict())
        self.assertTrue(torch.equal(adapter_before, model.aer.input_adapter.weight))
        ema_model = deepcopy(model).eval().requires_grad_(False)
        ema = EMA(decay=0.9, step_start=0)
        with torch.no_grad():
            model.aer.output_adapter.weight.add_(1)
        ema.step_ema(ema_model, model)
        buffer = BytesIO()
        torch.save({"model": model.state_dict(), "ema_model": ema_model.state_dict(), "ema": ema.state_dict()}, buffer)
        buffer.seek(0)
        checkpoint = torch.load(buffer, weights_only=True)
        x, t = sample("convnext_unet")
        for key, original in (("model", model.eval()), ("ema_model", ema_model)):
            restored = build_model(enabled, "cpu").eval()
            restored.load_state_dict(checkpoint[key], strict=True)
            with torch.inference_mode():
                self.assertTrue(torch.equal(original(x, t), restored(x, t)))
        self.assertEqual(checkpoint["ema"]["step"], 1)


if __name__ == "__main__":
    unittest.main()
