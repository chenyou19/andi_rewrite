"""Optional feature-only AER for timestep-conditioned 2D bottlenecks."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from math import isfinite, sqrt
from typing import Any, Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .aer_energy import EnergyTerms, LatentState, downsample

Mode = Literal["guarded", "reference", "unguarded", "pcg"]
MODES = ("guarded", "reference", "unguarded", "pcg")


def _validate_steps(steps: int) -> None:
    if type(steps) is not int or steps < 0:
        raise ValueError("model.aer.steps must be a nonnegative integer")


def _validate_nonnegative(value: float, name: str) -> None:
    if type(value) not in (int, float) or not isfinite(value) or value < 0:
        raise ValueError(f"model.aer.{name} must be finite and nonnegative")


@dataclass(frozen=True)
class AERConfig:
    enabled: bool = False
    steps: int = 3
    latent_channels: int = 16
    num_tokens: int = 4
    attention_heads: int = 2
    mode: Mode = "guarded"
    aux_loss_weight: float = 0.1
    residual_scale: float = 0.05
    residual_tolerance: float | None = None
    freeze_backbone: bool = False
    w_min: float = 0.8
    w_max: float = 1.2
    lambda_m: float = 0.7
    lambda_s: float = 0.5
    lambda_e: float = 0.4

    def __post_init__(self) -> None:
        for name in ("enabled", "freeze_backbone"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"model.aer.{name} must be a boolean")
        _validate_steps(self.steps)
        for name in ("latent_channels", "num_tokens", "attention_heads"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"model.aer.{name} must be a positive integer")
        if self.latent_channels % self.attention_heads:
            raise ValueError("model.aer.latent_channels must be divisible by attention_heads")
        if self.mode not in MODES:
            raise ValueError("model.aer.mode must be guarded, reference, unguarded or pcg")
        for name in ("aux_loss_weight", "residual_scale", "lambda_m", "lambda_s", "lambda_e", "w_min", "w_max"):
            _validate_nonnegative(getattr(self, name), name)
        if self.residual_tolerance is not None:
            _validate_nonnegative(self.residual_tolerance, "residual_tolerance")
        fp32 = torch.finfo(torch.float32)
        if not fp32.tiny <= self.w_min <= self.w_max <= fp32.max:
            raise ValueError("model.aer requires positive float32 precision bounds with w_min <= w_max")
        if self.freeze_backbone and not self.enabled:
            raise ValueError("model.aer.freeze_backbone requires enabled=true")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> AERConfig:
        if values is None:
            return cls()
        if not isinstance(values, Mapping):
            raise ValueError("model.aer must be a mapping")
        unknown = set(values) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"unknown model.aer keys: {', '.join(sorted(map(str, unknown)))}")
        return cls(**values)


@dataclass(frozen=True)
class RefinementContext:
    fine: Tensor
    coarse: Tensor
    energy: EnergyTerms
    feature_size: tuple[int, int]


@dataclass(frozen=True)
class AERTrace:
    energy: Tensor
    residual: Tensor
    reference_energy: Tensor
    candidate_energy: Tensor
    accepted: Tensor
    active: Tensor
    iterations: Tensor
    converged: Tensor

    @property
    def budget_exhausted(self) -> Tensor:
        return ~self.converged


@dataclass(frozen=True)
class AERResult:
    feature: Tensor
    trace: AERTrace | None
    proposal_loss: Tensor | None


class CoefficientHead(nn.Module):
    def __init__(self, config: AERConfig):
        super().__init__()
        self.config = config
        d, m = config.latent_channels, config.num_tokens
        self.fine_head = nn.Conv2d(d, d, 1)
        self.coarse_head = nn.Conv2d(d, 2 * d, 1)
        self.token_embeddings = nn.Parameter(torch.randn(m, d) * 0.1)
        self.token_head = nn.Linear(d, 2 * d)
        self.routing_head = nn.Conv2d(d, m, 1)
        self.edge_head = nn.Conv2d(d, 1, 1)

    def forward(self, fine: Tensor, coarse: Tensor) -> EnergyTerms:
        config = self.config
        raw_f = self.fine_head(fine)
        anchor_c, raw_c = self.coarse_head(coarse).chunk(2, dim=1)
        token_input = coarse.mean(dim=(2, 3)).unsqueeze(1) + self.token_embeddings.unsqueeze(0)
        anchor_s, raw_s = self.token_head(token_input).chunk(2, dim=-1)
        raw_routing = self.routing_head(coarse).flatten(2).transpose(1, 2)
        raw_edge_x = self.edge_head((fine[..., 1:] - fine[..., :-1]).abs())
        raw_edge_y = self.edge_head((fine[..., 1:, :] - fine[..., :-1, :]).abs())
        dtype = torch.float64 if fine.dtype == torch.float64 else torch.float32
        with torch.autocast(device_type=fine.device.type, enabled=False):
            precision = LatentState(*(
                (config.w_min + (config.w_max - config.w_min) * raw.to(dtype).sigmoid())
                .clamp(config.w_min, config.w_max)
                for raw in (raw_f, raw_c, raw_s)
            ))
            return EnergyTerms.from_coefficients(
                LatentState(fine.to(dtype), anchor_c.to(dtype), anchor_s.to(dtype)),
                precision,
                raw_routing.to(dtype).softmax(dim=-1),
                raw_edge_x.to(dtype).sigmoid(),
                raw_edge_y.to(dtype).sigmoid(),
                w_min=config.w_min,
                w_max=config.w_max,
                lambda_m=config.lambda_m,
                lambda_s=config.lambda_s,
                lambda_e=config.lambda_e,
            )


class LocalGate(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.conv = nn.Conv2d(4 * width, 2 * width, 3, padding=1)

    def forward(self, features: Tensor) -> Tensor:
        content, gate = self.conv(features).chunk(2, dim=1)
        return F.silu(content) * gate.sigmoid()


class SharedProposer(nn.Module):
    def __init__(self, config: AERConfig):
        super().__init__()
        width, heads = config.latent_channels, config.attention_heads
        self.fine_gate = LocalGate(width)
        self.coarse_gate = LocalGate(width)
        self.token_input = nn.Linear(3 * width, width)
        self.tokens_read_coarse = nn.MultiheadAttention(width, heads, batch_first=True)
        self.coarse_reads_tokens = nn.MultiheadAttention(width, heads, batch_first=True)
        self.coarse_norm = nn.LayerNorm(width)
        self.token_norm = nn.LayerNorm(width)
        self.fine_output = nn.Conv2d(width, width, 3, padding=1)
        self.coarse_output = nn.Conv2d(width, width, 3, padding=1)
        self.token_output = nn.Linear(width, width)
        for output in (self.fine_output, self.coarse_output, self.token_output):
            nn.init.normal_(output.weight, std=0.02)
            nn.init.zeros_(output.bias)

    def forward(self, state: LatentState, context: RefinementContext, reference_delta: LatentState) -> LatentState:
        fine, coarse, tokens = state
        dtype = context.fine.dtype
        coarse_up = F.interpolate(coarse, size=fine.shape[-2:], mode="nearest")
        fine_context = torch.cat((fine, context.fine, coarse_up, reference_delta.fine), dim=1).to(dtype)
        coarse_context = torch.cat((coarse, context.coarse, downsample(fine), reference_delta.coarse), dim=1).to(dtype)
        fine_hidden = self.fine_gate(fine_context)
        coarse_hidden = self.coarse_gate(coarse_context)
        coarse_sequence = coarse_hidden.flatten(2).transpose(1, 2)
        token_context = torch.cat((tokens, context.energy.anchor.tokens, reference_delta.tokens), dim=-1).to(dtype)
        token_sequence = self.token_input(token_context)

        coarse_normalized = self.coarse_norm(coarse_sequence)
        token_message, _ = self.tokens_read_coarse(
            self.token_norm(token_sequence), coarse_normalized, coarse_normalized, need_weights=False
        )
        token_hidden = token_sequence + token_message
        token_normalized = self.token_norm(token_hidden)
        coarse_message, _ = self.coarse_reads_tokens(
            coarse_normalized, token_normalized, token_normalized, need_weights=False
        )
        coarse_message = coarse_message.transpose(1, 2).reshape_as(coarse_hidden)
        delta = LatentState(
            self.fine_output(fine_hidden).to(fine.dtype),
            self.coarse_output(coarse_hidden + coarse_message).to(coarse.dtype),
            self.token_output(token_hidden).to(tokens.dtype),
        )
        # Learn a full direction and use the reference RMS only as a displacement scale.
        elements_per_image = sum(part[0].numel() for part in state)
        return delta.scale(reference_delta.norm() / sqrt(elements_per_image))


class AERFeatureRefiner2D(nn.Module):
    """Apply a shared finite-step rollout to one BCHW bottleneck feature."""

    def __init__(self, in_channels: int, config: AERConfig | None = None):
        super().__init__()
        self.config = config if config is not None else AERConfig(enabled=True)
        self.in_channels = in_channels
        width = self.config.latent_channels
        self.input_adapter = nn.Conv2d(in_channels, width, 1)
        self.coefficient_head = CoefficientHead(self.config)
        self.proposer = SharedProposer(self.config)
        self.output_adapter = nn.Conv2d(2 * width, in_channels, 1, bias=False)

    def encode(self, feature: Tensor) -> RefinementContext:
        if feature.ndim != 4 or feature.shape[1] != self.in_channels:
            raise ValueError(f"expected a BCHW feature with {self.in_channels} channels")
        if not feature.is_floating_point() or any(size == 0 for size in feature.shape):
            raise ValueError("feature must be a nonempty floating-point tensor")
        height, width = feature.shape[-2:]
        fine = self.input_adapter(feature)
        if height % 2 or width % 2:
            fine = F.pad(fine, (0, width % 2, 0, height % 2), mode="replicate")
        # The projected fine anchor also supplies the fixed coarse evidence.
        coarse = downsample(fine)
        return RefinementContext(fine, coarse, self.coefficient_head(fine, coarse), (height, width))

    def forward(
        self,
        feature: Tensor,
        *,
        steps: int | None = None,
        mode: Mode | None = None,
        bypass: bool = False,
        return_aux: bool = False,
        return_trace: bool = True,
        residual_tolerance: float | None = None,
    ) -> AERResult:
        steps = self.config.steps if steps is None else steps
        mode = self.config.mode if mode is None else mode
        tolerance = self.config.residual_tolerance if residual_tolerance is None else residual_tolerance
        _validate_steps(steps)
        if mode not in MODES:
            raise ValueError("mode must be guarded, reference, unguarded or pcg")
        if tolerance is not None:
            _validate_nonnegative(tolerance, "residual_tolerance")
        if bypass or (steps == 0 and not return_trace):
            return AERResult(feature, None, None)

        context = self.encode(feature)
        state, trace, proposal_loss = self._rollout(
            context, steps, mode, tolerance, return_trace, return_aux and self.training
        )
        if steps == 0:
            return AERResult(feature, trace, proposal_loss)
        delta = state.subtract(context.energy.anchor)
        routed = torch.bmm(context.energy.routing, delta.tokens).transpose(1, 2).reshape_as(delta.coarse)
        coarse = F.interpolate(delta.coarse + routed, size=delta.fine.shape[-2:], mode="nearest")
        update = self.output_adapter(torch.cat((delta.fine, coarse), dim=1).to(context.fine.dtype))
        height, width = context.feature_size
        refined = feature + (self.config.residual_scale * update[..., :height, :width]).to(feature.dtype)
        return AERResult(refined, trace, proposal_loss)

    def _rollout(
        self,
        context: RefinementContext,
        steps: int,
        mode: Mode,
        tolerance: float | None,
        return_trace: bool,
        return_aux: bool,
    ) -> tuple[LatentState, AERTrace | None, Tensor | None]:
        terms = context.energy
        state = terms.anchor
        track_gradients = torch.is_grad_enabled()
        with torch.set_grad_enabled(track_gradients and steps > 0):
            gradient = terms.gradient(state)
        with torch.no_grad():
            initial_energy = terms.energy(state)
            residual = gradient.norm()
            valid_initial = (
                torch.isfinite(initial_energy) & torch.isfinite(residual)
                & torch.isfinite(terms.lipschitz) & (terms.lipschitz > 0)
            )
            if not valid_initial.all():
                raise FloatingPointError("initial AER energy, gradient norm and step bound must be finite")
        energy_history, residual_history = [initial_energy], [residual]
        reference_history, candidate_history = [], []
        accept_history, active_history, auxiliary_losses = [], [], []
        iterations = torch.zeros_like(initial_energy, dtype=torch.long)
        converged = _converged(residual, terms.strong_convexity, tolerance)
        pcg = terms.pcg_iterates(state, steps) if mode == "pcg" else None

        for step in range(steps):
            active = ~converged
            if tolerance is not None and not active.any():
                break
            reference_delta = gradient.scale(-terms.lipschitz.reciprocal())
            reference = state.add(reference_delta)
            proposal_delta = None
            if mode in ("guarded", "unguarded"):
                proposal_delta = self.proposer(state, context, reference_delta)
                candidate = state.add(proposal_delta)
            elif pcg is not None:
                candidate = next(pcg)
            else:
                candidate = reference
            with torch.no_grad():
                reference_energy = terms.energy(reference)
                candidate_energy = reference_energy if mode == "reference" else terms.energy(candidate)
                finite_reference = torch.isfinite(reference_energy)
                finite_candidate = torch.isfinite(candidate_energy)
                if mode in ("guarded", "reference") and not (finite_reference | ~active).all():
                    raise FloatingPointError("nonfinite AER reference energy for an active sample")
                if self.training and track_gradients and not (finite_reference & finite_candidate).all():
                    raise FloatingPointError("nonfinite AER rollout energy during training")
                if mode == "guarded":
                    accepted = active & finite_candidate & (candidate_energy <= reference_energy)
                elif mode == "unguarded":
                    accepted = active
                else:
                    accepted = torch.zeros_like(active)
                use_candidate = active if pcg is not None else accepted
                if return_trace:
                    updated_energy = torch.where(use_candidate, candidate_energy, reference_energy)
                    energy_history.append(torch.where(active, updated_energy, energy_history[-1]))
                    reference_history.append(reference_energy)
                    candidate_history.append(candidate_energy)
                    accept_history.append(accepted)
                    active_history.append(active)
                    iterations = iterations + active.long()
            if return_aux and track_gradients and proposal_delta is not None:
                auxiliary_losses.append(_proposal_distillation(proposal_delta, state, terms, active))
            state = candidate.where(use_candidate, reference).where(active, state)
            if step + 1 < steps or return_trace or tolerance is not None:
                with torch.set_grad_enabled(track_gradients and step + 1 < steps):
                    gradient = terms.gradient(state)
                with torch.no_grad():
                    residual = gradient.norm()
                    converged = converged | _converged(residual, terms.strong_convexity, tolerance)
                    if return_trace:
                        residual_history.append(residual)

        trace = None
        if return_trace:
            empty = initial_energy.new_empty((0, initial_energy.numel()))
            trace = AERTrace(
                torch.stack(energy_history),
                torch.stack(residual_history),
                torch.stack(reference_history) if reference_history else empty,
                torch.stack(candidate_history) if candidate_history else empty,
                torch.stack(accept_history) if accept_history else empty.bool(),
                torch.stack(active_history) if active_history else empty.bool(),
                iterations,
                converged,
            )
        proposal_loss = torch.stack(auxiliary_losses).mean() if auxiliary_losses else None
        return state, trace, proposal_loss


def _converged(residual: Tensor, strong_convexity: float, tolerance: float | None) -> Tensor:
    if tolerance is None:
        return torch.zeros_like(residual, dtype=torch.bool)
    return residual / strong_convexity <= tolerance


def _proposal_distillation(delta: LatentState, state: LatentState, terms: EnergyTerms, active: Tensor) -> Tensor:
    with torch.no_grad():
        for target in terms.pcg_iterates(state, steps=3):
            pass
        target_delta = target.subtract(state)
        target_magnitude = target_delta.dot(target_delta).clamp_min(1e-6)
    error = delta.subtract(target_delta)
    per_image = error.dot(error) / target_magnitude
    return torch.where(active, per_image, 0).sum() / active.sum().clamp_min(1)


class AERDenoiser(nn.Module):
    """Shared integration for the two denoisers, with no AER parameters when disabled."""

    def _init_aer(self, channels: int, config: AERConfig | None) -> None:
        self.aer = AERFeatureRefiner2D(channels, config) if config is not None and config.enabled else None
        if self.aer is not None and self.aer.config.freeze_backbone:
            self.requires_grad_(False)
            self.aer.requires_grad_(True)
            self.train(self.training)

    def _refine_bottleneck(self, feature: Tensor, return_aux: bool) -> tuple[Tensor, Tensor | None]:
        if self.aer is None:
            return feature, None
        result = self.aer(feature, return_aux=return_aux, return_trace=False)
        return result.feature, result.proposal_loss

    def train(self, mode: bool = True) -> AERDenoiser:
        super().train(mode)
        if mode and self.aer is not None and self.aer.config.freeze_backbone:
            for module in self.children():
                if module is not self.aer:
                    module.eval()
        return self


def load_pretrained_backbone(model: nn.Module, state_dict: Mapping[str, Tensor]) -> None:
    """Warm start an enabled model from an exact backbone-only state dict."""
    if not isinstance(model, AERDenoiser) or model.aer is None:
        raise ValueError("warm start requires an unwrapped, AER-enabled denoiser")
    initialized = model.state_dict()
    backbone_keys = {name for name in initialized if not name.startswith("aer.")}
    missing = sorted(backbone_keys - state_dict.keys())
    unexpected = sorted(state_dict.keys() - backbone_keys)
    bad_shapes = sorted(
        name for name in backbone_keys & state_dict.keys()
        if not isinstance(state_dict[name], Tensor) or state_dict[name].shape != initialized[name].shape
    )
    if missing or unexpected or bad_shapes:
        raise RuntimeError(f"incompatible backbone weights (missing={missing}, unexpected={unexpected}, bad_shapes={bad_shapes})")
    model.load_state_dict({**initialized, **state_dict}, strict=True)
