"""Per-image quadratic energy and matrix-free operators for AER-Net."""

from dataclasses import dataclass
from typing import Iterator, NamedTuple

import torch
from torch import Tensor
from torch.nn import functional as F


def batch_broadcast(value: Tensor, like: Tensor) -> Tensor:
    return value.reshape(-1, *([1] * (like.ndim - 1)))


class LatentState(NamedTuple):
    fine: Tensor
    coarse: Tensor
    tokens: Tensor

    def add(self, other: "LatentState") -> "LatentState":
        return LatentState(*(a + b for a, b in zip(self, other)))

    def subtract(self, other: "LatentState") -> "LatentState":
        return LatentState(*(a - b for a, b in zip(self, other)))

    def scale(self, per_image: Tensor) -> "LatentState":
        return LatentState(*(a * batch_broadcast(per_image, a) for a in self))

    def where(self, mask: Tensor, other: "LatentState") -> "LatentState":
        return LatentState(*(torch.where(batch_broadcast(mask, a), a, b) for a, b in zip(self, other)))

    def dot(self, other: "LatentState") -> Tensor:
        return sum((a * b).flatten(1).sum(dim=1) for a, b in zip(self, other))

    def norm(self) -> Tensor:
        flattened = torch.cat([part.flatten(1) for part in self], dim=1)
        return torch.linalg.vector_norm(flattened, dim=1)


def downsample(fine: Tensor) -> Tensor:
    return F.avg_pool2d(fine, kernel_size=2, stride=2)


def downsample_adjoint(coarse: Tensor) -> Tensor:
    return coarse.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1) / 4


def weighted_degree(edge_x: Tensor, edge_y: Tensor) -> Tensor:
    return (
        F.pad(edge_x, (0, 1, 0, 0))
        + F.pad(edge_x, (1, 0, 0, 0))
        + F.pad(edge_y, (0, 0, 0, 1))
        + F.pad(edge_y, (0, 0, 1, 0))
    )


@dataclass(frozen=True)
class EnergyTerms:
    """Image-conditioned coefficients. All spatial/channel reductions are sums."""

    anchor: LatentState
    precision: LatentState
    routing: Tensor
    edge_x: Tensor
    edge_y: Tensor
    lipschitz: Tensor
    strong_convexity: float
    lambda_m: float
    lambda_s: float
    lambda_e: float

    @classmethod
    def from_coefficients(
        cls,
        anchor: LatentState,
        precision: LatentState,
        routing: Tensor,
        edge_x: Tensor,
        edge_y: Tensor,
        *,
        w_min: float,
        w_max: float,
        lambda_m: float,
        lambda_s: float,
        lambda_e: float,
    ) -> "EnergyTerms":
        # The caller supplies W in [w_min, w_max], nonnegative edges and lambdas.
        dtype = torch.float64 if anchor.fine.dtype == torch.float64 else torch.float32
        with torch.autocast(device_type=anchor.fine.device.type, enabled=False):
            anchor = LatentState(*(part.to(dtype) for part in anchor))
            precision = LatentState(*(part.to(dtype) for part in precision))
            routing, edge_x, edge_y = (part.to(dtype) for part in (routing, edge_x, edge_y))
            routing_norm_one = routing.abs().sum(dim=1).amax(dim=1)
            routing_norm_inf = routing.abs().sum(dim=2).amax(dim=1)
            max_degree = weighted_degree(edge_x, edge_y).flatten(1).amax(dim=1)
            # D has disjoint rows with four entries of 1/4, so ||D||^2 = 1/4.
            lipschitz = (
                w_max
                + lambda_m * 1.25
                + lambda_s * (1 + routing_norm_one * routing_norm_inf)
                + 2 * lambda_e * max_degree
            )
        return cls(anchor, precision, routing, edge_x, edge_y, lipschitz, w_min, lambda_m, lambda_s, lambda_e)

    def _cast_state(self, state: LatentState) -> LatentState:
        return LatentState(*(part.to(self.anchor.fine.dtype) for part in state))

    def _differences(self, state: LatentState) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        fine, coarse, tokens = state
        scale = downsample(fine) - coarse
        routed = torch.bmm(self.routing, tokens).transpose(1, 2).reshape_as(coarse)
        structure = coarse - routed
        horizontal = fine[..., :-1] - fine[..., 1:]
        vertical = fine[..., :-1, :] - fine[..., 1:, :]
        return scale, structure, horizontal, vertical

    def energy(self, state: LatentState) -> Tensor:
        with torch.autocast(device_type=self.routing.device.type, enabled=False):
            state = self._cast_state(state)
            delta = state.subtract(self.anchor)
            weighted_delta = LatentState(*(p * a for p, a in zip(self.precision, delta)))
            scale, structure, horizontal, vertical = self._differences(state)
            return 0.5 * (
                delta.dot(weighted_delta)
                + self.lambda_m * scale.square().flatten(1).sum(dim=1)
                + self.lambda_s * structure.square().flatten(1).sum(dim=1)
                + self.lambda_e * (self.edge_x * horizontal.square()).flatten(1).sum(dim=1)
                + self.lambda_e * (self.edge_y * vertical.square()).flatten(1).sum(dim=1)
            )

    def _coupling_gradient(self, state: LatentState) -> LatentState:
        scale, structure, horizontal, vertical = self._differences(state)
        flux_x = self.edge_x * horizontal
        flux_y = self.edge_y * vertical
        laplacian = (
            F.pad(flux_x, (0, 1, 0, 0))
            - F.pad(flux_x, (1, 0, 0, 0))
            + F.pad(flux_y, (0, 0, 0, 1))
            - F.pad(flux_y, (0, 0, 1, 0))
        )
        token_force = torch.bmm(self.routing.transpose(1, 2), structure.flatten(2).transpose(1, 2))
        return LatentState(
            self.lambda_m * downsample_adjoint(scale) + self.lambda_e * laplacian,
            -self.lambda_m * scale + self.lambda_s * structure,
            -self.lambda_s * token_force,
        )

    def gradient(self, state: LatentState) -> LatentState:
        with torch.autocast(device_type=self.routing.device.type, enabled=False):
            state = self._cast_state(state)
            delta = state.subtract(self.anchor)
            anchored = LatentState(*(p * a for p, a in zip(self.precision, delta)))
            return anchored.add(self._coupling_gradient(state))

    def reference_step(self, state: LatentState) -> LatentState:
        return state.add(self.gradient(state).scale(-self.lipschitz.reciprocal()))

    def hessian_vector(self, vector: LatentState) -> LatentState:
        with torch.autocast(device_type=self.routing.device.type, enabled=False):
            vector = self._cast_state(vector)
            anchored = LatentState(*(p * a for p, a in zip(self.precision, vector)))
            return anchored.add(self._coupling_gradient(vector))

    def _jacobi_diagonal(self) -> LatentState:
        fine, coarse, tokens = self.precision
        return LatentState(
            fine + self.lambda_m / 16 + self.lambda_e * weighted_degree(self.edge_x, self.edge_y),
            coarse + self.lambda_m + self.lambda_s,
            tokens + self.lambda_s * self.routing.square().sum(dim=1).unsqueeze(-1),
        )

    def pcg_iterates(self, state: LatentState, steps: int) -> Iterator[LatentState]:
        """Jacobi-preconditioned CG for H v = W b, with independent batch scalars."""
        diagonal = self._jacobi_diagonal()
        residual = self.gradient(state).scale(-torch.ones_like(self.lipschitz))
        direction = LatentState(*(r / d for r, d in zip(residual, diagonal)))
        rz = residual.dot(direction)
        for _ in range(steps):
            h_direction = self.hessian_vector(direction)
            alpha = _positive_ratio(rz, direction.dot(h_direction))
            state = state.add(direction.scale(alpha))
            yield state
            residual = residual.subtract(h_direction.scale(alpha))
            preconditioned = LatentState(*(r / d for r, d in zip(residual, diagonal)))
            next_rz = residual.dot(preconditioned)
            direction = preconditioned.add(direction.scale(_positive_ratio(next_rz, rz)))
            rz = next_rz


def _positive_ratio(numerator: Tensor, denominator: Tensor) -> Tensor:
    # A stationary sample leaves its state untouched without a 0/0 update.
    valid = torch.isfinite(denominator) & (denominator > 0)
    quotient = numerator / torch.where(valid, denominator, torch.ones_like(denominator))
    return torch.where(valid & torch.isfinite(quotient), quotient, torch.zeros_like(quotient))
