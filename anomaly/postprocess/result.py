"""Structured products from one postprocessing run."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PostprocessResult:
    """All score and selected-threshold products from one raw-map tensor."""

    score_raw: torch.Tensor
    score_mf: torch.Tensor
    thresholds_raw: torch.Tensor
    thresholds_mf: torch.Tensor
    binary_mask_raw: torch.Tensor
    binary_mask_mf: torch.Tensor
    binary_mask_raw_postprocessed: torch.Tensor
    binary_mask_mf_postprocessed: torch.Tensor
    threshold_method: str
    normalization_scope: str

    def as_dict(self) -> dict[str, torch.Tensor | str]:
        payload: dict[str, torch.Tensor | str] = {
            "score_raw": self.score_raw,
            "score_mf": self.score_mf,
            "thresholds_raw": self.thresholds_raw,
            "thresholds_mf": self.thresholds_mf,
            "binary_mask_raw": self.binary_mask_raw,
            "binary_mask_mf": self.binary_mask_mf,
            "binary_mask_raw_postprocessed": self.binary_mask_raw_postprocessed,
            "binary_mask_mf_postprocessed": self.binary_mask_mf_postprocessed,
            "threshold_method": self.threshold_method,
            "normalization_scope": self.normalization_scope,
        }
        if self.threshold_method == "yen":
            payload.update(
                {
                    "yen_thresholds_raw": self.thresholds_raw,
                    "yen_thresholds_mf": self.thresholds_mf,
                    "yen_mask_raw": self.binary_mask_raw,
                    "yen_mask_mf": self.binary_mask_mf,
                    "yen_mask_raw_postprocessed": self.binary_mask_raw_postprocessed,
                    "yen_mask_mf_postprocessed": self.binary_mask_mf_postprocessed,
                }
            )
        return payload

    # Compatibility aliases keep existing Yen callers byte-for-byte stable.
    # New code should use the method-neutral fields above.
    @property
    def yen_thresholds_raw(self) -> torch.Tensor:
        return self.thresholds_raw

    @property
    def yen_thresholds_mf(self) -> torch.Tensor:
        return self.thresholds_mf

    @property
    def yen_mask_raw(self) -> torch.Tensor:
        return self.binary_mask_raw

    @property
    def yen_mask_mf(self) -> torch.Tensor:
        return self.binary_mask_mf

    @property
    def yen_mask_raw_postprocessed(self) -> torch.Tensor:
        return self.binary_mask_raw_postprocessed

    @property
    def yen_mask_mf_postprocessed(self) -> torch.Tensor:
        return self.binary_mask_mf_postprocessed

    @classmethod
    def concatenate(cls, results: list["PostprocessResult"]) -> "PostprocessResult":
        if not results:
            raise ValueError("Cannot concatenate an empty list of postprocess results.")
        scopes = {result.normalization_scope for result in results}
        if len(scopes) != 1:
            raise ValueError(f"Cannot concatenate mixed normalization scopes: {sorted(scopes)}")
        methods = {result.threshold_method for result in results}
        if len(methods) != 1:
            raise ValueError(f"Cannot concatenate mixed threshold methods: {sorted(methods)}")

        def combine(name: str) -> torch.Tensor:
            return torch.cat([getattr(result, name) for result in results], dim=0)

        return cls(
            score_raw=combine("score_raw"),
            score_mf=combine("score_mf"),
            thresholds_raw=combine("thresholds_raw"),
            thresholds_mf=combine("thresholds_mf"),
            binary_mask_raw=combine("binary_mask_raw"),
            binary_mask_mf=combine("binary_mask_mf"),
            binary_mask_raw_postprocessed=combine("binary_mask_raw_postprocessed"),
            binary_mask_mf_postprocessed=combine("binary_mask_mf_postprocessed"),
            threshold_method=results[0].threshold_method,
            normalization_scope=results[0].normalization_scope,
        )
