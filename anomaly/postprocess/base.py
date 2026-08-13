"""Common postprocessing types, constants, and stable registries."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch


NORMALIZATION_SCOPES = {"dataset", "subject"}
SUPPORTED_THRESHOLD_METHODS = ("yen", "otsu")


class BasePostprocessor(ABC):
    """A single score-map or binary-mask postprocessing step."""

    name = "base"

    @abstractmethod
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply this postprocessing step."""

    def describe(self) -> dict[str, Any]:
        return {"type": self.name}


# These dictionaries are intentionally mutable, module-owned singletons.  Third
# parties may register aliases at runtime, and their insertion order is part of
# the compatibility surface used in validation messages.
SCORE_POSTPROCESSORS: dict[str, type[BasePostprocessor]] = {}
MASK_POSTPROCESSORS: dict[str, type[BasePostprocessor]] = {}


def register_score_postprocessor(*names: str):
    """Register a score-map postprocessor under one or more aliases."""

    def decorator(cls: type[BasePostprocessor]) -> type[BasePostprocessor]:
        for name in names:
            SCORE_POSTPROCESSORS[name.lower()] = cls
        return cls

    return decorator


def register_mask_postprocessor(*names: str):
    """Register a binary-mask postprocessor under one or more aliases."""

    def decorator(cls: type[BasePostprocessor]) -> type[BasePostprocessor]:
        for name in names:
            MASK_POSTPROCESSORS[name.lower()] = cls
        return cls

    return decorator
