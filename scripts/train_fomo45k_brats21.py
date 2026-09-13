"""Train ANDi with the FOMO45K BraTS21 adapter registered at runtime.

This wrapper keeps the shared dataset factory backward compatible while the
BraTS21 publication remains an optional, dependency-heavy preprocessing path.
All command-line arguments are handled by ``scripts.train``.
"""

from __future__ import annotations

try:
    from _bootstrap import bootstrap
except ImportError:
    from andi_rewrite.scripts._bootstrap import bootstrap

bootstrap()

from andi_rewrite.data.datasets.factory import DATASET_BUILDERS  # noqa: E402
from andi_rewrite.data.datasets.fomo45k_brats21 import (  # noqa: E402
    build_fomo45k_brats21_slice_dataset,
    build_fomo45k_brats21_volume_dataset,
)


def register_fomo45k_brats21() -> None:
    DATASET_BUILDERS.update(
        {
            "fomo45k_brats21": build_fomo45k_brats21_slice_dataset,
            "fomo45k_brats21_slices": build_fomo45k_brats21_slice_dataset,
            "fomo45k_brats21_volume": build_fomo45k_brats21_volume_dataset,
        }
    )


def main() -> None:
    register_fomo45k_brats21()
    from andi_rewrite.scripts.train import main as train_main

    train_main()


if __name__ == "__main__":
    main()
