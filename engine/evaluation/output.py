"""Legacy metric CSV and result-dictionary output contracts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def write_original_style_csv(scores: dict[Any, Any], path: str | Path) -> None:
    """Write the historical five-column metrics CSV without schema drift."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for key, value in scores.items():
        if isinstance(value, dict):
            rows.append(
                {
                    "thr": key,
                    "value": value.get("value"),
                    "dice": value.get("dice"),
                    "sensitivity": value.get("sensitivity"),
                    "precision": value.get("precision"),
                }
            )
        else:
            rows.append(
                {"thr": key, "value": value, "dice": None, "sensitivity": None, "precision": None}
            )
    frame = pd.DataFrame(rows, columns=["thr", "value", "dice", "sensitivity", "precision"])
    frame.to_csv(path, index=False)


def build_evaluation_result(
    *,
    output_csv: str | Path,
    output_mf_csv: str | Path,
    subjects: int,
    labels_available: bool,
    scores: dict[Any, Any],
    scores_mf: dict[Any, Any],
    compute_auprc: bool,
    auprc_mode: str,
    auprc_seed: int,
    memory_mode: str,
    threshold_method: str,
    postprocess_mode: str,
    normalization_scope: str,
    prediction_normalization_scope: str,
    postprocessing: dict[str, Any],
    cache_directory: str | Path | None = None,
    cache_hits: int | None = None,
    subjects_inferred: int | None = None,
) -> dict[str, Any]:
    """Assemble the stable evaluation result schema for both memory modes."""

    result: dict[str, Any] = {
        "output": str(output_csv),
        "output_mf": str(output_mf_csv),
        "subjects": subjects,
        "labels_available": labels_available,
    }
    if cache_directory is not None:
        result.update(
            {
                "memory_mode": memory_mode,
                "cache_directory": str(cache_directory),
                "cache_hits": cache_hits,
                "subjects_inferred": subjects_inferred,
            }
        )
    result.update(
        {
            "AUPRC": scores.get("AUPRC"),
            "AUPRC_mf": scores_mf.get("AUPRC"),
            "AUPRC_mode": auprc_mode if compute_auprc else None,
            "AUPRC_samples": scores.get("AUPRC_samples"),
            "AUPRC_seed": auprc_seed if compute_auprc and auprc_mode == "sampled" else None,
        }
    )
    if cache_directory is None:
        result["memory_mode"] = memory_mode
    result.update(
        {
            "threshold_method": threshold_method,
            "ThresholdDice": scores.get(threshold_method),
            "ThresholdDice_mf": scores_mf.get(threshold_method),
            "Threshold": scores.get(f"{threshold_method}thr"),
            "Threshold_mf": scores_mf.get(f"{threshold_method}thr"),
            "postprocess_mode": postprocess_mode,
            "normalization_scope": normalization_scope,
            "prediction_normalization_scope": prediction_normalization_scope,
            "postprocessing": postprocessing,
        }
    )
    method_label = threshold_method.title()
    result.update(
        {
            f"Dice{method_label}": scores.get(threshold_method),
            f"Dice{method_label}_mf": scores_mf.get(threshold_method),
            f"{method_label}Thr": scores.get(f"{threshold_method}thr"),
            f"{method_label}Thr_mf": scores_mf.get(f"{threshold_method}thr"),
        }
    )
    return result
