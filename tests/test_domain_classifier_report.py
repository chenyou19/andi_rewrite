"""Contracts for the domain-classifier report aggregator.

These tests use tiny deterministic fixtures only to exercise schema handling;
they are not empirical domain-classifier results.
"""

from __future__ import annotations

import csv
import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from domain_classifier.metrics import aggregate_subject_predictions
from domain_classifier.report import (
    PRIMARY_FULL_RETRAINED_HOLM4_FAMILY,
    ReportConfig,
    SECONDARY_48_FAMILY,
    SUMMARY_FIELDS,
    aggregate_fixed_seed_subject_ensemble,
    build_report,
    build_summary_rows,
    compute_andi_ap_correlation,
    export_model_grid_v3_rosters,
    generate_calibration_diagnostic_plots,
    load_andi_ap_points,
    summarise_secondary_48,
    write_summary_csv,
)


class DomainClassifierReportTest(unittest.TestCase):
    def _prediction_fixture(self, root: Path) -> Path:
        prediction_dir = root / "predictions"
        control_dir = root / "controls"
        prediction_dir.mkdir()
        control_dir.mkdir()
        path = prediction_dir / "per_slice_predictions.csv"
        rows = [
            # Two slices per subject make an accidental slice-level bootstrap
            # distinguishable from the required participant-level aggregation.
            {"healthy_domain": "FOMO", "input_modalities": "FLAIR+T1+T2", "classifier": "small_cnn", "seed": 73, "split": "train", "label": 0, "score": 0.1, "participant_id": "h1"},
            {"healthy_domain": "FOMO", "input_modalities": "FLAIR+T1+T2", "classifier": "small_cnn", "seed": 73, "split": "train", "label": 0, "score": 0.2, "participant_id": "h1"},
            {"healthy_domain": "FOMO", "input_modalities": "FLAIR+T1+T2", "classifier": "small_cnn", "seed": 73, "split": "train", "label": 1, "score": 0.8, "participant_id": "b1"},
            {"healthy_domain": "FOMO", "input_modalities": "FLAIR+T1+T2", "classifier": "small_cnn", "seed": 73, "split": "train", "label": 1, "score": 0.9, "participant_id": "b1"},
            {"healthy_domain": "FOMO", "input_modalities": "FLAIR+T1+T2", "classifier": "small_cnn", "seed": 73, "split": "test", "label": 0, "score": 0.2, "participant_id": "h2"},
            {"healthy_domain": "FOMO", "input_modalities": "FLAIR+T1+T2", "classifier": "small_cnn", "seed": 73, "split": "test", "label": 0, "score": 0.4, "participant_id": "h2"},
            {"healthy_domain": "FOMO", "input_modalities": "FLAIR+T1+T2", "classifier": "small_cnn", "seed": 73, "split": "test", "label": 1, "score": 0.6, "participant_id": "b2"},
            {"healthy_domain": "FOMO", "input_modalities": "FLAIR+T1+T2", "classifier": "small_cnn", "seed": 73, "split": "test", "label": 1, "score": 0.8, "participant_id": "b2"},
        ]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        (control_dir / "tiny_overfit.json").write_text(
            json.dumps({"status": "PASS", "train_accuracy": 1.0, "train_bce": 0.001}),
            encoding="utf-8",
        )
        (control_dir / "positive_control.json").write_text(
            json.dumps({"status": "PASS", "subject_auc": 0.95, "ci_low": 0.8}),
            encoding="utf-8",
        )
        (control_dir / "negative_control.json").write_text(
            json.dumps({"status": "PASS", "subject_auc": 0.5, "ci_low": 0.4, "ci_high": 0.6}),
            encoding="utf-8",
        )
        (control_dir / "label_permutation.json").write_text(
            json.dumps({"status": "PASS", "subject_auc": 0.5, "ci_low": 0.4, "ci_high": 0.6}),
            encoding="utf-8",
        )
        return path

    def test_prediction_aggregation_uses_test_subjects_and_required_columns(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._prediction_fixture(root)
            rows, context = build_summary_rows(root, config=ReportConfig(bootstrap_replicates=20))
            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertEqual(row["test_subjects"], 2)
            self.assertEqual(row["test_slices"], 4)
            self.assertAlmostEqual(row["subject_roc_auc"], 1.0)
            self.assertEqual(row["bootstrap_resampling_unit"], "subject")
            self.assertEqual(row["control_gate_status"], "PASS")
            self.assertIn("subject_roc_auc", SUMMARY_FIELDS)
            self.assertTrue(context["prediction_paths"])

    def test_subject_aggregation_keeps_one_score_per_participant(self) -> None:
        result = aggregate_subject_predictions(
            [0, 0, 1, 1],
            [0.1, 0.9, 0.2, 0.8],
            ["h", "h", "b", "b"],
        )
        self.assertEqual(result["participant_ids"].tolist(), ["h", "b"])
        np.testing.assert_allclose(result["scores"], [0.5, 0.5])
        self.assertEqual(result["slice_counts"].tolist(), [2, 2])

    def test_missing_controls_are_inconclusive_and_do_not_pass_gate(self) -> None:
        records = [
            {"healthy_domain": "FOMO", "input_modalities": "FLAIR", "classifier": "logistic", "seed": 73,
             "subject_roc_auc": 0.5, "bootstrap_ci_low": 0.49, "bootstrap_ci_high": 0.51}
        ]
        rows, _ = build_summary_rows(records)
        self.assertEqual(rows[0]["control_gate_status"], "INCONCLUSIVE")
        self.assertEqual(rows[0]["low_separability_status"], "INCONCLUSIVE")

    def test_correlation_reports_small_n_caveat(self) -> None:
        rows = [
            {
                "healthy_domain": domain,
                "stage": "final",
                "input_modalities": "FLAIR+T1+T2",
                "classifier": "small_cnn",
                "seed": 73,
                "subject_roc_auc": auc,
            }
            for domain, auc in zip(("FOMO", "MPI", "OASIS3", "Mixed"), (0.8, 0.7, 0.6, 0.65))
        ]
        points = {
            "FOMO": {"mf_ap": 0.79},
            "MPI": {"mf_ap": 0.75},
            "OASIS3": {"mf_ap": 0.71},
            "Mixed": {"mf_ap": 0.67},
        }
        result = compute_andi_ap_correlation(rows, points)
        self.assertEqual(result["n_points"], 4)
        self.assertIn("4", result["caveat"])
        self.assertIsNotNone(result["pearson_r"])

    def test_correlation_uses_exact_primary_selector_and_is_order_invariant(self) -> None:
        primary = [
            {
                "healthy_domain": domain,
                "stage": "final",
                "input_modalities": "FLAIR+T1+T2",
                "classifier": "small_cnn",
                "seed": 73,
                "subject_roc_auc": 1.0,
                "run_id": f"primary-{domain}",
            }
            for domain in ("FOMO", "MPI", "OASIS3", "Mixed")
        ]
        distractors = [
            {
                "healthy_domain": domain,
                "stage": "final",
                "input_modalities": "FLAIR",
                "classifier": "small_cnn",
                "seed": 73,
                "subject_roc_auc": 0.51,
            }
            for domain in ("FOMO", "MPI", "OASIS3", "Mixed")
        ]
        points = {domain: {"mf_ap": value} for domain, value in zip(("FOMO", "MPI", "OASIS3", "Mixed"), (0.79, 0.75, 0.71, 0.67))}
        result = compute_andi_ap_correlation(list(reversed(distractors + primary)), points)
        self.assertEqual(result["n_points"], 4)
        self.assertEqual(result["status"], "INCONCLUSIVE_CONSTANT_INPUT")
        self.assertTrue(all(point["domain_auc"] == 1.0 for point in result["points"]))
        self.assertIsNone(result["pearson_r"])
        self.assertIsNone(result["spearman_rho"])

    def test_correlation_duplicate_primary_or_missing_primary_fails_closed(self) -> None:
        base = {
            "stage": "final",
            "input_modalities": "FLAIR+T1+T2",
            "classifier": "small_cnn",
            "seed": 73,
        }
        rows = [
            {**base, "healthy_domain": "FOMO", "subject_roc_auc": 0.8},
            {**base, "healthy_domain": "FOMO", "subject_roc_auc": 0.9},
            {**base, "healthy_domain": "MPI", "subject_roc_auc": 0.7},
            {**base, "healthy_domain": "OASIS3", "subject_roc_auc": 0.6},
            # Mixed has another model, which must not be used as fallback.
            {**base, "healthy_domain": "Mixed", "classifier": "resnet18", "subject_roc_auc": 0.95},
        ]
        points = {domain: {"mf_ap": value} for domain, value in zip(("FOMO", "MPI", "OASIS3", "Mixed"), (0.79, 0.75, 0.71, 0.67))}
        duplicate = compute_andi_ap_correlation(rows, points)
        self.assertEqual(duplicate["status"], "FAIL_CLOSED_DUPLICATE_OR_CONFLICT")
        missing = compute_andi_ap_correlation(rows[1:], points)
        self.assertEqual(missing["status"], "INCONCLUSIVE_SELECTOR_PENDING")
        self.assertIn("Mixed", missing["missing_domains"])
        conflicting_points = [(domain, {"mf_ap": value}) for domain, value in zip(("FOMO", "MPI", "OASIS3", "Mixed"), (0.79, 0.75, 0.71, 0.67))]
        conflicting_points.append(("fomo", {"mf_ap": 0.80}))
        conflict = compute_andi_ap_correlation(rows[:1] + rows[2:], dict(conflicting_points))
        self.assertEqual(conflict["status"], "FAIL_CLOSED_DUPLICATE_OR_CONFLICT")
        self.assertIn("fomo", conflict["duplicate_ap_domains"])

    def test_build_report_writes_artifacts_without_claiming_unexecuted_plots(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._prediction_fixture(root)
            output = root / "report"
            result = build_report(root, output, config=ReportConfig(bootstrap_replicates=10))
            self.assertTrue((output / "summary.csv").is_file())
            self.assertTrue((output / "report_audit.json").is_file())
            self.assertTrue((output / "report_manifest.json").is_file())
            self.assertTrue((output / "report.md").is_file())
            report_text = (output / "report.md").read_text(encoding="utf-8")
            self.assertIn("low-separability", report_text)
            self.assertIn("BraTS subjects are tumor patients", report_text)
            self.assertIn("AUC=1 alone", report_text)
            self.assertIn("roc_curves", result["plots"])

    def test_control_rows_do_not_contaminate_formal_questions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._prediction_fixture(root)
            control_dir = root / "controls" / "negative_control"
            control_dir.mkdir(parents=True)
            (control_dir / "summary.json").write_text(
                json.dumps(
                    {
                        "healthy_domain": "FOMO",
                        "input_modalities": "FLAIR+T1+T2",
                        "classifier": "small_cnn",
                        "seed": 73,
                        "stage": "final",
                        "subject_roc_auc": 0.99,
                        "control_type": "negative_control",
                    }
                ),
                encoding="utf-8",
            )
            result = build_report(
                root,
                root / "report",
                config=ReportConfig(bootstrap_replicates=10),
                andi_ap_source={"FOMO": {"mf_ap": 0.8}},
                make_plots=False,
            )
            self.assertEqual(result["correlation"]["n_points"], 1)
            report_manifest = json.loads(
                (root / "report" / "report_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(report_manifest["formal_final_rows"], 1)
            self.assertEqual(report_manifest["excluded_nonformal_rows"], 1)
            report_text = (root / "report" / "report.md").read_text(encoding="utf-8")
            self.assertIn("formal final rows used for scientific questions: **1**", report_text)

    def test_stage1_audit_and_manifest_are_preserved_when_output_is_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._prediction_fixture(root)
            audit = root / "audit.json"
            manifest = root / "experiment_manifest.json"
            audit.write_text(json.dumps({"authority": "stage1-audit"}), encoding="utf-8")
            manifest.write_text(json.dumps({"authority": "stage1-manifest"}), encoding="utf-8")
            build_report(root, root, config=ReportConfig(bootstrap_replicates=10), make_plots=False)
            self.assertEqual(json.loads(audit.read_text(encoding="utf-8"))["authority"], "stage1-audit")
            self.assertEqual(json.loads(manifest.read_text(encoding="utf-8"))["authority"], "stage1-manifest")
            self.assertTrue((root / "report_audit.json").is_file())
            self.assertTrue((root / "report_manifest.json").is_file())

    def test_runner_result_schema_is_consumed_from_test_predictions(self) -> None:
        payload = {
            "seed": 73,
            "split_seed": 73,
            "config": {
                "model": "small_cnn",
                "modalities": ["flair", "t1", "t2"],
                "stage": "final",
                "split_seed": 73,
            },
            "test_predictions": [
                {"record_index": 0, "label": 0, "probability": 0.1, "participant_id": "h1", "pair_id": "p1", "case_id": "h1"},
                {"record_index": 1, "label": 1, "probability": 0.9, "participant_id": "b1", "pair_id": "p1", "case_id": "b1"},
            ],
            "test_subject_predictions": [
                {"label": 0, "probability": 0.1, "participant_id": "h1", "pair_id": "p1", "slice_count": 1},
                {"label": 1, "probability": 0.9, "participant_id": "b1", "pair_id": "p1", "slice_count": 1},
            ],
            "test_statistics": {"subject_bootstrap": {"resampling_unit": "matched_pair"}},
        }
        rows, _ = build_summary_rows(payload, config=ReportConfig(bootstrap_replicates=10))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["classifier"], "small_cnn")
        self.assertEqual(rows[0]["input_modalities"], "FLAIR+T1+T2")
        self.assertEqual(rows[0]["stage"], "final")
        self.assertAlmostEqual(rows[0]["subject_roc_auc"], 1.0)

    def test_control_mode_and_recorded_bootstrap_are_visible(self) -> None:
        payload = {
            "seed": 73,
            "config": {
                "model": "small_cnn",
                "modalities": ["flair", "t1", "t2"],
                "stage": "final",
                "control_type": "tiny",
            },
            "test_predictions": [
                {"label": 0, "probability": 0.1, "participant_id": "h1"},
                {"label": 1, "probability": 0.9, "participant_id": "b1"},
            ],
            "train_final": {
                "slice": {"accuracy": 1.0, "bce": 0.001},
                "subject": {"accuracy": 1.0, "bce": 0.002},
            },
            "test_statistics": {
                "subject_bootstrap": {
                    "ci_low": 0.41,
                    "ci_high": 0.59,
                    "n_bootstrap": 37,
                    "n_valid": 35,
                    "resampling_unit": "subject",
                }
            },
        }
        rows, _ = build_summary_rows(payload, config=ReportConfig(bootstrap_replicates=10))
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["control_mode"], "tiny_overfit")
        self.assertEqual(row["result_kind"], "control")
        self.assertEqual(row["train_final_accuracy"], 1.0)
        self.assertEqual(row["train_final_bce"], 0.001)
        self.assertEqual(row["bootstrap_source"], "recorded_subject_bootstrap")
        self.assertEqual(row["bootstrap_ci_low"], 0.41)
        self.assertEqual(row["bootstrap_ci_high"], 0.59)
        self.assertEqual(row["bootstrap_n"], 37.0)
        with tempfile.TemporaryDirectory() as directory:
            summary_path = Path(directory) / "summary.csv"
            write_summary_csv(summary_path, rows)
            serialized = next(csv.DictReader(summary_path.open(encoding="utf-8")))
            self.assertEqual(serialized["formal_final"], "false")

        negative_payload = dict(payload)
        negative_payload["config"] = {**payload["config"], "control_type": "negative"}
        negative_rows, _ = build_summary_rows(negative_payload, config=ReportConfig(bootstrap_replicates=10))
        self.assertEqual(negative_rows[0]["control_mode"], "same_cohort_negative")
        shuffle_payload = dict(payload)
        shuffle_payload["config"] = {**payload["config"], "control_type": "label_shuffle"}
        shuffle_rows, _ = build_summary_rows(shuffle_payload, config=ReportConfig(bootstrap_replicates=10))
        self.assertEqual(shuffle_rows[0]["control_mode"], "label_shuffle")

    def test_report_surfaces_stage1_and_parity_status_without_copying_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stage1 = root / "audit.json"
            stage1.write_text(
                json.dumps(
                    {
                        "status": "PASS",
                        "source_inventories": {"fomo45k": {"fomo45k": {"participants": 240, "records": 10}}},
                        "comparisons": {
                            "fomo45k": {
                                "pair_count": 240,
                                "records": 20,
                                "native_tumor_free_records": 10,
                                "model_mask_false_records": 10,
                            }
                        },
                        "brats_inventories": {"fomo45k": {"participants": 938, "records": 20}},
                    }
                ),
                encoding="utf-8",
            )
            parity = root / "parity.json"
            parity.write_text(
                json.dumps(
                    {
                        "run": {
                            "status": "INCOMPLETE",
                            "cohorts": [
                                {
                                    "cohort": "FOMO",
                                    "status": "PASS",
                                    "selected_pairs": 1,
                                    "actual_pair_count": 1,
                                    "actual_slice_count": 3,
                                    "pairs": [{"slices": [{"checks": {"max_abs_error": 0.0}}]}],
                                },
                                {
                                    "cohort": "MPI",
                                    "status": "INCOMPLETE",
                                    "selected_pairs": 1,
                                    "pairs": [{"slices": [{"reason": "PermissionError: WinError 5 ACL"}]}],
                                },
                            ],
                        }
                    }
                ),
                encoding="utf-8",
            )
            control_dir = root / "controls" / "negative_control"
            control_dir.mkdir(parents=True)
            (control_dir / "gate.json").write_text(
                json.dumps(
                    {
                        "mode": "negative",
                        "permutation_mode": "smoke",
                        "gate": {
                            "status": "PASS",
                            "name": "negative_or_shuffle",
                            "rows": [{"paired_swap_status": "complete", "seed": 73}],
                        },
                        "retrained_permutations": {
                            "status": "incomplete",
                            "completed": 0,
                            "requested": 0,
                            "unit": "whole_pair_label_swap",
                        },
                    }
                ),
                encoding="utf-8",
            )
            output = root / "report"
            build_report([root, stage1, parity], output, config=ReportConfig(bootstrap_replicates=10), make_plots=False)
            markdown = (output / "report.md").read_text(encoding="utf-8")
            manifest = json.loads((output / "report_manifest.json").read_text(encoding="utf-8"))
            self.assertIn("FOMO healthy participants=240, matched pairs=240", markdown)
            self.assertIn("FOMO PASS (3 slices, max_abs_error=0)", markdown)
            self.assertIn("MPI INCOMPLETE (ACL/PermissionError (WinError 5))", markdown)
            self.assertIn("combined control gate", markdown)
            self.assertIn("same_cohort_negative (gate=negative_or_shuffle)", markdown)
            self.assertIn("heldout pair-swap sanity (fixed-classifier association): **complete**", markdown)
            self.assertIn("full-retrained paired-label null (conditional matched sample): **incomplete**；permutation_mode=smoke", markdown)
            self.assertNotIn("unconditional C2ST control", markdown)
            self.assertEqual(manifest["stage1"]["pair_count"], 240)
            self.assertEqual(manifest["parity"]["cohorts"][0]["actual_slice_count"], 3)

    def test_v3_build_summary_is_build_only_and_reports_candidate_scope(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "model_grid_v3"
            audits = root / "audits"
            audits.mkdir(parents=True)
            comparisons = {}
            for cohort, records, pairs in (
                ("fomo45k", 5422, 240),
                ("mpi", 2500, 115),
                ("oasis3", 5388, 240),
                ("mixed", 5404, 240),
            ):
                audit_path = audits / f"{cohort}.json"
                audit = {
                    "comparison": cohort,
                    "canonical_parity": {
                        "status": "PASS",
                        "checks": {
                            "canonical_dtype": "float32",
                            "canonical_shape": [3, 128, 128],
                            "normalization": "robust_iqr",
                        },
                    },
                    "historical_control_reuse": {
                        "brats_candidate_tensors": "historical selected candidate source",
                        "old_fomo_pair_assignments_reused": False,
                        "results_reused": False,
                        "selection_equivalence_proof": False,
                    },
                    "pair_z_histogram": {"status": "PASS", "pair_count": pairs},
                }
                if cohort == "mixed":
                    audit["eligibility_composition"] = {
                        "by_underlying_cohort_split": {
                            "fomo45k:train": {
                                "eligible_participants": 168,
                                "selected_participants": 70,
                                "selected_records": 812,
                            },
                            "fomo45k:val": {
                                "eligible_participants": 36,
                                "selected_participants": 14,
                                "selected_records": 137,
                            },
                            "fomo45k:test": {
                                "eligible_participants": 36,
                                "selected_participants": 14,
                                "selected_records": 157,
                            },
                        }
                    }
                audit_path.write_text(json.dumps(audit), encoding="utf-8")
                comparisons[cohort] = {
                    "audit": str(audit_path),
                    "domain_counts": {cohort: records // 2, "brats21": records // 2},
                    "records": records,
                    "split_counts": {"train": records // 2, "val": 0, "test": 0},
                    "status": "PASS",
                }
            (root / "build_summary.json").write_text(
                json.dumps(
                    {
                        "schema_version": 3,
                        "status": "PASS",
                        "elapsed_seconds": 200.19,
                        "training_started": False,
                        "no_training": True,
                        "comparisons": comparisons,
                    }
                ),
                encoding="utf-8",
            )
            output = root / "report"
            result = build_report(root, output, config=ReportConfig(bootstrap_replicates=10), make_plots=False)
            manifest = json.loads((output / "report_manifest.json").read_text(encoding="utf-8"))
            markdown = (output / "report.md").read_text(encoding="utf-8")
            self.assertEqual(result["rows"], [])
            self.assertEqual(manifest["formal_final_rows"], 0)
            model_grid = manifest["model_grid_v3"]
            self.assertEqual(model_grid["classification"], "BUILD_ONLY")
            self.assertEqual(model_grid["scope_status"], "DESIGN_AUDIT_PENDING")
            self.assertEqual(model_grid["candidate_coverage_status"], "AUDIT_PENDING")
            self.assertEqual(model_grid["comparisons"][0]["audit"]["pair_count"], 240)
            self.assertIn("v3 model-grid build (build-only)", markdown)
            self.assertIn("FOMO", markdown)
            self.assertIn("selected healthy source composition", markdown)
            self.assertIn("Mixed source selection", markdown)
            self.assertIn("exhaustive tensor parity", markdown)
            self.assertNotIn("5422 |", (output / "summary.csv").read_text(encoding="utf-8"))

    def test_v3_full_candidate_coverage_passes_without_selection_equivalence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "model_grid_v3"
            audits = root / "audits"
            audits.mkdir(parents=True)
            audit_path = audits / "fomo45k.json"
            audit_path.write_text(
                json.dumps(
                    {
                        "canonical_parity": {"status": "PASS"},
                        "pair_z_histogram": {"status": "PASS", "pair_count": 240},
                        "historical_control_reuse": {
                            "selection_equivalence_proof": False,
                            "results_reused": False,
                        },
                        "full_candidate_coverage": {
                            "source_full_inventory_verified": True,
                            "source_csv_count": 2727,
                            "source_map_count": 2727,
                            "source_processed_count": 2727,
                            "status": "PASS",
                        },
                    }
                ),
                encoding="utf-8",
            )
            (root / "build_summary.json").write_text(
                json.dumps(
                    {
                        "schema_version": 3,
                        "status": "PASS",
                        "training_started": False,
                        "comparisons": {
                            "fomo45k": {
                                "audit": str(audit_path),
                                "records": 5422,
                                "domain_counts": {"brats21": 2711, "fomo45k": 2711},
                                "split_counts": {"train": 3806, "val": 768, "test": 848},
                                "status": "PASS",
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            output = root / "report"
            build_report(root, output, config=ReportConfig(bootstrap_replicates=10), make_plots=False)
            manifest = json.loads((output / "report_manifest.json").read_text(encoding="utf-8"))
            audit_summary = manifest["model_grid_v3"]["comparisons"][0]["audit"]
            self.assertEqual(manifest["model_grid_v3"]["candidate_coverage_status"], "PASS")
            self.assertEqual(manifest["model_grid_v3"]["scope_status"], "CANDIDATE_COVERAGE_VERIFIED_BUILD_ONLY")
            self.assertFalse(audit_summary["selection_equivalence_proof"])
            self.assertEqual(audit_summary["controls_reuse_status"], "NOT_ALLOWED")
            self.assertEqual(audit_summary["candidate_coverage_status"], "PASS")
            self.assertTrue(audit_summary["source_full_inventory_verified"])
            self.assertEqual(audit_summary["candidate_inventory_counts"]["source_processed_count"], 2727)

    def test_v3_subject_roster_export_is_split_disjoint_and_source_fingerprinted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "model_grid_v3"
            for comparison in ("fomo45k", "mpi", "oasis3", "mixed"):
                for split in ("train", "val", "test"):
                    path = root / "manifests" / comparison / f"{split}.jsonl"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    pair_id = f"{comparison}:{split}:pair"
                    healthy = {
                        "participant_id": f"{comparison}:healthy:{split}",
                        "domain": comparison,
                        "label": 0,
                        "pair_id": pair_id,
                        "source_dataset": comparison if comparison != "mixed" else "mixed",
                        "source_split": split,
                        "source_key": "1",
                        "z": 10,
                        "z_bin": 1,
                        "case_id": "case-h",
                        "session_id": "ses-h",
                        "metadata": {
                            "lmdb_path": f"{comparison}.lmdb",
                            **(
                                {
                                    "underlying_source_dataset": "fomo45k",
                                    "underlying_source_split": split,
                                    "underlying_source_key": "2",
                                }
                                if comparison == "mixed"
                                else {}
                            ),
                        },
                    }
                    brats = {
                        "participant_id": f"brats21:{split}",
                        "domain": "brats21",
                        "label": 1,
                        "pair_id": pair_id,
                        "source_dataset": "brats21",
                        "source_split": split,
                        "source_key": "3",
                        "z": 10,
                        "z_bin": 1,
                        "case_id": "case-b",
                        "session_id": "ses-b",
                        "metadata": {"lmdb_path": "brats.lmdb"},
                    }
                    path.write_text(json.dumps(healthy) + "\n" + json.dumps(brats) + "\n", encoding="utf-8")
            output = Path(directory) / "report" / "v3_subject_rosters"
            result = export_model_grid_v3_rosters(root, output)
            self.assertEqual(result["status"], "PASS")
            self.assertTrue((output / "train_subjects.csv").is_file())
            self.assertTrue((output / "mixed" / "test_subjects.csv").is_file())
            self.assertTrue((output / "subject_roster_manifest.json").is_file())
            audit = json.loads((output / "subject_roster_audit.json").read_text(encoding="utf-8"))
            self.assertEqual(audit["status"], "PASS")
            self.assertEqual(audit["comparison_overlap"]["fomo45k"]["participant_overlap"]["train:val"], [])
            with (output / "mixed" / "train_subjects.csv").open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            healthy_row = next(row for row in rows if row["label"] == "0")
            self.assertEqual(healthy_row["selected_slice_count"], "1")
            self.assertIn("fomo45k", healthy_row["underlying_source_datasets"])
            with self.assertRaises(ValueError):
                export_model_grid_v3_rosters(root, root / "rosters_inside_frozen_root")

    def test_build_report_passes_revision_directory_to_roster_export(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "model_grid_v3_fullcandidate_20260917_final"
            for comparison in ("fomo45k", "mpi", "oasis3", "mixed"):
                manifest_dir = root / "manifests" / comparison
                manifest_dir.mkdir(parents=True)
                for split in ("train", "val", "test"):
                    (manifest_dir / f"{split}.jsonl").write_text("", encoding="utf-8")
            (root / "build_summary.json").write_text(
                json.dumps({"schema_version": 3, "status": "PASS", "comparisons": {}}), encoding="utf-8"
            )
            output = Path(directory) / "report"
            build_report(root / "build_summary.json", output, config=ReportConfig(bootstrap_replicates=10), make_plots=False)
            manifest = json.loads((output / "report_manifest.json").read_text(encoding="utf-8"))
            roster = manifest["model_grid_v3"]["subject_rosters"]
            self.assertEqual(roster["status"], "PASS")
            self.assertTrue((output / "v3_subject_rosters" / "subject_roster_audit.json").is_file())

    def test_v3_superseded_pretrain_pool_marker_blocks_candidate_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "model_grid_v3"
            audits = root / "audits"
            audits.mkdir(parents=True)
            audit_path = audits / "fomo45k.json"
            audit_path.write_text(
                json.dumps(
                    {
                        "canonical_parity": {"status": "PASS"},
                        "pair_z_histogram": {"status": "PASS", "pair_count": 240},
                        "candidate_pool_audit": {"scope_status": "SUPERSEDED_PRETRAIN_SELECTION_BUG"},
                    }
                ),
                encoding="utf-8",
            )
            (root / "build_summary.json").write_text(
                json.dumps(
                    {
                        "schema_version": 3,
                        "status": "PASS",
                        "comparisons": {
                            "fomo45k": {
                                "audit": str(audit_path),
                                "records": 1,
                                "domain_counts": {"brats21": 1, "fomo45k": 1},
                                "split_counts": {"train": 1},
                                "status": "PASS",
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            output = root / "report"
            build_report(root, output, config=ReportConfig(bootstrap_replicates=10), make_plots=False)
            manifest = json.loads((output / "report_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["model_grid_v3"]["candidate_coverage_status"], "BLOCKED_SUPERSEDED_PRETRAIN_SELECTION_BUG")
            self.assertEqual(manifest["model_grid_v3"]["scope_status"], "BLOCKED_SUPERSEDED_PRETRAIN_SELECTION_BUG")

    def test_timestamped_v3_revision_is_read_only_when_explicitly_supplied(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "model_grid_v3_fullcandidate_20260917"
            root.mkdir(parents=True)
            (root / "build_summary.json").write_text(
                json.dumps({"schema_version": 3, "status": "RUNNING", "comparisons": {}}),
                encoding="utf-8",
            )
            output = root / "report"
            build_report(root, output, config=ReportConfig(bootstrap_replicates=10), make_plots=False)
            manifest = json.loads((output / "report_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["model_grid_v3"]["status"], "RUNNING")
            self.assertEqual(manifest["model_grid_v3"]["comparisons"], [])

    def test_v3_final_revision_propagates_build_coverage_and_execution_sidecars(self) -> None:
        """Build evidence must remain separate from fits and tensor parity."""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "model_grid_v3_fullcandidate_20260917_final"
            root.mkdir(parents=True)
            audit_path = root / "audit_fomo.json"
            audit_path.write_text(
                json.dumps(
                    {
                        "canonical_parity": {"status": "PASS"},
                        "pair_z_histogram": {"status": "PASS", "pair_count": 240},
                        "historical_control_reuse": {
                            "selection_equivalence_proof": False,
                            "results_reused": False,
                        },
                    }
                ),
                encoding="utf-8",
            )
            (root / "build_summary.json").write_text(
                json.dumps(
                    {
                        "schema_version": 3,
                        "status": "PASS",
                        "training_started": False,
                        "no_training": True,
                        "full_candidate_coverage": {
                            "status": "PASS",
                            "verified": True,
                            "csv_rows": 938,
                            "processed_subjects": 938,
                            "candidate_rows": 38967,
                            "old_selected_pool_used_as_source": False,
                        },
                        "comparisons": {
                            "fomo45k": {
                                "audit": str(audit_path),
                                "records": 5896,
                                "domain_counts": {"fomo45k": 2948, "brats21": 2948},
                                "split_counts": {"train": 4206, "val": 804, "test": 886},
                                "status": "PASS",
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            (root / "stage_a_commands.json").write_text(
                json.dumps({"status": "PARTIAL_STAGE_A_TINY_RUNNING", "training_started": True}),
                encoding="utf-8",
            )
            (root / "training_protocol.json").write_text(
                json.dumps({"status": "FROZEN_PRETRAIN_REVIEW_PENDING", "training_started": False}),
                encoding="utf-8",
            )
            (root / "final_external_audit.json").write_text(
                json.dumps(
                    {
                        "status": "FAIL",
                        "failures": [{"reason": "candidate_rows_do_not_match_support_counts"}],
                        "registered_materialization": {
                            "status": "PASS",
                            "protocol": "registered_reader_v1",
                            "manifest_root": str(root / "registered"),
                            "total_records": 5896,
                            "manifest_sha256": {"train": "abc"},
                            "split_audits": {"train": {"failure_count": 0}},
                        },
                    }
                ),
                encoding="utf-8",
            )
            output = root / "report"
            build_report(root, output, config=ReportConfig(bootstrap_replicates=10), make_plots=False)
            manifest = json.loads((output / "report_manifest.json").read_text(encoding="utf-8"))
            markdown = (output / "report.md").read_text(encoding="utf-8")
            model_grid = manifest["model_grid_v3"]
            audit = model_grid["comparisons"][0]["audit"]
            self.assertEqual(model_grid["candidate_coverage_status"], "PASS")
            self.assertEqual(model_grid["scope_status"], "CANDIDATE_COVERAGE_VERIFIED_BUILD_ONLY")
            self.assertEqual(audit["candidate_coverage_status"], "PASS")
            self.assertFalse(audit["selection_equivalence_proof"])
            self.assertEqual(model_grid["candidate_inventory_counts"]["candidate_rows"], 38967)
            self.assertEqual(model_grid["stage_a_status"], "PARTIAL_STAGE_A_TINY_RUNNING")
            self.assertEqual(model_grid["tiny_gate_status"], "NOT_RECORDED_AS_OF_REPORT_GENERATION")
            self.assertEqual(model_grid["formal_fit_count"], 0)
            self.assertEqual(model_grid["registered_materialization"]["status"], "PASS")
            self.assertEqual(model_grid["independent_full_mixed_review_status"], "PENDING")
            self.assertEqual(model_grid["external_audit_status"], "FAIL")
            self.assertTrue(manifest["generated_at_utc"])
            audit_payload = json.loads((output / "report_audit.json").read_text(encoding="utf-8"))
            self.assertEqual(audit_payload["generated_at_utc"], manifest["generated_at_utc"])
            self.assertIn("registered materialization=**PASS**", markdown)
            self.assertIn("confirmatory fit count=**0**", markdown)
            self.assertIn("candidate_rows_do_not_match_support_counts", markdown)

    def test_v3_report_does_not_promote_build_counts_to_formal_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "model_grid_v3_fullcandidate_20260917_final"
            root.mkdir(parents=True)
            (root / "build_summary.json").write_text(
                json.dumps(
                    {
                        "status": "PASS",
                        "no_training": True,
                        "training_started": False,
                        "full_candidate_coverage": {"status": "PASS", "verified": True, "candidate_rows": 3},
                        "comparisons": {},
                    }
                ),
                encoding="utf-8",
            )
            output = root / "report"
            result = build_report(root, output, config=ReportConfig(bootstrap_replicates=10), make_plots=False)
            manifest = json.loads((output / "report_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(result["rows"], [])
            self.assertEqual(manifest["formal_final_rows"], 0)
            self.assertEqual(manifest["model_grid_v3"]["formal_fit_count"], 0)
            self.assertEqual(manifest["model_grid_v3"]["candidate_coverage_status"], "AUDIT_PENDING")

    def test_v3_observed_fit_is_separate_from_legacy_calibration_and_pending_null(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "model_grid_v3_fullcandidate_20260917_final"
            observed_root = root / "stage_a_fomo45k_observed_v3_20260917"
            observed_root.joinpath("observed").mkdir(parents=True)
            observed_root.joinpath("retrained_null").mkdir()
            (root / "build_summary.json").write_text(
                json.dumps({"schema_version": 3, "status": "PASS", "comparisons": {}}),
                encoding="utf-8",
            )
            (observed_root / "protocol.json").write_text(
                json.dumps({"protocol": "v3_stage_a_observed", "status": "PRE_REGISTERED"}),
                encoding="utf-8",
            )
            (observed_root / "observed" / "result.json").write_text(
                json.dumps(
                    {
                        "protocol": "v3_stage_a_observed",
                        "seed": 73,
                        "split_seed": 73,
                        "best_epoch": 40,
                        "epochs_completed": 40,
                        "config": {"model": "small_cnn", "modalities": ["flair", "t1", "t2"], "permutation_replicates": 199},
                        "test": {
                            "slice": {"roc_auc": 0.9982, "n": 886},
                            "subject": {"roc_auc": 1.0, "n": 72},
                        },
                        "validation": {"subject": {"roc_auc": 1.0}},
                        "test_statistics": {
                            "subject_bootstrap": {"ci_low": 1.0, "ci_high": 1.0, "n_bootstrap": 2000, "n_valid": 2000, "resampling_unit": "subject"},
                            "heldout_pair_swap": {"status": "COMPLETE", "p_value": 0.001, "n_swaps": 1000},
                        },
                    }
                ),
                encoding="utf-8",
            )
            (observed_root / "retrained_null" / "status.json").write_text(
                json.dumps({"status": "incomplete", "completed": 2, "requested": 199}), encoding="utf-8"
            )
            (observed_root / "input_binding_audit.json").write_text(
                json.dumps(
                    {
                        "status": "PASS",
                        "observed_rows_checked": 4,
                        "braTS_rows_checked_through_canonical_reader": 2,
                        "shape_counts": {"[3, 128, 128]": 4},
                        "dtype_counts": {"torch.float32": 4},
                        "digest_equal": True,
                        "observed_source_cache_digest": {"rows": 4, "sha256": "cache-sha"},
                        "recomputed_source_cache_digest": {"rows": 4, "sha256": "cache-sha"},
                        "frozen_healthy_ledger": {"sha256": "ledger-sha", "healthy_ledger_rows": 4},
                        "errors": [],
                    }
                ),
                encoding="utf-8",
            )
            (observed_root / "immutable_test_label_sanity").mkdir()
            (observed_root / "immutable_test_label_sanity" / "result.json").write_text(
                json.dumps(
                    {
                        "status": "COMPLETE",
                        "probabilities_immutable": True,
                        "true_test_labels_preserved": True,
                        "random_test_label_gate": {
                            "status": "INCONCLUSIVE",
                            "passed": False,
                            "closeness_status": "INCONCLUSIVE",
                            "reason": "no narrow band",
                            "rows": [
                                {"seed": 10073, "subject_auc": 0.55, "bootstrap_ci_low": 0.4, "bootstrap_ci_high": 0.7},
                            ],
                        },
                    }
                ),
                encoding="utf-8",
            )
            output = Path(directory) / "report"
            build_report(root, output, config=ReportConfig(bootstrap_replicates=10), make_plots=False)
            manifest = json.loads((output / "report_manifest.json").read_text(encoding="utf-8"))
            markdown = (output / "report.md").read_text(encoding="utf-8")
            observed = manifest["model_grid_v3"]["v3_observed"]
            self.assertEqual(observed["classification"], "V3_STAGE_A_OBSERVED_EXCLUDED_FROM_CONFIRMATORY")
            self.assertEqual(observed["subject_auc"], 1.0)
            self.assertEqual(observed["test_subject_count"], 72)
            self.assertEqual(observed["full_retrained_null"]["status"], "INCOMPLETE")
            self.assertEqual(observed["full_retrained_null"]["completed"], 2)
            self.assertEqual(observed["input_binding"]["status"], "PASS")
            self.assertEqual(observed["input_binding"]["observed_rows_checked"], 4)
            self.assertEqual(observed["immutable_test_label_sanity"]["status"], "COMPLETE")
            self.assertEqual(observed["immutable_test_label_sanity"]["gate_status"], "INCONCLUSIVE")
            self.assertEqual(observed["immutable_test_label_sanity"]["rows"][0]["seed"], 10073)
            self.assertNotIn("random_test_label_results", json.dumps(observed))
            self.assertEqual(manifest["formal_final_rows"], 0)
            self.assertEqual(manifest["v3_primary_observed_fit_count"], 1)
            self.assertEqual(manifest["v3_confirmatory_fit_count"], 0)
            self.assertEqual(manifest["v3_stage_a_gate_status"], "INCONCLUSIVE_CONTROLS")
            self.assertIn("v3 FOMO Stage-A observed fit", markdown)
            self.assertIn("72 個 held-out subjects", markdown)
            self.assertIn("Input binding audit: status=**PASS**", markdown)
            self.assertIn("random-label gate=**INCONCLUSIVE**", markdown)

    def test_v3_prefers_corrected_external_audit_and_keeps_superseded_v1(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "model_grid_v3_fullcandidate_20260917_final"
            audits = root / "audits"
            audits.mkdir(parents=True)
            mixed_audit = audits / "mixed.json"
            mixed_audit.write_text(
                json.dumps(
                    {
                        "canonical_parity": {"status": "PASS"},
                        "pair_z_histogram": {"status": "PASS", "pair_count": 240},
                        "mixed_tensor_parity": {
                            "status": "PASS",
                            "selected_records": 4,
                            "checked_records": 4,
                            "mismatch_count": 0,
                            "aggregate_sha256": "mixed-sha",
                        },
                        "brats_selected_tensor_cache": {"status": "PASS", "participants": 1, "records": 4, "cache_file_count": 1},
                        "standalone_map_consistency": {"status": "PASS", "failure_count": 0},
                        "mixed_source_join": {"status": "PASS", "failure_count": 0},
                    }
                ),
                encoding="utf-8",
            )
            (root / "build_summary.json").write_text(
                json.dumps(
                    {
                        "schema_version": 3,
                        "status": "PASS",
                        "training_started": False,
                        "no_training": True,
                        "full_candidate_coverage": {"status": "PASS", "verified": True, "csv_rows": 938, "processed_subjects": 938, "candidate_rows": 38967},
                        "comparisons": {
                            "mixed": {
                                "audit": str(mixed_audit),
                                "records": 15110,
                                "domain_counts": {"mixed": 7555, "brats21": 7555},
                                "split_counts": {"train": 10000, "val": 2500, "test": 2610},
                                "status": "PASS",
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            v1 = root / "final_external_audit.json"
            v2 = root / "final_external_audit_v2.json"
            v1.write_text(json.dumps({"status": "FAIL", "failures": ["candidate_rows_do_not_match_support_counts"]}), encoding="utf-8")
            v2.write_text(
                json.dumps(
                    {
                        "status": "PASS",
                        "failures": [],
                        "comparisons": {"mixed": {"status": "PASS", "mixed_tensor_parity": "PASS", "records": 15110}},
                    }
                ),
                encoding="utf-8",
            )
            # Make the version ordering explicit so the test does not depend
            # on filesystem timestamp resolution.
            os.utime(v1, ns=(1_000_000_000, 1_000_000_000))
            os.utime(v2, ns=(2_000_000_000, 2_000_000_000))
            output = root / "report"
            build_report(root, output, config=ReportConfig(bootstrap_replicates=10), make_plots=False)
            manifest = json.loads((output / "report_manifest.json").read_text(encoding="utf-8"))
            markdown = (output / "report.md").read_text(encoding="utf-8")
            model_grid = manifest["model_grid_v3"]
            self.assertEqual(model_grid["selected_external_audit"]["name"], "final_external_audit_v2.json")
            self.assertEqual(model_grid["external_audit_status"], "PASS")
            self.assertEqual(model_grid["historical_external_audits"]["final_external_audit.json"]["classification"], "SUPERSEDED")
            self.assertEqual(model_grid["independent_full_mixed_review_status"], "PASS")
            self.assertEqual(model_grid["independent_full_mixed_review"]["exact_tensor_records"], 4)
            self.assertIn("Historical external audit (final_external_audit.json) is **SUPERSEDED**", markdown)
            self.assertIn("selected healthy exact tensor records=4", markdown)

    def test_legacy_control_fail_is_separate_from_v3_stage_a_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "model_grid_v3_fullcandidate_20260917_final"
            legacy_root = Path(directory) / "legacy_controls"
            (legacy_root / "controls" / "label_shuffle").mkdir(parents=True)
            (legacy_root / "controls" / "label_shuffle" / "gate.json").write_text(
                json.dumps({"mode": "shuffle", "gate": {"status": "FAIL"}}), encoding="utf-8"
            )
            (root / "stage_a_fomo45k" / "tiny" / "tiny").mkdir(parents=True)
            (root / "stage_a_fomo45k_acl_retry_20260917" / "positive").mkdir(parents=True)
            (root / "stage_a_fomo45k_negative_v3_20260917" / "negative").mkdir(parents=True)
            for relative in (
                "stage_a_fomo45k/tiny/tiny/gate.json",
                "stage_a_fomo45k_acl_retry_20260917/positive/gate.json",
                "stage_a_fomo45k_negative_v3_20260917/negative/gate.json",
            ):
                (root / relative).write_text(json.dumps({"gate": {"status": "PASS"}}), encoding="utf-8")
            observed_root = root / "stage_a_fomo45k_observed_v3_20260917"
            (observed_root / "observed").mkdir(parents=True)
            (observed_root / "retrained_null").mkdir()
            (observed_root / "observed" / "result.json").write_text(
                json.dumps(
                    {
                        "seed": 73,
                        "config": {"model": "small_cnn", "modalities": ["flair", "t1", "t2"]},
                        "test": {"slice": {"roc_auc": 0.9, "n": 2}, "subject": {"roc_auc": 1.0, "n": 2}},
                    }
                ),
                encoding="utf-8",
            )
            (observed_root / "retrained_null" / "status.json").write_text(
                json.dumps({"status": "incomplete", "completed": 0, "requested": 199}), encoding="utf-8"
            )
            (root / "build_summary.json").write_text(
                json.dumps({"schema_version": 3, "status": "PASS", "comparisons": {}}), encoding="utf-8"
            )
            output = Path(directory) / "report"
            build_report([legacy_root, root], output, config=ReportConfig(bootstrap_replicates=10), make_plots=False)
            manifest = json.loads((output / "report_manifest.json").read_text(encoding="utf-8"))
            markdown = (output / "report.md").read_text(encoding="utf-8")
            self.assertEqual(manifest["legacy_rollout_gate_status"], "HALTED_CONTROL_FAIL")
            self.assertEqual(manifest["v3_stage_a_gate_status"], "CONTROLS_PASS_OBSERVED_NULL_PENDING")
            self.assertEqual(manifest["rollout_gate_status"], "CONTROLS_PASS_OBSERVED_NULL_PENDING")
            self.assertIn("Legacy rollout gate", markdown)
            self.assertIn("v3 Stage-A gate", markdown)

    def test_failed_shuffle_halts_rollout_without_claiming_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            control_dir = root / "controls" / "label_shuffle"
            control_dir.mkdir(parents=True)
            (control_dir / "gate.json").write_text(
                json.dumps(
                    {
                        "mode": "shuffle",
                        "permutation_mode": "smoke",
                        "gate": {
                            "status": "FAIL",
                            "name": "negative_or_shuffle",
                            "failure_p_value_seeds": [273],
                            "holm_adjusted_p_values": {"273": 0.002997},
                            "rows": [{"paired_swap_status": "complete", "seed": 273}],
                        },
                    }
                ),
                encoding="utf-8",
            )
            output = root / "report"
            build_report(root, output, config=ReportConfig(bootstrap_replicates=10), make_plots=False)
            markdown = (output / "report.md").read_text(encoding="utf-8")
            manifest = json.loads((output / "report_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["formal_final_rows"], 0)
            self.assertEqual(manifest["rollout_gate_status"], "HALTED_CONTROL_FAIL")
            self.assertIn("label_shuffle (gate=negative_or_shuffle)", markdown)
            self.assertIn("rollout halted by control FAIL", markdown)
            self.assertIn("不單獨證明 information leakage", markdown)

    def test_shuffle_diagnostic_addendum_is_compact_and_hashed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audit = {
                "status": "diagnostic_only_gate_unchanged",
                "base_key_count": 5454,
                "duplicate_base_keys": [],
                "tensor_cache_note": "one-time exact loader output; per-row tensor hashes not persisted",
                "saved_model_orientation": {
                    "status": "diagnostic_only_no_retraining",
                    "cache_tensor_digest": {"rows": 5454, "sha256": "cache-digest"},
                },
                "seed_reports": {
                    "273": {
                        "split_integrity": {
                            "invariant_field_mismatches": 0,
                            "missing_base_keys": 0,
                            "pair_id_mismatches": 0,
                            "test_true_label_mismatches": 0,
                            "unexpected_seed_keys": 0,
                            "duplicate_cache_keys": {},
                            "participant_overlap": {"train_val": [], "train_test": [], "val_test": []},
                            "pair_overlap": {"train_val": [], "train_test": [], "val_test": []},
                        },
                        "prediction_join_and_orientation": {
                            "train": {"join_error_count": 0},
                            "val": {"join_error_count": 0},
                            "test": {"join_error_count": 0},
                        },
                        "contingency": {
                            "train": {"subjects": {"brats21": {"0": 54, "1": 42}, "fomo45k": {"0": 42, "1": 54}}},
                            "val": {"subjects": {"brats21": {"0": 17, "1": 7}, "fomo45k": {"0": 7, "1": 17}}},
                            "test": {"n_slices": 2842, "subjects": {"brats21": {"0": 0, "1": 120}, "fomo45k": {"0": 120, "1": 0}}},
                        },
                    }
                },
            }
            (root / "shuffle_diagnostic_audit.json").write_text(json.dumps(audit), encoding="utf-8")
            (root / "shuffle_diagnostic_audit.md").write_text("diagnostic addendum", encoding="utf-8")
            orientation = {
                "status": "diagnostic_only_no_retraining",
                "seeds": {
                    "273": {
                        "train": {"subject_true_auc": 0.331814, "subject_pseudo_auc": 0.932834},
                        "val": {"subject_true_auc": 0.385416, "subject_pseudo_auc": 0.491319},
                        "test": {"subject_true_auc": 0.303681, "subject_pseudo_auc": 0.303681},
                    }
                },
            }
            (root / "shuffle_saved_model_orientation.json").write_text(json.dumps(orientation), encoding="utf-8")
            output = root / "report"
            build_report(root, output, config=ReportConfig(bootstrap_replicates=10), make_plots=False)
            markdown = (output / "report.md").read_text(encoding="utf-8")
            manifest = json.loads((output / "report_manifest.json").read_text(encoding="utf-8"))
            diagnostic = manifest["shuffle_diagnostic"]
            self.assertEqual(diagnostic["status"], "diagnostic_only_gate_unchanged")
            self.assertEqual(diagnostic["integrity_issue_count"], 0)
            self.assertEqual(diagnostic["v1_test_slice_counts"], {"273": 2842})
            self.assertEqual(diagnostic["v1_test_pair_counts"], {"273": 120})
            self.assertEqual(len(manifest["referenced_files"]["shuffle_diagnostic"]), 3)
            self.assertIn("seed273 true-label subject AUC train/val/test=0.3318/0.3854/0.3037", markdown)
            self.assertIn("not uniquely proven cause", markdown)
            self.assertIn("status=**PENDING**", markdown)
            self.assertIn("cache-digest", json.dumps(diagnostic))

    def test_protocol_v2_gates_are_diagnostic_and_not_formal_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "protocol.json").write_text(
                json.dumps(
                    {
                        "protocol": "balanced_powered_shuffle_v2",
                        "status": "PRE_REGISTERED",
                        "split_seed": 73,
                        "fit_seeds": [73],
                        "retrained_permutation_null": "not_run",
                    }
                ),
                encoding="utf-8",
            )
            (root / "preflight.json").write_text(
                json.dumps(
                    {
                        "status": "PASS",
                        "same_target_key_order_across_fit_seeds": True,
                        "same_target_pair_order_across_fit_seeds": True,
                        "same_true_test_labels_across_fit_seeds": True,
                    }
                ),
                encoding="utf-8",
            )

            def gate(status: str, auc: float) -> dict[str, object]:
                return {
                    "status": status,
                    "passed": status == "PASS",
                    "closeness_status": "PASS" if status == "PASS" else "INCONCLUSIVE",
                    "rows": [
                        {
                            "seed": 73,
                            "subject_auc": auc,
                            "bootstrap_ci": [0.3, 0.6],
                            "paired_swap_p": 0.01,
                            "holm_p": 0.02,
                        }
                    ],
                }

            (root / "gate_true_test.json").write_text(json.dumps(gate("FAIL", 0.38)), encoding="utf-8")
            (root / "gate_random_test_labels.json").write_text(json.dumps(gate("PASS", 0.50)), encoding="utf-8")
            (root / "gate.json").write_text(json.dumps({"protocol": "balanced_powered_shuffle_v2"}), encoding="utf-8")
            (root / "supplementary_diagnostic_summary.json").write_text(
                json.dumps({"protocol": "supplementary_balanced_v2", "interpretation": "diagnostic only; not confirmatory"}),
                encoding="utf-8",
            )
            (root / "supplementary_diagnostic_summary.md").write_text("diagnostic", encoding="utf-8")
            seed_dir = root / "seed_73"
            seed_dir.mkdir()
            (seed_dir / "result.json").write_text(
                json.dumps(
                    {
                        "seed": 73,
                        "config": {"model": "small_cnn", "modalities": ["flair", "t1", "t2"], "stage": "final"},
                        "test_predictions": [
                            {"label": 0, "probability": 0.4, "participant_id": "h1"},
                            {"label": 1, "probability": 0.6, "participant_id": "b1"},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            output = root / "report"
            result = build_report(root, output, config=ReportConfig(bootstrap_replicates=10), make_plots=False)
            manifest = json.loads((output / "report_manifest.json").read_text(encoding="utf-8"))
            markdown = (output / "report.md").read_text(encoding="utf-8")
            self.assertEqual(manifest["formal_final_rows"], 0)
            self.assertEqual(manifest["protocol_v2"]["true_test_gate"]["status"], "FAIL")
            self.assertEqual(manifest["protocol_v2"]["random_test_label_gate"]["status"], "PASS")
            self.assertTrue(any(row["result_kind"] == "diagnostic_formalfit" for row in result["rows"]))
            self.assertIn("protocol v2 gate", markdown)
            self.assertIn("true test labels retained | FAIL", markdown)
            self.assertIn("independent random held-out test labels | PASS", markdown)
            self.assertIn("classification=EXCLUDED_DIAGNOSTIC", markdown)

    def test_observed_calibration_is_separate_and_null_status_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "calibration"
            (root / "observed").mkdir(parents=True)
            (root / "retrained_null").mkdir()
            (root / "protocol.json").write_text(
                json.dumps({"protocol": "fomo45k_calibration_smallcnn_full_retrained_pair_null"}),
                encoding="utf-8",
            )
            (root / "observed" / "result.json").write_text(
                json.dumps(
                    {
                        "protocol": "fomo45k_calibration_smallcnn_full_retrained_pair_null",
                        "run_fingerprint": "calibration-fixture",
                        "seed": 73,
                        "split_seed": 73,
                        "best_epoch": 2,
                        "epochs_completed": 2,
                        "config": {
                            "model": "small_cnn",
                            "modalities": ["flair", "t1", "t2"],
                            "permutation_mode": "full_retrained_pair_swap_all_splits",
                            "permutation_replicates": 199,
                        },
                        "test": {
                            "slice": {"roc_auc": 0.91, "bce": 0.1},
                            "subject": {"roc_auc": 0.8, "bce": 0.2, "accuracy": 0.75},
                        },
                        "validation": {"subject": {"roc_auc": 0.77, "bce": 0.25}},
                        "test_statistics": {
                            "heldout_pair_swap": {
                                "status": "complete",
                                "conditional_on_matched_pairs": True,
                                "n_pairs": 2,
                                "n_swaps": 10,
                                "observed_auc": 0.8,
                                "p_value": 0.09,
                            },
                            "subject_bootstrap": {
                                "ci_low": 0.65,
                                "ci_high": 0.91,
                                "n_bootstrap": 20,
                                "n_valid": 20,
                                "resampling_unit": "matched_pair",
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            (root / "retrained_null" / "status.json").write_text(
                json.dumps({"status": "incomplete", "completed": 4, "requested": 199, "remaining": 195}),
                encoding="utf-8",
            )
            result = build_report(root, root / "report", make_plots=False)
            manifest = json.loads((root / "report" / "report_manifest.json").read_text(encoding="utf-8"))
            markdown = (root / "report" / "report.md").read_text(encoding="utf-8")
            self.assertEqual(result["rows"], [])
            self.assertEqual(manifest["formal_final_rows"], 0)
            self.assertEqual(manifest["calibration"]["classification"], "DIAGNOSTIC_CALIBRATION_EXCLUDED_FROM_CONFIRMATORY")
            self.assertEqual(manifest["calibration"]["full_retrained_null"]["status"], "INCOMPLETE")
            self.assertIn("Observed calibration (diagnostic only)", markdown)
            self.assertIn("0.8000", markdown)
            self.assertIn("INCOMPLETE (4/199)", markdown)

    def test_calibration_plots_are_diagnostic_and_defer_partial_null_histogram(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "calibration"
            observed = root / "observed"
            null_root = root / "retrained_null"
            observed.mkdir(parents=True)
            null_root.mkdir()
            (root / "protocol.json").write_text(json.dumps({"protocol": "calibration"}), encoding="utf-8")
            (observed / "result.json").write_text(
                json.dumps(
                    {
                        "seed": 73,
                        "test_subject_predictions": [
                            {"label": 0, "probability": 0.1},
                            {"label": 0, "probability": 0.2},
                            {"label": 1, "probability": 0.8},
                            {"label": 1, "probability": 0.9},
                        ],
                        "test": {
                            "subject": {"roc_auc": 1.0, "tn": 2, "fp": 0, "fn": 0, "tp": 2},
                        },
                        "history": [
                            {"epoch": 1, "train_loss": 0.4, "val_loss": 0.5, "val_subject": {"roc_auc": 0.8}},
                            {"epoch": 2, "train_loss": 0.1, "val_loss": 0.2, "val_subject": {"roc_auc": 1.0}},
                        ],
                        "test_statistics": {
                            "subject_bootstrap": {
                                "samples": [0.8, 0.9, 1.0, 1.0],
                                "observed_auc": 1.0,
                                "ci_low": 0.8,
                                "ci_high": 1.0,
                                "n_bootstrap": 4,
                                "n_valid": 4,
                                "resampling_unit": "matched_pair",
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            (null_root / "status.json").write_text(
                json.dumps({"status": "incomplete", "completed": 4, "requested": 199}),
                encoding="utf-8",
            )
            output = root / "calibration_figures"
            result = generate_calibration_diagnostic_plots(observed / "result.json", output)
            manifest = json.loads((output / "plot_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["formal_matrix_rows"], 0)
            self.assertEqual(manifest["classification"], "DIAGNOSTIC_CALIBRATION_EXCLUDED_FROM_CONFIRMATORY")
            for name in ("observed_roc", "observed_pr", "observed_confusion", "observed_training_curves", "observed_subject_bootstrap"):
                self.assertEqual(result["plots"][name]["status"], "COMPLETE")
            self.assertEqual(result["plots"]["full_retrained_null"]["status"], "MISSING")
            self.assertFalse((output / "full_retrained_null.png").exists())

            # The terminal branch consumes the planned statistic schema only
            # after all requested retrained null fits are present.
            (null_root / "status.json").write_text(
                json.dumps({"status": "complete", "completed": 199, "requested": 199}),
                encoding="utf-8",
            )
            (null_root / "statistic.json").write_text(
                json.dumps(
                    {
                        "conditional_on_fixed_matched_pairs": True,
                        "label_scope": "train_val_test_complete_pair_swaps",
                        "null_T_values": [0.1] * 199,
                        "observed_T": 0.5,
                        "p_plus_one": 0.005,
                        "statistic": "T=abs(subject_mean_score_roc_auc-0.5)",
                    }
                ),
                encoding="utf-8",
            )
            complete_output = root / "calibration_figures_complete"
            complete_result = generate_calibration_diagnostic_plots(observed / "result.json", complete_output)
            self.assertEqual(complete_result["plots"]["full_retrained_null"]["status"], "COMPLETE")
            self.assertTrue((complete_output / "full_retrained_null.png").is_file())
            complete_manifest = json.loads((complete_output / "plot_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(complete_manifest["null_status"]["minimum_attainable_p_for_199"], 0.005)

    def test_known_andi_paths_are_provenanced_when_present(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "ap.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["version", "AUPRC"])
                writer.writeheader()
                writer.writerow({"version": "raw", "AUPRC": "0.4"})
                writer.writerow({"version": "median_filter", "AUPRC": "0.8"})
            points = load_andi_ap_points({"FOMO": {"mf_ap": 0.8, "path": str(path)}})
            self.assertEqual(points["FOMO"]["mf_ap"], 0.8)

    @staticmethod
    def _secondary_cell(*, seeds=(73, 173, 273), probabilities=None, p_value=0.02, classifier="small_cnn"):
        values = probabilities or {
            73: (0.10, 0.90),
            173: (0.20, 0.80),
            273: (0.30, 0.70),
        }
        seed_results = []
        for seed in seeds:
            h_score, b_score = values[seed]
            seed_results.append(
                {
                    "seed": seed,
                    "result": {
                        "seed": seed,
                        "test_subject_predictions": [
                            {"participant_id": "h1", "label": 0, "probability": h_score, "pair_id": "p1"},
                            {"participant_id": "b1", "label": 1, "probability": b_score, "pair_id": "p1"},
                        ],
                    },
                }
            )
        return {
            "cell_id": "FOMO|FLAIR|small_cnn|test",
            "cohort": "FOMO",
            "modalities": ["flair"],
            "classifier": classifier,
            "evaluation_split": "test",
            "seed_results": seed_results,
            "secondary_p_value": p_value,
        }

    def test_secondary_subject_ensemble_and_disjoint_holm_families(self) -> None:
        cell = self._secondary_cell()
        primary = {
            "cells": [
                {"cell_id": f"primary-{index}", "full_retrained_p_value": value, "p_value_source": "full_retrained_pair_null"}
                for index, value in enumerate((0.001, 0.01, 0.04, 0.2))
            ]
        }
        result = summarise_secondary_48(
            {"family": SECONDARY_48_FAMILY, "cells": [cell], "primary_full_retrained": primary},
            expected_cells=1,
        )
        self.assertEqual(result["status"], "PASS")
        self.assertTrue(result["formal_eligible"])
        self.assertEqual(result["cells"][0]["status"], "PASS")
        self.assertAlmostEqual(result["cells"][0]["metrics"]["subject_roc_auc"], 1.0)
        self.assertEqual(result["holm"]["family"], SECONDARY_48_FAMILY)
        self.assertEqual(result["holm"]["status"], "PASS")
        self.assertAlmostEqual(result["holm"]["adjusted_p_values"][cell["cell_id"]], 0.02)
        self.assertEqual(result["primary_full_retrained"]["family"], PRIMARY_FULL_RETRAINED_HOLM4_FAMILY)
        self.assertEqual(result["primary_full_retrained"]["status"], "PASS")
        self.assertNotIn(cell["cell_id"], result["primary_full_retrained"]["adjusted_p_values"])

    def test_secondary_logistic_uses_only_seed_73(self) -> None:
        cell = self._secondary_cell(seeds=(73,), classifier="statistical_logistic")
        result = aggregate_fixed_seed_subject_ensemble(cell)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["expected_seeds"], [73])
        self.assertEqual(result["seed_results"][0]["seed"], 73)

    def test_secondary_missing_seed_fails_closed_without_metrics(self) -> None:
        cell = self._secondary_cell(seeds=(73, 173))
        result = aggregate_fixed_seed_subject_ensemble(cell)
        self.assertEqual(result["status"], "FAIL_CLOSED")
        self.assertFalse(result["formal_eligible"])
        self.assertIn(273, result["missing_seeds"])
        self.assertNotIn("metrics", result)

    def test_secondary_subject_mismatch_fails_closed_without_metrics(self) -> None:
        cell = self._secondary_cell()
        cell["seed_results"][1]["result"]["test_subject_predictions"][1]["participant_id"] = "b2"
        result = aggregate_fixed_seed_subject_ensemble(cell)
        self.assertEqual(result["status"], "FAIL_CLOSED")
        self.assertFalse(result["formal_eligible"])
        self.assertTrue(any("subject identifier set mismatch" in error for error in result["errors"]))
        self.assertNotIn("metrics", result)

    def test_secondary_holm_excludes_conditional_heldout_swap_p(self) -> None:
        cell = self._secondary_cell()
        cell.pop("secondary_p_value")
        cell["p_value"] = 0.001
        cell["p_value_source"] = "heldout_pair_swap"
        result = summarise_secondary_48(
            {"family": SECONDARY_48_FAMILY, "cells": [cell]},
            expected_cells=1,
        )
        self.assertEqual(result["cells"][0]["status"], "PASS")
        self.assertEqual(result["holm"]["status"], "INCONCLUSIVE")
        self.assertEqual(result["holm"]["adjusted_p_values"], {})
        self.assertEqual(result["holm"]["missing_p_value_cells"], [cell["cell_id"]])


if __name__ == "__main__":
    unittest.main()
