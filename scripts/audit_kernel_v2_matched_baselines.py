#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sphererate.baselines.matched_protocol import load_matched_protocol
from sphererate.utils import ensure_dir


SEEDS = (20260710, 20260711, 20260712)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _check(condition: bool, label: str, checks: dict[str, bool]) -> None:
    checks[label] = bool(condition)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit split/start/source integrity of matched baselines.")
    parser.add_argument("--output-dir", default="reports/kernel_v2_matched_baseline_audit")
    parser.add_argument("--allow-incomplete-scandmm", action="store_true")
    args = parser.parse_args()
    protocol = load_matched_protocol(ROOT / "configs/kernel_v2_matched_baselines.yaml")
    export = _load(ROOT / "data/processed/kernel_v2_matched_scanpaths/audit.json")
    checks: dict[str, bool] = {}
    pending: list[str] = []
    _check(export["content_counts"] == {"train": 50, "val": 9, "gate": 13}, "export_content_counts", checks)
    _check(export["scanpath_counts"] == {"train": 1840, "val": 322, "gate": 467}, "export_scanpath_counts", checks)
    _check(export["audit"]["test_target_rows_loaded"] == 0, "export_test_targets_zero", checks)
    _check(export["initial_gaze"]["yaw_deg"] == 90.0, "export_declared_start", checks)
    ids = {key: set(value) for key, value in export["content_ids"].items()}
    _check(ids["train"].isdisjoint(ids["val"]), "train_val_disjoint", checks)
    _check(ids["train"].isdisjoint(ids["gate"]), "train_gate_disjoint", checks)
    _check(ids["val"].isdisjoint(ids["gate"]), "val_gate_disjoint", checks)

    for seed in SEEDS:
        path_run = ROOT / f"runs/kernel_v2_pathformer3d_retrained_seed{seed}"
        provenance = _load(path_run / "provenance.json")
        selection = _load(path_run / "selection.json")
        _check(provenance["split_counts"] == {"train": 1840, "val": 9, "gate": 13}, f"pathformer_{seed}_counts", checks)
        _check(provenance["test_target_loads"] == 0, f"pathformer_{seed}_test_zero", checks)
        _check(provenance["source_commit"] == protocol.methods["pathformer3d_retrained_v2"].source_commit, f"pathformer_{seed}_source", checks)
        _check(selection["test_target_loads"] == 0, f"pathformer_{seed}_selection_test_zero", checks)
        _check(int(selection["selected"]["epoch"]) in range(0, 41, 5), f"pathformer_{seed}_val_selected", checks)
        for split, expected_count in (("val", 9), ("gate", 13)):
            run = ROOT / f"runs/kernel_v2_pathformer3d_retrained_seed{seed}_{split}"
            metadata = _load(run / "inference_metadata.json")
            paths = list((run / "scanpaths").glob("*.npy"))
            valid = len(paths) == expected_count
            for path in paths:
                values = np.load(path)
                valid &= bool(
                    values.shape == (200, 20, 2)
                    and np.isfinite(values).all()
                    and np.all(values >= 0.0)
                    and np.all(values <= 1.0)
                )
            _check(valid, f"pathformer_{seed}_{split}_normalized_paths", checks)
            _check(metadata["start_mode"] == "declared" and metadata["start_yaw_deg"] == 90.0, f"pathformer_{seed}_{split}_start", checks)

        scan_run = ROOT / f"runs/kernel_v2_scandmm_declared_start_seed{seed}"
        scan_provenance = _load(scan_run / "provenance.json")
        _check(scan_provenance["train_contents"] == 50 and scan_provenance["val_contents"] == 9, f"scandmm_{seed}_counts", checks)
        _check(scan_provenance["gate_used_for_training_or_selection"] is False, f"scandmm_{seed}_gate_not_selected", checks)
        _check(scan_provenance["test_target_rows_loaded"] == 0, f"scandmm_{seed}_test_zero", checks)
        _check(
            scan_provenance["source_commit"]
            == protocol.methods["scandmm_declared_start_v3"].source_commit,
            f"scandmm_{seed}_source",
            checks,
        )
        _check(
            scan_provenance["conditioning_semantics"]["legacy_first_target_as_initializer"]
            is False,
            f"scandmm_{seed}_independent_training_start",
            checks,
        )
        scan_selection = scan_run / "selection.json"
        if scan_selection.exists():
            selected = _load(scan_selection)
            _check(selected["gate_used"] is False, f"scandmm_{seed}_selection_gate_false", checks)
            _check(selected["test_target_rows_loaded"] == 0, f"scandmm_{seed}_selection_test_zero", checks)
            _check(
                selected["initialization_semantics"] == "independent_declared_tensor",
                f"scandmm_{seed}_selection_start_semantics",
                checks,
            )
            for split, expected_count in (("val", 9), ("gate", 13)):
                final = ROOT / f"runs/kernel_v2_scandmm_declared_start_seed{seed}_{split}"
                metadata = _load(final / "inference_metadata.json")
                _check(len(list((final / "scanpaths").glob("*.npy"))) == expected_count, f"scandmm_{seed}_{split}_count", checks)
                _check(metadata["start_mode"] == "declared" and metadata["start_yaw_deg"] == 90.0, f"scandmm_{seed}_{split}_start", checks)
                _check(
                    metadata["initialization_semantics"] == "independent_declared_tensor",
                    f"scandmm_{seed}_{split}_independent_start",
                    checks,
                )
        else:
            pending.append(f"scandmm_{seed}_selection_complete")
            _check(bool(args.allow_incomplete_scandmm), f"scandmm_{seed}_selection_complete", checks)

        hat_run = ROOT / f"runs/kernel_v2_hat_seed{seed}_train"
        hat_provenance = _load(hat_run / "provenance.json")
        hat_selection = _load(hat_run / "selection.json")
        _check(
            hat_provenance["split_counts"] == {"train": 50, "val": 9, "gate": 13},
            f"hat_{seed}_counts",
            checks,
        )
        _check(hat_provenance["test_target_loads"] == 0, f"hat_{seed}_test_zero", checks)
        _check(
            hat_provenance["source_commit"]
            == protocol.methods["hat_erp_declared_start_v1"].source_commit,
            f"hat_{seed}_source",
            checks,
        )
        hat_semantics = hat_provenance["conditioning_semantics"]
        _check(
            hat_semantics["declared_start_is_first_history_token"] is True
            and hat_semantics["declared_start_is_not_a_supervised_target"] is True,
            f"hat_{seed}_independent_training_start",
            checks,
        )
        _check(hat_selection["gate_used"] is False, f"hat_{seed}_selection_gate_false", checks)
        _check(hat_selection["test_target_loads"] == 0, f"hat_{seed}_selection_test_zero", checks)
        _check(
            int(hat_selection["selected"]["step"]) in range(1000, 5001, 1000),
            f"hat_{seed}_val_selected",
            checks,
        )
        _check(
            int(hat_selection["sampling_ior_radius"]) == 0,
            f"hat_{seed}_second_sampled_ior_disabled",
            checks,
        )
        for split, expected_count in (("val", 9), ("gate", 13)):
            final = ROOT / f"runs/kernel_v2_hat_seed{seed}_{split}"
            metadata = _load(final / "inference_metadata.json")
            paths = list((final / "scanpaths").glob("*.npy"))
            valid = len(paths) == expected_count
            copied_start = 0
            total_points = 0
            for path in paths:
                values = np.load(path)
                valid &= bool(
                    values.shape == (200, 20, 2)
                    and np.isfinite(values).all()
                    and np.all(values >= 0.0)
                    and np.all(values <= 1.0)
                )
                copied_start += int(
                    np.count_nonzero(
                        np.isclose(values[..., 0], 0.5, atol=1e-6)
                        & np.isclose(values[..., 1], 0.75, atol=1e-6)
                    )
                )
                total_points += int(values.shape[0] * values.shape[1])
            _check(valid, f"hat_{seed}_{split}_normalized_paths", checks)
            _check(
                metadata["initialization_semantics"] == "declared_start_history_token"
                and metadata["start_yaw_deg"] == 90.0,
                f"hat_{seed}_{split}_start",
                checks,
            )
            _check(metadata["test_target_loads"] == 0, f"hat_{seed}_{split}_test_zero", checks)
            _check(
                int(metadata["sampling_ior_radius"]) == 0,
                f"hat_{seed}_{split}_ior_disabled",
                checks,
            )
            _check(
                total_points > 0 and copied_start / total_points < 0.1,
                f"hat_{seed}_{split}_not_copied_start",
                checks,
            )

    for method in ("bms360", "gbvs360"):
        for split, expected_count in (("val", 9), ("gate", 13)):
            report = _load(ROOT / f"reports/kernel_v2_{method}_{split}/results.json")
            metadata = _load(ROOT / f"runs/kernel_v2_bms_gbvs/{method}/{split}/inference_metadata.json")
            _check(len(report["content_ids"]) == expected_count, f"{method}_{split}_count", checks)
            _check(metadata["test_target_loads"] == 0, f"{method}_{split}_test_zero", checks)
            _check(report["source"]["commit"] == protocol.methods[method].source_commit, f"{method}_{split}_source", checks)

    for source in ("panoramic_cnn", "bms360", "gbvs360"):
        for split in ("val", "gate"):
            reachable = _load(ROOT / f"reports/kernel_v2_reachable_{source}_{split}/results.json")
            calibration = reachable["calibration"]
            _check(calibration["calibration_split"] == "train", f"reachable_{source}_{split}_train_only", checks)
            _check(calibration["quantile"] == 0.95, f"reachable_{source}_{split}_q95", checks)
            _check(calibration["test_target_loads"] == 0, f"reachable_{source}_{split}_test_zero", checks)

    msdb_method = protocol.methods["deepgaze_msdb_erp_v2"]
    msdb_parameters = []
    for split, expected_count in (("train", 50), ("val", 9), ("gate", 13)):
        metadata = _load(ROOT / f"runs/kernel_v2_deepgaze_msdb/raw/{split}/metadata.json")
        _check(metadata["content_count"] == expected_count, f"deepgaze_msdb_{split}_count", checks)
        _check(metadata["source_commit"] == msdb_method.source_commit, f"deepgaze_msdb_{split}_source", checks)
        _check(metadata["training_target_loads"] == 0, f"deepgaze_msdb_{split}_cache_target_zero", checks)
        _check(metadata["test_target_loads"] == 0, f"deepgaze_msdb_{split}_test_zero", checks)
        _check(
            metadata["adapter"]["reference_ppd_is_physical_hmd_measurement"] is False,
            f"deepgaze_msdb_{split}_ppd_caveat",
            checks,
        )
    for split in ("val", "gate"):
        report = _load(ROOT / f"reports/kernel_v2_deepgaze_msdb_{split}/results.json")
        adapter = report["adapter"]
        _check(report["source_commit"] == msdb_method.source_commit, f"deepgaze_msdb_{split}_report_source", checks)
        _check(adapter["calibration_split"] == "train", f"deepgaze_msdb_{split}_train_only", checks)
        _check(adapter["train_content_count"] == 50, f"deepgaze_msdb_{split}_train_count", checks)
        _check(adapter["val_or_gate_used_for_calibration"] is False, f"deepgaze_msdb_{split}_selection_zero", checks)
        _check(report["test_target_loads"] == 0, f"deepgaze_msdb_{split}_report_test_zero", checks)
        msdb_parameters.append(adapter["calibration"])
    _check(msdb_parameters[0] == msdb_parameters[1], "deepgaze_msdb_frozen_calibration", checks)

    failed = sorted(key for key, value in checks.items() if not value)
    caveats: list[str] = []
    motion_path = ROOT / "reports/kernel_v2_scanpath_motion_audit/results.json"
    if motion_path.exists():
        motion = _load(motion_path)
        _check(
            motion["status"] == "MOTION_PASS",
            "scanpath_generated_motion_valid",
            checks,
        )
        _check(
            motion.get("method_roots", {}).get("ScanDMM")
            == "kernel_v2_scandmm_declared_start",
            "scanpath_motion_uses_corrected_scandmm",
            checks,
        )
        failed = sorted(key for key, value in checks.items() if not value)
    result = {
        "protocol": protocol.name,
        "checks": checks,
        "passed": len(checks) - len(failed),
        "total": len(checks),
        "failed": failed,
        "pending": pending,
        "caveats": caveats,
        "status": (
            "FAIL"
            if failed
            else ("INCOMPLETE" if pending else ("PASS_WITH_CAVEAT" if caveats else "PASS"))
        ),
        "test_target_loads": 0,
    }
    output = ensure_dir(args.output_dir)
    (output / "results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    lines = [
        "# Kernel-v2 matched-baseline integrity audit",
        "",
        f"Status: **{result['status']}** ({result['passed']}/{result['total']} checks passed).",
        "",
        "The audit covers content-disjoint train/val/gate roles, zero v2-test target access, source commits, val-only checkpoint selection, declared-start inference inputs, normalized scanpath coordinates, generated-path motion, and train-only reachability calibration.",
        "",
        f"Failed checks: `{failed}`.",
        f"Pending checks: `{pending}`.",
        f"Caveats: `{caveats}`.",
    ]
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(output / "summary.md"), "status": result["status"]}, indent=2))
    if failed and not args.allow_incomplete_scandmm:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
