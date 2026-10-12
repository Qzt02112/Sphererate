#!/usr/bin/env python
"""Integrity audit for the frozen V12-C cross-dataset evaluations."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def main() -> None:
    sitzmann = _load("reports/v12c_frozen_transfer_sitzmann/results.json")
    etdk2 = _load("reports/v12c_frozen_transfer_etdk2/results.json")
    sitz_checkpoints = sitzmann["checkpoints"]
    etdk2_checkpoints = etdk2["checkpoints"]["v12c"]
    actual_hashes = {
        int(row["seed"]): hashlib.sha256((ROOT / row["path"]).read_bytes()).hexdigest()
        for row in sitz_checkpoints
    }
    checks = {
        "same_v12c_checkpoint_hashes_across_datasets": {
            int(left["seed"]): left["sha256"] for left in sitz_checkpoints
        }
        == {int(right["seed"]): right["sha256"] for right in etdk2_checkpoints},
        "reported_hashes_match_files": actual_hashes
        == {int(row["seed"]): row["sha256"] for row in sitz_checkpoints},
        "all_checkpoint_epochs_frozen_at_one": all(
            int(row["epoch"]) == 1 for row in sitz_checkpoints + etdk2_checkpoints
        ),
        "sitzmann_target_calibration_forbidden": not bool(
            sitzmann["data_audit"]["target_domain_calibration"]
        ),
        "sitzmann_train_val_target_loads_zero": int(
            sitzmann["data_audit"]["target_domain_train_or_val_loaded_by_v12c"]
        )
        == 0,
        "sitzmann_outer_loads_zero": not bool(sitzmann["data_audit"]["outer_test_loaded"]),
        "etdk2_target_calibration_forbidden": not bool(
            etdk2["data_audit"]["target_domain_calibration"]
        ),
        "etdk2_checkpoint_reselection_forbidden": not bool(
            etdk2["data_audit"]["checkpoint_selection_after_external_evaluation"]
        ),
        "etdk2_test_target_loads_zero": int(
            etdk2["access_audit"]["etdk2_test_target_rows_loaded"]
        )
        == 0,
        "salient360_outer_target_loads_zero": int(
            etdk2["access_audit"]["salient360_outer_test_target_rows_loaded"]
        )
        == 0,
        "etdk2_source_train_and_outer_excluded": etdk2["access_audit"][
            "excluded_etdk2_val_sessions_by_salient360_source_role"
        ]
        == {"test": 18, "train": 36},
        "etdk2_cross_content_claim_disabled": not bool(
            etdk2["data_audit"]["cross_content_generalization_claim"]
        ),
    }
    result = {
        "protocol": "v12c_frozen_transfer_integrity_v1",
        "overall": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "checkpoint_sha256": actual_hashes,
        "selection_source": "Salient360 val only; frozen before both external evaluations",
        "forbidden_feedback": [
            "checkpoint reselection",
            "target-domain calibration",
            "reachability refit",
            "loss or soft-topk tuning",
            "tile-grid or budget tuning",
        ],
    }
    output = ROOT / "reports" / "v12c_frozen_transfer"
    output.mkdir(parents=True, exist_ok=True)
    (output / "integrity_audit.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    lines = ["# V12-C Frozen Transfer Integrity Audit", ""]
    lines.extend(
        f"- {name}: **{'PASS' if passed else 'FAIL'}**" for name, passed in checks.items()
    )
    lines.extend(["", f"Overall: **{result['overall']}**"])
    (output / "integrity_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"overall": result["overall"], "report": str(output / "integrity_audit.md")}, indent=2))


if __name__ == "__main__":
    main()
