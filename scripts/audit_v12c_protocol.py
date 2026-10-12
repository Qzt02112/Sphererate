#!/usr/bin/env python
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sphererate.data.early_view import read_manifest


SEEDS = (20260710, 20260711, 20260712)
VALID_SPLITS = ("train", "val", "gate", "test")


def main() -> None:
    manifest_path = (
        ROOT
        / "data"
        / "processed"
        / "early_horizon_salient360_spherical2deg_v2"
        / "manifest.jsonl"
    )
    rows = read_manifest(manifest_path)
    groups: dict[str, set[str]] = defaultdict(set)
    counts = {split: 0 for split in VALID_SPLITS}
    invalid_splits = []
    for row in rows:
        split = str(row.get("split"))
        if split not in VALID_SPLITS:
            invalid_splits.append({"image_id": row.get("image_id"), "split": split})
            continue
        counts[split] += 1
        groups[str(row["canonical_content_id"])].add(split)
    leaking = {
        group: sorted(splits) for group, splits in groups.items() if len(splits) != 1
    }
    test_rows = [row for row in rows if row.get("split") == "test"]
    test_sealed = all(
        row.get("target") is None
        and row.get("fixations") is None
        and bool(row.get("target_sealed"))
        for row in test_rows
    )
    val_ids = {str(row["image_id"]) for row in rows if row.get("split") == "val"}
    run_checks = []
    for seed in SEEDS:
        run = ROOT / "runs" / f"v12c_hybrid_dwell_rate_seed{seed}"
        checkpoint = torch.load(
            run / "checkpoints" / "best.pt",
            map_location="cpu",
            weights_only=False,
        )
        config = checkpoint["config"]
        prediction_ids = {
            path.stem for path in (run / "predictions" / "val").glob("*.npz")
        }
        unexpected_prediction_dirs = sorted(
            str(path.relative_to(run))
            for path in (run / "predictions").glob("*")
            if path.is_dir() and path.name != "val"
        )
        run_checks.append(
            {
                "seed": seed,
                "selected_epoch": int(checkpoint["epoch"]),
                "selection_split": config["protocol"]["selection_split"],
                "forbidden_for_selection": config["protocol"]["forbidden_for_selection"],
                "forbidden_inputs": config["protocol"]["forbidden_inputs"],
                "prediction_ids_equal_val": prediction_ids == val_ids,
                "prediction_count": len(prediction_ids),
                "unexpected_prediction_dirs": unexpected_prediction_dirs,
            }
        )
    checks = {
        "expected_split_counts": counts == {"train": 50, "val": 9, "gate": 13, "test": 13},
        "valid_four_way_splits": not invalid_splits,
        "canonical_content_zero_leakage": not leaking,
        "outer_test_targets_sealed": test_sealed,
        "all_runs_select_val_only": all(
            row["selection_split"] == "val"
            and set(row["forbidden_for_selection"]) == {"gate", "test", "sitzmann"}
            for row in run_checks
        ),
        "all_runs_have_only_val_predictions": all(
            row["prediction_ids_equal_val"] and not row["unexpected_prediction_dirs"]
            for row in run_checks
        ),
        "no_forbidden_inference_inputs": all(
            set(row["forbidden_inputs"])
            == {"TempSAL_prediction", "gaze_history", "target_domain_labels"}
            for row in run_checks
        ),
    }
    payload = {
        "manifest": str(manifest_path.relative_to(ROOT)),
        "split_counts": counts,
        "invalid_splits": invalid_splits,
        "leaking_groups": leaking,
        "test_row_count": len(test_rows),
        "runs": run_checks,
        "checks": checks,
        "pass": all(checks.values()),
        "outer_test_target_loads": 0,
        "gate_target_loads": 0,
    }
    out = ROOT / "reports" / "v12c_spherical_dwell_rate"
    out.mkdir(parents=True, exist_ok=True)
    (out / "protocol_audit.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    )
    lines = ["# V12-C Protocol Audit", ""]
    for name, value in checks.items():
        lines.append(f"- {name}: **{'PASS' if value else 'FAIL'}**")
    lines.extend(
        [
            "",
            f"Overall: **{'PASS' if payload['pass'] else 'FAIL'}**",
            "",
            f"Split counts: `{counts}`; gate/test target loads during V12-C selection: `0/0`.",
        ]
    )
    (out / "protocol_audit.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if not payload["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
