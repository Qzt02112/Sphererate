#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (20260710, 20260711, 20260712)


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the standard 59/13/13 benchmark.")
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--output-dir", default="reports/standard_split_protocol_audit")
    args = parser.parse_args()
    checks: list[dict[str, Any]] = []

    def check(name: str, condition: bool, detail: Any = None) -> None:
        checks.append({"name": name, "pass": bool(condition), "detail": detail})

    data = _load(ROOT / "reports/standard_split_data_audit/audit.json")["audit"]
    check("data_audit_pass", data.get("decision") == "PASS", data.get("split_counts"))
    check("split_counts_59_13_13", data.get("split_counts") == {"train": 59, "val": 13, "test": 13})
    check("test_targets_materialized", bool(data.get("checks", {}).get("test_targets_materialized")))
    check("templates_train_only", bool(data.get("checks", {}).get("all_templates_train_only")))

    for split in ("val", "test"):
        primary = _load(ROOT / f"reports/standard_split_primary_{split}/results.json")
        check(f"primary_{split}_role", primary.get("split") == split)
        check(f"primary_{split}_test_not_selection", primary.get("test_used_for_selection") is False)
        check(f"primary_{split}_three_seeds", tuple(primary.get("seeds", [])) == SEEDS)

    for seed in SEEDS:
        path = ROOT / f"runs/standard_split_v3/pathformer3d_train_seed{seed}/selection.json"
        selection = _load(path)
        check(f"pathformer_{seed}_val_selection", str(selection.get("criterion", "")).startswith("val early-macro"))
        check(f"pathformer_{seed}_complete_epoch40", (ROOT / f"runs/standard_split_v3/pathformer3d_train_seed{seed}/checkpoints/epoch_0040.pt").is_file())

        scandmm_run = ROOT / f"runs/standard_split_v3/scandmm_train_seed{seed}"
        scandmm_selection = _load(scandmm_run / "selection.json")
        check(
            f"scandmm_{seed}_val_selection",
            scandmm_selection.get("selection_split") == "val"
            and scandmm_selection.get("selection_metric") == "early_macro_kld_sph",
        )
        check(
            f"scandmm_{seed}_complete_epoch499",
            (scandmm_run / "checkpoints/epoch_0499.pkl").is_file(),
        )
        scandmm_test_meta = _load(
            ROOT
            / f"runs/standard_split_v3/scandmm_declared_start_seed{seed}_test/inference_metadata.json"
        )
        check(
            f"scandmm_{seed}_test_not_selection",
            scandmm_test_meta.get("test_used_for_selection") is False,
        )

        hat_run = ROOT / f"runs/standard_split_v3/hat_train_seed{seed}"
        hat_selection = _load(hat_run / "selection.json")
        check(
            f"hat_{seed}_val_selection",
            str(hat_selection.get("criterion", "")).startswith("val early-macro")
            and int(hat_selection.get("test_target_loads", -1)) == 0,
        )
        check(
            f"hat_{seed}_complete_step5000",
            (hat_run / "checkpoints/step_005000.pt").is_file(),
        )
        hat_test_meta = _load(
            ROOT / f"runs/standard_split_v3/hat_seed{seed}_test/inference_metadata.json"
        )
        check(
            f"hat_{seed}_test_not_selection",
            hat_test_meta.get("test_used_for_selection") is False,
        )

    optional = {
        "scandmm": ROOT / "reports/standard_split_scandmm_test/results.json",
        "hat": ROOT / "reports/standard_split_hat_test/results.json",
    }
    missing = [name for name, path in optional.items() if not path.is_file()]
    check("long_baselines_complete", not missing, missing)
    full = _load(ROOT / "reports/standard_split_full_benchmark_test/results.json")
    check("full_benchmark_no_missing_rows", not full.get("missing_methods"), full.get("missing_methods"))
    check("full_benchmark_test_not_selection", full.get("test_used_for_selection") is False)
    check("full_benchmark_all_14_rows", len(full.get("methods", {})) == 14, len(full.get("methods", {})))

    passed = all(row["pass"] for row in checks)
    complete = not missing and not full.get("missing_methods")
    if args.require_complete and not complete:
        passed = False
    output = ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    result = {"pass": passed, "complete": complete, "checks": checks}
    (output / "audit.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# Standard 59/13/13 protocol audit",
        "",
        f"Status: **{'PASS' if passed else 'IN PROGRESS' if not args.require_complete else 'FAIL'}**.",
        "",
        "| Check | Status | Detail |",
        "|---|---|---|",
    ]
    for row in checks:
        detail = "" if row["detail"] is None else f"`{row['detail']}`"
        lines.append(f"| {row['name']} | {'PASS' if row['pass'] else 'PENDING/FAIL'} | {detail} |")
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(output / "summary.md"), "pass": passed, "complete": complete}, indent=2))
    if args.require_complete and not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
