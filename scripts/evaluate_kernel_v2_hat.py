#!/usr/bin/env python
"""Aggregate three validation-selected HAT runs under the matched protocol."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (20260710, 20260711, 20260712)


def _load(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate matched HAT val/gate results.")
    parser.add_argument("--run-prefix", default="kernel_v2_hat")
    parser.add_argument("--output-prefix", default="reports/kernel_v2_hat")
    args = parser.parse_args()
    selections = []
    parameter_counts = set()
    trainable_counts = set()
    for seed in SEEDS:
        train = ROOT / f"runs/{args.run_prefix}_seed{seed}_train"
        provenance = _load(train / "provenance.json")
        selection = _load(train / "selection.json")
        if provenance["method"] != "HAT-ERP-declared-start-v1":
            raise ValueError(f"Wrong HAT provenance for seed {seed}")
        if int(provenance["training"]["max_iters"]) != 5000:
            raise ValueError(f"HAT seed {seed} does not use the frozen 5k budget")
        semantics = provenance["conditioning_semantics"]
        if not semantics["declared_start_is_first_history_token"]:
            raise ValueError(f"HAT seed {seed} omitted the declared start")
        if not semantics["declared_start_is_not_a_supervised_target"]:
            raise ValueError(f"HAT seed {seed} predicts rather than conditions on the start")
        if selection["gate_used"] or int(selection["test_target_loads"]) != 0:
            raise ValueError(f"HAT seed {seed} used a forbidden split for selection")
        if int(selection.get("sampling_ior_radius", -1)) != 0:
            raise ValueError(f"HAT seed {seed} uses fixation-level IOR on second-sampled gaze")
        selections.append(
            {
                "seed": seed,
                "selected_step": int(selection["selected"]["step"]),
                "val_selection_kld_sph": float(selection["selected"]["kld_sph"]),
                "checkpoint_sha256": str(selection["selected"]["checkpoint_sha256"]),
            }
        )

    for split in ("val", "gate"):
        command = [
            sys.executable,
            "scripts/evaluate_eh6_scandmm.py",
            "--split",
            split,
            "--output-dir",
            f"{args.output_prefix}_{split}",
            "--data-root",
            "data/processed/early_horizon_salient360_spherical2deg_v2",
            "--renderer",
            "sphere_chord",
            "--sigma-deg",
            "2",
            "--baseline-name",
            "HAT (ERP-adapted)",
            "--tier",
            "secondary_history_conditioned_scanpath",
            "--time-semantics",
            "one generated gaze point per second",
            "--require-initialization-semantics",
            "declared_start_history_token",
        ]
        for seed in SEEDS:
            run = ROOT / f"runs/{args.run_prefix}_seed{seed}_{split}"
            metadata = _load(run / "inference_metadata.json")
            if metadata["initialization_semantics"] != "declared_start_history_token":
                raise ValueError(f"HAT seed {seed} has wrong inference start semantics")
            if int(metadata.get("sampling_ior_radius", -1)) != 0:
                raise ValueError(f"HAT seed {seed} has unmatched sampling IOR")
            if int(metadata["test_target_loads"]) != 0:
                raise ValueError(f"HAT seed {seed} loaded a sealed target")
            parameter_counts.add(int(metadata["parameter_count"]))
            trainable_counts.add(int(metadata["trainable_parameter_count"]))
            command.extend(("--scanpath-run", str(run)))
        subprocess.run(command, cwd=ROOT, check=True)

    if len(parameter_counts) != 1 or len(trainable_counts) != 1:
        raise ValueError("HAT parameter counts differ across seeds/splits")
    audit_root = ROOT / args.output_prefix
    audit_root.mkdir(parents=True, exist_ok=True)
    audit = {
        "method": "HAT (ERP-adapted)",
        "selection_split": "val",
        "evaluation_splits": ["val", "gate"],
        "seeds": selections,
        "declared_start": {"pitch_deg": 0.0, "yaw_deg": 90.0},
        "initialization_semantics": "declared_start_history_token",
        "first_supervised_target": "first_one_second_gaze",
        "sampling_ior_radius": 0,
        "sampling_ior_semantics": "disabled_for_second_sampled_gaze",
        "generated_steps": 20,
        "parameters": next(iter(parameter_counts)),
        "trainable_parameters": next(iter(trainable_counts)),
        "gate_used_for_selection": False,
        "test_target_loads": 0,
    }
    (audit_root / "selection_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    lines = [
        "# HAT matched declared-start audit",
        "",
        "HAT is retrained on the same 50-content training split. The declared start is the first history token and is not a supervised gaze target.",
        "",
        "| seed | selected step | val selection KLD |",
        "|---:|---:|---:|",
    ]
    for row in selections:
        lines.append(
            f"| {row['seed']} | {row['selected_step']} | {row['val_selection_kld_sph']:.6f} |"
        )
    lines.extend(
        [
            "",
            f"Total/trainable parameters: `{audit['parameters']}` / `{audit['trainable_parameters']}`.",
            "",
            "Each final run contains 200 paths per content and 20 one-second predictions. No sealed test target was loaded.",
        ]
    )
    (audit_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"audit": str(audit_root / "summary.md")}, indent=2))


if __name__ == "__main__":
    main()
