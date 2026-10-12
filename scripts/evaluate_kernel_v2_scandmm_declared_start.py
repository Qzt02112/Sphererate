#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
FROZEN_SEEDS = (20260710, 20260711, 20260712)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate corrected three-seed ScanDMM val/gate results."
    )
    parser.add_argument("--output-prefix", default="reports/kernel_v2_scandmm_declared_start")
    args = parser.parse_args()
    adapter = ROOT / "src/sphererate/baselines/scandmm_declared_start.py"
    adapter_hash = hashlib.sha256(adapter.read_bytes()).hexdigest()
    selections = []
    for seed in FROZEN_SEEDS:
        training = ROOT / f"runs/kernel_v2_scandmm_declared_start_seed{seed}"
        provenance = _load(training / "provenance.json")
        selection = _load(training / "selection.json")
        if provenance["method"] != "ScanDMM-declared-start-v3":
            raise ValueError(f"Seed {seed} has wrong training method metadata")
        semantics = provenance["conditioning_semantics"]
        if semantics["legacy_first_target_as_initializer"] is not False:
            raise ValueError(f"Seed {seed} did not disable legacy first-target initialization")
        if selection["initialization_semantics"] != "independent_declared_tensor":
            raise ValueError(f"Seed {seed} selection has wrong initialization semantics")
        if selection["declared_start_adapter"]["sha256"] != adapter_hash:
            raise ValueError(f"Seed {seed} adapter hash differs from the current audited adapter")
        if selection["gate_used"] or selection["test_target_rows_loaded"] != 0:
            raise ValueError(f"Seed {seed} selection accessed a forbidden split")
        selections.append(
            {
                "seed": seed,
                "selected_epoch": int(selection["best"]["epoch"]),
                "val_selection_kld_sph": float(selection["best"]["kld_sph"]),
                "checkpoint": selection["best"]["checkpoint"],
                "checkpoint_sha256": selection["best"]["checkpoint_sha256"],
            }
        )

    output_prefix = Path(args.output_prefix)
    for split in ("val", "gate"):
        command = [
            sys.executable,
            "scripts/evaluate_eh6_scandmm.py",
            "--split",
            split,
        ]
        for seed in FROZEN_SEEDS:
            run = ROOT / f"runs/kernel_v2_scandmm_declared_start_seed{seed}_{split}"
            metadata = _load(run / "inference_metadata.json")
            if metadata["initialization_semantics"] != "independent_declared_tensor":
                raise ValueError(f"{run} does not use independent declared-start inference")
            if metadata["declared_start_adapter"]["sha256"] != adapter_hash:
                raise ValueError(f"{run} adapter hash mismatch")
            command.extend(("--scanpath-run", str(run)))
        command.extend(
            (
                "--output-dir",
                f"{output_prefix}_{split}",
                "--data-root",
                "data/processed/early_horizon_salient360_spherical2deg_v2",
                "--renderer",
                "sphere_chord",
                "--sigma-deg",
                "2",
                "--baseline-name",
                "ScanDMM (independent declared start)",
                "--require-initialization-semantics",
                "independent_declared_tensor",
            )
        )
        subprocess.run(command, cwd=ROOT, check=True)

    audit_dir = output_prefix
    audit_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "method": "ScanDMM-declared-start-v3",
        "selection_split": "val",
        "evaluation_splits": ["val", "gate"],
        "seeds": selections,
        "declared_start": {"pitch_deg": 0.0, "yaw_deg": 90.0},
        "initialization_semantics": "independent_declared_tensor",
        "first_supervised_target": "first_one_second_gaze",
        "supervised_steps": 20,
        "adapter_sha256": adapter_hash,
        "gate_used_for_selection": False,
        "test_target_loads": 0,
    }
    (audit_dir / "selection_audit.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    lines = [
        "# ScanDMM independent declared-start selection audit",
        "",
        "All checkpoints were selected by 9-content val early-macro KLD; gate was evaluated only after selection.",
        "",
        "| seed | selected epoch | val selection KLD |",
        "|---:|---:|---:|",
    ]
    for row in selections:
        lines.append(
            f"| {row['seed']} | {row['selected_epoch']} | {row['val_selection_kld_sph']:.6f} |"
        )
    lines += [
        "",
        f"Adapter SHA-256: `{adapter_hash}`.",
        "",
        "Every final run contains 200 paths per content, 20 one-second gaze points per path, "
        "and explicit `independent_declared_tensor` metadata. No v2 test target was loaded.",
    ]
    (audit_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"selection_audit": str(audit_dir / "summary.md")}, indent=2))


def _load(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
