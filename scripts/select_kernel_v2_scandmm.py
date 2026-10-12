#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select a leak-free declared-start ScanDMM checkpoint using val only."
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--training-run", required=True)
    parser.add_argument("--selection-root")
    parser.add_argument("--n-scanpaths", type=int, default=50)
    parser.add_argument("--final-scanpaths", type=int, default=200)
    parser.add_argument("--required-final-epoch", type=int, default=499)
    parser.add_argument(
        "--evaluate-available-only",
        action="store_true",
        help="Cache val metrics for currently available checkpoints without selecting a winner.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data-root", default="data/processed/early_horizon_salient360_spherical2deg_v2")
    parser.add_argument("--start-pitch-deg", type=float, default=0.0)
    parser.add_argument("--start-yaw-deg", type=float, default=90.0)
    parser.add_argument("--expected-val", type=int, default=9)
    parser.add_argument("--expected-gate", type=int, default=13)
    parser.add_argument("--expected-test", type=int, default=13)
    parser.add_argument("--data-config", default="configs/early_horizon_spherical2deg_v2.yaml")
    parser.add_argument("--run-prefix", default="kernel_v2_scandmm_declared_start")
    parser.add_argument("--report-prefix", default="kernel_v2_scandmm_declared_start")
    args = parser.parse_args()
    training_run = Path(args.training_run)
    checkpoints = sorted((training_run / "checkpoints").glob("epoch_*.pkl"), key=_epoch)
    if not checkpoints:
        raise FileNotFoundError(f"No sparse checkpoints under {training_run}")
    if (
        not args.evaluate_available_only
        and _epoch(checkpoints[-1]) != int(args.required_final_epoch)
    ):
        raise RuntimeError(
            f"Training is incomplete: last checkpoint epoch {_epoch(checkpoints[-1])}, "
            f"required {args.required_final_epoch}"
        )
    selection_root = Path(
        args.selection_root or f"runs/{args.run_prefix}_selection_seed{args.seed}"
    )
    selection_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for checkpoint in checkpoints:
        epoch = _epoch(checkpoint)
        inference_run = selection_root / f"epoch_{epoch:04d}_n{args.n_scanpaths}"
        report_dir = (
            Path("reports")
            / f"{args.report_prefix}_selection_seed{args.seed}_epoch{epoch:04d}"
        )
        results_path = report_dir / "results.json"
        if not results_path.exists():
            _run(
                [
                    sys.executable,
                    "scripts/run_scandmm_inference.py",
                    "--config",
                    str(args.data_config),
                    "--data-root",
                    str(args.data_root),
                    "--split",
                    "val",
                    "--model",
                    str(checkpoint),
                    "--run-dir",
                    str(inference_run),
                    "--n-scanpaths",
                    str(args.n_scanpaths),
                    "--length",
                    "20",
                    "--seed",
                    str(args.seed),
                    "--start-yaw-deg",
                    str(args.start_yaw_deg),
                    "--start-pitch-deg",
                    str(args.start_pitch_deg),
                    "--independent-declared-start",
                ]
            )
            _run(
                [
                    sys.executable,
                    "scripts/evaluate_eh6_scandmm.py",
                    "--split",
                    "val",
                    "--scanpath-run",
                    str(inference_run),
                    "--output-dir",
                    str(report_dir),
                    "--data-root",
                    str(args.data_root),
                    "--renderer",
                    "sphere_chord",
                    "--sigma-deg",
                    "2",
                    "--require-initialization-semantics",
                    "independent_declared_tensor",
                ]
            )
        result = json.loads(results_path.read_text(encoding="utf-8"))
        metric = result["method_early_macro"]
        rows.append(
            {
                "epoch": epoch,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "inference_run": str(inference_run),
                "report": str(report_dir),
                "n_scanpaths": int(args.n_scanpaths),
                **{key: float(metric[key]) for key in ("kld_sph", "sim_sph", "cc_sph")},
            }
        )
        print(json.dumps(rows[-1]), flush=True)
    if args.evaluate_available_only:
        print(
            json.dumps(
                {
                    "status": "available_checkpoints_evaluated_without_selection",
                    "seed": int(args.seed),
                    "candidates": len(rows),
                    "latest_epoch": int(rows[-1]["epoch"]),
                    "selection_written": False,
                    "gate_used": False,
                },
                indent=2,
            )
        )
        return
    best = min(rows, key=lambda row: (row["kld_sph"], row["epoch"]))
    selection = {
        "method": "ScanDMM-declared-start-v3",
        "initialization_semantics": "independent_declared_tensor",
        "first_supervised_target": "first_one_second_gaze",
        "seed": int(args.seed),
        "selection_split": "val",
        "selection_metric": "early_macro_kld_sph",
        "selection_scanpaths_per_content": int(args.n_scanpaths),
        "start": {"pitch_deg": float(args.start_pitch_deg), "yaw_deg": float(args.start_yaw_deg)},
        "renderer": {"family": "sphere_chord", "sigma_deg": 2.0},
        "best": best,
        "candidates": rows,
        "gate_used": False,
        "test_target_rows_loaded": 0,
    }
    adapter_path = ROOT / "src/sphererate/baselines/scandmm_declared_start.py"
    selection["declared_start_adapter"] = {
        "path": str(adapter_path),
        "sha256": hashlib.sha256(adapter_path.read_bytes()).hexdigest(),
    }
    path = training_run / "selection.json"
    path.write_text(json.dumps(selection, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    provenance_path = training_run / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["declared_start_adapter"] = selection["declared_start_adapter"]
    provenance_path.write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    final_splits = ("val", "test") if "standard_v3" in str(args.data_root) else ("val", "gate")
    for offset, split in enumerate(final_splits):
        final_run = Path(f"runs/{args.run_prefix}_seed{args.seed}_{split}")
        expected_count = {
            "val": int(args.expected_val),
            "gate": int(args.expected_gate),
            "test": int(args.expected_test),
        }[split]
        scanpath_dir = final_run / "scanpaths"
        if len(list(scanpath_dir.glob("*.npy"))) != expected_count:
            _run(
                [
                    sys.executable,
                    "scripts/run_scandmm_inference.py",
                    "--config",
                    str(args.data_config),
                    "--data-root",
                    str(args.data_root),
                    "--split",
                    split,
                    "--model",
                    str(best["checkpoint"]),
                    "--run-dir",
                    str(final_run),
                    "--n-scanpaths",
                    str(args.final_scanpaths),
                    "--length",
                    "20",
                    "--seed",
                    str(args.seed + 1901 + offset),
                    "--start-yaw-deg",
                    str(args.start_yaw_deg),
                    "--start-pitch-deg",
                    str(args.start_pitch_deg),
                    "--independent-declared-start",
                ]
            )
        metadata_path = final_run / "inference_metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata.update(
            {
                "split": split,
                "selection_split": "val",
                "selection_metric": "early_macro_kld_sph",
                "selected_epoch": int(best["epoch"]),
                "selected_checkpoint_sha256": str(best["checkpoint_sha256"]),
                "test_evaluated": bool(split == "test"),
                "test_used_for_selection": False,
            }
        )
        metadata_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    print(json.dumps({"selection": str(path), "best": best}, indent=2, ensure_ascii=False))


def _epoch(path: Path) -> int:
    match = re.search(r"epoch_(\d+)\.pkl$", path.name)
    if match is None:
        raise ValueError(f"Invalid sparse checkpoint name: {path}")
    return int(match.group(1))


def _run(command: list[str]) -> None:
    env = dict(**__import__("os").environ)
    env["PYTHONPATH"] = "src" + ((":" + env["PYTHONPATH"]) if env.get("PYTHONPATH") else "")
    subprocess.run(command, cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
