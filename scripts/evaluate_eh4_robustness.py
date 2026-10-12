#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from sphererate.data.early_dataset import EarlyHorizonDataset
from sphererate.evaluation.early_model import compare_coverage_rows, compare_method_rows
from sphererate.utils import ensure_dir
from scripts.evaluate_eh3_gate import (
    _build_model_from_state,
    _predict_learned,
    _score_methods,
)


SEEDS = (20260710, 20260711, 20260712)


def _average_rows(rows_by_seed: list[list[dict[str, Any]]], value_keys: tuple[str, ...]) -> list[dict[str, Any]]:
    keyed: dict[tuple[str, float], list[dict[str, Any]]] = {}
    for rows in rows_by_seed:
        for row in rows:
            keyed.setdefault((str(row["content_id"]), float(row["horizon_s"])), []).append(row)
    output = []
    for (content, horizon), rows in sorted(keyed.items()):
        item = {"content_id": content, "horizon_s": horizon}
        for key in value_keys:
            item[key] = float(np.mean([float(row[key]) for row in rows]))
        output.append(item)
    return output


def _efficiency(checkpoint: Path, sample: dict[str, Any], device: torch.device) -> dict[str, Any]:
    state = torch.load(checkpoint, map_location=device)
    model = _build_model_from_state(state, device)
    image = sample["image"].unsqueeze(0).to(device)
    template = sample["template"].unsqueeze(0).to(device)
    with torch.no_grad():
        for _ in range(10):
            model(image, template)
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        timings = []
        with torch.no_grad():
            for _ in range(50):
                starter.record()
                model(image, template)
                ender.record()
                torch.cuda.synchronize()
                timings.append(starter.elapsed_time(ender))
        peak_memory = int(torch.cuda.max_memory_allocated(device))
    else:
        timings = []
        with torch.no_grad():
            for _ in range(20):
                start = time.perf_counter()
                model(image, template)
                timings.append((time.perf_counter() - start) * 1000.0)
        peak_memory = 0
    profiler_flops = 0
    try:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(activities=activities, with_flops=True) as profile:
            with torch.no_grad():
                model(image, template)
        profiler_flops = int(sum(int(item.flops or 0) for item in profile.key_averages()))
    except Exception:
        profiler_flops = 0
    return {
        "total_parameters": int(state["parameters"]["total"]),
        "trainable_parameters": int(state["parameters"]["trainable"]),
        "batch_size": 1,
        "input_size": list(image.shape),
        "output_horizons": 4,
        "latency_ms_mean": float(np.mean(timings)),
        "latency_ms_std": float(np.std(timings)),
        "peak_allocated_bytes": peak_memory,
        "profiler_supported_ops_flops": profiler_flops,
        "flops_note": "PyTorch profiler supported operations only; use as an approximate lower-bound accounting.",
    }


def _format_ci(result: dict[str, Any]) -> str:
    return f"{result['mean']:.6f} [{result['ci95_low']:.6f}, {result['ci95_high']:.6f}]"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate EH4 three-seed robustness and efficiency.")
    parser.add_argument("--output-dir", default="reports/gate_eh4_robustness")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--split", choices=("val", "gate"), default="gate")
    parser.add_argument("--data-root", default="data/processed/early_horizon_salient360_v1")
    parser.add_argument(
        "--checkpoint-pattern",
        default="runs/eh3_candidate_correct_seed{seed}/checkpoints/best.pt",
    )
    args = parser.parse_args()
    root = Path(args.data_root)
    dataset = EarlyHorizonDataset(root, split=args.split, target_mode="correct")
    samples = [dataset[index] for index in range(len(dataset))]
    device = torch.device(args.device)
    template_predictions = {
        str(sample["image_id"]): sample["template"].numpy().astype(np.float32) for sample in samples
    }
    seed_results: dict[str, Any] = {}
    seed_score_rows: list[list[dict[str, Any]]] = []
    seed_coverage_rows: list[list[dict[str, Any]]] = []
    template_score_rows: list[dict[str, Any]] | None = None
    template_coverage_rows: list[dict[str, Any]] | None = None
    for seed in SEEDS:
        checkpoint = Path(str(args.checkpoint_pattern).format(seed=seed))
        predictions, provenance = _predict_learned(checkpoint, samples, device)
        scores, coverage = _score_methods(
            {"candidate": predictions, "template": template_predictions},
            samples,
        )
        comparison = compare_method_rows(scores["candidate"], scores["template"], iterations=10000, seed=20260711)
        coverage_comparison = compare_coverage_rows(
            coverage["candidate"], coverage["template"], iterations=10000, seed=20260711
        )
        seed_results[str(seed)] = {
            "checkpoint": provenance,
            "map_vs_template": comparison,
            "coverage_vs_template": coverage_comparison,
        }
        seed_score_rows.append(scores["candidate"])
        seed_coverage_rows.append(coverage["candidate"])
        template_score_rows = scores["template"]
        template_coverage_rows = coverage["template"]
    assert template_score_rows is not None and template_coverage_rows is not None
    average_scores = _average_rows(seed_score_rows, ("kld_sph", "sim_sph", "cc_sph"))
    average_coverage = _average_rows(seed_coverage_rows, ("coverage",))
    aggregate_map = compare_method_rows(
        average_scores, template_score_rows, iterations=10000, seed=20260711
    )
    aggregate_coverage = compare_coverage_rows(
        average_coverage, template_coverage_rows, iterations=10000, seed=20260711
    )
    all_seed_point_positive = all(
        result["map_vs_template"]["early_macro"]["kld_sph"]["mean"] > 0
        and result["map_vs_template"]["early_macro"]["sim_sph"]["mean"] > 0
        and result["coverage_vs_template"]["mean"] > 0
        for result in seed_results.values()
    )
    checks = {
        "all_seed_point_estimates_positive": all_seed_point_positive,
        "seed_average_kld_ci_positive": aggregate_map["early_macro"]["kld_sph"]["ci95_low"] > 0,
        "seed_average_sim_ci_positive": aggregate_map["early_macro"]["sim_sph"]["ci95_low"] > 0,
        "seed_average_coverage_ci_positive": aggregate_coverage["ci95_low"] > 0,
        "outer_test_not_used": True,
    }
    efficiency = _efficiency(
        Path(str(args.checkpoint_pattern).format(seed=20260711)), samples[0], device
    )
    result = {
        "gate": "EH4-three-seed-robustness",
        "decision": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "seeds": seed_results,
        "seed_average_map_vs_template": aggregate_map,
        "seed_average_coverage_vs_template": aggregate_coverage,
        "efficiency": efficiency,
        "development_gate_fresh": False,
        "split": str(args.split),
        "outer_test_used": False,
    }
    output = ensure_dir(args.output_dir)
    (output / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# EH4: Three-Seed Robustness and Efficiency",
        "",
        f"Decision: **{result['decision']}**",
        "",
        "## Per-seed early-macro improvements over early template",
        "",
        "| Seed | KLD | SIM | CC | 10% coverage |",
        "|---:|---:|---:|---:|---:|",
    ]
    for seed, current in seed_results.items():
        macro = current["map_vs_template"]["early_macro"]
        lines.append(
            f"| {seed} | {_format_ci(macro['kld_sph'])} | {_format_ci(macro['sim_sph'])} | "
            f"{_format_ci(macro['cc_sph'])} | {_format_ci(current['coverage_vs_template'])} |"
        )
    macro = aggregate_map["early_macro"]
    lines.extend(
        [
            "",
            "## Seed-averaged evidence",
            "",
            f"- KLD vs template: {_format_ci(macro['kld_sph'])}",
            f"- SIM vs template: {_format_ci(macro['sim_sph'])}",
            f"- CC vs template: {_format_ci(macro['cc_sph'])}",
            f"- 10% coverage vs template: {_format_ci(aggregate_coverage)}",
            "",
            "## Efficiency",
            "",
            f"- parameters: `{efficiency['total_parameters']}` total / `{efficiency['trainable_parameters']}` trainable",
            f"- batch-1 latency for four maps: `{efficiency['latency_ms_mean']:.3f} ± {efficiency['latency_ms_std']:.3f} ms`",
            f"- peak allocated memory: `{efficiency['peak_allocated_bytes'] / 2**20:.1f} MiB`",
            f"- profiler-supported FLOPs: `{efficiency['profiler_supported_ops_flops']}`",
            "",
            "## Checks",
            "",
        ]
    )
    lines.extend(f"- {name}: **{'PASS' if passed else 'FAIL'}**" for name, passed in checks.items())
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"decision": result["decision"], "report": str(output / "summary.md")}, indent=2))


if __name__ == "__main__":
    main()
