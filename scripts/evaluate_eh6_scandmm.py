#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sphererate.baselines.scanpath_time import second_sampled_scanpaths_to_cumulative_maps
from sphererate.data.early_dataset import EarlyHorizonDataset
from sphererate.evaluation.early_model import compare_coverage_rows, compare_method_rows
from sphererate.evaluation.metrics import cc, kld, sim
from sphererate.evaluation.tile_prefetch import gaze_mass_coverage, rank_tiles
from sphererate.utils import ensure_dir


def _score(
    predictions: dict[str, np.ndarray], samples: list[dict[str, Any]], budgets: tuple[float, ...]
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    scores: list[dict[str, Any]] = []
    coverages: dict[str, list[dict[str, Any]]] = {str(value): [] for value in budgets}
    for sample in samples:
        image_id = str(sample["image_id"])
        target = sample["target"].numpy()
        area = sample["area_weights"].numpy()[0]
        horizons = sample["horizons_s"].numpy()
        with np.load(str(sample["fixations_path"])) as data:
            fixations = np.asarray(data["fixations"], dtype=np.float32)
        observers = np.unique(fixations[:, 3]).astype(int).tolist()
        for index, horizon in enumerate(horizons):
            prediction = predictions[image_id][index]
            scores.append(
                {
                    "content_id": str(sample["canonical_content_id"]),
                    "image_id": image_id,
                    "horizon_s": float(horizon),
                    "kld_sph": kld(prediction, target[index], area),
                    "sim_sph": sim(prediction, target[index], area),
                    "cc_sph": cc(prediction, target[index], area),
                }
            )
            for budget in budgets:
                selected = rank_tiles(
                    prediction,
                    area,
                    tile_rows=8,
                    tile_cols=16,
                    budget_percent=budget,
                )
                coverages[str(budget)].append(
                    {
                        "content_id": str(sample["canonical_content_id"]),
                        "image_id": image_id,
                        "horizon_s": float(horizon),
                        "coverage": gaze_mass_coverage(
                            fixations,
                            observer_ids=observers,
                            horizon_s=float(horizon),
                            selected_tile_ids=selected,
                            tile_rows=8,
                            tile_cols=16,
                        ),
                    }
                )
    return scores, coverages


def _average_rows(rows_by_seed: list[list[dict[str, Any]]], value_keys: tuple[str, ...]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for rows in rows_by_seed:
        for row in rows:
            grouped[(str(row["content_id"]), float(row["horizon_s"]))].append(row)
    return [
        {
            "content_id": content,
            "horizon_s": horizon,
            **{key: float(np.mean([float(row[key]) for row in group])) for key in value_keys},
        }
        for (content, horizon), group in sorted(grouped.items())
    ]


def _macro(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> dict[str, float]:
    primary = [row for row in rows if float(row["horizon_s"]) in {0.5, 1.0, 3.0}]
    return {key: float(np.mean([float(row[key]) for row in primary])) for key in keys}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate scanpath models under the matched secondary protocol.")
    parser.add_argument("--split", choices=("val", "gate", "test"), default="val")
    parser.add_argument("--scanpath-run", action="append", required=True, help="Run directory containing scanpaths/*.npy")
    parser.add_argument("--output-dir", default="reports/eh6_scandmm_val")
    parser.add_argument("--allow-outer-test", action="store_true")
    parser.add_argument("--data-root", default="data/processed/early_horizon_salient360_v1")
    parser.add_argument("--renderer", choices=("legacy_erp", "sphere_chord"), default="legacy_erp")
    parser.add_argument("--sigma-px", type=float, default=10.0)
    parser.add_argument("--sigma-deg", type=float, default=2.0)
    parser.add_argument("--baseline-name", default="ScanDMM")
    parser.add_argument("--tier", default="secondary_scanpath")
    parser.add_argument("--time-semantics", default="official one gaze point per second")
    parser.add_argument("--half-second-policy", default="first-second approximation; 0.5 s is not native")
    parser.add_argument(
        "--require-initialization-semantics",
        default=None,
        help="Fail if inference metadata does not declare this exact initialization mode.",
    )
    args = parser.parse_args()
    standard_protocol = "standard_v3" in str(args.data_root)
    if args.split == "test" and not args.allow_outer_test and not standard_protocol:
        raise PermissionError("Legacy EH6 outer-test evaluation requires explicit --allow-outer-test")
    dataset = EarlyHorizonDataset(
        args.data_root,
        split=args.split,
        target_mode="correct",
        allow_outer_test=True if standard_protocol else bool(args.allow_outer_test),
    )
    samples = [dataset[index] for index in range(len(dataset))]
    expected_ids = {str(sample["image_id"]) for sample in samples}
    budgets = (5.0, 10.0, 15.0, 20.0)
    score_seeds: list[list[dict[str, Any]]] = []
    coverage_seeds: dict[str, list[list[dict[str, Any]]]] = {str(value): [] for value in budgets}
    run_records = []
    for raw_run in args.scanpath_run:
        run = Path(raw_run)
        scanpath_dir = run / "scanpaths" if (run / "scanpaths").is_dir() else run
        found_ids = {path.stem for path in scanpath_dir.glob("*.npy")}
        if found_ids != expected_ids:
            raise ValueError(
                f"{scanpath_dir}: ID mismatch; missing={sorted(expected_ids-found_ids)}, extra={sorted(found_ids-expected_ids)}"
            )
        predictions = {}
        for sample in samples:
            image_id = str(sample["image_id"])
            predictions[image_id] = second_sampled_scanpaths_to_cumulative_maps(
                np.load(scanpath_dir / f"{image_id}.npy"),
                horizons_s=sample["horizons_s"].numpy().tolist(),
                height=int(sample["target"].shape[-2]),
                width=int(sample["target"].shape[-1]),
                sigma_px=float(args.sigma_px) if args.renderer == "legacy_erp" else None,
                renderer=str(args.renderer),
                sigma_deg=float(args.sigma_deg) if args.renderer == "sphere_chord" else None,
                coordinate_order="yx",
            )
        scores, coverages = _score(predictions, samples, budgets)
        score_seeds.append(scores)
        for budget in budgets:
            coverage_seeds[str(budget)].append(coverages[str(budget)])
        first = np.load(next(scanpath_dir.glob("*.npy")))
        metadata_path = run / "inference_metadata.json"
        metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        run_records.append(
            {
                "path": str(run),
                "content_count": len(found_ids),
                "scanpaths_per_content": int(first.shape[0]),
                "steps": int(first.shape[1]),
                "start_mode": metadata.get("start_mode", "unknown_legacy_run"),
                "start_yaw_deg": metadata.get("start_yaw_deg"),
                "start_pitch_deg": metadata.get("start_pitch_deg"),
                "initialization_semantics": metadata.get(
                    "initialization_semantics", "unknown_legacy_run"
                ),
            }
        )
    start_modes = {record["start_mode"] for record in run_records}
    if len(start_modes) != 1:
        raise ValueError(f"All ScanDMM runs must share one start mode, found {sorted(start_modes)}")
    initialization_semantics = {
        record["initialization_semantics"] for record in run_records
    }
    if len(initialization_semantics) != 1:
        raise ValueError(
            "All ScanDMM runs must share initialization semantics, found "
            f"{sorted(initialization_semantics)}"
        )
    observed_initialization = next(iter(initialization_semantics))
    if (
        args.require_initialization_semantics is not None
        and observed_initialization != args.require_initialization_semantics
    ):
        raise ValueError(
            f"Required initialization semantics {args.require_initialization_semantics!r}, "
            f"found {observed_initialization!r}"
        )
    averaged_scores = _average_rows(score_seeds, ("kld_sph", "sim_sph", "cc_sph"))
    averaged_coverages = {
        budget: _average_rows(rows, ("coverage",)) for budget, rows in coverage_seeds.items()
    }
    template_scores = []
    template_coverages: dict[str, list[dict[str, Any]]] = {str(value): [] for value in budgets}
    template_predictions = {str(sample["image_id"]): sample["template"].numpy() for sample in samples}
    template_scores, template_coverages = _score(template_predictions, samples, budgets)
    comparisons = {
        "map_vs_early_template": compare_method_rows(
            averaged_scores, template_scores, iterations=10000, seed=20260711
        ),
        "coverage_vs_early_template": {
            budget: compare_coverage_rows(
                averaged_coverages[budget], template_coverages[budget], iterations=10000, seed=20260711
            )
            for budget in averaged_coverages
        },
    }
    result = {
        "baseline": str(args.baseline_name),
        "tier": str(args.tier),
        "split": args.split,
        "time_semantics": str(args.time_semantics),
        "half_second_policy": str(args.half_second_policy),
        "coordinate_order": "yx (official inference source)",
        "start_condition": run_records[0]["start_mode"],
        "initialization_semantics": observed_initialization,
        "renderer": str(args.renderer),
        "sigma_px": float(args.sigma_px) if args.renderer == "legacy_erp" else None,
        "sigma_deg": float(args.sigma_deg) if args.renderer == "sphere_chord" else None,
        "runs": run_records,
        "method_early_macro": _macro(averaged_scores, ("kld_sph", "sim_sph", "cc_sph")),
        "coverage_early_macro": {
            budget: _macro(rows, ("coverage",))["coverage"] for budget, rows in averaged_coverages.items()
        },
        "score_rows": averaged_scores,
        "coverage_rows": averaged_coverages,
        "comparisons": comparisons,
        "evaluation_role": (
            "validation_selected_test_report"
            if standard_protocol and args.split == "test"
            else "validation"
            if standard_protocol
            else "legacy_frozen_post_hoc_benchmark"
            if args.split == "test"
            else "legacy_adapter_validation"
        ),
    }
    output = ensure_dir(args.output_dir)
    (output / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    metric = result["method_early_macro"]
    lines = [
        f"# EH6 {args.baseline_name} — {args.split}",
        "",
        "Tier: **secondary scanpath-to-prefetch baseline**.",
        "",
        f"Time semantics: {args.time_semantics}. {args.half_second_policy}.",
        f"Start condition: **{result['start_condition']}**.",
        "",
        "## Early macro",
        "",
        "| KLD_sph ↓ | SIM_sph ↑ | CC_sph ↑ | 5% coverage | 10% coverage | 15% coverage | 20% coverage |",
        "|---:|---:|---:|---:|---:|---:|---:|",
        f"| {metric['kld_sph']:.6f} | {metric['sim_sph']:.6f} | {metric['cc_sph']:.6f} | "
        f"{result['coverage_early_macro']['5.0']:.6f} | {result['coverage_early_macro']['10.0']:.6f} | "
        f"{result['coverage_early_macro']['15.0']:.6f} | "
        f"{result['coverage_early_macro']['20.0']:.6f} |",
        "",
        f"This result is a scanpath-to-cumulative-map adaptation ({args.tier}), not a native real-time cumulative population-map output.",
    ]
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(output / "summary.md"), "split": args.split}, indent=2))


if __name__ == "__main__":
    main()
