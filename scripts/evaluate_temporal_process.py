#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sphererate.config import apply_overrides, load_config, load_config as load_yaml
from sphererate.evaluation.process import evaluate_process_prediction_dir
from sphererate.evaluation.summary import summarize_reports


PROCESS_METRICS = [
    "temporal_kld_sph",
    "temporal_sim_sph",
    "temporal_cc_sph",
    "temporal_step_tv_pred",
    "temporal_step_tv_gt",
    "temporal_step_tv_abs_gap",
    "cumulative_kld_sph",
    "cumulative_sim_sph",
    "cumulative_cc_sph",
    "cumulative_step_tv_abs_gap",
]


def _runs(batch_path: str | Path) -> list[dict]:
    batch = load_yaml(batch_path)
    return list(batch.get("runs", []))


def _load_image_ids(path: str | Path | None) -> set[str] | None:
    if path is None:
        return None
    with open(path, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def _history_is_complete(run_dir: Path, expected_epochs: int) -> bool:
    history_path = run_dir / "history.json"
    if not history_path.exists():
        return False
    import json

    with open(history_path, "r", encoding="utf-8") as f:
        history = json.load(f)
    if not history:
        return False
    last_epoch = int(history[-1].get("epoch", 0))
    return len(history) >= expected_epochs and last_epoch >= expected_epochs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--batch", required=True)
    parser.add_argument("--splits", nargs="+", default=["val", "test"], choices=["train", "val", "test"])
    parser.add_argument("--report-root", default=None)
    parser.add_argument("--summary-out", default=None)
    parser.add_argument("--image-id-file", default=None)
    parser.add_argument("--skip-missing", action="store_true")
    parser.add_argument("--skip-incomplete", action="store_true")
    args = parser.parse_args()

    base = load_config(args.config)
    processed_root = Path(base["paths"]["processed_root"]) / base["data"]["processed_name"]
    runs_root = Path(base["paths"]["runs_root"])
    report_root = Path(args.report_root or Path(base["paths"]["reports_root"]) / (Path(args.batch).stem + "_process"))
    image_ids = _load_image_ids(args.image_id_file)

    for run in _runs(args.batch):
        run_name = str(run["name"])
        expected_cfg = apply_overrides(base, run.get("overrides", {}))
        expected_epochs = int(expected_cfg.get("train", {}).get("epochs", 0))
        condition_prefix_bins = int(expected_cfg.get("train", {}).get("condition_prefix_bins", 0) or 0)
        run_dir = runs_root / run_name
        if args.skip_incomplete and expected_epochs > 0 and not _history_is_complete(run_dir, expected_epochs):
            print(f"Skipping incomplete run: {run_name} (expected {expected_epochs} epochs)")
            continue
        for split in args.splits:
            pred_dir = runs_root / run_name / "predictions" / split
            if not pred_dir.exists():
                if args.skip_missing:
                    print(f"Skipping missing predictions: {pred_dir}")
                    continue
                raise FileNotFoundError(f"Missing predictions: {pred_dir}")
            out_dir = report_root / f"{run_name}_{split}"
            summary = evaluate_process_prediction_dir(
                processed_root,
                pred_dir,
                split=split,
                output_dir=out_dir,
                image_ids=image_ids,
                condition_prefix_bins=condition_prefix_bins,
            )
            print(f"{run_name}/{split}: {summary['scores']}")

    summary_out = args.summary_out or report_root / "summary.md"
    summarize_reports(str(report_root), str(summary_out), metrics=PROCESS_METRICS)
    print(f"Wrote {summary_out}")


if __name__ == "__main__":
    main()
