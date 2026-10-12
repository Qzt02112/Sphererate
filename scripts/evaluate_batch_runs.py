#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sphererate.config import apply_overrides, load_config, load_config as load_yaml
from sphererate.evaluation.evaluate import evaluate_prediction_dir
from sphererate.evaluation.summary import summarize_reports
from sphererate.train import _loader, load_model_for_inference, save_predictions


def _evaluation_config(config: dict) -> dict:
    cfg = dict(config)
    train_cfg = dict(cfg.get("train", {}))
    train_cfg["max_train_samples"] = None
    train_cfg["max_val_samples"] = None
    train_cfg["shuffle_temporal_labels"] = False
    cfg["train"] = train_cfg
    return cfg


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
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--report-root", default=None)
    parser.add_argument("--summary-out", default=None)
    parser.add_argument("--image-id-file", default=None)
    parser.add_argument("--skip-missing", action="store_true")
    parser.add_argument("--skip-incomplete", action="store_true")
    args = parser.parse_args()

    base = load_config(args.config)
    report_root = Path(args.report_root or Path(base["paths"]["reports_root"]) / Path(args.batch).stem)
    runs = _runs(args.batch)
    image_ids = _load_image_ids(args.image_id_file)

    for run in runs:
        run_name = str(run["name"])
        expected_cfg = apply_overrides(base, run.get("overrides", {}))
        expected_epochs = int(expected_cfg.get("train", {}).get("epochs", 0))
        run_dir = Path(base["paths"]["runs_root"]) / run_name
        checkpoint = run_dir / "checkpoints" / "best.pt"
        if not checkpoint.exists():
            if args.skip_missing:
                print(f"Skipping missing checkpoint: {checkpoint}")
                continue
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
        if args.skip_incomplete and expected_epochs > 0 and not _history_is_complete(run_dir, expected_epochs):
            print(f"Skipping incomplete run: {run_name} (expected {expected_epochs} epochs)")
            continue

        model, cfg, device = load_model_for_inference(checkpoint, args.device)
        cfg = _evaluation_config(cfg)
        condition_prefix_bins = int(cfg.get("train", {}).get("condition_prefix_bins", 0) or 0)
        show_progress = bool(cfg.get("train", {}).get("progress_bar", True))
        processed_root = Path(cfg["paths"]["processed_root"]) / cfg["data"]["processed_name"]
        for split in args.splits:
            loader = _loader(cfg, split, processed_root, augment=False)
            pred_dir = checkpoint.parents[1] / "predictions" / split
            save_predictions(model, loader, device, pred_dir, show_progress=show_progress)
            out_dir = report_root / f"{run_name}_{split}"
            summary = evaluate_prediction_dir(
                processed_root,
                pred_dir,
                split=split,
                output_dir=out_dir,
                image_ids=image_ids,
                condition_prefix_bins=condition_prefix_bins,
            )
            print(f"{run_name}/{split}: {summary['scores']}")

    summary_out = args.summary_out or report_root / "summary.md"
    summarize_reports(str(report_root), str(summary_out))
    print(f"Wrote {summary_out}")


if __name__ == "__main__":
    main()
