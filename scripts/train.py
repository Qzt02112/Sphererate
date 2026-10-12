#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sphererate.config import load_config, set_by_dotted
from sphererate.train import train_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--model",
        default=None,
        choices=[
            "static",
            "semantic_static",
            "semantic_static_aux",
            "parallel",
            "directsum",
            "tempsal_mixer",
            "semantic_tempsal_temporal",
            "semantic_tempsal_fusion",
            "residual_mixer",
            "dytsaldmm",
            "semantic_dytsaldmm",
            "semantic_teacher_residual",
        ],
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--experiment", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.model:
        config = set_by_dotted(config, "train.model", args.model)
        config = set_by_dotted(config, "model.num_bins", config["data"]["num_bins"])
    if args.device:
        config = set_by_dotted(config, "train.device", args.device)
    if args.experiment:
        config = set_by_dotted(config, "train.experiment", args.experiment)
    if args.epochs is not None:
        config = set_by_dotted(config, "train.epochs", args.epochs)
    if args.batch_size is not None:
        config = set_by_dotted(config, "train.batch_size", args.batch_size)
    if args.max_train_samples is not None:
        config = set_by_dotted(config, "train.max_train_samples", args.max_train_samples)
    if args.max_val_samples is not None:
        config = set_by_dotted(config, "train.max_val_samples", args.max_val_samples)
    run_dir = train_model(config, model_name=args.model, device_name=args.device)
    print(f"Run directory: {run_dir}")


if __name__ == "__main__":
    main()
