#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sphererate.config import apply_overrides, load_config, load_config as load_yaml, set_by_dotted
from sphererate.train import train_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--batch", default="configs/batch_smoke.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-existing", action="store_true", help="Skip runs whose best checkpoint already exists.")
    parser.add_argument("--only", nargs="*", default=None, help="Optional run names to execute from the batch file.")
    args = parser.parse_args()

    base = load_config(args.config)
    batch = load_yaml(args.batch)
    only = set(args.only or [])
    for run in batch.get("runs", []):
        run_name = str(run["name"])
        if only and run_name not in only:
            continue
        cfg = apply_overrides(base, run.get("overrides", {}))
        cfg = set_by_dotted(cfg, "train.model", run["model"])
        cfg = set_by_dotted(cfg, "train.device", args.device)
        cfg = set_by_dotted(cfg, "train.experiment", run_name)
        checkpoint = Path(cfg["paths"]["runs_root"]) / run_name / "checkpoints" / "best.pt"
        if args.skip_existing and checkpoint.exists():
            print(f"Skipping existing batch run: {run_name} ({checkpoint})")
            continue
        print(f"Starting batch run: {run_name} ({run['model']})")
        train_model(cfg, model_name=run["model"], device_name=args.device)


if __name__ == "__main__":
    main()
