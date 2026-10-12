#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sphererate.baselines.registry import baseline_rows, load_baseline_registry
from sphererate.baselines.runner import build_baseline_task, load_batch_tasks, parse_cli_vars, run_task


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    list_p = sub.add_parser("list")
    list_p.add_argument("--config", default="configs/baselines.yaml")
    show_p = sub.add_parser("show")
    show_p.add_argument("name")
    show_p.add_argument("--config", default="configs/baselines.yaml")

    run_p = sub.add_parser("run")
    run_p.add_argument("name")
    run_p.add_argument("--config", default="configs/baselines.yaml", help="Baseline registry YAML.")
    run_p.add_argument("--stage", default="train", choices=["train", "infer", "postprocess"])
    run_p.add_argument("--dataset", default="salient360")
    run_p.add_argument("--split", default="val")
    run_p.add_argument("--out")
    run_p.add_argument("--device", default="cuda")
    run_p.add_argument("--experiment-config", default="configs/default.yaml")
    run_p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="Template variable override.")
    run_p.add_argument("--dry-run", action="store_true")

    batch_p = sub.add_parser("batch")
    batch_p.add_argument("batch_config")
    batch_p.add_argument("--config", default="configs/baselines.yaml", help="Baseline registry YAML.")
    batch_p.add_argument("--device", default="cuda")
    batch_p.add_argument("--experiment-config", default="configs/default.yaml")
    batch_p.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    registry = load_baseline_registry(args.config)
    if args.cmd == "list":
        for row in baseline_rows(registry):
            exists = "ok" if row["exists"] else "missing"
            print(f"{row['name']:16s} {exists:7s} {row['category']:24s} {row['status']} :: {row['path']}")
    elif args.cmd == "show":
        item = registry.get("baselines", {}).get(args.name)
        if not item:
            raise SystemExit(f"Unknown baseline: {args.name}")
        for key, value in item.items():
            print(f"{key}: {value}")
    elif args.cmd == "run":
        task = build_baseline_task(
            registry,
            name=args.name,
            stage=args.stage,
            dataset=args.dataset,
            split=args.split,
            out=args.out,
            device=args.device,
            experiment_config=args.experiment_config,
            extra_vars=parse_cli_vars(args.set),
        )
        if not task.exists and args.name not in registry.get("ours", {}):
            raise SystemExit(f"Baseline source is missing for {args.name}: {task.status}")
        raise SystemExit(run_task(task, dry_run=args.dry_run))
    elif args.cmd == "batch":
        tasks = load_batch_tasks(
            registry,
            args.batch_config,
            default_device=args.device,
            default_experiment_config=args.experiment_config,
        )
        for task in tasks:
            if not task.exists and task.name not in registry.get("ours", {}):
                raise SystemExit(f"Baseline source is missing for {task.name}: {task.status}")
            code = run_task(task, dry_run=args.dry_run)
            if code != 0:
                raise SystemExit(code)


if __name__ == "__main__":
    main()
