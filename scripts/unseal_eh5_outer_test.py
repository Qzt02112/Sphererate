#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sphererate.data.early_view import build_train_only_baselines, materialize_eh5_outer_test
from sphererate.utils import ensure_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="One-shot EH5 outer-test materialization.")
    parser.add_argument("--authorization", required=True)
    parser.add_argument(
        "--source-manifest",
        default="data/processed/p2_salient360_temporal_aux_seed20260711/manifest.jsonl",
    )
    parser.add_argument("--view-root", default="data/processed/early_horizon_salient360_v1")
    parser.add_argument("--report-dir", default="reports/eh5_outer_test_access")
    args = parser.parse_args()
    result = materialize_eh5_outer_test(
        source_manifest=args.source_manifest,
        view_root=args.view_root,
        authorization=args.authorization,
    )
    baseline_root = build_train_only_baselines(
        args.view_root,
        Path(args.view_root) / "baselines",
        include_outer_test=True,
        authorization=args.authorization,
    )
    metadata = json.loads((baseline_root / "metadata.json").read_text())
    if metadata["test_prediction_count"] != 13:
        raise RuntimeError("EH5 must produce exactly 13 train-only test baseline predictions")
    result.update(
        {
            "stage": "EH5",
            "authorization": args.authorization,
            "source_manifest": args.source_manifest,
            "view_root": args.view_root,
            "test_baseline_prediction_count": metadata["test_prediction_count"],
            "method_frozen_before_access": True,
            "outer_test_access_is_irreversible": True,
        }
    )
    report_dir = ensure_dir(args.report_dir)
    (report_dir / "access_log.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (report_dir / "summary.md").write_text(
        "# EH5 Outer-Test Access Log\n\n"
        "The frozen method passed EH0–EH4 before access.\n\n"
        f"- materialized targets: `{result['materialized_outer_test_count']}`\n"
        f"- train-only baseline predictions: `{result['test_baseline_prediction_count']}`\n"
        "- method/checkpoints/metrics frozen before access: `true`\n"
        "- status: outer-test is now used and must never be treated as unopened again\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
