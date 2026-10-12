#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sphererate.data.cross_dataset_view import build_cross_dataset_view


def main() -> None:
    parser = argparse.ArgumentParser(description="Build coordinate-only views for frozen cross-dataset tests.")
    parser.add_argument("--contract", default="configs/sitzmann_cross_dataset_benchmark.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    contract = yaml.safe_load((ROOT / args.contract).read_text(encoding="utf-8"))
    results = {}
    for name, view in contract["cross_views"].items():
        source = contract["datasets"][view["source_dataset"]]
        model_source = contract["datasets"][view["model_source"]]
        output = ROOT / view["output_root"]
        if output.exists():
            if not args.force:
                raise FileExistsError(f"{output} exists; pass --force to rebuild")
            shutil.rmtree(output)
        results[name] = build_cross_dataset_view(
            ROOT / source["root"],
            output,
            shift_px=int(view["shift_px"]),
            model_source=str(view["model_source"]),
            target_start_x_norm=float(model_source["start_x_norm"]),
            external_scanpath_start_yaw_deg=float(model_source["external_scanpath_start_yaw_deg"]),
            workspace_root=ROOT,
        )
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
