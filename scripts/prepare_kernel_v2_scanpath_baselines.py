#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sphererate.baselines.matched_protocol import load_matched_protocol
from sphererate.baselines.scanpath_export import export_matched_scanpaths


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the frozen matched scanpath baseline view.")
    parser.add_argument("--protocol", default="configs/kernel_v2_matched_baselines.yaml")
    parser.add_argument("--output-root", default="data/processed/kernel_v2_matched_scanpaths")
    parser.add_argument("--manifest")
    parser.add_argument("--length", type=int)
    parser.add_argument("--step-s", type=float)
    parser.add_argument("--start-pitch-deg", type=float)
    parser.add_argument("--start-yaw-deg", type=float)
    parser.add_argument("--expected-train", type=int)
    parser.add_argument("--expected-val", type=int)
    parser.add_argument("--expected-gate", type=int)
    args = parser.parse_args()
    protocol = load_matched_protocol(args.protocol)
    result = export_matched_scanpaths(
        args.manifest or protocol.manifest,
        args.output_root,
        length=int(args.length if args.length is not None else protocol.scanpath_length),
        step_s=float(args.step_s if args.step_s is not None else protocol.scanpath_step_s),
        start_pitch_deg=float(
            args.start_pitch_deg if args.start_pitch_deg is not None else protocol.start_pitch_deg
        ),
        start_yaw_deg=float(
            args.start_yaw_deg if args.start_yaw_deg is not None else protocol.start_yaw_deg
        ),
        allowed_splits=tuple(protocol.expected_content_counts),
    )
    expected = dict(protocol.expected_content_counts)
    for split, override in (
        ("train", args.expected_train),
        ("val", args.expected_val),
        ("gate", args.expected_gate),
    ):
        if override is not None:
            expected[split] = int(override)
    if result["content_counts"] != expected:
        raise RuntimeError(f"Content counts changed: {result['content_counts']} != {expected}")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
