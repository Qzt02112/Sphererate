#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sphererate.evaluation.summary import DEFAULT_METRICS, summarize_reports


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", action="append", required=True, help="Report root to scan; can be passed multiple times.")
    parser.add_argument("--out", default="reports/summary.md")
    parser.add_argument("--metrics", nargs="*", default=DEFAULT_METRICS)
    args = parser.parse_args()

    out = summarize_reports(args.root, args.out, metrics=args.metrics)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
