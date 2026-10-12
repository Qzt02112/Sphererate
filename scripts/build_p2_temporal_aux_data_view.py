#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sphererate.data.p2_view import build_p2_temporal_aux_view


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the nested Salient360 view for P2 temporal auxiliary training.")
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--dev-count", type=int, default=9)
    args = parser.parse_args()
    manifest = build_p2_temporal_aux_view(
        args.source_manifest,
        args.out_root,
        seed=args.seed,
        dev_count=args.dev_count,
    )
    print(f"Prepared P2 view at {manifest.parent}")


if __name__ == "__main__":
    main()

