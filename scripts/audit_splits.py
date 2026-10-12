#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sphererate.data.splits import audit_group_disjoint_splits
from sphererate.utils import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a SphereRate manifest for content/scene split leakage.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--fail-on-leak", action="store_true")
    args = parser.parse_args()

    manifest = Path(args.manifest)
    audit = audit_group_disjoint_splits(read_jsonl(manifest))
    payload = json.dumps(audit, indent=2, ensure_ascii=False)
    print(payload)
    if args.json_out:
        output = Path(args.json_out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload + "\n", encoding="utf-8")
    if args.fail_on_leak and not audit["zero_leakage"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
