#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sphererate.data.future_audit import audit_future_saliency_data_view, write_future_saliency_audit


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit an explicitly supplied historical combined future-saliency view."
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Combined view root; no default is provided because the old view was invalidated.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("reports/data_readiness/future_saliency_v1.json"),
    )
    parser.add_argument(
        "--markdown-out",
        type=Path,
        default=Path("reports/data_readiness/future_saliency_v1.md"),
    )
    args = parser.parse_args()

    report = audit_future_saliency_data_view(args.root)
    write_future_saliency_audit(report, args.json_out, args.markdown_out)
    print(f"Data readiness: {'PASS' if report['pass'] else 'FAIL'}; issues={report['critical_issue_count']}")
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
