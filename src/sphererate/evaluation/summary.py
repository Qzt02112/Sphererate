from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_METRICS = ["auc_judd", "nss", "sim", "kld", "sim_sph", "kld_sph", "macro_kld_sph"]


def discover_summary_files(root: str | Path) -> list[Path]:
    root = Path(root)
    return sorted(root.rglob("*summary.json"))


def load_summary_row(path: str | Path, name: str | None = None) -> dict[str, Any]:
    return load_summary_rows(path, name=name)[0]


def load_summary_rows(path: str | Path, name: str | None = None) -> list[dict[str, Any]]:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    row_name = name or path.parent.name
    base = {
        "name": row_name,
        "split": data.get("split", ""),
        "source": str(path),
    }

    rows: list[dict[str, Any]] = []
    mixed_row = {
        **base,
        "scope": "mixed",
        "num_images": data.get("num_images", ""),
    }
    mixed_row.update(data.get("scores", {}))
    for key, value in data.get("macro_scores", {}).items():
        mixed_row[f"macro_{key}"] = value
    rows.append(mixed_row)

    macro_scores = data.get("macro_scores") or {}
    if macro_scores:
        macro_row = {
            **base,
            "scope": "macro",
            "num_images": data.get("num_images", ""),
        }
        macro_row.update(macro_scores)
        for key, value in macro_scores.items():
            macro_row[f"macro_{key}"] = value
        rows.append(macro_row)

    for dataset_name, dataset_data in sorted((data.get("datasets") or {}).items()):
        dataset_row = {
            **base,
            "scope": f"dataset:{dataset_name}",
            "num_images": dataset_data.get("num_images", ""),
        }
        dataset_row.update(dataset_data.get("scores", {}))
        rows.append(dataset_row)
    for condition_name, condition_data in sorted((data.get("conditions") or {}).items()):
        condition_row = {
            **base,
            "scope": f"condition:{condition_name}",
            "num_images": condition_data.get("num_images", ""),
        }
        condition_row.update(condition_data.get("scores", {}))
        rows.append(condition_row)
    return rows


def _scope_order(scope: str) -> int:
    if scope == "mixed":
        return 0
    if scope == "macro":
        return 1
    return 2


def rows_to_markdown(rows: list[dict[str, Any]], metrics: list[str] | None = None) -> str:
    metrics = metrics or DEFAULT_METRICS
    headers = ["Method", "Split", "Scope", "N"] + metrics
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        vals = [
            str(row.get("name", "")),
            str(row.get("split", "")),
            str(row.get("scope", "mixed")),
            str(row.get("num_images", "")),
        ]
        for metric in metrics:
            value = row.get(metric, "")
            vals.append(f"{float(value):.4f}" if isinstance(value, (int, float)) else str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def summarize_reports(
    report_roots: list[str | Path] | str | Path,
    output_path: str | Path,
    metrics: list[str] | None = None,
) -> Path:
    rows: list[dict[str, Any]] = []
    if isinstance(report_roots, (str, Path)):
        report_roots = [report_roots]
    for root in report_roots:
        for path in discover_summary_files(root):
            rows.extend(load_summary_rows(path))
    rows.sort(
        key=lambda row: (
            row.get("split", ""),
            row.get("name", ""),
            _scope_order(str(row.get("scope", ""))),
            row.get("scope", ""),
        )
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Experiment Summary\n\n")
        f.write(rows_to_markdown(rows, metrics=metrics))
        f.write("\n## Sources\n\n")
        seen_sources: set[tuple[Any, Any, Any]] = set()
        for row in rows:
            key = (row.get("name"), row.get("split"), row.get("source"))
            if key in seen_sources:
                continue
            seen_sources.add(key)
            f.write(f"- {row.get('name')} ({row.get('split')}): `{row.get('source')}`\n")
    return output_path
