from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
import re
from typing import Any

from sphererate.data.splits import apply_group_disjoint_splits
from sphererate.utils import ensure_dir, write_jsonl


ALLOWED_DATASETS = {"etdk2", "salient360", "sitzmann"}
FORBIDDEN_DATASETS = {"aoi", "jufe", "jufe_start_conditioned"}


def _read_manifest(root: Path) -> list[dict[str, Any]]:
    path = root / "manifest.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Missing component manifest: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _strip_prefix(value: str, prefix: str) -> str:
    return value[len(prefix) :] if value.startswith(prefix) else value


def _canonicalize(row: dict[str, Any], expected_dataset: str) -> dict[str, Any]:
    out = dict(row)
    dataset = str(out.get("dataset") or expected_dataset).lower()
    if dataset != expected_dataset:
        raise ValueError(f"Expected {expected_dataset} row, found dataset={dataset}")
    out["dataset"] = dataset

    if dataset == "etdk2":
        source = str(out.get("source_image_id") or out.get("base_image_id") or "")
        if not source.startswith("P"):
            raise ValueError(f"ET-DK2 primary view only accepts public P# stimuli, got {source!r}")
        canonical = str(out.get("canonical_content_id") or f"salient360:{source}")
        out["task_role"] = "primary_session"
    elif dataset == "salient360":
        source = str(out.get("source_image_id") or out.get("image_id") or "")
        source = _strip_prefix(source, "salient360_")
        source = source if source.startswith("P") else f"P{source}"
        canonical = f"salient360:{source}"
        out["task_role"] = "static_reference"
    else:
        source = str(out.get("source_image_id") or out.get("image_id") or "")
        source = _strip_prefix(source, "sitzmann_")
        if re.fullmatch(r"cubemap_\d{4}", source):
            scene = source
        else:
            scene = re.sub(r"_\d+$", "", source)
        if not scene:
            raise ValueError(f"Cannot derive Sitzmann scene from {source!r}")
        canonical = f"sitzmann:{scene}"
        out["task_role"] = "external_temporal"

    out["canonical_content_id"] = canonical
    out["group_id"] = f"content:{canonical}"
    return out


def build_future_saliency_data_view(
    *,
    etdk2_root: Path,
    salient360_root: Path,
    sitzmann_root: Path,
    out_root: Path,
    split_file: Path,
    seed: int = 20260711,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> Path:
    components = {
        "etdk2": Path(etdk2_root),
        "salient360": Path(salient360_root),
        "sitzmann": Path(sitzmann_root),
    }
    rows: list[dict[str, Any]] = []
    for dataset, root in components.items():
        rows.extend(_canonicalize(row, dataset) for row in _read_manifest(root))

    forbidden = [row for row in rows if str(row.get("dataset") or "").lower() in FORBIDDEN_DATASETS]
    unknown = [row for row in rows if str(row.get("dataset") or "").lower() not in ALLOWED_DATASETS]
    if forbidden or unknown:
        raise ValueError(f"Future-saliency view contains forbidden/unknown dataset rows: {len(forbidden) + len(unknown)}")

    assigned, split_spec, split_audit = apply_group_disjoint_splits(
        rows,
        val_fraction=float(val_fraction),
        test_fraction=float(test_fraction),
        seed=int(seed),
    )
    if not split_audit["zero_leakage"]:
        raise RuntimeError(f"Canonical-content split leaked {split_audit['leaking_group_count']} groups")

    out_root = ensure_dir(Path(out_root))
    write_jsonl(assigned, out_root / "manifest.jsonl")
    filenames = {
        "etdk2": "etdk2_sessions.jsonl",
        "salient360": "salient360_static.jsonl",
        "sitzmann": "sitzmann_external.jsonl",
    }
    for dataset, filename in filenames.items():
        write_jsonl([row for row in assigned if row["dataset"] == dataset], out_root / filename)

    split_file = Path(split_file)
    split_file.parent.mkdir(parents=True, exist_ok=True)
    with split_file.open("w", encoding="utf-8") as handle:
        json.dump(split_spec, handle, indent=2, ensure_ascii=False)

    row_counts = Counter(row["dataset"] for row in assigned)
    split_counts = {
        dataset: {
            split: sum(row["dataset"] == dataset and row["split"] == split for row in assigned)
            for split in ("train", "val", "test")
        }
        for dataset in sorted(ALLOWED_DATASETS)
    }
    content_datasets: dict[str, set[str]] = {}
    for row in assigned:
        content_datasets.setdefault(row["canonical_content_id"], set()).add(row["dataset"])
    info = {
        "dataset": "future_saliency_data_view",
        "version": 1,
        "seed": int(seed),
        "components": {dataset: str(root) for dataset, root in components.items()},
        "row_counts": dict(sorted(row_counts.items())),
        "split_counts": split_counts,
        "canonical_content_count": len(content_datasets),
        "shared_cross_dataset_content_count": sum(len(names) > 1 for names in content_datasets.values()),
        "forbidden_dataset_rows": len(forbidden),
        "split_file": str(split_file),
        "split_audit": split_audit,
        "task_roles": {
            "etdk2": "primary prefix-conditioned viewing sessions",
            "salient360": "static content reference only",
            "sitzmann": "external temporal benchmark; legacy rotation variants remain grouped",
        },
        "notes": "Manifest-only research view. Do not pass the mixed manifest directly to one training loader.",
    }
    with (out_root / "dataset_info.json").open("w", encoding="utf-8") as handle:
        json.dump(info, handle, indent=2, ensure_ascii=False)
    return out_root
