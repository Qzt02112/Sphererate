from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any

import numpy as np

from sphererate.utils import ensure_dir, read_jsonl, write_jsonl


def _require_unique(rows: list[dict[str, Any]], key: str) -> None:
    values = [str(row.get(key) or "") for row in rows]
    duplicates = [value for value, count in Counter(values).items() if not value or count > 1]
    if duplicates:
        raise ValueError(f"P2 view requires unique non-empty {key}; invalid={duplicates[:5]}")


def _audit_paths(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        for key in ("image", "target", "fixations"):
            if not Path(str(row.get(key) or "")).is_file():
                raise FileNotFoundError(f"Missing P2 {key} for {row.get('image_id')}: {row.get(key)}")


def build_p2_temporal_aux_view(
    source_manifest: str | Path,
    out_root: str | Path,
    *,
    seed: int = 20260711,
    dev_count: int = 9,
) -> Path:
    source_manifest = Path(source_manifest)
    rows = read_jsonl(source_manifest)
    if not rows:
        raise ValueError(f"Empty source manifest: {source_manifest}")
    if any(str(row.get("dataset") or "").lower() != "salient360" for row in rows):
        raise ValueError("P2 training view accepts Salient360 rows only")
    _require_unique(rows, "image_id")
    _require_unique(rows, "canonical_content_id")
    _audit_paths(rows)

    outer_train = sorted((row for row in rows if row.get("split") == "train"), key=lambda row: row["image_id"])
    outer_val = sorted((row for row in rows if row.get("split") == "val"), key=lambda row: row["image_id"])
    outer_test = sorted((row for row in rows if row.get("split") == "test"), key=lambda row: row["image_id"])
    if not 0 < int(dev_count) < len(outer_train):
        raise ValueError(f"dev_count must be inside outer-train size, got {dev_count}/{len(outer_train)}")

    rng = np.random.default_rng(int(seed))
    dev_indices = set(int(value) for value in rng.choice(len(outer_train), size=int(dev_count), replace=False))
    assigned: list[dict[str, Any]] = []
    for index, row in enumerate(outer_train):
        assigned.append({**row, "outer_split": "train", "split": "val" if index in dev_indices else "train"})
    assigned.extend({**row, "outer_split": "val", "split": "gate"} for row in outer_val)
    assigned.extend({**row, "outer_split": "test", "split": "test"} for row in outer_test)
    assigned.sort(key=lambda row: (str(row["split"]), str(row["image_id"])))

    split_groups = {
        split: {str(row["canonical_content_id"]) for row in assigned if row["split"] == split}
        for split in ("train", "val", "gate", "test")
    }
    zero_leakage = all(
        split_groups[left].isdisjoint(split_groups[right])
        for left_index, left in enumerate(split_groups)
        for right in list(split_groups)[left_index + 1 :]
    )
    if not zero_leakage:
        raise RuntimeError("P2 nested split leaked canonical contents")

    out_root = ensure_dir(out_root)
    manifest = out_root / "manifest.jsonl"
    write_jsonl(assigned, manifest)
    counts = Counter(str(row["split"]) for row in assigned)
    info = {
        "dataset": "p2_salient360_temporal_aux",
        "seed": int(seed),
        "source_manifest": str(source_manifest),
        "counts": {split: int(counts.get(split, 0)) for split in ("train", "val", "gate", "test")},
        "zero_leakage": bool(zero_leakage),
        "test_used": False,
        "split_roles": {"train": "fit", "val": "dev_checkpoint", "gate": "frozen_decision", "test": "unopened"},
    }
    with (out_root / "dataset_info.json").open("w", encoding="utf-8") as handle:
        json.dump(info, handle, indent=2, ensure_ascii=False)
    return manifest


def build_standard_salient360_view(
    source_manifest: str | Path,
    split_file: str | Path,
    out_root: str | Path,
) -> Path:
    """Build the paper-facing 59/13/13 Salient360 source view.

    The source manifest supplies files only.  Its historical ``split`` field is
    deliberately ignored: assignments come from the project-wide canonical
    content split so that the former outer train/validation/test partition is
    preserved exactly.
    """

    source_manifest = Path(source_manifest)
    split_file = Path(split_file)
    rows = read_jsonl(source_manifest)
    if not rows:
        raise ValueError(f"Empty source manifest: {source_manifest}")
    if any(str(row.get("dataset") or "").lower() != "salient360" for row in rows):
        raise ValueError("Standard view accepts Salient360 rows only")
    _require_unique(rows, "image_id")
    _audit_paths(rows)

    split_payload = json.loads(split_file.read_text(encoding="utf-8"))
    group_to_split = {
        str(group): str(split)
        for group, split in split_payload.get("group_to_split", {}).items()
        if str(group).startswith("content:salient360:")
    }
    assigned: list[dict[str, Any]] = []
    for source in rows:
        image_id = str(source["image_id"])
        canonical = str(source.get("canonical_content_id") or f"salient360:{image_id}")
        group = str(source.get("group_id") or f"content:{canonical}")
        if group not in group_to_split:
            raise KeyError(f"No canonical split assignment for {group}")
        split = group_to_split[group]
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported standard split {split!r} for {group}")
        assigned.append(
            {
                **source,
                "canonical_content_id": canonical,
                "group_id": group,
                "source_manifest_split": source.get("split"),
                "split": split,
                "split_protocol": "salient360_standard_train59_val13_test13",
            }
        )
    assigned.sort(key=lambda row: (str(row["split"]), str(row["image_id"])))

    split_groups = {
        split: {str(row["canonical_content_id"]) for row in assigned if row["split"] == split}
        for split in ("train", "val", "test")
    }
    zero_leakage = all(
        split_groups[left].isdisjoint(split_groups[right])
        for left_index, left in enumerate(split_groups)
        for right in list(split_groups)[left_index + 1 :]
    )
    if not zero_leakage:
        raise RuntimeError("Standard split leaked canonical contents")

    counts = Counter(str(row["split"]) for row in assigned)
    out_root = ensure_dir(out_root)
    manifest = out_root / "manifest.jsonl"
    write_jsonl(assigned, manifest)
    info = {
        "dataset": "salient360_standard_source_view",
        "protocol": "train59_val13_test13",
        "source_manifest": str(source_manifest),
        "split_file": str(split_file),
        "split_file_version": split_payload.get("version"),
        "split_file_seed": split_payload.get("seed"),
        "counts": {split: int(counts.get(split, 0)) for split in ("train", "val", "test")},
        "zero_leakage": bool(zero_leakage),
        "split_roles": {"train": "fit", "val": "checkpoint_and_hyperparameter_selection", "test": "report"},
        "test_access_policy": "ordinary_repeatable_evaluation; never_model_selection",
    }
    (out_root / "dataset_info.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest
