from __future__ import annotations

from collections import Counter, defaultdict
import re
from typing import Any, Iterable

import numpy as np


VALID_SPLITS = ("train", "val", "test")
GROUP_RULES = {
    "jufe": "numeric content id before the first underscore",
    "sitzmann": "panorama/scene id after removing the final numeric variant suffix",
    "salient360": "image id",
    "jufe_start_conditioned": "numeric JUFE content id before the first underscore",
    "etdk2": "explicit Salient360-linked canonical panorama id",
    "canonical": "explicit canonical_content_id shared across dataset aliases",
}


def _source_id(row: dict[str, Any]) -> str:
    source = str(row.get("source_image_id") or row.get("image_id") or "")
    if not source:
        raise ValueError("Split row has no source_image_id or image_id.")
    return source


def _strip_dataset_prefix(source: str, dataset: str) -> str:
    prefixes = [f"{dataset}_"]
    if dataset == "jufe_start_conditioned":
        prefixes.append("jufe_")
    for prefix in prefixes:
        if source.startswith(prefix):
            return source[len(prefix) :]
    return source


def group_id_for_row(row: dict[str, Any]) -> str:
    dataset = str(row.get("dataset") or "").lower()
    canonical = str(row.get("canonical_content_id") or "").strip()
    if canonical:
        return f"content:{canonical}"
    source = _strip_dataset_prefix(_source_id(row), dataset)

    if dataset in {"jufe", "jufe_start_conditioned"}:
        match = re.match(r"^(?P<content>\d+)(?:_|$)", source)
        if not match:
            raise ValueError(f"Cannot derive JUFE content group from source id: {source}")
        return f"jufe:{match.group('content')}"

    if dataset == "sitzmann":
        if re.fullmatch(r"cubemap_\d{4}", source):
            scene = source
        else:
            scene = re.sub(r"_\d+$", "", source)
        if not scene:
            raise ValueError(f"Cannot derive Sitzmann scene group from source id: {source}")
        return f"sitzmann:{scene}"

    if dataset == "salient360":
        image_id = source if source.startswith("P") else f"P{source}"
        return f"salient360:{image_id}"

    if dataset == "etdk2":
        image_id = source if source.startswith("P") else f"P{source}"
        return f"content:salient360:{image_id}"

    raise ValueError(f"Unsupported dataset for group-disjoint split: {dataset or '<missing>'}")


def _group_family(group_id: str) -> str:
    parts = str(group_id).split(":")
    if len(parts) >= 3 and parts[0] == "content":
        return parts[1]
    return parts[0]


def _heldout_group_counts(num_groups: int, val_fraction: float, test_fraction: float) -> tuple[int, int]:
    if num_groups <= 1:
        return 0, 0
    n_test = max(1, int(round(num_groups * test_fraction))) if test_fraction > 0 else 0
    n_val = max(1, int(round(num_groups * val_fraction))) if val_fraction > 0 else 0
    while num_groups - n_test - n_val < 1:
        if n_test >= n_val and n_test > 0:
            n_test -= 1
        elif n_val > 0:
            n_val -= 1
        else:
            break
    return n_val, n_test


def assign_group_disjoint_splits(
    rows: Iterable[dict[str, Any]],
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> dict[str, str]:
    if not 0.0 <= val_fraction < 1.0 or not 0.0 <= test_fraction < 1.0:
        raise ValueError("val_fraction and test_fraction must be in [0, 1).")
    if val_fraction + test_fraction >= 1.0:
        raise ValueError("val_fraction + test_fraction must be less than 1.")

    groups_by_dataset: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        group_id = group_id_for_row(row)
        groups_by_dataset[_group_family(group_id)].add(group_id)

    rng = np.random.default_rng(int(seed))
    group_to_split: dict[str, str] = {}
    for dataset in sorted(groups_by_dataset):
        groups = sorted(groups_by_dataset[dataset])
        rng.shuffle(groups)
        n_val, n_test = _heldout_group_counts(len(groups), val_fraction, test_fraction)
        test_groups = set(groups[:n_test])
        val_groups = set(groups[n_test : n_test + n_val])
        for group_id in groups:
            group_to_split[group_id] = (
                "test" if group_id in test_groups else "val" if group_id in val_groups else "train"
            )
    return group_to_split


def audit_group_disjoint_splits(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    materialized = list(rows)
    splits_by_group: dict[str, set[str]] = defaultdict(set)
    samples_by_group: dict[str, list[str]] = defaultdict(list)
    dataset_by_group: dict[str, str] = {}
    sample_counts_by_dataset: dict[str, Counter[str]] = defaultdict(Counter)
    groups_by_dataset_split: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))

    for row in materialized:
        split = str(row.get("split") or "")
        if split not in VALID_SPLITS:
            raise ValueError(f"Invalid or missing split for {row.get('image_id')}: {split!r}")
        group_id = str(row.get("group_id") or group_id_for_row(row))
        dataset = _group_family(group_id)
        splits_by_group[group_id].add(split)
        samples_by_group[group_id].append(str(row.get("image_id") or row.get("source_image_id")))
        dataset_by_group[group_id] = dataset
        sample_counts_by_dataset[dataset][split] += 1
        groups_by_dataset_split[dataset][split].add(group_id)

    leaking_groups = [
        {
            "dataset": dataset_by_group[group_id],
            "group_id": group_id,
            "splits": sorted(splits),
            "sample_ids": sorted(samples_by_group[group_id]),
        }
        for group_id, splits in sorted(splits_by_group.items())
        if len(splits) > 1
    ]

    per_dataset: dict[str, Any] = {}
    for dataset in sorted(sample_counts_by_dataset):
        dataset_groups = [group_id for group_id, name in dataset_by_group.items() if name == dataset]
        per_dataset[dataset] = {
            "sample_count": int(sum(sample_counts_by_dataset[dataset].values())),
            "group_count": len(dataset_groups),
            "samples_by_split": {
                split: int(sample_counts_by_dataset[dataset].get(split, 0)) for split in VALID_SPLITS
            },
            "groups_by_split": {
                split: len(groups_by_dataset_split[dataset].get(split, set())) for split in VALID_SPLITS
            },
            "leaking_group_count": sum(item["dataset"] == dataset for item in leaking_groups),
        }

    return {
        "sample_count": len(materialized),
        "group_count": len(splits_by_group),
        "leaking_group_count": len(leaking_groups),
        "zero_leakage": len(leaking_groups) == 0,
        "leaking_groups": leaking_groups,
        "per_dataset": per_dataset,
    }


def audit_conditioned_pairs(
    rows: Iterable[dict[str, Any]],
    required_conditions: tuple[str, ...] = ("good", "bad"),
) -> dict[str, Any]:
    """Audit group leakage and paired-condition completeness for a conditioned manifest."""
    materialized = list(rows)
    group_audit = audit_group_disjoint_splits(materialized)
    conditions_by_base: dict[str, Counter[str]] = defaultdict(Counter)
    splits_by_base: dict[str, set[str]] = defaultdict(set)
    groups_by_base: dict[str, set[str]] = defaultdict(set)
    for row in materialized:
        base_id = str(row.get("base_image_id") or row.get("source_image_id") or "")
        if not base_id:
            raise ValueError(f"Conditioned row has no base/source image id: {row.get('image_id')}")
        condition = str(row.get("condition") or "").lower()
        conditions_by_base[base_id][condition] += 1
        splits_by_base[base_id].add(str(row.get("split") or ""))
        groups_by_base[base_id].add(str(row.get("group_id") or group_id_for_row(row)))

    required = set(required_conditions)
    incomplete_pairs: list[dict[str, Any]] = []
    duplicate_conditions: list[dict[str, Any]] = []
    split_mismatches: list[dict[str, Any]] = []
    group_mismatches: list[dict[str, Any]] = []
    for base_id in sorted(conditions_by_base):
        counter = conditions_by_base[base_id]
        present = {condition for condition, count in counter.items() if count > 0}
        if present != required:
            incomplete_pairs.append(
                {"base_image_id": base_id, "present": sorted(present), "required": sorted(required)}
            )
        duplicates = {condition: count for condition, count in sorted(counter.items()) if count != 1}
        if duplicates:
            duplicate_conditions.append({"base_image_id": base_id, "counts": duplicates})
        if len(splits_by_base[base_id]) != 1:
            split_mismatches.append({"base_image_id": base_id, "splits": sorted(splits_by_base[base_id])})
        if len(groups_by_base[base_id]) != 1:
            group_mismatches.append({"base_image_id": base_id, "groups": sorted(groups_by_base[base_id])})

    pair_complete = not (incomplete_pairs or duplicate_conditions or split_mismatches or group_mismatches)
    return {
        **group_audit,
        "base_image_count": len(conditions_by_base),
        "required_conditions": list(required_conditions),
        "pair_complete": pair_complete,
        "incomplete_pair_count": len(incomplete_pairs),
        "duplicate_condition_count": len(duplicate_conditions),
        "pair_split_mismatch_count": len(split_mismatches),
        "pair_group_mismatch_count": len(group_mismatches),
        "incomplete_pairs": incomplete_pairs,
        "duplicate_conditions": duplicate_conditions,
        "pair_split_mismatches": split_mismatches,
        "pair_group_mismatches": group_mismatches,
    }


def apply_group_disjoint_splits(
    rows: Iterable[dict[str, Any]],
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    materialized = [dict(row) for row in rows]
    group_to_split = assign_group_disjoint_splits(materialized, val_fraction, test_fraction, seed)
    for row in materialized:
        group_id = group_id_for_row(row)
        row["group_id"] = group_id
        row["split"] = group_to_split[group_id]

    audit = audit_group_disjoint_splits(materialized)
    spec = {
        "version": 1,
        "seed": int(seed),
        "val_fraction": float(val_fraction),
        "test_fraction": float(test_fraction),
        "group_rules": GROUP_RULES,
        "group_to_split": dict(sorted(group_to_split.items())),
    }
    return materialized, spec, audit
