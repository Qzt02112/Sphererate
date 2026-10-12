from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from sphererate.data.early_view import read_manifest


def audit_early_horizon_view(
    root: str | Path,
    *,
    expected_horizons_s: Sequence[float],
    integral_tolerance: float = 1e-4,
) -> dict[str, Any]:
    root = Path(root)
    rows = read_manifest(root / "manifest.jsonl")
    split_counts = Counter(str(row["split"]) for row in rows)
    groups: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        groups[str(row["canonical_content_id"])].add(str(row["split"]))
    leaking = {group: sorted(splits) for group, splits in groups.items() if len(splits) > 1}
    invalid_integrals = 0
    invalid_horizons = 0
    missing_non_test = 0
    outer_test_loaded = 0
    observer_counts: list[int] = []
    expected = np.asarray(expected_horizons_s, dtype=np.float32)

    for row in rows:
        if row["split"] == "test":
            if row.get("target") is not None or row.get("fixations") is not None or not row.get("target_sealed"):
                outer_test_loaded += 1
            continue
        target_path = Path(row["target"])
        if not target_path.is_file():
            missing_non_test += 1
            continue
        with np.load(target_path) as target:
            cumulative = np.asarray(target["cumulative"], dtype=np.float64)
            final = np.asarray(target["final"], dtype=np.float64)
            area = np.asarray(target["area_weights"], dtype=np.float64)
            horizons = np.asarray(target["horizons_s"], dtype=np.float32)
        if horizons.shape != expected.shape or not np.allclose(horizons, expected):
            invalid_horizons += 1
        integrals = np.concatenate(((cumulative * area).sum(axis=(1, 2)), [(final * area).sum()]))
        invalid_integrals += int(np.sum(np.abs(integrals - 1.0) > float(integral_tolerance)))
        fixation_path = row.get("fixations")
        if fixation_path and Path(fixation_path).is_file():
            with np.load(fixation_path) as data:
                observer_counts.append(len(np.unique(data["fixations"][:, 3])))

    zero_leakage = not leaking
    return {
        "row_count": len(rows),
        "split_counts": dict(split_counts),
        "zero_leakage": zero_leakage,
        "leaking_groups": leaking,
        "missing_non_test_target_count": missing_non_test,
        "outer_test_target_files_loaded": outer_test_loaded,
        "invalid_horizon_files": invalid_horizons,
        "invalid_spherical_integrals": invalid_integrals,
        "physical_observer_count": {
            "min": min(observer_counts) if observer_counts else None,
            "median": float(np.median(observer_counts)) if observer_counts else None,
            "max": max(observer_counts) if observer_counts else None,
        },
        "pass": bool(
            zero_leakage
            and missing_non_test == 0
            and outer_test_loaded == 0
            and invalid_horizons == 0
            and invalid_integrals == 0
        ),
    }
