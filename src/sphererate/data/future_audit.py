from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from sphererate.data.etdk2 import start_longitude_deg
from sphererate.data.future_view import ALLOWED_DATASETS, FORBIDDEN_DATASETS


VALID_SPLITS = {"train", "val", "test"}


def _read_rows(root: Path) -> list[dict[str, Any]]:
    path = root / "manifest.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Missing future-saliency manifest: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _quantiles(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(array)),
        "p25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "p75": float(np.quantile(array, 0.75)),
        "max": float(np.max(array)),
    }


def audit_future_saliency_data_view(root: str | Path) -> dict[str, Any]:
    root = Path(root)
    rows = _read_rows(root)
    issues: list[dict[str, Any]] = []
    splits_by_content: dict[str, set[str]] = defaultdict(set)
    dataset_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    durations: dict[str, list[float]] = defaultdict(list)
    gaze_valid_fractions: list[float] = []

    def issue(code: str, row: dict[str, Any], detail: str) -> None:
        issues.append({"code": code, "image_id": str(row.get("image_id") or ""), "detail": detail})

    for row in rows:
        dataset = str(row.get("dataset") or "").lower()
        if dataset in FORBIDDEN_DATASETS:
            issue("forbidden_dataset", row, f"dataset={dataset}")
            continue
        if dataset not in ALLOWED_DATASETS:
            issue("unknown_dataset", row, f"dataset={dataset}")
            continue
        dataset_rows[dataset].append(row)

        split = str(row.get("split") or "")
        if split not in VALID_SPLITS:
            issue("invalid_split", row, f"split={split!r}")
        canonical = str(row.get("canonical_content_id") or "")
        if not canonical:
            issue("missing_canonical_content", row, "canonical_content_id is empty")
        else:
            splits_by_content[canonical].add(split)

        required_paths = ["image", "target"]
        if dataset == "etdk2":
            required_paths.append("trace")
        else:
            required_paths.append("fixations")
        for key in required_paths:
            value = row.get(key)
            if not value or not Path(str(value)).is_file():
                issue("missing_artifact", row, f"{key}={value!r}")

        if row.get("duration_s") is not None:
            durations[dataset].append(float(row["duration_s"]))

        if dataset != "etdk2":
            continue
        try:
            start_index = int(row["start_index"])
            expected_lon = start_longitude_deg(start_index)
            declared_lon = float(row["start_longitude_deg"])
            if abs(expected_lon - declared_lon) > 1e-6:
                issue(
                    "start_longitude_mismatch",
                    row,
                    f"start_index={start_index}, expected={expected_lon}, declared={declared_lon}",
                )
            if abs(float(row.get("start_latitude_deg", 0.0))) > 1e-6:
                issue("invalid_start_latitude", row, f"latitude={row.get('start_latitude_deg')}")
            expected_erp_lon = float(((-declared_lon + 180.0) % 360.0) - 180.0)
            actual_erp_lon = float(row["start_erp_longitude_deg"])
            if abs(expected_erp_lon - actual_erp_lon) > 1e-6:
                issue(
                    "start_erp_longitude_mismatch",
                    row,
                    f"declared={declared_lon}, expected_erp={expected_erp_lon}, actual={actual_erp_lon}",
                )
        except (KeyError, TypeError, ValueError) as exc:
            issue("invalid_start_condition", row, str(exc))

        trace_path = Path(str(row.get("trace") or ""))
        if not trace_path.is_file():
            continue
        try:
            with np.load(trace_path) as trace:
                required_trace = {
                    "time_s", "gaze_texture_xy", "head_texture_xy", "gaze_erp_xy", "head_erp_xy",
                    "gaze_valid", "head_valid",
                }
                missing = sorted(required_trace - set(trace.files))
                if missing:
                    issue("trace_missing_arrays", row, f"missing={missing}")
                    continue
                time_s = np.asarray(trace["time_s"], dtype=np.float64)
                gaze_xy = np.asarray(trace["gaze_texture_xy"], dtype=np.float32)
                head_xy = np.asarray(trace["head_texture_xy"], dtype=np.float32)
                gaze_erp_xy = np.asarray(trace["gaze_erp_xy"], dtype=np.float32)
                head_erp_xy = np.asarray(trace["head_erp_xy"], dtype=np.float32)
                gaze_valid = np.asarray(trace["gaze_valid"], dtype=bool)
                head_valid = np.asarray(trace["head_valid"], dtype=bool)
                length = len(time_s)
                if any(
                    len(array) != length
                    for array in (gaze_xy, head_xy, gaze_erp_xy, head_erp_xy, gaze_valid, head_valid)
                ):
                    issue("trace_length_mismatch", row, f"time length={length}")
                if row.get("num_samples") is not None and int(row["num_samples"]) != length:
                    issue("declared_sample_count_mismatch", row, f"declared={row['num_samples']}, actual={length}")
                if not np.isfinite(time_s).all() or np.any(np.diff(time_s) < -1e-9):
                    issue("non_monotonic_timestamp", row, "time_s is non-finite or decreases")
                for name, xy, valid in (("gaze", gaze_xy, gaze_valid), ("head", head_xy, head_valid)):
                    valid_xy = xy[valid]
                    if not np.isfinite(valid_xy).all() or np.any(valid_xy < 0.0) or np.any(valid_xy > 1.0):
                        issue("invalid_valid_coordinates", row, f"{name} valid coordinates leave [0,1]")
                if not np.allclose(gaze_erp_xy[:, 0], gaze_xy[:, 0], equal_nan=True) or not np.allclose(
                    gaze_erp_xy[:, 1], 1.0 - gaze_xy[:, 1], equal_nan=True
                ):
                    issue("gaze_erp_coordinate_mismatch", row, "ERP coordinates do not flip Unity texture Y")
                if not np.allclose(head_erp_xy[:, 0], head_xy[:, 0], equal_nan=True) or not np.allclose(
                    head_erp_xy[:, 1], 1.0 - head_xy[:, 1], equal_nan=True
                ):
                    issue("head_erp_coordinate_mismatch", row, "ERP coordinates do not flip Unity texture Y")
                if length:
                    gaze_valid_fractions.append(float(gaze_valid.mean()))
        except Exception as exc:
            issue("unreadable_trace", row, repr(exc))

        target_path = Path(str(row.get("target") or ""))
        if not target_path.is_file():
            continue
        try:
            with np.load(target_path) as target:
                required_target = {"temporal", "cumulative", "final", "fixation", "area_weights"}
                missing = sorted(required_target - set(target.files))
                if missing:
                    issue("target_missing_arrays", row, f"missing={missing}")
                for key in required_target & set(target.files):
                    if not np.isfinite(np.asarray(target[key])).all():
                        issue("non_finite_target", row, f"array={key}")
        except Exception as exc:
            issue("unreadable_target", row, repr(exc))

    for canonical, splits in sorted(splits_by_content.items()):
        if len(splits) > 1:
            issues.append(
                {
                    "code": "canonical_content_leakage",
                    "image_id": canonical,
                    "detail": f"splits={sorted(splits)}",
                }
            )

    summaries: dict[str, Any] = {}
    for dataset in sorted(dataset_rows):
        subset = dataset_rows[dataset]
        summary: dict[str, Any] = {
            "rows": len(subset),
            "contents": len({row.get("canonical_content_id") for row in subset}),
            "splits": {split: sum(row.get("split") == split for row in subset) for split in sorted(VALID_SPLITS)},
        }
        if dataset == "etdk2":
            sessions_by_content = Counter(str(row.get("canonical_content_id")) for row in subset)
            starts_by_content: dict[str, set[int]] = defaultdict(set)
            sessions_by_content_start: Counter[tuple[str, int]] = Counter()
            for row in subset:
                content = str(row.get("canonical_content_id"))
                start = int(row.get("start_index"))
                starts_by_content[content].add(start)
                sessions_by_content_start[(content, start)] += 1
            summary.update(
                observers=len({row.get("subject_id") for row in subset}),
                start_indices=dict(sorted(Counter(str(row.get("start_index")) for row in subset).items())),
                duration_s=_quantiles(durations[dataset]),
                gaze_valid_fraction=_quantiles(gaze_valid_fractions),
                sessions_per_content=_quantiles([float(value) for value in sessions_by_content.values()]),
                observed_starts_per_content=_quantiles(
                    [float(len(values)) for values in starts_by_content.values()]
                ),
                sessions_per_nonempty_content_start=_quantiles(
                    [float(value) for value in sessions_by_content_start.values()]
                ),
                empty_content_start_cells=len(starts_by_content) * 8 - len(sessions_by_content_start),
            )
        summaries[dataset] = summary

    code_counts = Counter(item["code"] for item in issues)
    return {
        "dataset": "future_saliency_data_view",
        "root": str(root),
        "pass": not issues,
        "row_count": len(rows),
        "canonical_content_count": len(splits_by_content),
        "critical_issue_count": len(issues),
        "critical_issue_codes": dict(sorted(code_counts.items())),
        "critical_issues": issues,
        "datasets": summaries,
        "test_targets_used_for_model_selection": False,
        "notes": "This is an integrity/provenance audit only; no model scores or target-driven selection are computed.",
    }


def write_future_saliency_audit(report: dict[str, Any], json_path: Path, markdown_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    lines = [
        "# Future Saliency Data Readiness Audit",
        "",
        f"- Decision: **{'PASS' if report['pass'] else 'FAIL'}**",
        f"- Rows: {report['row_count']}",
        f"- Canonical contents: {report['canonical_content_count']}",
        f"- Critical issues: {report['critical_issue_count']}",
        f"- Test targets used for model selection: `{str(report['test_targets_used_for_model_selection']).lower()}`",
        "",
        "## Dataset summary",
        "",
        "| Dataset | Rows | Contents | Train | Val | Test |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for dataset, data in sorted(report["datasets"].items()):
        splits = data["splits"]
        lines.append(
            f"| {dataset} | {data['rows']} | {data['contents']} | "
            f"{splits.get('train', 0)} | {splits.get('val', 0)} | {splits.get('test', 0)} |"
        )
    etdk2 = report["datasets"].get("etdk2")
    if etdk2:
        lines.extend(
            [
                "",
                "ET-DK2 has 18 sessions per content, but start-conditioned cells are sparse: "
                f"observed starts per content range from {etdk2['observed_starts_per_content']['min']:.0f} "
                f"to {etdk2['observed_starts_per_content']['max']:.0f}, with "
                f"{etdk2['empty_content_start_cells']} empty content-start cells. "
                "D1 must therefore avoid treating each image-start cell as a reliable population target.",
            ]
        )
    lines.extend(["", "## Critical issues", ""])
    if report["critical_issues"]:
        for item in report["critical_issues"][:100]:
            lines.append(f"- `{item['code']}` `{item['image_id']}`: {item['detail']}")
    else:
        lines.append("None. Canonical-content leakage is zero and all required artifacts passed integrity checks.")
    lines.extend(["", report["notes"], ""])
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
