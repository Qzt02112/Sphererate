from __future__ import annotations

import json

import numpy as np

from sphererate.data.future_audit import audit_future_saliency_data_view


def _write_view(root, rows) -> None:
    root.mkdir(parents=True)
    (root / "manifest.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def _etdk2_row(tmp_path, *, split="train", start_index=3, start_lon=0.0):
    image = tmp_path / "P1.png"
    image.write_bytes(b"image")
    target = tmp_path / "target.npz"
    np.savez_compressed(
        target,
        temporal=np.ones((2, 2, 4), dtype=np.float32),
        cumulative=np.ones((2, 2, 4), dtype=np.float32),
        final=np.ones((2, 4), dtype=np.float32),
        fixation=np.ones((2, 4), dtype=np.float32),
        area_weights=np.ones((2, 4), dtype=np.float32),
    )
    trace = tmp_path / "trace.npz"
    np.savez_compressed(
        trace,
        time_s=np.array([0.0, 1.0]),
        gaze_texture_xy=np.array([[0.2, 0.5], [0.3, 0.5]], dtype=np.float32),
        head_texture_xy=np.array([[0.2, 0.5], [0.3, 0.5]], dtype=np.float32),
        gaze_erp_xy=np.array([[0.2, 0.5], [0.3, 0.5]], dtype=np.float32),
        head_erp_xy=np.array([[0.2, 0.5], [0.3, 0.5]], dtype=np.float32),
        gaze_valid=np.array([True, True]),
        head_valid=np.array([True, True]),
    )
    return {
        "dataset": "etdk2",
        "image_id": "etdk2_P1_S0002_t00",
        "canonical_content_id": "salient360:P1",
        "group_id": "content:salient360:P1",
        "subject_id": "S0002",
        "start_index": start_index,
        "start_longitude_deg": start_lon,
        "start_erp_longitude_deg": -start_lon,
        "start_latitude_deg": 0.0,
        "duration_s": 1.0,
        "num_samples": 2,
        "split": split,
        "image": str(image),
        "trace": str(trace),
        "target": str(target),
    }


def test_future_data_audit_passes_valid_view(tmp_path) -> None:
    root = tmp_path / "view"
    _write_view(root, [_etdk2_row(tmp_path)])

    report = audit_future_saliency_data_view(root)

    assert report["pass"] is True
    assert report["critical_issue_count"] == 0
    assert report["datasets"]["etdk2"]["rows"] == 1


def test_future_data_audit_detects_start_mismatch_and_content_leakage(tmp_path) -> None:
    root = tmp_path / "view"
    train = _etdk2_row(tmp_path, split="train", start_index=3, start_lon=45.0)
    test = dict(train, image_id="etdk2_P1_S0003_t00", subject_id="S0003", split="test")
    _write_view(root, [train, test])

    report = audit_future_saliency_data_view(root)

    assert report["pass"] is False
    codes = {issue["code"] for issue in report["critical_issues"]}
    assert "start_longitude_mismatch" in codes
    assert "canonical_content_leakage" in codes


def test_future_data_audit_rejects_forbidden_dataset(tmp_path) -> None:
    root = tmp_path / "view"
    _write_view(
        root,
        [{"dataset": "aoi", "image_id": "aoi_1", "canonical_content_id": "aoi:1", "split": "train"}],
    )

    report = audit_future_saliency_data_view(root)

    assert report["pass"] is False
    assert report["critical_issues"][0]["code"] == "forbidden_dataset"
