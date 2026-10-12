from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from sphererate.data.prepare import prepare_etdk2


ETDK2_COLUMNS = [
    "SUB_ID", "IMG_INDEX", "IS_GRAY", "GRAY_IDX", "UNITY_TIMESTAMP", "SMI_TIMESTAMP",
    "GIW_X", "GIW_Y", "GIW_Z", "GIW_TEXTURE_X", "GIW_TEXTURE_Y", "SCREEN_X", "SCREEN_Y",
    "HIW_X", "HIW_Y", "HIW_Z", "HIW_TEXTURE_X", "HIW_TEXTURE_Y",
    "GAZE_DIR_X", "GAZE_DIR_Y", "GAZE_DIR_Z", "HEAD_POS_X", "HEAD_POS_Y", "HEAD_POS_Z",
    "HEAD_EUL_X", "HEAD_EUL_Y", "HEAD_EUL_Z",
]


def _row(subject: int, image: int, gray: bool, start: int, timestamp: float) -> dict[str, object]:
    row: dict[str, object] = {name: 0.0 for name in ETDK2_COLUMNS}
    row.update(
        SUB_ID=subject,
        IMG_INDEX=image,
        IS_GRAY=gray,
        GRAY_IDX=start,
        UNITY_TIMESTAMP=timestamp,
        SMI_TIMESTAMP=int(timestamp * 1e9),
        GIW_X=1.0,
        GIW_TEXTURE_X=0.5,
        GIW_TEXTURE_Y=0.5,
        HIW_X=1.0,
        HIW_TEXTURE_X=0.5,
        HIW_TEXTURE_Y=0.5,
        GAZE_DIR_X=1.0,
    )
    return row


def _write_subject(path: Path, subject: int) -> None:
    rows = []
    time = 0.0
    for image, start in [(-1, 3), (0, 3), (-1, 5), (1, 5), (-1, 1), (40, 1)]:
        gray = image == -1
        repeats = 1 if gray else 3
        for _ in range(repeats):
            rows.append(_row(subject, image, gray, start, time))
            time += 0.5
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ETDK2_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def test_prepare_etdk2_emits_shared_session_rows_and_artifacts(tmp_path) -> None:
    raw_root = tmp_path / "raw"
    etdk2_root = raw_root / "etdk2"
    etdk2_root.mkdir(parents=True)
    (etdk2_root / "img_index.csv").write_text(
        "img_index,IMG\n0,P2\n1,P3\n40,C0\n", encoding="utf-8"
    )
    _write_subject(etdk2_root / "gaze" / "S0002.csv", 2)
    _write_subject(etdk2_root / "gaze" / "S0003.csv", 3)

    stimuli = raw_root / "salient360" / "stimuli"
    stimuli.mkdir(parents=True)
    for image_id in (2, 3):
        Image.fromarray(np.full((8, 16, 3), image_id, dtype=np.uint8)).save(
            stimuli / f"P{image_id}_16x8.jpg"
        )

    config = {
        "paths": {"raw_root": str(raw_root), "processed_root": str(tmp_path / "processed")},
        "data": {
            "dataset": "etdk2",
            "processed_name": "etdk2_test",
            "image_size": [8, 16],
            "num_bins": 2,
            "split_seed": 7,
            "val_fraction": 0.0,
            "test_fraction": 0.0,
            "sigma_px": 1.0,
            "area_weighted": True,
            "etdk2_min_session_samples": 2,
            "etdk2_min_session_duration_s": 0.5,
        },
    }

    out_root = prepare_etdk2(config)
    rows = [json.loads(line) for line in (out_root / "manifest.jsonl").read_text().splitlines()]
    info = json.loads((out_root / "dataset_info.json").read_text())

    assert len(rows) == 4
    assert {row["source_image_id"] for row in rows} == {"P2", "P3"}
    assert {row["canonical_content_id"] for row in rows} == {"salient360:P2", "salient360:P3"}
    assert {row["subject_id"] for row in rows} == {"S0002", "S0003"}
    assert {row["start_erp_longitude_deg"] for row in rows} == {0.0, -90.0}
    assert all(row["split"] == "train" for row in rows)
    assert info["excluded_sessions"]["construction_stimulus_unavailable"] == 2

    for row in rows:
        assert Path(row["image"]).is_file()
        assert Path(row["trace"]).is_file()
        assert Path(row["target"]).is_file()
        with np.load(row["trace"]) as trace:
            assert trace["gaze_texture_xy"].shape == (3, 2)
            assert trace["gaze_erp_xy"].shape == (3, 2)
            assert np.all(np.diff(trace["time_s"]) >= 0)
        with np.load(row["fixations"]) as fixation_data:
            np.testing.assert_allclose(fixation_data["fixations"][:, 1], 0.5)
        with np.load(row["target"]) as target:
            assert target["temporal"].shape == (2, 8, 16)
            assert target["final"].shape == (8, 16)
