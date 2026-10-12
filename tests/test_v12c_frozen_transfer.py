from __future__ import annotations

import json
from pathlib import Path

from scripts.evaluate_v12c_frozen_etdk2 import _read_rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_etdk2_transfer_excludes_source_train_and_sealed_outer(tmp_path: Path) -> None:
    salient = tmp_path / "salient.jsonl"
    _write_jsonl(
        salient,
        [
            {"image_id": "P1", "split": "train"},
            {"image_id": "P2", "split": "val"},
            {"image_id": "P3", "split": "gate"},
            {"image_id": "P4", "split": "test"},
        ],
    )
    etdk2 = tmp_path / "etdk2.jsonl"
    _write_jsonl(
        etdk2,
        [
            {
                "dataset": "etdk2",
                "split": "val",
                "source_image_id": f"P{index}",
                "canonical_content_id": f"salient360:P{index}",
            }
            for index in range(1, 5)
        ]
        + [
            {
                "dataset": "etdk2",
                "split": "test",
                "source_image_id": "P2",
                "canonical_content_id": "salient360:P2",
            }
        ],
    )

    rows, audit = _read_rows(etdk2, salient)

    assert [row["source_image_id"] for row in rows] == ["P2", "P3"]
    assert audit["excluded_etdk2_val_sessions_by_salient360_source_role"] == {
        "test": 1,
        "train": 1,
    }
    assert audit["etdk2_test_target_rows_loaded"] == 0
    assert audit["salient360_outer_test_target_rows_loaded"] == 0
