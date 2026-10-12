from __future__ import annotations

import json

from sphererate.data.future_view import build_future_saliency_data_view


def _write_manifest(root, rows) -> None:
    root.mkdir(parents=True)
    (root / "manifest.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def test_future_view_unifies_shared_content_and_excludes_jufe_aoi(tmp_path) -> None:
    etdk2 = tmp_path / "etdk2"
    salient = tmp_path / "salient"
    sitzmann = tmp_path / "sitzmann"
    _write_manifest(
        etdk2,
        [
            {
                "dataset": "etdk2",
                "image_id": f"etdk2_P1_S000{subject}_t00",
                "source_image_id": "P1",
                "canonical_content_id": "salient360:P1",
                "subject_id": f"S000{subject}",
                "split": "test",
            }
            for subject in (2, 3)
        ],
    )
    _write_manifest(
        salient,
        [
            {"dataset": "salient360", "image_id": "P1", "split": "train"},
            {"dataset": "salient360", "image_id": "P2", "split": "val"},
        ],
    )
    _write_manifest(
        sitzmann,
        [
            {
                "dataset": "sitzmann",
                "image_id": f"sitzmann_room_{variant}",
                "source_image_id": f"room_{variant}",
                "split": "train" if variant == 0 else "test",
            }
            for variant in (0, 1)
        ],
    )

    out = build_future_saliency_data_view(
        etdk2_root=etdk2,
        salient360_root=salient,
        sitzmann_root=sitzmann,
        out_root=tmp_path / "out",
        split_file=tmp_path / "split.json",
        seed=11,
        val_fraction=0.2,
        test_fraction=0.2,
    )

    rows = [json.loads(line) for line in (out / "manifest.jsonl").read_text().splitlines()]
    info = json.loads((out / "dataset_info.json").read_text())
    shared = [row for row in rows if row["canonical_content_id"] == "salient360:P1"]
    rotations = [row for row in rows if row["canonical_content_id"] == "sitzmann:room"]

    assert len(shared) == 3
    assert len({row["split"] for row in shared}) == 1
    assert len(rotations) == 2
    assert len({row["split"] for row in rotations}) == 1
    assert {row["dataset"] for row in rows} == {"etdk2", "salient360", "sitzmann"}
    assert info["split_audit"]["zero_leakage"] is True
    assert info["forbidden_dataset_rows"] == 0
    assert (out / "etdk2_sessions.jsonl").is_file()
    assert (out / "salient360_static.jsonl").is_file()
    assert (out / "sitzmann_external.jsonl").is_file()
