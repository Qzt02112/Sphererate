from __future__ import annotations

import json
from pathlib import Path

import pytest

from sphererate.data.early_view import build_early_horizon_view, materialize_eh5_outer_test, read_manifest


def _scanpath(path: Path) -> None:
    path.write_text(
        "Idx, longitude, latitude, start timestamp\n"
        "0,0.75,0.5,100\n1,0.8,0.5,900\n"
        "0,0.74,0.5,50\n1,0.7,0.5,1000\n",
        encoding="utf-8",
    )


def test_outer_test_materialization_requires_explicit_eh5_authorization(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    rows = []
    # Production guard requires exactly 13 test rows.
    for index in range(13):
        raw = tmp_path / f"P{index}_R.txt"
        _scanpath(raw)
        rows.append(
            {
                "image_id": f"P{index}",
                "canonical_content_id": f"salient360:P{index}",
                "split": "test",
                "image": str(tmp_path / f"P{index}.png"),
                "raw_scanpath_r": str(raw),
                "raw_scanpath_l": str(raw),
            }
        )
    source.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    root = build_early_horizon_view(
        source_manifest=source,
        out_root=tmp_path / "view",
        horizons_s=(0.5, 1.0),
        height=8,
        width=16,
        sigma_px=1.0,
        seal_test_targets=True,
    )
    with pytest.raises(PermissionError):
        materialize_eh5_outer_test(source_manifest=source, view_root=root, authorization="wrong")

    result = materialize_eh5_outer_test(
        source_manifest=source,
        view_root=root,
        authorization="EH5_OUTER_TEST_ONCE",
    )

    assert result["materialized_outer_test_count"] == 13
    assert all(not row["target_sealed"] for row in read_manifest(root / "manifest.jsonl"))
    with pytest.raises(RuntimeError, match="already materialized"):
        materialize_eh5_outer_test(
            source_manifest=source,
            view_root=root,
            authorization="EH5_OUTER_TEST_ONCE",
        )
