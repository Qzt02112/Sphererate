from __future__ import annotations

import json

import pytest

from sphererate.data.prepare import _load_split_file


def test_load_split_file_accepts_prefixed_ids(tmp_path) -> None:
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps({"train": ["P1"], "val": ["P2"], "test": ["P3"]}), encoding="utf-8")

    assert _load_split_file(split_path, ["1", "2", "3"]) == {"1": "train", "2": "val", "3": "test"}


def test_load_split_file_requires_all_ids(tmp_path) -> None:
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps({"train": ["P1"], "val": [], "test": []}), encoding="utf-8")

    with pytest.raises(ValueError, match="missing"):
        _load_split_file(split_path, ["1", "2"])
