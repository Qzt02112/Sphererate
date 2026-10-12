from __future__ import annotations

import hashlib
import io
import zipfile

import pytest

from sphererate.data.download_etdk2 import (
    REQUIRED_FILES,
    file_md5,
    select_required_files,
    verified_file,
    safe_extract_zip,
)


def test_select_required_files_uses_official_metadata() -> None:
    metadata = {
        "files": [
            {
                "key": name,
                "checksum": f"md5:{index:032x}",
                "links": {"content": f"https://example.test/{name}"},
                "size": index + 1,
            }
            for index, name in enumerate(sorted(REQUIRED_FILES), start=1)
        ]
    }

    selected = select_required_files(metadata)

    assert set(selected) == REQUIRED_FILES
    assert selected["gaze.zip"]["checksum"].startswith("md5:")


def test_select_required_files_rejects_missing_file() -> None:
    with pytest.raises(ValueError, match="missing required files"):
        select_required_files({"files": []})


def test_verified_file_checks_md5(tmp_path) -> None:
    path = tmp_path / "payload.bin"
    path.write_bytes(b"etdk2")
    expected = hashlib.md5(b"etdk2").hexdigest()

    assert file_md5(path) == expected
    assert verified_file(path, f"md5:{expected}") is True
    assert verified_file(path, "md5:" + "0" * 32) is False


def test_safe_extract_zip_rejects_path_traversal(tmp_path) -> None:
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../escape.csv", "bad")

    with pytest.raises(ValueError, match="Unsafe ZIP member"):
        safe_extract_zip(archive, tmp_path / "out")


def test_safe_extract_zip_extracts_regular_members(tmp_path) -> None:
    archive = tmp_path / "good.zip"
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as zf:
        zf.writestr("gaze/S0002.csv", "SUB_ID,IMG_INDEX\n2,1\n")
    archive.write_bytes(payload.getvalue())

    names = safe_extract_zip(archive, tmp_path / "out")

    assert names == ["gaze/S0002.csv"]
    assert (tmp_path / "out" / "gaze" / "S0002.csv").is_file()
