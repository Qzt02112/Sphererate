from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import zipfile

import requests


RECORD_ID = "4642612"
API_URL = f"https://zenodo.org/api/records/{RECORD_ID}"
REQUIRED_FILES = {"gaze.zip", "img_index.csv", "readme.txt", "AOIs.csv"}


def file_md5(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checksum_value(checksum: str) -> str:
    algorithm, separator, value = str(checksum).partition(":")
    if separator != ":" or algorithm.lower() != "md5" or len(value) != 32:
        raise ValueError(f"Expected an md5 checksum, got {checksum!r}")
    return value.lower()


def verified_file(path: Path, checksum: str) -> bool:
    return path.is_file() and file_md5(path) == _checksum_value(checksum)


def select_required_files(metadata: dict[str, Any]) -> dict[str, dict[str, Any]]:
    files = {str(item.get("key")): dict(item) for item in metadata.get("files", [])}
    missing = sorted(REQUIRED_FILES - set(files))
    if missing:
        raise ValueError(f"Zenodo record {RECORD_ID} is missing required files: {missing}")
    selected = {name: files[name] for name in sorted(REQUIRED_FILES)}
    for name, item in selected.items():
        _checksum_value(str(item.get("checksum", "")))
        links = item.get("links") or {}
        content_url = links.get("content") or links.get("self")
        if not content_url:
            raise ValueError(f"Zenodo file {name} has no content URL")
        item["content_url"] = content_url
    return selected


def safe_extract_zip(archive: Path, out_dir: Path) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    root = out_dir.resolve()
    extracted: list[str] = []
    with zipfile.ZipFile(archive, "r") as handle:
        for member in handle.infolist():
            target = (out_dir / member.filename).resolve()
            if target != root and root not in target.parents:
                raise ValueError(f"Unsafe ZIP member: {member.filename}")
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with handle.open(member, "r") as src, target.open("wb") as dst:
                while chunk := src.read(1024 * 1024):
                    dst.write(chunk)
            extracted.append(member.filename)
    return sorted(extracted)


def _download_file(url: str, destination: Path, expected_checksum: str, timeout: float) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.unlink(missing_ok=True)
    try:
        with requests.get(url, stream=True, timeout=timeout) as response:
            response.raise_for_status()
            with temporary.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
        if not verified_file(temporary, expected_checksum):
            actual = file_md5(temporary)
            raise RuntimeError(
                f"Checksum mismatch for {destination.name}: expected {_checksum_value(expected_checksum)}, got {actual}"
            )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def download_etdk2(out_root: Path, timeout: float = 120.0, extract: bool = True) -> dict[str, Any]:
    out_root.mkdir(parents=True, exist_ok=True)
    response = requests.get(API_URL, timeout=timeout)
    response.raise_for_status()
    metadata = response.json()
    selected = select_required_files(metadata)

    file_records: dict[str, Any] = {}
    for name, item in selected.items():
        destination = out_root / name
        checksum = str(item["checksum"])
        reused = verified_file(destination, checksum)
        if not reused:
            _download_file(str(item["content_url"]), destination, checksum, timeout)
        file_records[name] = {
            "path": str(destination),
            "size": int(destination.stat().st_size),
            "checksum": checksum,
            "verified": True,
            "reused": reused,
        }

    extracted_members: list[str] = []
    participant_files: list[str] = []
    if extract:
        gaze_root = out_root / "gaze"
        extracted_members = safe_extract_zip(out_root / "gaze.zip", gaze_root)
        participant_files = sorted(str(path.relative_to(out_root)) for path in gaze_root.rglob("S*.csv"))

    source_record = {
        "dataset": "ET-DK2",
        "record_id": RECORD_ID,
        "doi": metadata.get("doi") or "10.5281/zenodo.4642612",
        "version": (metadata.get("metadata") or {}).get("version") or "v3",
        "license": ((metadata.get("metadata") or {}).get("license") or {}).get("id") or "cc-by-4.0",
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "api_url": API_URL,
        "files": file_records,
        "extracted_members": extracted_members,
        "participant_files": participant_files,
    }
    with (out_root / "source_record.json").open("w", encoding="utf-8") as handle:
        json.dump(source_record, handle, indent=2, ensure_ascii=False)
    return source_record
