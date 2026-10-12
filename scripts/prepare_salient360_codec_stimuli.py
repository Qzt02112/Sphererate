#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

from PIL import Image
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
Image.MAX_IMAGE_PIXELS = None
EXPECTED_MD5 = "ac1a471e058177705ea96120af19ff2c"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def digest(path: Path, algorithm: str) -> str:
    result = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def image_id_from_path(path: Path) -> str | None:
    exact = re.fullmatch(r"[Pp]0*(\d+)", path.stem)
    if exact:
        return f"P{int(exact.group(1))}"
    token = re.search(r"(?:^|[_-])[Pp]0*(\d+)(?:[_-]|$)", path.stem)
    return f"P{int(token.group(1))}" if token else None


def discover_originals(extract_root: Path) -> dict[str, Path]:
    grouped: dict[str, list[Path]] = {}
    for path in sorted(extract_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        image_id = image_id_from_path(path)
        if image_id is not None:
            grouped.setdefault(image_id, []).append(path)
    ambiguous = {key: values for key, values in grouped.items() if len(values) != 1}
    if ambiguous:
        detail = {key: [str(path) for path in value] for key, value in ambiguous.items()}
        raise RuntimeError(f"Ambiguous original stimuli: {detail}")
    return {key: value[0] for key, value in grouped.items()}


def read_processed_manifest(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def alignment_audit(original_path: Path, processed_path: Path) -> tuple[int, float]:
    with Image.open(original_path) as image:
        original = np.asarray(
            image.convert("L").resize((512, 256), Image.Resampling.LANCZOS),
            dtype=np.float64,
        )
    with Image.open(processed_path) as image:
        processed = np.asarray(image.convert("L"), dtype=np.float64)
    if processed.shape != (256, 512):
        raise RuntimeError(f"Unexpected processed ERP size: {processed.shape}")
    original -= original.mean()
    processed -= processed.mean()
    correlation = np.fft.ifft(
        np.sum(
            np.conj(np.fft.fft(original, axis=1)) * np.fft.fft(processed, axis=1),
            axis=0,
        )
    ).real
    shift = int(np.argmax(correlation))
    denominator = float(np.sqrt(np.sum(original**2) * np.sum(processed**2)))
    ncc = float(correlation[shift] / max(denominator, 1e-12))
    return shift, ncc


def build_manifest(
    originals: dict[str, Path], processed_rows: list[dict[str, object]]
) -> list[dict[str, object]]:
    expected = {str(row["image_id"]) for row in processed_rows}
    missing = sorted(expected - set(originals))
    if missing:
        raise RuntimeError(f"Missing official ERP stimuli: {missing}")
    rows: list[dict[str, object]] = []
    processed_by_id = {str(row["image_id"]): row for row in processed_rows}
    for image_id in sorted(expected, key=lambda value: int(value[1:])):
        path = originals[image_id]
        with Image.open(path) as image:
            width, height = image.size
            mode = image.mode
        if width <= 512 and height <= 256:
            raise RuntimeError(f"{image_id} is not an original-resolution ERP: {width}x{height}")
        processed_path = processed_by_id[image_id].get("image")
        shift, ncc = (None, None)
        if processed_path is not None:
            shift, ncc = alignment_audit(path, Path(str(processed_path)))
            if shift != 0 or ncc < 0.98:
                raise RuntimeError(
                    f"Original/processed ERP misalignment for {image_id}: shift={shift}, NCC={ncc:.4f}"
                )
        rows.append(
            {
                "image_id": image_id,
                "canonical_content_id": f"salient360:{image_id}",
                "split": str(processed_by_id[image_id]["split"]),
                "source_path": str(path),
                "sha256": digest(path, "sha256"),
                "width": width,
                "height": height,
                "mode": mode,
                "processed_alignment_shift_px_512": shift,
                "processed_alignment_ncc": ncc,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit and index official Salient360 ERPs.")
    parser.add_argument(
        "--archive", default="data/raw/salient360_official/Image_Stimuli.7z"
    )
    parser.add_argument(
        "--extract-root", default="data/raw/salient360_official/Image_Stimuli"
    )
    parser.add_argument(
        "--processed-manifest",
        default="data/processed/early_horizon_salient360_spherical2deg_standard_v3/manifest.jsonl",
    )
    parser.add_argument(
        "--output", default="data/interim/salient360_codec_originals/manifest.jsonl"
    )
    parser.add_argument("--skip-extract", action="store_true")
    args = parser.parse_args()

    archive = Path(args.archive)
    if not archive.is_file():
        raise FileNotFoundError(archive)
    actual_md5 = digest(archive, "md5")
    if actual_md5 != EXPECTED_MD5:
        raise RuntimeError(f"Archive MD5 mismatch: {actual_md5} != {EXPECTED_MD5}")

    extract_root = Path(args.extract_root)
    if not args.skip_extract:
        extract_root.mkdir(parents=True, exist_ok=True)
        executable = shutil.which("7z") or shutil.which("7zz") or shutil.which("7za")
        if executable is not None:
            result = subprocess.run(
                [executable, "x", "-y", str(archive), f"-o{extract_root}"],
                check=False,
            )
            if result.returncode != 0:
                raise RuntimeError(f"7z extraction failed with code {result.returncode}")
        else:
            try:
                import py7zr
            except ImportError as error:
                raise RuntimeError("Install 7z/7zz or py7zr to extract the official archive") from error
            with py7zr.SevenZipFile(archive, mode="r") as handle:
                handle.extractall(path=extract_root)

    processed_rows = read_processed_manifest(Path(args.processed_manifest))
    if len(processed_rows) != 85:
        raise RuntimeError(f"Expected 85 processed contents, found {len(processed_rows)}")
    rows = build_manifest(discover_originals(extract_root), processed_rows)
    if sum(row["split"] == "test" for row in rows) != 13:
        raise RuntimeError("Expected exactly 13 standard-test contents")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "archive_md5": actual_md5,
                "contents": len(rows),
                "test_contents": sum(row["split"] == "test" for row in rows),
                "output": str(output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    sys.exit(main())
