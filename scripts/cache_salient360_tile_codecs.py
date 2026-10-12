#!/usr/bin/env python
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
Image.MAX_IMAGE_PIXELS = None

from sphererate.evaluation.codec_prefetch import (
    decode_hevc_intra,
    encode_hevc_intra,
    tile_bounds,
)


def _encode_job(
    source: Path,
    bitstream: Path,
    decoded: Path,
    qp: int,
    height: int,
    width: int,
    threads: int,
) -> dict[str, object]:
    if not bitstream.is_file() or bitstream.stat().st_size == 0:
        encode_hevc_intra(source, bitstream, qp, threads=threads)
    if not decoded.is_file():
        decode_hevc_intra(
            bitstream,
            decoded,
            crop_height=height,
            crop_width=width,
        )
    with Image.open(decoded) as image:
        if image.size != (width, height):
            raise RuntimeError(f"Wrong decoded dimensions for {decoded}: {image.size}")
    return {
        "qp": int(qp),
        "bitstream": str(bitstream),
        "decoded": str(decoded),
        "bytes": int(bitstream.stat().st_size),
    }


def _read_manifest(path: Path, split: str) -> list[dict[str, object]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    selected = [row for row in rows if split == "all" or row["split"] == split]
    if split == "test" and len(selected) != 13:
        raise RuntimeError(f"Expected 13 test stimuli, found {len(selected)}")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache independently coded HEVC ERP tiles.")
    parser.add_argument(
        "--manifest", default="data/interim/salient360_codec_originals/manifest.jsonl"
    )
    parser.add_argument("--output-root", default="data/interim/salient360_tile_hevc_qp22_qp42")
    parser.add_argument("--split", default="test", choices=("train", "val", "test", "all"))
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--cols", type=int, default=16)
    parser.add_argument("--qps", type=int, nargs="+", default=(22, 42))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--ffmpeg-threads", type=int, default=1)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    rows = _read_manifest(Path(args.manifest), args.split)
    if args.limit is not None:
        rows = rows[: args.limit]
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    for image_index, row in enumerate(rows, start=1):
        image_id = str(row["image_id"])
        with Image.open(str(row["source_path"])) as image:
            source_image = image.convert("RGB")
            width, height = source_image.size
            image_root = output_root / image_id
            image_root.mkdir(parents=True, exist_ok=True)
            bounds = tile_bounds(height, width, args.rows, args.cols)
            sources: list[Path] = []
            for tile_id, (top, left, bottom, right) in enumerate(bounds):
                tile_root = image_root / f"tile_{tile_id:03d}"
                tile_root.mkdir(parents=True, exist_ok=True)
                source_path = tile_root / "source.png"
                if not source_path.is_file():
                    source_image.crop((left, top, right, bottom)).save(source_path)
                sources.append(source_path)

        futures = {}
        with ThreadPoolExecutor(max_workers=int(args.workers)) as pool:
            for tile_id, ((top, left, bottom, right), source_path) in enumerate(
                zip(bounds, sources)
            ):
                tile_root = source_path.parent
                for qp in args.qps:
                    future = pool.submit(
                        _encode_job,
                        source_path,
                        tile_root / f"qp{qp}.hevc",
                        tile_root / f"qp{qp}.png",
                        int(qp),
                        bottom - top,
                        right - left,
                        int(args.ffmpeg_threads),
                    )
                    futures[future] = (tile_id, int(qp))
            records: dict[tuple[int, int], dict[str, object]] = {}
            for future in as_completed(futures):
                records[futures[future]] = future.result()

        metadata = {
            "image_id": image_id,
            "source_path": str(row["source_path"]),
            "source_sha256": row["sha256"],
            "height": height,
            "width": width,
            "tile_rows": int(args.rows),
            "tile_cols": int(args.cols),
            "pixel_format": "yuv420p",
            "codec": "libx265_hevc_intra",
            "tiles": [
                {
                    "tile_id": tile_id,
                    "bounds": list(bounds[tile_id]),
                    "representations": {
                        f"qp{qp}": records[(tile_id, int(qp))] for qp in args.qps
                    },
                }
                for tile_id in range(len(bounds))
            ],
        }
        (image_root / "metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        for source_path in sources:
            source_path.unlink(missing_ok=True)
        low = sum(
            record["bytes"]
            for (tile_id, qp), record in records.items()
            if qp == max(args.qps)
        )
        high = sum(
            record["bytes"]
            for (tile_id, qp), record in records.items()
            if qp == min(args.qps)
        )
        if high < low:
            raise RuntimeError(f"All-high representation is smaller than all-low for {image_id}")
        print(
            f"[{image_index}/{len(rows)}] {image_id}: {width}x{height}, "
            f"low={low} bytes, high={high} bytes",
            flush=True,
        )


if __name__ == "__main__":
    main()
