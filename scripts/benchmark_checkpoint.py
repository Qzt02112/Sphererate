#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sphererate.train import load_model_for_inference
from sphererate.utils import ensure_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark model-only single-image inference from a saved checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.warmup < 0 or args.iterations <= 0:
        raise ValueError("warmup must be non-negative and iterations must be positive")

    model, config, device = load_model_for_inference(args.checkpoint, args.device)
    height, width = [int(v) for v in config["data"]["image_size"]]
    in_channels = int(config["model"].get("in_channels", 8))
    image = torch.zeros((1, in_channels, height, width), dtype=torch.float32, device=device)

    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    checkpoint_bytes = Path(args.checkpoint).stat().st_size

    timings_ms: list[float] = []
    with torch.inference_mode():
        for _ in range(args.warmup):
            model(image)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            for _ in range(args.iterations):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                model(image)
                end.record()
                torch.cuda.synchronize(device)
                timings_ms.append(float(start.elapsed_time(end)))
            peak_allocated_bytes = int(torch.cuda.max_memory_allocated(device))
            peak_reserved_bytes = int(torch.cuda.max_memory_reserved(device))
            gpu_name = torch.cuda.get_device_name(device)
        else:
            for _ in range(args.iterations):
                start_time = time.perf_counter()
                model(image)
                timings_ms.append((time.perf_counter() - start_time) * 1000.0)
            peak_allocated_bytes = None
            peak_reserved_bytes = None
            gpu_name = None

    ordered = sorted(timings_ms)
    p90_index = min(len(ordered) - 1, max(0, int(round(0.9 * (len(ordered) - 1)))))
    result = {
        "checkpoint": str(args.checkpoint),
        "model": config["train"]["model"],
        "device": str(device),
        "gpu_name": gpu_name,
        "input_shape": list(image.shape),
        "precision": "float32",
        "warmup": args.warmup,
        "iterations": args.iterations,
        "parameters": {
            "total": total_params,
            "trainable": trainable_params,
        },
        "checkpoint_bytes": checkpoint_bytes,
        "latency_ms": {
            "mean": statistics.fmean(timings_ms),
            "median": statistics.median(timings_ms),
            "p90": ordered[p90_index],
            "min": ordered[0],
            "max": ordered[-1],
        },
        "peak_memory_bytes": {
            "allocated": peak_allocated_bytes,
            "reserved": peak_reserved_bytes,
        },
        "scope": "model forward only; batch size 1; excludes data loading and metric computation",
    }
    output = Path(args.output)
    ensure_dir(output.parent)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
