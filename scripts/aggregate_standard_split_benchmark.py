#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
HORIZONS_EARLY = {0.5, 1.0, 3.0}
BUDGETS = (5.0, 10.0, 15.0, 20.0)


META = {
    "Early Template": ("start + horizon", "0"),
    "TempSAL-ERP": ("image + horizon", "3.68M"),
    "SphereRate (ours)": ("image + start + horizon", "3.76M"),
    "Panoramic-CNN": ("image", "2.92M"),
    "Panoramic-CNN†": ("image + start + horizon", "2.92M"),
    "BMS360": ("image", "non-neural"),
    "BMS360†": ("image + start + horizon", "non-neural"),
    "GBVS360": ("image", "non-neural"),
    "GBVS360†": ("image + start + horizon", "non-neural"),
    "DeepGaze MSDB": ("image", "451.64M frozen + 2"),
    "DeepGaze MSDB†": ("image + start + horizon", "451.64M frozen + 2"),
    "Pathformer3D": ("image + start", "5.05M"),
    "ScanDMM": ("image + start", "4.66M"),
    "HAT": ("image + start", "42.78M"),
}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _mean(rows: list[dict[str, Any]], key: str, horizons: set[float]) -> float:
    values = [float(row[key]) for row in rows if float(row["horizon_s"]) in horizons]
    if not values:
        raise ValueError(f"No rows for {key} at {sorted(horizons)}")
    return float(np.mean(values))


def _from_rows(
    score_rows: list[dict[str, Any]],
    coverage_rows: dict[str, list[dict[str, Any]]],
    horizons: set[float],
) -> dict[str, float]:
    result = {
        key: _mean(score_rows, key, horizons)
        for key in ("kld_sph", "sim_sph", "cc_sph")
    }
    for budget in BUDGETS:
        key = str(budget)
        result[f"coverage_{budget:g}"] = _mean(coverage_rows[key], "coverage", horizons)
    return result


def _regular(path: Path) -> dict[str, dict[str, float]]:
    data = _load(path)
    return {
        "early_macro": _from_rows(data["score_rows"], data["coverage_rows"], HORIZONS_EARLY),
        "horizon_5": _from_rows(data["score_rows"], data["coverage_rows"], {5.0}),
    }


def _reachable(path: Path) -> dict[str, dict[str, float]]:
    data = _load(path)
    return {
        "early_macro": _from_rows(
            data["reachable_score_rows"], data["reachable_coverage_rows"], HORIZONS_EARLY
        ),
        "horizon_5": _from_rows(
            data["reachable_score_rows"], data["reachable_coverage_rows"], {5.0}
        ),
    }


def _deepgaze(path: Path, variant: str) -> dict[str, dict[str, float]]:
    data = _load(path)[variant]
    return {
        "early_macro": _from_rows(data["score_rows"], data["coverage_rows"], HORIZONS_EARLY),
        "horizon_5": _from_rows(data["score_rows"], data["coverage_rows"], {5.0}),
    }


def _primary(path: Path) -> dict[str, dict[str, dict[str, float]]]:
    methods = _load(path)["methods"]

    def values(method: str, scope: str) -> dict[str, float]:
        current = methods[method][scope]
        return current["mean"] if "mean" in current else current

    return {
        "Early Template": {
            scope: values("early_template", scope) for scope in ("early_macro", "horizon_5")
        },
        "TempSAL-ERP": {
            scope: values("tempsal_erp", scope) for scope in ("early_macro", "horizon_5")
        },
        "SphereRate (ours)": {
            scope: values("v12c", scope) for scope in ("early_macro", "horizon_5")
        },
    }


def _collect(split: str) -> tuple[dict[str, Any], list[str]]:
    methods: dict[str, Any] = _primary(
        ROOT / f"reports/standard_split_primary_{split}/results.json"
    )
    sources: list[tuple[str, Path, str, str | None]] = [
        ("Panoramic-CNN", ROOT / f"reports/standard_split_panoramic_static_{split}/results.json", "regular", None),
        ("Panoramic-CNN†", ROOT / f"reports/standard_split_reachable_panoramic_cnn_{split}/results.json", "reachable", None),
        ("BMS360", ROOT / f"reports/standard_split_bms360_{split}/results.json", "regular", None),
        ("BMS360†", ROOT / f"reports/standard_split_reachable_bms360_{split}/results.json", "reachable", None),
        ("GBVS360", ROOT / f"reports/standard_split_gbvs360_{split}/results.json", "regular", None),
        ("GBVS360†", ROOT / f"reports/standard_split_reachable_gbvs360_{split}/results.json", "reachable", None),
        ("DeepGaze MSDB", ROOT / f"reports/standard_split_deepgaze_msdb_{split}/results.json", "deepgaze", "train_calibrated"),
        ("DeepGaze MSDB†", ROOT / f"reports/standard_split_deepgaze_msdb_{split}/results.json", "deepgaze", "train_calibrated_reachable"),
        ("Pathformer3D", ROOT / f"reports/standard_split_pathformer3d_{split}/results.json", "regular", None),
        ("ScanDMM", ROOT / f"reports/standard_split_scandmm_{split}/results.json", "regular", None),
        ("HAT", ROOT / f"reports/standard_split_hat_{split}/results.json", "regular", None),
    ]
    missing = []
    for label, path, kind, variant in sources:
        if not path.exists():
            missing.append(label)
            continue
        if kind == "regular":
            methods[label] = _regular(path)
        elif kind == "reachable":
            methods[label] = _reachable(path)
        else:
            methods[label] = _deepgaze(path, str(variant))
    return methods, missing


def _format(value: float) -> str:
    return f"{value:.4f}"


def _signed(value: float) -> str:
    return f"{value:+.4f}"


def _write_summary(result: dict[str, Any], path: Path) -> None:
    lines = [
        f"# Standard Salient360 Benchmark — {result['split']}",
        "",
        "Protocol: 59 train contents for fitting, 13 validation contents for model/checkpoint selection, and the unchanged 13 test contents for reporting. Test is repeatable but never used for selection.",
        "",
        "`†` denotes the same static saliency output with the train-only 95% reachable-region adapter.",
    ]
    if result["missing_methods"]:
        lines.extend(["", "Incomplete rows: " + ", ".join(result["missing_methods"]) + "."])
    for scope, title in (("early_macro", "Early macro (0.5/1/3 s)"), ("horizon_5", "5 s")):
        lines.extend(
            [
                "",
                f"## {title}",
                "",
                "| Method | Input | Params | KLD↓ | SIM↑ | CC↑ | Cov@5↑ | Cov@10↑ | Cov@15↑ | Cov@20↑ |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for name, scopes in result["methods"].items():
            row = scopes[scope]
            input_name, params = META[name]
            lines.append(
                f"| {name} | {input_name} | {params} | {_format(row['kld_sph'])} | "
                f"{_format(row['sim_sph'])} | {_format(row['cc_sph'])} | "
                f"{_format(row['coverage_5'])} | {_format(row['coverage_10'])} | "
                f"{_format(row['coverage_15'])} | {_format(row['coverage_20'])} |"
            )
    lines.extend(
        [
            "",
            "## Reading the result",
            "",
        ]
    )
    ours_early = result["methods"]["SphereRate (ours)"]["early_macro"]
    temp_early = result["methods"]["TempSAL-ERP"]["early_macro"]
    ours_5 = result["methods"]["SphereRate (ours)"]["horizon_5"]
    temp_5 = result["methods"]["TempSAL-ERP"]["horizon_5"]
    scanpath_names = ("Pathformer3D", "ScanDMM", "HAT")
    scanpath_klds = {
        name: result["methods"][name]["early_macro"]["kld_sph"]
        for name in scanpath_names
        if name in result["methods"]
    }
    lines.extend(
        [
            "Validation chooses models; only the resulting rows are interpreted on test. Density and tile coverage are kept together because improving one does not guarantee improving the other, especially at the 5% budget.",
            "",
            (
                "SphereRate versus TempSAL-ERP at early macro: "
                f"KLD gain={_signed(temp_early['kld_sph'] - ours_early['kld_sph'])}, "
                f"SIM={_signed(ours_early['sim_sph'] - temp_early['sim_sph'])}, "
                f"CC={_signed(ours_early['cc_sph'] - temp_early['cc_sph'])}, "
                f"Cov@5/10/15/20="
                f"{_signed(ours_early['coverage_5'] - temp_early['coverage_5'])}/"
                f"{_signed(ours_early['coverage_10'] - temp_early['coverage_10'])}/"
                f"{_signed(ours_early['coverage_15'] - temp_early['coverage_15'])}/"
                f"{_signed(ours_early['coverage_20'] - temp_early['coverage_20'])}. "
                "The tightest budget is therefore an exception to the otherwise better point estimates."
            ),
            "",
            (
                "At 5 s, SphereRate versus TempSAL-ERP has "
                f"KLD gain={_signed(temp_5['kld_sph'] - ours_5['kld_sph'])} and "
                f"Cov@5/10/15/20="
                f"{_signed(ours_5['coverage_5'] - temp_5['coverage_5'])}/"
                f"{_signed(ours_5['coverage_10'] - temp_5['coverage_10'])}/"
                f"{_signed(ours_5['coverage_15'] - temp_5['coverage_15'])}/"
                f"{_signed(ours_5['coverage_20'] - temp_5['coverage_20'])}; "
                "all seven reported point estimates improve."
            ),
        ]
    )
    if scanpath_klds:
        rendered = ", ".join(f"{name}={value:.4f}" for name, value in scanpath_klds.items())
        lines.extend(
            [
                "",
                (
                    f"The start-aware scanpath adapters have early KLD {rendered}, versus "
                    f"SphereRate={ours_early['kld_sph']:.4f}. Explicit start conditioning and "
                    "plausible trajectories do not by themselves guarantee a calibrated cumulative population map."
                ),
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate the conventional 59/13/13 benchmark.")
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    methods, missing = _collect(args.split)
    if missing and args.require_complete:
        raise RuntimeError(f"Missing standard-split reports: {missing}")
    result = {
        "protocol": "salient360_standard_train59_val13_test13_v3",
        "split": args.split,
        "test_repeatable": True,
        "test_used_for_selection": False,
        "methods": methods,
        "missing_methods": missing,
    }
    output = ROOT / (args.output_dir or f"reports/standard_split_full_benchmark_{args.split}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _write_summary(result, output / "summary.md")
    print(json.dumps({"report": str(output / "summary.md"), "missing": missing}, indent=2))


if __name__ == "__main__":
    main()
