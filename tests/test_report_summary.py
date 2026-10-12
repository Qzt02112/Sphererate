from __future__ import annotations

import json

from sphererate.evaluation.summary import load_summary_rows, summarize_reports


def test_summary_expands_macro_and_dataset_rows(tmp_path) -> None:
    report_dir = tmp_path / "run_val"
    report_dir.mkdir()
    summary_path = report_dir / "val_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "split": "val",
                "num_images": 3,
                "scores": {"kld_sph": 1.0, "sim_sph": 0.4},
                "macro_scores": {"kld_sph": 0.7, "sim_sph": 0.5},
                "datasets": {
                    "jufe": {"num_images": 2, "scores": {"kld_sph": 0.8, "sim_sph": 0.45}},
                    "salient360": {"num_images": 1, "scores": {"kld_sph": 0.6, "sim_sph": 0.55}},
                },
            }
        ),
        encoding="utf-8",
    )

    rows = load_summary_rows(summary_path)
    scopes = [row["scope"] for row in rows]
    assert scopes == ["mixed", "macro", "dataset:jufe", "dataset:salient360"]
    assert rows[1]["macro_kld_sph"] == 0.7

    out_path = tmp_path / "summary.md"
    summarize_reports(tmp_path, out_path)
    text = out_path.read_text(encoding="utf-8")
    assert "| run_val | val | macro | 3 |" in text
    assert "| run_val | val | dataset:jufe | 2 |" in text
    assert text.count("val_summary.json") == 1
