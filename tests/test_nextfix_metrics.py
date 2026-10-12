from __future__ import annotations

import numpy as np

from sphererate.evaluation.nextfix import angular_error_deg, next_fixation_scores, topk_hit


def test_nextfix_metrics_prefer_peak_on_fixation() -> None:
    fixation = np.zeros((8, 16), dtype=np.float32)
    fixation[4, 8] = 1.0
    good = np.zeros_like(fixation)
    good[4, 8] = 1.0
    bad = np.zeros_like(fixation)
    bad[1, 1] = 1.0

    assert angular_error_deg(good, fixation) < angular_error_deg(bad, fixation)
    assert topk_hit(good, fixation, 0.01) == 1.0
    assert topk_hit(bad, fixation, 0.01) == 0.0

    scores = next_fixation_scores(good, fixation)
    assert scores["peak_error_deg"] == 0.0
    assert scores["hit_top1pct"] == 1.0
