# Temporal Improvement Retrospective

更新时间：2026-07-10

> Canonical task, method, and Gates are locked in [RESEARCH_PROTOCOL.md](../RESEARCH_PROTOCOL.md). This document records lessons, not future branches.

## Current Conclusion

The previous deterministic temporal path is no longer the candidate mainline. It exposed two useful facts:

1. A strong static/semantic shortcut explains most final-map performance and can ignore temporal heads.
2. Aggregating observers and start conditions into one deterministic temporal target suppresses multimodal viewing dynamics.

The revised hypothesis is therefore **condition decomposition followed by image-only marginalization**, not a larger temporal transition network.

## Retained Findings

- Semantic/static modeling and yaw-roll augmentation are strong baseline ingredients.
- TempSAL-style staged fusion can learn small map motion but the final head can ignore it.
- Step-TV constrains movement magnitude, not correct semantic location.
- Teacher-residual bounded the correction but did not establish a stable temporal-order gain under yaw augmentation.
- Temporal-hardness and final-map difficulty are not equivalent.
- JUFE same-image good/bad start conditions produce different gaze distributions and provide a controlled way to test condition signal.
- V7 keeps start metadata out of the static trunk and is suitable as a P1 conditioned oracle after clean data reconstruction.

## Protocol Lessons

### Final saliency is largely order-invariant

Shuffling temporal bins can preserve total fixation mass. A final-map tie therefore does not by itself prove that temporal supervision is useless. Order controls should be judged primarily on temporal/process maps.

For final-map claims, stronger controls are:

- real time grouping vs equal-mass random fixation partitions;
- actual start condition vs swapped/random condition;
- temporal auxiliary supervision vs a capacity-matched ordinary multi-head model.

### Conditioned performance is not image-only performance

Given-start JUFE prediction can establish an oracle and show that the condition matters. It supports the main paper only when an image-only model marginalizes the unknown condition and improves population final saliency.

### Protocol validity precedes architecture

Current Multi360 content/scene leakage, repeated test inspection, and invalid CC implementation block paper-level interpretation. Old results remain useful only for diagnosis.

## Avoid Repeating

Do not spend mainline compute on:

- More teacher-residual top-k, peak, region, or localization-loss sweeps before Gate A-C.
- Larger GRU/Transformer/Mamba/diffusion temporal branches.
- More than `K=3` temporal stages without evidence.
- A second semantic encoder in the residual branch.
- Free fusion that can ignore condition maps.
- Claims based only on increased step-TV or qualitative movement.
- Treating normal-vs-shuffled final-map score as the only temporal control.
- Selecting methods or hyperparameters on the old test split.
- Expanding next-fixation, start-only, dual-path, or motion-renderer branches.

## Revised Experiment Sequence

1. Repair group-disjoint split, CC/weighted CC, priors, and untouched final test.
2. Rerun the strong image-only static baseline.
3. Rebuild and run V7 P1 actual/static/swapped/random controls.
4. If V7 passes, implement one shared-backbone start-marginalized residual model.
5. Compare against ordinary temporal auxiliary, random condition, and random fixation partitions.
6. Only after a P0 win, restore external TempSAL/ScanDMM/static baselines.

## Decision Rules

- P1 actual does not beat swapped/random: stop condition modeling.
- P1 wins but P0 marginalization does not: report conditioned saliency only; do not claim image-only improvement.
- P0 matches random condition: interpret the gain as capacity/ensemble.
- Process improves but final does not: do not claim final saliency improvement.
- Group-disjoint results remove the gain: stop model expansion.
- P0 wins static and matched controls across 3 seeds: proceed to paper comparison.

## Required Future Report Header

Every new report must begin with:

```text
protocol: P0 / P1 / D1
gate: A / B / C / D
claim_tested: one sentence
locked_baseline: run name
matched_control: run name
decision_rule: preregistered threshold or CI rule
test_status: unopened / frozen / evaluated-after-freeze
```
