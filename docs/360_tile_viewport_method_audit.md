# 360° Tile / Viewport Prediction Method Audit

Updated: 2026-07-14

## Question

Are there published methods that predict 360° image tiles directly, and should any of them enter the matched SphereHorizon comparison?

## Finding

Yes, but the closest literature mainly studies **360° video viewport prediction with several seconds of observed head/eye history**. It is adjacent to, rather than identical with, our setting:

```text
SphereHorizon: one static ERP + one declared start + future horizon
Viewport literature: video frames/saliency + a measured trajectory prefix + future horizon
```

The distinction is not cosmetic. A measured trajectory prefix contains direction and velocity that are unavailable from a declared start. Such methods must not be placed in the matched main table as if their inputs were equivalent.

## Taxonomy and decision

| Method | Output | Required temporal evidence | Content-aware | Main-table decision |
|---|---|---:|---:|---|
| MFTR (ACM MM 2023) | Per-future-step binary tile scores, then a viewport | Past 5 s head and eye trajectories plus video content | Yes | Related work/application reference only; input advantage is too large |
| STAR-VP (ACM MM 2024) | Future viewport trajectory | Historical viewpoint trajectory plus video content/saliency | Yes | Related work only; prefix-conditioned video task |
| Cross-modal multiscale transformer (Scientific Reports 2025) | Future viewport tile index on a 16×9 grid | Past 1 s head trajectory plus video frames extending into the future | Yes | Related work only; prefix and future frames are unavailable here |
| DiffVP (Electronics 2026) | Probabilistic future viewport trajectory | Past 3 s viewing trajectory plus saliency; predicts the next 5 s | Yes | Recent related work only; explicitly prefix-conditioned rather than cold-start |
| Hybrid tile prediction (MMSys 2024) | Tiles selected for streaming | Recent viewport motion/history | Partly | Streaming-system reference, not an attention baseline |
| Pano2Vid / Deep 360 Pilot | Content-driven NFOV camera path/viewing angle | Video sequence; Deep 360 Pilot also uses previous selected angles | Yes | Task-definition precedent, not a population-attention baseline |
| Snap Angle Prediction (ECCV 2018) | One panorama rotation angle | None | Yes | Static-panorama precedent, but optimizes object-preserving projection rather than human early attention or tile coverage |

Primary sources:

- [MFTR: Tile Classification Based Viewport Prediction with Multi-modal Fusion Transformer](https://arxiv.org/abs/2309.14704)
- [STAR-VP: Improving Long-term Viewport Prediction in 360° Videos](https://openreview.net/forum?id=Mc3G3ciRSG)
- [Viewport prediction with cross-modal multiscale transformer](https://www.nature.com/articles/s41598-025-16011-7)
- [DiffVP: A Diffusion-Based Framework for Long-Term Viewport Prediction](https://www.mdpi.com/2079-9292/15/6/1326)
- [Efficient viewport prediction and tiling schemes for 360 degree video streaming](https://doi.org/10.1145/3625468.3653425)
- [Pano2Vid: Automatic Cinematography for Watching 360° Videos](https://arxiv.org/abs/1612.02335)
- [Deep 360 Pilot](https://openaccess.thecvf.com/content_cvpr_2017/html/Hu_Deep_360_Pilot_CVPR_2017_paper.html)
- [Snap Angle Prediction for 360° Panoramas](https://openaccess.thecvf.com/content_ECCV_2018/html/Bo_Xiong_Snap_Angle_Prediction_ECCV_2018_paper.html)

## Consequence for the paper

1. Keep Table 1 matched: saliency methods receive only the ERP; start-aware scanpath methods receive the same declared start; no baseline receives a hidden trajectory prefix.
2. Cite MFTR as evidence that direct tile prediction is a valid application formulation, while explicitly contrasting its history-conditioned video setting with our history-free early-attention setting.
3. Use the application experiment to compare **the same tile-ranking operator** across all saliency/scanpath maps. Do not claim the first direct 360° tile predictor.
4. A future extension may evaluate MFTR/STAR-VP after collecting a true prefix, but that would define a different task and should not delay the current ICASSP submission.

## Candidate wording

> Direct tile classification has been explored for 360° video viewport prediction, but existing methods condition on several seconds of measured head/eye history and video frames. We instead study the cold-start regime: before any trajectory prefix is available, the system must rank early tiles from a static panorama and a declared initial orientation alone.
