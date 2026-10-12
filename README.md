# SphereRate

**English** | [简体中文](README.zh-CN.md)

Reference implementation of **SphereRate (V12-C)** — start-aware **early cumulative
saliency prediction** for 360° ERP images, and the code used to produce the training,
ablation, benchmark and real-codec experiments of the accompanying paper.

> **Status.** This is a *research code release*, not an official paper artifact.
> The associated paper is **currently under review (ICASSP)**; results below are the
> ones reported in the current manuscript. Per-sample prediction caches, datasets and
> trained checkpoints are **not** included in this repository (see [Data](#data) and
> [Reproducing the reported numbers](#reproducing-the-reported-numbers)).

---

## Task

```text
input  = ERP RGB + ERP geometry + declared equatorial start + horizon t
output = cumulative population attention C_t,  t in {0.5, 1, 3, 5} s
use    = rank 8x16 ERP tiles under 5 / 10 / 15 / 20 % budgets
```

All four horizons lie inside the **early viewing window**. The model deliberately does
**not** consume a gaze prefix, the true first fixation, observer identity, a historical
scanpath, a TempSAL prediction, or early-template values. The only conditioning signals
are the image, the sphere geometry, a system-declared start position, and the horizon.

## Method

SphereRate predicts a horizon-dependent, non-negative **interval dwell rate** over the
sphere, accumulates it in un-normalized mass space, and normalizes once:

```text
ERP RGB + geometry
    -> frozen same-split semantic anchor
    -> image-dependent semantic utility U_i(I)

declared start
    -> tangent-plane east/north geometry
    -> horizon-dependent movement cost R_i(x; s)

semantic utility - start cost
    -> non-negative interval dwell rate r_i(x)
    -> raw mass M_k = sum_{i<=k} dt_i * r_i
    -> spherical normalization C_k
    -> one shared map for density and tile ranking
```

$$
r_i(x)=\operatorname{Softmax}_{\mathrm{sph}}\bigl(\alpha_i[U(x;I)+E_i(x;I)]-R_i(x;s)\bigr),
\qquad
M_k(x)=\sum_{i=1}^{k}\Delta t_i\, r_i(x),
\qquad
C_k=\frac{M_k}{\int_{\mathbb S^2}M_k\,dA}.
$$

Design points:

- **start cost** answers *"can this region be reached within the current interval?"*;
- **image semantic** answers *"within reach, where is it worth looking?"*;
- the **non-negative rate** is accumulated in mass space, so accumulation never
  subtracts already-observed dwell;
- the **multi-budget tile loss** only aligns the ranking of the single shared output map
  under deployment budgets — it adds no budget-specific head;
- total parameters `3,755,545`, of which `74,264` are trainable.

Target renderer (fixed protocol):

```text
target renderer = unit-sphere chord Gaussian
sigma           = 2 deg visual angle
support         = 4 sigma
weighting       = clipped dwell duration
normalization   = spherical area
resolution      = 256 x 512
```

## Results

Early macro over the four horizons, three random seeds, content-disjoint Salient360
standard split `59 train / 13 val / 13 test` (checkpoint selection on `val` only):

| Method | KLD↓ | SIM↑ | CC↑ | Cov.@5↑ | Cov.@10↑ | Cov.@15↑ | Cov.@20↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Early template | 1.231 | .431 | .533 | .596 | .721 | .803 | .848 |
| TempSAL-ERP | 1.131 | .441 | .568 | .633 | .748 | .828 | .872 |
| **SphereRate** | **1.072** | **.464** | **.584** | **.635** | **.765** | **.840** | **.880** |

Mechanism ablation (matched parameters, semantic anchor, 25 epochs, optimizer, sampling
order and val-only selection; exactly one mechanism disabled per row):

| Variant | KLD↓ | SIM↑ | CC↑ | Cov.@5↑ | Cov.@10↑ | Cov.@15↑ | Cov.@20↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Full** | **1.072** | **.464** | **.584** | .635 | **.765** | .840 | **.880** |
| w/o rate accumulation | 1.085 | .454 | .573 | **.640** | .749 | .837 | .874 |
| w/o declared-start cost | 2.287 | .219 | .404 | .570 | .655 | .718 | .770 |
| w/o tile supervision | 1.075 | .463 | .583 | .635 | .764 | **.840** | .880 |

Real HEVC tile-prefetch application (native-resolution Salient360 ERP, 8×16 independent
tiles, `13 × 128 × 2 = 3328` bitstreams, gaze-PSNR / gaze coverage under measured
byte caps):

| Method | 5 % G-PSNR / Cov. | 10 % | 15 % | 20 % |
|---|---:|---:|---:|---:|
| Early template | 33.002 / .472 | 34.969 / .655 | 35.886 / .737 | 36.539 / .794 |
| TempSAL-ERP | **33.470 / .525** | 35.241 / .687 | 36.225 / .765 | 36.794 / .818 |
| **SphereRate** | 33.217 / .499 | **35.311 / .696** | **36.395 / .779** | **36.931 / .829** |
| GT-attention oracle | 33.970 / .605 | 36.082 / .793 | 37.018 / .863 | 37.493 / .904 |

## Scope and honest limitations

These boundary statements are part of the released result, not caveats added after the
fact. Please do not cite the method as uniformly stronger than the baselines.

- **Declared start is the decisive mechanism.** Removing it collapses every metric.
- **Early `Cov.@5` is not a significant win.** The four paired macro coverage CIs
  straddle zero; gains concentrate at `3 s` and `5 s`, while `0.5 s` is still dominated
  by the start template.
- **Rate accumulation** improves SIM / CC / Cov.@10 / Cov.@20 but **not** Cov.@5.
- **Tile supervision** has a small, budget-dependent positive effect and is *not* the
  main source of the improvement.
- **Real codec: medium budgets only.** SphereRate beats the early template by
  `.342 / .509 / .392 dB` at 10 / 15 / 20 %, but the **5 % budget is an explicit
  failure** (`.026` less gaze coverage than TempSAL, CI `[-.055, -.003]`).
- **Frozen cross-dataset transfer does not beat TempSAL.** On Sitzmann and on the
  legal ET-DK2 sub-split the early metrics are lower; the Salient360 in-domain
  benchmark is the primary evidence.

## Repository layout

| Path | Contents |
|---|---|
| `src/sphererate/models/` | SphereRate model, spherical layers, UNet backbones |
| `src/sphererate/data/` | dataset views, splits, ERP/geometry preparation |
| `src/sphererate/evaluation/` | spherical metrics, tile ranking, codec prefetch |
| `src/sphererate/baselines/` | baseline runner and adapters |
| `scripts/` | data / training / evaluation / audit / plotting entry points |
| `configs/` | mainline and baseline configs (`configs/standard_split/` is current) |
| `tests/` | unit and contract tests |
| `docs/` | protocol, framework, metrics and version notes |
| `data/splits/`, `data/manifests/` | frozen split definitions and data manifests |

## Installation

Requires **Python ≥ 3.10** and CUDA for training. An editable install needs a
reasonably recent `pip` / `setuptools`; on an older interpreter, upgrade them first:

```bash
git clone https://github.com/Qzt02112/Sphererate.git
cd Sphererate
python -m pip install -U pip setuptools   # needed on pip < 23 / setuptools < 68
python -m pip install -e .
```

Dependencies: `numpy`, `Pillow>=9.1`, `PyYAML`, `requests`, `torch`, `torchvision`,
`tqdm`.

### Configuring paths

Several configs and helper scripts keep machine-specific roots as `${...}` placeholders.
They are expanded from your environment, so export them instead of editing every file:

```bash
export SCANDMM_ROOT=/path/to/ScanDMM    # raw data + third-party baseline checkouts
export TEMPSAL_ROOT=/path/to/SphereRate # this project's root on your machine
```

An unset placeholder is left as the literal `${SCANDMM_ROOT}` string, so it stays
visible rather than silently resolving to the wrong place.

## Data

All three datasets are **publicly downloadable**; none of them is redistributed here.

| Dataset | Role | Source |
|---|---|---|
| **Salient360!** | primary early-horizon population target | <https://www.interdigital.com/data_sets/salient-360-dataset> |
| **Sitzmann VR saliency** | secondary in-domain track / frozen transfer | <https://github.com/vsitzmann/vr-saliency> |
| **ET-DK2** | external multi-start sessions (CC-BY-4.0) | Zenodo record `10.5281/zenodo.4642612` |

Rebuild the processed views (paths are configurable in `configs/`):

```bash
# Salient360: standard 59/13/13 content-disjoint split, 256x512, spherical 2-deg target
python scripts/build_standard_salient360_v3.py --config configs/standard_split/v12c_hybrid_dwell_rate.yaml

# Sitzmann: official standing-VR reconstruction (expects data/sitzmann_raw_data.zip)
python scripts/prepare_sitzmann_official.py --archive data/sitzmann_raw_data.zip

# ET-DK2: Zenodo downloader verifies the official MD5, then preprocesses 720 public sessions
python scripts/download_etdk2.py --out-root "${SCANDMM_ROOT}/data/raw/etdk2"
python scripts/prepare_dataset.py --config configs/etdk2_256x512.yaml
```

Expected processed layout (adjust the YAML roots after cloning):

```text
data/processed/early_horizon_salient360_spherical2deg_standard_v3/
data/processed/salient360_256x512_k6_order/
data/processed/standard_split_v3_matched_scanpaths/
data/processed/sitzmann_official_256x512_spherical2deg_standard_v2/
data/interim/salient360_codec_originals/
```

## Training and evaluation

Three fixed seeds (`20260710 / 20260711 / 20260712`); checkpoints are selected on
`val` only. The `test` split is evaluated repeatedly but **never** used for selection.

```bash
# mainline SphereRate / V12-C, three seeds
python scripts/batch_train.py \
  --config configs/standard_split/v12c_hybrid_dwell_rate.yaml \
  --batch  configs/standard_split/batch_v12c_hybrid_dwell_rate.yaml \
  --device cuda

# mechanism ablation
python scripts/batch_train.py --config configs/standard_split/v12c_hybrid_dwell_rate.yaml \
  --batch configs/standard_split/batch_v12c_ablation.yaml --device cuda

# aggregate the primary test numbers
python scripts/aggregate_standard_split_primary.py

# real HEVC tile-prefetch application
python scripts/evaluate_codec_tile_prefetch.py --config configs/standard_split/codec_application.yaml

# matched external baselines (ScanDMM / Pathformer3D / HAT / Panoramic-CNN / ...)
python scripts/batch_train.py --config configs/standard_split/v12c_hybrid_dwell_rate.yaml \
  --batch configs/standard_split/matched_baselines.yaml --device cuda
```

Metrics are defined in [docs/metrics.md](docs/metrics.md): spherical KLD / SIM / CC plus
tile `Cov.@5/10/15/20`.

## Tests

```bash
python -m pip install pytest
python -m pytest -q
```

Expected on a minimal checkout: `241 passed, 8 skipped`. The skips are the adapter
contract tests that import the upstream baseline sources (ScanDMM / Pathformer3D /
HAT), which are **not** vendored here — set `SCANDMM_ROOT` and `TEMPSAL_ROOT` to a
checkout that contains `third_party/` to run them.

## Reproducing the reported numbers

This repository contains **code, configs, fixed splits and data manifests**. To
reproduce the exact tables you also need the datasets above, then to re-run training and
evaluation. The frozen evidence used by the paper (validation-selected checkpoints plus
the per-report summaries and paired-bootstrap results) lives in the **full delivery
package** and is intentionally not committed here because of size and dataset terms.

## Third-party baselines

The comparison methods are **not** redistributed in this repository. Their source is
referenced here for attribution; please obtain each one from its own repository and
respect its own license.

| Baseline | Category | Source |
|---|---|---|
| TempSAL | temporal saliency | <https://github.com/IVRL/Tempsal> |
| Panoramic-CNN-360-Saliency | static 360 saliency | <https://github.com/DaniMS-ZGZ/Panoramic-CNN-360-Saliency> |
| GBVS360 / BMS360 / ProSal | traditional saliency | <https://github.com/Telecommunication-Telemedia-Assessment/GBVS360-BMS360-ProSal> |
| DeepGaze MSDB | static saliency | <https://github.com/matthias-k/deepgaze/tree/feature-deepgaze-msdb> |
| ScanDMM | scanpath-to-saliency | <https://github.com/xiangjieSui/ScanDMM> |
| Pathformer3D | scanpath-to-saliency | <https://github.com/lsztzp/Pathformer3D> |
| HAT | transformer scanpath-to-saliency | <https://github.com/cvlab-stonybrook/HAT> |
| 360-aware saliency | traditional saliency | <https://github.com/MikhailStartsev/360_aware_saliency> |
| SalBiNet360 | static 360 saliency | <https://github.com/githubcbob/SalBiNet360> |

We thank the authors of all of the above for releasing their code and data.

## Citation

The paper is under review. A BibTeX entry will be added once it is public; until then
please cite this repository.

## Contact

Questions and bug reports: **271218869@qq.com**

## License

Released under the [MIT License](LICENSE) — © 2026 Qiu Zhaotian. This covers only the
code in this repository; third-party baselines remain under their own licenses.
