# Baseline Source Registry

The current machine already contains these third-party repositories under:

`${SCANDMM_ROOT}/baselines/third_party`

| Baseline | Local status | Category | Source |
|---|---|---|---|
| ScanDMM | available | scanpath-to-saliency | `https://github.com/xiangjieSui/ScanDMM.git` |
| Pathformer3D | available | scanpath-to-saliency | `https://github.com/lsztzp/Pathformer3D.git` |
| ScanGAN360 | available | scanpath-to-saliency | `https://github.com/DaniMS-ZGZ/ScanGAN360.git` |
| SaltiNet | available, legacy | temporal saliency volume | `https://github.com/massens/saliency-360salient-2017.git` |
| PathGAN | available, legacy | scanpath-to-saliency | `https://github.com/imatge-upc/pathgan.git` |
| ScanDiff | available | diffusion scanpath | `https://github.com/aimagelab/ScanDiff.git` |
| DiffEye | available | diffusion gaze | `https://github.com/RehgLab/diffeye.git` |
| HAT | evaluated, three seeds | transformer scanpath-to-saliency | `https://github.com/cvlab-stonybrook/HAT.git` |
| SalDiff-DTM | adapter only | scanpath-to-saliency | no official local repo recorded |
| TempSAL | cloned to workspace | temporal saliency | `https://github.com/IVRL/Tempsal.git` |
| SalBiNet360 | cloned to workspace | static 360 saliency | `https://github.com/githubcbob/SalBiNet360.git` |
| Panoramic-CNN-360-Saliency | cloned to workspace | static 360 saliency | `https://github.com/DaniMS-ZGZ/Panoramic-CNN-360-Saliency.git` |
| 360-aware saliency | cloned to workspace | traditional saliency | `https://github.com/MikhailStartsev/360_aware_saliency.git` |
| SalyPath360 | source not found | joint saliency-scanpath | paper page found, no stable source found |
| MRGAN360 | source not found | recurrent saliency | no reliable official source found |

Unified status command:

```bash
python scripts/baseline_cli.py list --config configs/baselines.yaml
```

Unified dry-run / batch command templates:

```bash
python scripts/baseline_cli.py run scandmm --stage infer --split val --out runs/scandmm_val_200 --set n_scanpaths=200 --dry-run
python scripts/baseline_cli.py batch configs/baseline_batch_dryrun.yaml --dry-run
python scripts/baseline_cli.py batch configs/baseline_batch_scandmm_200x3.yaml --dry-run
```

The formal comparison protocol will convert scanpath outputs from each model into saliency maps with the same `scripts/postprocess_scanpaths.py` settings before evaluation.

Adapter audit notes after local source inspection:

| Baseline | Immediate runnable? | Blocker |
|---|---|---|
| ScanDMM | yes | Local checkpoint exists; adapter supports `--seed`, GPU inference, and the model's initial 3D gaze-point interface through `--start-yaw-deg/--start-pitch-deg`. |
| Panoramic-CNN-360-Saliency | no | Train/test code exists, but `models/` and `checkpoints/` contain README placeholders only. |
| SalBiNet360 | no | Requires Matlab preprocessing and Baidu-hosted pre-trained weights; no local `.pth` file present. |
| TempSAL | no | README points to Google Drive checkpoint; no local weights and dataset adapter needed. |
| 360-aware saliency | no | Full model depends on old Matlab / eDN / SAM Keras1-Theano stack. |

Public-source audit notes:

- TempSAL GitHub and project page were located during the first audit.
- SalBiNet360 and Panoramic-CNN-360-Saliency GitHub repositories were located and shallow-cloned into the current workspace.
- SalyPath360 has an arXiv page, but no stable official source appeared in the first audit.

EH6 correction: ScanDMM is not inherently random-start. Its DMM accepts the first 3D gaze point through `scanpaths[:, 0, :]`; the public inference helper merely samples that point randomly by default. Formal EH6 runs override the helper with the system-declared equatorial start `(yaw=+90°, pitch=0°)`.

HAT integration note: the official model accepts fixation history. In the matched adapter, the system-declared start is inserted as the first observed history token and is never used as a supervised target; the first supervised target is the first one-second gaze sample. Three seeds use the frozen `50 train / 9 val / 13 gate` content split, val-only selection, 200 generated paths per content, and zero gate/test trajectory loads during fitting. The fixation-event IOR is disabled because the adapter emits one gaze point per second. The final gate aggregate is `KLD/SIM/CC=1.7243/0.3531/0.4674` and `Cov@5/10/20=0.7194/0.8314/0.9172`.

Direct 360° tile-prediction methods were also audited. MFTR predicts binary future tile interest directly, but requires five seconds of measured head/eye history and video content; STAR-VP and recent viewport models likewise use a trajectory prefix. They are therefore application/related-work references rather than matched baselines for the history-free static-ERP task. See [360° tile / viewport method audit](360_tile_viewport_method_audit.md).
