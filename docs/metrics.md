# Evaluation Interface Notes

Implemented metrics:

| Metric | Direction | Input | Notes |
|---|---:|---|---|
| AUC-Judd | higher | predicted saliency + binary fixation map | Thresholds at fixation saliency values |
| NSS | higher | predicted saliency + binary fixation map | Z-scored prediction sampled at fixation pixels |
| CC | higher | predicted map + continuous GT map | Pearson correlation |
| SIM | higher | predicted distribution + GT distribution | Histogram intersection after normalization |
| KLD | lower | predicted distribution + GT distribution | `KL(GT || prediction)` |
| CC/SIM/KLD spherical | same | same + latitude area weights | Uses ERP latitude weights `cos(phi)` |

The code lives in `src/sphererate/evaluation/metrics.py` and is exposed through:

```bash
python scripts/evaluate.py --config configs/default.yaml --pred-dir <prediction_dir> --split val
```

For scanpath models, first convert generated scanpaths into saliency maps with:

```bash
python scripts/postprocess_scanpaths.py --scanpath-dir <scanpaths> --out-dir <pred_maps> --order yx
```
