# HAT runtime environment

更新时间：2026-08-05

标准 `59/13/13` 重跑使用工作区隔离环境：

```text
third_party/.envs/hat
Python 3.12 (system-site-packages)
NumPy 1.26.4
SciPy 1.13.1
PyTorch 2.8.0+cu128
Detectron2 0.6, CPU extension build
fvcore 0.1.5.post20221221
iopath 0.1.10
```

原因：base 环境的 NumPy 2.2.6 与现有 SciPy 二进制 ABI 不兼容。隔离环境只覆盖 NumPy，不修改 base，也不影响 ScanDMM/Pathformer3D 正在运行的进程。

Detectron2 0.6 在 Pillow 10 下唯一需要的导入兼容修复是：

```text
detectron2/data/transforms/transform.py
Image.LINEAR -> Image.BILINEAR
```

HAT 官方源码仍来自：

```text
${SCANDMM_ROOT}/baselines/third_party/HAT
commit 557205d3136a63120041c700ee717196a429e89b
```

训练与选择均必须用同一解释器：

```bash
third_party/.envs/hat/bin/python scripts/train_kernel_v2_hat.py ...
third_party/.envs/hat/bin/python scripts/select_kernel_v2_hat.py ...
```

导入验证已覆盖 `scipy.ndimage`、Detectron2、HAT data loader 和 `HumanAttnTransformer`。首次因 ABI 错误和资源竞争而中断的日志保留为 `runs/standard_split_v3/logs/hat_train_seed20260710_aborted_contention.log`，不进入正式结果。
