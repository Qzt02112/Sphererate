# Current Effective Framework

更新时间：2026-08-06

> 唯一有效协议见 [RESEARCH_PROTOCOL.md](../RESEARCH_PROTOCOL.md)。当前主 benchmark 是 Salient360 content-disjoint `59 train / 13 val / 13 test`：train 拟合参数，val 选择 checkpoint，test 可重复评估但不参与选择。当前方法为 **V12-C SphereRate**；V10、V11 只作历史对照。

## 1. 任务与输入输出

```text
input  = ERP RGB + ERP geometry + declared equatorial start + horizon t
output = cumulative population attention C_t, t in {0.5, 1, 3, 5} s
use    = rank 8x16 ERP tiles under 5/10/15/20% budgets
```

四个时刻都属于 early viewing window。模型不输入 gaze prefix、真实 first fixation、observer identity、历史 scanpath、TempSAL prediction 或 Early-template 数值。

## 2. V12-C SphereRate

```text
ERP RGB + geometry
    -> frozen same-split semantic anchor
    -> image-dependent semantic utility U_i(I)

declared start
    -> tangent-plane east/north geometry
    -> horizon-dependent movement cost R_i(x; s)

semantic utility - start cost
    -> non-negative interval dwell rate r_i(x)
    -> raw mass M_k = sum_{i<=k} delta_i r_i
    -> spherical normalization C_k
    -> one shared map for density and tile ranking
```

核心机制：

$$
r_i(x)=\operatorname{Softmax}_{\mathrm{sph}}
\left(\alpha_i[U(x;I)+E_i(x;I)]-R_i(x;s)\right),
\qquad
M_k(x)=\sum_{i=1}^{k}\Delta t_i r_i(x),
\qquad
C_k=\frac{M_k}{\int_{\mathbb S^2}M_k\,dA}.
$$

- 起点成本回答“当前时段能否合理到达”；
- 图像语义回答“可达范围内哪里值得看”；
- 非负 rate 在未归一化 mass 空间累加，保证不减去历史 dwell；
- multi-budget tile loss 只对齐统一输出图的部署排序，不增加预算专用 head；
- 总参数 `3,755,545`，其中可训练 `74,264`。

## 3. Standard-test 主结果

四时刻、三随机种子 early macro：

| Method | KLD↓ | SIM↑ | CC↑ | Cov.@5↑ | Cov.@10↑ | Cov.@15↑ | Cov.@20↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Early template | 1.231 | .431 | .533 | .596 | .721 | .803 | .848 |
| TempSAL-ERP | 1.131 | .441 | .568 | .633 | .748 | .828 | .872 |
| **SphereRate** | **1.072** | **.464** | **.584** | **.635** | **.765** | **.840** | **.880** |

Density 点估计和 Cov.@5/10/15/20 均高于 TempSAL；但原 fixed-count coverage 的四个 paired macro CI 均跨 0，因此不能写成“所有预算稳定显著领先”。逐时刻证据显示提升主要出现在 3/5 s；0.5 s 仍由 start template 主导。

## 4. 机制消融（matched 3 seeds）

所有消融保持相同参数量、semantic anchor、25 epochs、optimizer、采样顺序和 val-only 选模，只关闭一个机制。

| Variant | KLD↓ | SIM↑ | CC↑ | Cov.@5↑ | Cov.@10↑ | Cov.@15↑ | Cov.@20↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Full** | **1.072** | **.464** | **.584** | .635 | **.765** | .840 | **.880** |
| w/o rate accumulation | 1.085 | .454 | .573 | **.640** | .749 | .837 | .874 |
| w/o declared-start cost | 2.287 | .219 | .404 | .570 | .655 | .718 | .770 |
| w/o tile supervision | 1.075 | .463 | .583 | .635 | .764 | **.840** | .880 |

结论边界：

- declared start 是决定性机制，七项 paired CI 均支持 full；
- accumulation 显著改善 SIM、CC、Cov.@10、Cov.@20，但不改善 Cov.@5；
- tile supervision 对 KLD/SIM/CC/Cov.@10 有统计上很小的正效应，其余 coverage 为中性，不能把它写成主要创新来源。

完整结果：[mechanism ablation](../reports/sphererate_ablation_test/summary.md)。

## 5. 真实 HEVC tile 传输实验

应用实验使用官方原始分辨率 Salient360 ERP，而不是 512x256 网络输入。13 个 test 图宽度为 `3584--18332 px`，与处理后 ERP 的最佳循环位移全部为 0，NCC 为 `.996--.999`。

```text
grid       = 8x16 independent tiles
codec      = ffmpeg/libx265 HEVC-Intra, YUV420
low/high   = QP42 / QP22
payload    = measured elementary-stream bytes
byte cap   = R42 + b * (R22 - R42), b in {5,10,15,20}%
allocation = predicted tile-mass order, never exceed cap
```

已生成并审计 `13 x 128 x 2 = 3328` 条独立可解码码流；各方法平均 cap utilization 超过 `99.7%`。

| Method | 5% G-PSNR/Cov. | 10% | 15% | 20% |
|---|---:|---:|---:|---:|
| Early template | 33.002/.472 | 34.969/.655 | 35.886/.737 | 36.539/.794 |
| TempSAL-ERP | **33.470/.525** | 35.241/.687 | 36.225/.765 | 36.794/.818 |
| **SphereRate** | 33.217/.499 | **35.311/.696** | **36.395/.779** | **36.931/.829** |
| GT-attention oracle | 33.970/.605 | 36.082/.793 | 37.018/.863 | 37.493/.904 |

- 相对 Early template，SphereRate 在 10/15/20% 提升 `.342/.509/.392 dB`，paired CI 均高于 0；
- 相对 TempSAL，10/15/20% 点提升 `.070/.170/.136 dB`，但 CI 跨 0；
- 5% 下 SphereRate 比 TempSAL 少 `.026` dwell coverage，CI `[-.055,-.003]`，是明确失败；
- whole-sphere PSNR 在 deployable methods 间最大只差 `.021 dB`，收益来自把相同码率放到更有价值的位置，而不是提高总码率。

完整结果：[real-codec application](../reports/codec_tile_prefetch_test/summary.md)。

## 6. 当前允许主张与边界

允许：

> SphereRate uses declared-start cost and non-negative dwell-rate accumulation to improve early cumulative attention density; under measured HEVC byte caps, the improved ranking yields real attention-weighted reconstruction gains at medium budgets.

不允许：

- 所有预算均优于 TempSAL；
- tile loss 是主要性能来源；
- 任意纬度起点、个性化 gaze、跨域普遍泛化；
- 已完成 viewport rendering、latency、container overhead 或 end-to-end QoE 测试。

下一优化目标只剩两项：修复 5% 极紧预算排序，并把目前对 TempSAL 的 10--20% 点提升变成 content-bootstrap 可确认的收益。不得通过增加复杂 backbone 掩盖这两个明确问题。

## 7. 关键入口

- [论文 PDF](../output/pdf/SphereHorizon_ICASSP_draft.pdf)
- [模型实现](../src/sphererate/models/spherical_dwell_rate.py)
- [标准训练配置](../configs/standard_split/v12c_hybrid_dwell_rate.yaml)
- [消融 batch](../configs/standard_split/batch_v12c_ablation.yaml)
- [codec 配置](../configs/standard_split/codec_application.yaml)
- [消融结果](../reports/sphererate_ablation_test/results.json)
- [codec 结果](../reports/codec_tile_prefetch_test/results.json)
