# SphereRate Mainline Dashboard

更新时间：2026-07-11

> [RESEARCH_PROTOCOL.md](../RESEARCH_PROTOCOL.md) 是唯一权威协议。Outer-test 已在 EH5 使用一次，不再是未读集。

## One-Screen Status

| Item | State | Result |
|---|---|---|
| Primary task | Locked | start-aligned `0.5/1/3 s` cumulative saliency + tile prefetch |
| EH0 target reliability | PASS | early-macro KLD/SIM CI positive |
| EH1 application oracle | PASS | 10% coverage vs static/template CI positive |
| EH2 data/leakage | PASS | 72 development targets, 13 test sealed until EH5, binocular fix |
| EH3 minimal model | PASS | map、coverage、wrong-time/start/image controls pass |
| EH4 robustness | PASS | 3 seeds + ET-DK2 126 sessions / 8 starts |
| EH5 outer-test | **PASS / USED ONCE** | KLD、SIM、coverage confirmatory CI positive |
| Active experiment gate | None | only paper/figure/reproducibility work authorized |

## Frozen Evidence

| Evidence | Improvement | 95% CI |
|---|---:|---:|
| Outer-test KLD vs early template | 0.032211 | [0.014578, 0.052557] |
| Outer-test SIM vs early template | 0.013376 | [0.008309, 0.018520] |
| Outer-test 10% coverage vs template | 0.013268 | [0.001425, 0.026123] |
| Outer-test 10% coverage vs static-final | 0.427983 | [0.307969, 0.540779] |
| ET-DK2 coverage vs template | 0.013821 | [0.002709, 0.026385] |
| ET-DK2 correct vs wrong image | 0.015302 | [0.002673, 0.026926] |

## Claim Limits

- candidate 不显著优于 capacity fixed codes；
- 5 s 不优于 template；
- CC 只作 guard；
- only equatorial yaw；
- no post-test tuning or retest。

## Key Links

- [Goal](../Goal.md)
- EH0（报告见完整交付包 reports/）
- EH1（报告见完整交付包 reports/）
- EH3（报告见完整交付包 reports/）
- EH4 robustness（报告见完整交付包 reports/）
- EH4 ET-DK2（报告见完整交付包 reports/）
- EH5 outer-test（报告见完整交付包 reports/）
