#!/usr/bin/env python
"""Create the decision report for frozen V12-C external evaluations."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
METRICS = ("kld_sph", "sim_sph", "cc_sph", "coverage_5", "coverage_10", "coverage_20")


def _load(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def _format(values: dict) -> str:
    return " / ".join(f"{float(values[key]):.4f}" for key in METRICS)


def main() -> None:
    sitz = _load("reports/v12c_frozen_transfer_sitzmann/results.json")
    et = _load("reports/v12c_frozen_transfer_etdk2/results.json")
    integrity = _load("reports/v12c_frozen_transfer/integrity_audit.json")
    result = {
        "decision": "FROZEN_CROSS_DATASET_DOMINANCE_FAIL_RETAIN_IN_DOMAIN_CLAIM",
        "integrity": integrity["overall"],
        "sitzmann": {
            "methods": sitz["methods"],
            "v12c_over_tempsal": sitz["comparisons"]["tempsal_erp_salient360_frozen"],
            "wrong_image_3s": sitz["wrong_image_3s"],
        },
        "etdk2": {
            "methods": et["methods"],
            "v12c_over_tempsal": et["comparisons"]["tempsal_erp"],
            "attribution_3s": et["attribution_3s"],
            "access_audit": et["access_audit"],
        },
        "claim_boundary": {
            "supported": [
                "Salient360 in-domain density, Cov@10 and five-second improvement",
                "Sitzmann frozen correct-image attribution",
                "ET-DK2 arbitrary-start conditioning sensitivity",
                "partial five-second external gains",
            ],
            "unsupported": [
                "overall frozen cross-dataset superiority to TempSAL-ERP",
                "cross-dataset early Cov@5/10/20 dominance",
                "ET-DK2 image-content attribution",
                "ET-DK2 independent cross-content generalization",
            ],
        },
    }
    output = ROOT / "reports" / "v12c_frozen_transfer"
    (output / "results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    sitz_methods = sitz["methods"]
    et_methods = et["methods"]
    lines = [
        "# V12-C 冻结跨数据集测试总报告",
        "",
        "Decision: **FROZEN CROSS-DATASET DOMINANCE FAIL / RETAIN IN-DOMAIN CLAIM**",
        "",
        "三个 V12-C checkpoint 在任何外部结果产生前已经冻结；Sitzmann 和 ET-DK2 均未做模型、loss、球面参数、tile grid、预算或 checkpoint 校准。完整性审计为 **PASS**。",
        "",
        "指标顺序：`KLD↓ / SIM↑ / CC↑ / Cov@5↑ / Cov@10↑ / Cov@20↑`。",
        "",
        "## Sitzmann：Salient360 → official four-start scenes",
        "",
        "| Method | Early macro | 5 s |",
        "|---|---|---|",
        f"| TempSAL-ERP frozen | {_format(sitz_methods['tempsal_erp_salient360_frozen']['early_macro'])} | {_format(sitz_methods['tempsal_erp_salient360_frozen']['horizon_5'])} |",
        f"| V12-C frozen | {_format(sitz_methods['v12c_salient360_frozen']['early_macro'])} | {_format(sitz_methods['v12c_salient360_frozen']['horizon_5'])} |",
        "",
        "Early V12-C 相对 TempSAL 的 KLD/SIM/CC/Cov@5/10/20 差值为 `-0.0947/-0.0246/-0.0222/-0.0305/-0.0089/-0.0098`。除 Cov@10 外，其余置信区间均严格为负。5 s 只有 KLD `+0.0075` 和 Cov@10 `+0.0174` 点估计为正，置信区间均跨零。",
        "",
        "不过，3 s correct-image 相对 wrong-image 的 KLD/SIM/CC gain 为 `+0.3320/+0.0364/+0.0748`，置信区间全正：V12-C 在 Sitzmann 仍使用图像内容，只是其排序/校准不如 TempSAL。",
        "",
        "## ET-DK2：real multi-start collection protocol",
        "",
        "当前 ET-DK2 local val 与 Salient360 split 不一致。已排除 36 个 source-train sessions 和 18 个 sealed-outer sessions，只评测 P13/P3/P76 的 54 sessions；因此这是跨起点/跨采集协议测试，不是独立跨内容测试。",
        "",
        "| Method | Early macro | 5 s |",
        "|---|---|---|",
        f"| TempSAL-ERP frozen | {_format(et_methods['tempsal_erp']['early_macro'])} | {_format(et_methods['tempsal_erp']['horizon_5'])} |",
        f"| V12-C frozen | {_format(et_methods['v12c']['early_macro'])} | {_format(et_methods['v12c']['horizon_5'])} |",
        "",
        "Early V12-C 的 CC 高 `+0.0160` 且 CI 为正，但 KLD 与 Cov@5/10/20 点估计均更差，其中 Cov@20 的差值 `-0.0074` CI 严格为负。5 s Cov@20 高 `+0.0174` 且 CI 为正，其余指标互有胜负。",
        "",
        "3 s wrong-start 会让 KLD/SIM/CC 分别恶化 `2.9232/0.1134/0.2505`，说明显式起点几何能泛化到任意 yaw；但 wrong-image 对照没有稳定差异，说明跨采集协议下的 content re-ranking 没有建立。",
        "",
        "## 结论与方向",
        "",
        "1. **域内目标成立，跨域总体领先不成立。** Salient360 上的 density、Cov@10 和 5 s 优势不能外推为跨数据集 SOTA。",
        "2. **泛化的是起点动力学，不是内容重排。** ET-DK2 wrong-start 强通过、wrong-image 失败；Sitzmann虽有 content signal，但相对 TempSAL 排序更差。",
        "3. **证据更符合 prior/protocol shift。** Sitzmann target-train template 的 early KLD/CC 为 `1.3273/.7459`，明显优于两个 Salient360-frozen模型；ET-DK2 source template 也在多项 density 上优于 V12-C。不能把退化简单归因于网络容量。",
        "4. **禁止外部调参。** 当前 checkpoint 保持冻结；论文主张收缩为 Salient360 in-domain 主结果 + external limitation。若未来继续解决，只能另行预注册训练侧的 multi-domain/domain-robust 改进，再使用新的合法验证协议。",
        "",
        "## Reproducibility",
        "",
        "- [Sitzmann result](../v12c_frozen_transfer_sitzmann/summary.md)",
        "- [ET-DK2 result](../v12c_frozen_transfer_etdk2/summary.md)",
        "- [Integrity audit](integrity_audit.md)",
    ]
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"decision": result["decision"], "report": str(output / "summary.md")}, indent=2))


if __name__ == "__main__":
    main()
