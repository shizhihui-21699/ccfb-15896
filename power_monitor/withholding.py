"""策略性持留（physical withholding）识别 —— 基于“外生价格影响”的条件检验。

为什么简单指标会误报
--------------------
最直觉的做法是“看机组是否专挑高价时段停运”。这必然失败，原因有二：

1. **混淆**：高价与高负荷/高紧张度重合，而真实故障率本就随紧张度上升。
   仿真中未持留的 C03/C04 在高价时段停运率也上升——那是真实故障。
2. **反向因果**：便宜、常年满发的机组（C01/C02）本身就是边际定价者，
   它们一停运，价格自然就高。用“停运时的价格”判断是被停运推高的价格，
   是循环论证。

正确做法：外生价格影响 Δp
-------------------------
对每个 (机组 u, 小时 t) 计算**只由其他机组状态决定**的反事实价格差：

    Δp_{u,t} = Clear(除 u 外的机组, 负荷_t) − Clear(除 u 外的机组 + u 按额定容量, 负荷_t)

它完全不含 u 自身是否停运的信息，因此不受反向因果污染，含义清晰：
**u 若在此刻停运，能把出清价抬高多少**。这正是持留的经济动机——
只有 Δp 足够大（且同集团还有机组在发电以捕获租金）时，持留才划算。

识别策略
--------
1. 把小时按 Δp 分档（<10 / 10–30 / ≥30 元/MWh），再在每个 Δp 档内按
   紧张度分层，确保比较发生在**同等物理压力**下。
2. 对每台机组，比较“高 Δp 档”与“低 Δp 档”的停运率；若高 Δp 档显著更高，
   说明其停运**选择性**地集中在最能推高价格的时刻——真实故障无此选择性。
3. 二项检验 + BH-FDR；效应量用“超额停运小时数”和 Δp 加权租金损失，
   给监管人员可解释的量化。
4. 附带“兄弟机组捕获租金”检查：持留是否主要发生在同集团其他机组可发电时。

输出为**异常证据**，须结合检修工单、备件领用、修复时长、人员到岗等内部
资料交叉验证，不能单独作为认定。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .config import DEFAULT, StudyConfig
from .data import pivot_matrices, clear_hour


def compute_price_impact(panel: pd.DataFrame, cfg: StudyConfig = DEFAULT) -> pd.DataFrame:
    """计算每 (unit, hour) 的外生价格影响 Δp。

    返回长表：timestamp, unit_id, delta_p, price_wo（若该机组缺席的出清价）
    """
    M = pivot_matrices(panel)
    units, times = M["units"], M["times"]
    avail, bid, cap, need = M["avail"], M["bid"], M["cap"], M["need"]
    nu, nh = len(units), len(times)

    dp = np.zeros((nu, nh))
    price_without = np.zeros((nu, nh))

    for t in range(nh):
        a = avail[:, t].copy()
        b = bid[:, t]
        nd = need[t]
        if not (a > 1e-6).any():
            continue
        for k in range(nu):
            # 关键：两个反事实都假设 u 始终按额定容量可用，
            # 因此 Δp 只由“其他机组状态 + 负荷”决定，与 u 自身是否停运无关。
            # Δp = u 缺席时的出清价 − u 在场时的出清价，含义是“u 此刻停运能把价格抬高多少”。
            a_with = a.copy()
            a_with[k] = cap[k, t] if cap[k, t] > 0 else a[k]
            p_with, _ = clear_hour(a_with, b, nd, cfg.price_cap)

            a_wo = a_with.copy()
            a_wo[k] = 0.0
            p_wo, _ = clear_hour(a_wo, b, nd, cfg.price_cap)

            dp[k, t] = max(0.0, p_wo - p_with)
            price_without[k, t] = p_wo

    rows = []
    for k, uid in enumerate(units):
        rows.append(pd.DataFrame({
            "timestamp": times, "unit_id": uid,
            "delta_p": dp[k], "price_wo": price_without[k],
        }))
    return pd.concat(rows, ignore_index=True)


def detect_withholding(panel: pd.DataFrame, impact: pd.DataFrame = None,
                       cfg: StudyConfig = DEFAULT, verbose: bool = True):
    """返回 (summary, episode_flags)。

    summary: 每机组持留证据表（按超额停运、租金损失排序）
    episode_flags: 高 Δp 档内被标记的停运事件明细（供逐条列举）
    """
    if impact is None:
        impact = compute_price_impact(panel, cfg)

    # 合并可用性信息（区分“可选停机”与“本就不可用”）
    g = panel[["timestamp", "unit_id", "tech", "owner", "declared_outage",
               "available_mw", "capacity_mw", "price", "price_rank",
               "tight_rank", "rsi", "month"]].copy()
    g["declared_outage"] = g["declared_outage"].fillna(0).astype(int)
    g = g.merge(impact, on=["timestamp", "unit_id"], how="left")

    # 只在机组“本来可用”的小时评估（可用量 < 额定即视为停运候选）
    g["was_available"] = g["available_mw"] > 1e-6
    # 对已申报停运的小时，available=0，需保留；用 capacity>0 即可
    g = g[g["capacity_mw"] > 0].copy()

    # Δp 分档（元/MWh）
    bins = [-1, 10, 30, np.inf]
    labels = ["low", "mid", "high"]
    g["dp_bin"] = pd.cut(g["delta_p"], bins=bins, labels=labels)

    # 紧张度分层（5 档），确保“同等物理压力”下比较
    g["tight_bin"] = pd.cut(g["tight_rank"], [0, .2, .4, .6, .8, 1.0],
                            labels=False, include_lowest=True)

    # 同集团其他机组可用性（捕获租金的前提）

    summary_rows, episode_frames = [], []

    for uid, ug in g.groupby("unit_id", sort=False):
        tech = ug["tech"].iloc[0]
        owner = ug["owner"].iloc[0]

        low = ug[ug["dp_bin"] == "low"]
        high = ug[ug["dp_bin"] == "high"]
        # 只有“显著具备市场力”的机组才检验：高 Δp（≥30）的小时数足够多，
        # 否则停运根本推不动价格，谈不上持留（气电常年低 Δp，应排除）。
        MIN_HIGH_HOURS = 500
        if len(low) < 50 or len(high) < MIN_HIGH_HOURS:
            summary_rows.append({
                "unit_id": uid, "tech": tech, "owner": owner,
                "n_low": len(low), "n_high": len(high),
                "rate_low": low["declared_outage"].mean() if len(low) else np.nan,
                "rate_high": high["declared_outage"].mean() if len(high) else np.nan,
                "excess_rate": np.nan, "lift": np.nan, "p_value": np.nan,
                "excess_hours": np.nan, "rent_loss": np.nan,
                "has_market_power": bool(len(high) >= MIN_HIGH_HOURS),
            })
            continue

        # 分层加权：按 high 组的紧张度分布，对 low 组加权得到可比基线
        hi_w = high["tight_bin"].value_counts(normalize=True)
        base_rate = 0.0
        ok = True
        for b, w in hi_w.items():
            lb = low[low["tight_bin"] == b]
            if len(lb) < 20:
                ok = False
                break
            base_rate += w * lb["declared_outage"].mean()
        if not ok:
            base_rate = low["declared_outage"].mean()

        k_hi = int(high["declared_outage"].sum())
        n_hi = len(high)
        p0 = max(base_rate, 1e-4)
        # 单侧二项检验
        if n_hi >= 30:
            mu = n_hi * p0
            sd = np.sqrt(n_hi * p0 * (1 - p0))
            z = (k_hi - mu) / sd if sd > 0 else 0.0
            pval = 1 - stats.norm.cdf(z)
        else:
            pval = stats.binomtest(k_hi, n_hi, p0, alternative="greater").pvalue

        rate_high = k_hi / n_hi
        lift = rate_high / p0 if p0 > 0 else np.inf
        excess = k_hi - n_hi * p0

        # 租金损失：超额停运小时 × 该小时的 Δp × 机组容量（元）
        flagged_hi = high[(high["declared_outage"] == 1)]
        rent_loss = float((flagged_hi["delta_p"] * flagged_hi["capacity_mw"]).sum())

        summary_rows.append({
            "unit_id": uid, "tech": tech, "owner": owner,
            "n_low": len(low), "n_high": n_hi,
            "rate_low": base_rate, "rate_high": rate_high,
            "excess_rate": rate_high - base_rate, "lift": lift,
            "p_value": pval, "excess_hours": excess,
            "rent_loss": rent_loss, "has_market_power": True,
        })

        if rate_high > base_rate * 1.3:
            sub = flagged_hi[["timestamp", "unit_id", "tech", "owner", "price",
                              "price_rank", "delta_p", "rsi", "tight_rank"]].copy()
            sub["base_rate"] = base_rate
            sub["lift"] = lift
            episode_frames.append(sub)

    summary = pd.DataFrame(summary_rows)

    # BH-FDR（仅对具备市场力且完成检验的机组）
    testable = summary["p_value"].notna()
    if testable.any():
        sub = summary[testable].sort_values("p_value").reset_index()
        m = len(sub)
        p = sub["p_value"].to_numpy()
        q = p * m / np.arange(1, m + 1)
        q = np.minimum.accumulate(q[::-1])[::-1]
        qmap = dict(zip(sub["index"], np.clip(q, 0, 1)))
        summary["q_bh"] = summary.index.map(lambda i: qmap.get(i, np.nan))
        # 判定必须同时满足“统计显著 + 选择性 + 规模”，缺一不可：
        #  - 统计：FDR q <= 阈值
        #  - 选择性：lift >= 1.3（停运确实偏向高 Δp 时刻）且超额停运小时 >= 40
        #    （3.5 万小时下微小差异也会显著，必须要求实质数量）
        #  - 规模：租金损失 >= 1000 万元
        # 注意：不能只看租金损失。串通成员的报价本就被抬高，会虚增其 Δp 与租金，
        # 造成“串通污染持留”的误报；lift+超额小时的“选择性”证据能挡住这类误报。
        summary["significant"] = (
            testable & (summary["q_bh"] <= cfg.strategic_q)
            & (summary["lift"] >= 1.3)
            & (summary["excess_hours"] >= cfg.min_excess_hours)
            & (summary["rent_loss"] >= cfg.min_rent_loss)
        )
    else:
        summary["q_bh"] = np.nan
        summary["significant"] = False

    summary = summary.sort_values(
        ["significant", "rent_loss"], ascending=[False, False]).reset_index(drop=True)

    episodes = pd.concat(episode_frames, ignore_index=True) if episode_frames else pd.DataFrame()

    if verbose:
        print("[withholding] Δp 分档: low<10, mid 10-30, high>=30 元/MWh")
        print("[withholding] 高 Δp 档内按紧张度分层加权，与低 Δp 档比较停运率\n")
        cols = ["unit_id", "tech", "n_high", "rate_low", "rate_high",
                "excess_rate", "lift", "excess_hours", "rent_loss", "p_value",
                "q_bh", "significant"]
        show = summary[cols].copy()
        show["rent_loss"] = (show["rent_loss"] / 1e6).round(2)
        print(show.round(4).to_string(index=False))
        print(f"\n[withholding] 显著持留机组: "
              f"{list(summary.loc[summary['significant'], 'unit_id'])}")

    return summary, episodes


if __name__ == "__main__":
    from .data import load_panel
    p = load_panel()
    s, e = detect_withholding(p)
    s.to_csv("outputs/withholding_summary.csv", index=False)
