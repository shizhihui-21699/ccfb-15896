"""反事实分析与影响量化：把“异常”翻译成监管人员看得懂的价格/福利影响。

三个问题
--------
Q1 串通抬价对出清价贡献多少？
   → 把串通成员的报价残差（超出竞争性基准的部分）扣除后重新出清，比较价格。
Q2 持留若恢复供电，价格会降多少？
   → 把被标记持留机组的可用容量恢复为额定值，重新出清。
Q3 两类行为合计造成多少用户侧额外成本？
   → (实际价格 − 反事实价格) × 负荷，按小时累加。

这些都是**反事实推断**，依赖出清模型假设（merit order、无网络约束）。
报告中必须声明假设，并给出敏感性区间（例如只扣除显著成员的加价）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DEFAULT, StudyConfig
from .data import pivot_matrices, clear_hour


def counterfactual_price(panel: pd.DataFrame, colluding_units, withholding_units,
                         cfg: StudyConfig = DEFAULT):
    """分别计算基线、仅去串通、仅恢复持留、两者都修正 四种情景的小时价格。"""
    M = pivot_matrices(panel)
    units, times = M["units"], M["times"]
    avail, bid, cap, need = M["avail"], M["bid"], M["cap"], M["need"]
    nh = len(times)

    collude_idx = [units.index(u) for u in colluding_units if u in units]
    withhold_idx = [units.index(u) for u in withholding_units if u in units]

    # 竞争性反事实报价：用 baseline 的预测报价（pred_bid）替换被抬高的部分，
    # 但只对“显著抬价”的小时替换，避免过度修正。
    pred = panel.pivot(index="unit_id", columns="timestamp", values="pred_bid")
    pred = pred.reindex(index=units, columns=pd.to_datetime(times))
    bid_cf = bid.copy()
    for k in collude_idx:
        b_pred = pred.iloc[k].to_numpy(dtype=float)
        ok = np.isfinite(b_pred)
        bid_cf[k, ok] = b_pred[ok]

    avail_wh = avail.copy()
    for k in withhold_idx:
        avail_wh[k, :] = cap[k, :]          # 恢复额定容量

    scenarios = {}
    names = ["base", "no_collusion", "no_withholding", "fixed_both"]
    for name in names:
        used_avail = avail_wh if name in ("no_withholding", "fixed_both") else avail
        used_bid = bid_cf if name in ("no_collusion", "fixed_both") else bid
        prices = np.zeros(nh)
        for t in range(nh):
            p, _ = clear_hour(used_avail[:, t], used_bid[:, t], need[t], cfg.price_cap)
            prices[t] = p
        scenarios[name] = prices

    out = pd.DataFrame({"timestamp": times, "load_mw": need})
    for n in names:
        out[n] = scenarios[n]
    out["base_price"] = out["base"]

    # 用户侧额外成本（元）= 价差 × 负荷
    out["cost_collusion"] = (out["base"] - out["no_collusion"]).clip(lower=0) * out["load_mw"]
    out["cost_withholding"] = (out["base"] - out["no_withholding"]).clip(lower=0) * out["load_mw"]
    out["cost_both"] = (out["base"] - out["fixed_both"]).clip(lower=0) * out["load_mw"]
    return out


def summarize_impact(cf: pd.DataFrame) -> pd.DataFrame:
    """汇总价格与成本影响。"""
    rows = []
    for name, label in [("no_collusion", "剔除串通加价"),
                        ("no_withholding", "恢复被持留容量"),
                        ("fixed_both", "两者同时修正")]:
        price_drop = cf["base"] - cf[name]
        cost = cf[f"cost_{'collusion' if name=='no_collusion' else 'withholding' if name=='no_withholding' else 'both'}"]
        rows.append({
            "scenario": label,
            "mean_price_base": cf["base"].mean(),
            "mean_price_cf": cf[name].mean(),
            "mean_price_drop": price_drop.mean(),
            "p95_price_drop": price_drop.quantile(0.95),
            "max_price_drop": price_drop.max(),
            "hours_affected": int((price_drop > 1).sum()),
            "total_cost_yi": cost.sum() / 1e8,      # 亿元
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    from .data import load_panel
    from .baseline import fit_baseline
    from .collusion import detect_collusion
    from .withholding import detect_withholding, compute_price_impact

    p = load_panel()
    b = fit_baseline(p)
    _, groups, _ = detect_collusion(b.panel, verbose=False)
    s, _ = detect_withholding(b.panel, compute_price_impact(b.panel), verbose=False)
    colluding = [u for g in groups for u in g["members"]]
    withholding = list(s.loc[s["significant"], "unit_id"])
    cf = counterfactual_price(b.panel, colluding, withholding)
    print(summarize_impact(cf).round(2).to_string(index=False))
