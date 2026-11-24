"""基本面残差模型（整个方法的地基）。

核心思想
--------
电价/报价本来就会随负荷、燃料成本、新能源出力、时段与季节剧烈波动。
直接比较“谁报价高”或“谁和谁同步”会把正常的市场反应当成串通。

因此第一步对每台机组拟合**竞争性报价函数**：

    bid_{u,t} = f_u( 边际成本, 紧张度, 新能源占比, 时段谐波, 季节谐波, 周末 )
                + ε_{u,t}

用**分位数回归（τ=0.5，中位数）**估计 f_u，而非最小二乘：
分位数回归对少数被抬高的观测（串通时段约占 3%）不敏感，
不会被“可疑样本”把基准线拉高——这是保证反事实干净的关键。

残差 ε 做稳健标准化（MAD）得到 z 分数：z 越大，报价越偏离
“同等市场条件下的竞争性水平”。z 是后续串通/持留识别的共同输入。

重要：这里是**统计异常**，不是违法认定。抬价也可能是合法的市场力运用，
是否构成串通则需要“协调性”证据（见 collusion.py）。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .config import DEFAULT, StudyConfig


@dataclass
class BaselineResult:
    panel: pd.DataFrame                 # 含 pred_bid / resid / z
    coefs: pd.DataFrame                 # 每机组回归系数
    fit_quality: pd.DataFrame           # 每机组 R²(伪)、残差自相关
    pooled_r2: float = 0.0


def _design(df: pd.DataFrame) -> np.ndarray:
    """构建设计矩阵（不含常数项，统一在 fit 中加）。"""
    h = df["hour"].to_numpy(dtype=float)
    mo = df["month"].to_numpy(dtype=float)
    tight = df["tightness"].to_numpy(dtype=float)
    ren = (df["renewable_mw"] / df["load_mw"].clip(lower=1)).to_numpy(dtype=float)
    mc = df["mc_obs"].to_numpy(dtype=float)
    wk = (df["dow"] >= 5).to_numpy(dtype=float)
    net_ratio = (df["net_load_mw"] / df["capacity_mw"].sum()).to_numpy(dtype=float)

    cols = [
        mc,
        tight,
        tight ** 2,
        ren,
        np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24),
        np.sin(4 * np.pi * h / 24), np.cos(4 * np.pi * h / 24),
        np.sin(2 * np.pi * mo / 12), np.cos(2 * np.pi * mo / 12),
        wk,
        net_ratio,
    ]
    return np.column_stack(cols)


FEATURE_NAMES = [
    "mc_obs", "tightness", "tightness^2", "renewable_share",
    "hour_sin", "hour_cos", "hour2_sin", "hour2_cos",
    "month_sin", "month_cos", "weekend", "net_load_ratio",
]


def _robust_scale(x: np.ndarray) -> float:
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    s = 1.4826 * mad
    return float(s) if s > 1e-9 else float(np.std(x) + 1e-9)


def fit_baseline(panel: pd.DataFrame, cfg: StudyConfig = DEFAULT) -> BaselineResult:
    """逐机组拟合竞争性报价函数，返回带残差与 z 分数的面板。"""
    panel = panel.copy()
    panel["pred_bid"] = np.nan
    panel["resid"] = np.nan
    panel["z"] = np.nan
    panel["resid_scale"] = np.nan

    coef_rows, quality_rows = [], []

    for uid, g in panel.groupby("unit_id", sort=False):
        g = g.sort_values("timestamp")
        y = g["bid_price"].to_numpy(dtype=float)
        X = _design(g)
        ok = np.isfinite(y) & np.all(np.isfinite(X), axis=1) & np.isfinite(g["mc_obs"].to_numpy())
        if ok.sum() < 200:
            # 样本太少，退化为稳健常数
            pred = np.full_like(y, np.median(y[ok]) if ok.any() else np.nan)
        else:
            Xs = X[ok]
            ys = y[ok]
            # 标准化以改善数值稳定性
            mu = Xs.mean(0)
            sd = Xs.std(0)
            sd[sd < 1e-9] = 1.0
            Xn = (Xs - mu) / sd
            Xn = sm.add_constant(Xn)
            try:
                res = sm.QuantReg(ys, Xn).fit(q=0.5, max_iter=5000)
                pred_ok = Xn @ res.params
                pred = np.full_like(y, np.nan)
                pred[ok] = pred_ok
                # 记录系数（转换回原尺度：仅记录标准化系数便于比较）
                coefs = dict(zip(["const"] + FEATURE_NAMES, res.params))
                pseudo_r2 = 1 - (np.sum(np.abs(ys - pred_ok)) /
                                 max(np.sum(np.abs(ys - np.median(ys))), 1e-9))
                resid_ok = ys - pred_ok
                # 一阶自相关
                ac1 = np.corrcoef(resid_ok[:-1], resid_ok[1:])[0, 1] if len(resid_ok) > 2 else np.nan
            except Exception:
                pred = np.full_like(y, np.nan)
                pred[ok] = np.median(ys)
                coefs = {k: np.nan for k in ["const"] + FEATURE_NAMES}
                pseudo_r2, ac1 = np.nan, np.nan
        resid = y - pred
        scale = _robust_scale(resid[ok]) if ok.any() else np.nan
        z = resid / scale if np.isfinite(scale) else np.full_like(resid, np.nan)

        idx = g.index
        panel.loc[idx, "pred_bid"] = pred
        panel.loc[idx, "resid"] = resid
        panel.loc[idx, "z"] = z
        panel.loc[idx, "resid_scale"] = scale

        coef_row = {"unit_id": uid}
        coef_row.update(coefs)
        coef_rows.append(coef_row)
        quality_rows.append({
            "unit_id": uid, "n_obs": int(ok.sum()),
            "pseudo_r2": pseudo_r2, "resid_ac1": ac1,
            "resid_scale": scale,
            "mean_bid": float(np.nanmean(y)),
        })

    coefs_df = pd.DataFrame(coef_rows)
    quality_df = pd.DataFrame(quality_rows)

    # 合并参照：pooled 模型的解释力，用于报告“共同因素解释了大部分波动”
    pooled_r2 = _pooled_fit_r2(panel)

    return BaselineResult(panel=panel, coefs=coefs_df, fit_quality=quality_df, pooled_r2=pooled_r2)


def _pooled_fit_r2(panel: pd.DataFrame) -> float:
    """池化稳健拟合的解释力，用于说明“共同市场条件”能解释多少报价波动。"""
    g = panel.dropna(subset=["bid_price"]).copy()
    if len(g) < 1000:
        return float("nan")
    y = g["bid_price"].to_numpy(dtype=float)
    X = _design(g)
    units = pd.get_dummies(g["unit_id"], drop_first=True).to_numpy(dtype=float)
    X = np.column_stack([X, units])
    ok = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    if ok.sum() < 1000:
        return float("nan")
    Xn = (X[ok] - X[ok].mean(0)) / (X[ok].std(0) + 1e-9)
    Xn = sm.add_constant(Xn)
    try:
        res = sm.QuantReg(y[ok], Xn).fit(q=0.5)
        pred = Xn @ res.params
        return float(1 - np.sum(np.abs(y[ok] - pred)) /
                     max(np.sum(np.abs(y[ok] - np.median(y[ok]))), 1e-9))
    except Exception:
        return float("nan")


if __name__ == "__main__":
    from .data import load_panel
    p = load_panel()
    r = fit_baseline(p)
    print(r.fit_quality.round(3).to_string(index=False))
    print("\npooled 伪R² =", round(r.pooled_r2, 3))
    print("\n各机组 |z|>3 的小时占比:")
    z = r.panel.groupby("unit_id")["z"].apply(lambda s: (s.abs() > 3).mean())
    print(z.round(4).to_string())
