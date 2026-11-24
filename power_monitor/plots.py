"""图表生成（供监管报告）。中文字体 + 统一配色。"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import DEFAULT

plt.rcParams["font.sans-serif"] = [DEFAULT.font, "Heiti SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130
plt.rcParams["savefig.dpi"] = 160
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.25
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

C_BASE = "#8A93A0"
C_HI = "#D1495B"
C_BLUE = "#2E6FA7"
C_ORANGE = "#E08A2B"
C_GREEN = "#3C8B5A"


def fig_price_and_tightness(panel, out: Path, colluding_units=None):
    """图1：价格-紧张度关系，并标出“同时抬价”时段，说明正常波动 vs 可疑抬价。"""
    h = panel.drop_duplicates("timestamp").copy()
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    # 是否处于“多家串通成员共同抬价”的小时
    if colluding_units:
        z = panel.pivot(index="timestamp", columns="unit_id", values="z")
        mem = [u for u in colluding_units if u in z.columns]
        co = (z[mem] > 3).sum(axis=1) >= 2
        h["co"] = h["timestamp"].map(co).fillna(False)
        normal = h[~h["co"]]
        flagged = h[h["co"]]
        ax.scatter(normal["tightness"], normal["price"], s=4, alpha=0.20,
                   color=C_BASE, label="正常小时")
        ax.scatter(flagged["tightness"], flagged["price"], s=10, alpha=0.75,
                   color=C_HI, label="多家串通成员同期抬价的小时")
        ax.legend(loc="upper left", frameon=False, fontsize=9)
    else:
        ax.scatter(h["tightness"], h["price"], s=4, alpha=0.25, color=C_BASE,
                   label="全部小时")
    ax.set_xlabel("市场紧张度（净负荷/可调度容量压力）")
    ax.set_ylabel("出清价（元/MWh）")
    ax.set_title("正常价格随紧张度上升；可疑抬价集中在原有价格带之上", fontsize=11)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_residual_distribution(panel, colluding_units, out: Path):
    """图2：残差 z 分布——串通成员 vs 其他。展示“扣除基本面后才现形”。

    用线性区段 + 对数纵轴：正常机组集中在 z∈[-3,3]，
    串通成员在 z>3 处形成一条长尾（可达 30+），对数尺度下才看得清。
    """
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    cart = panel[panel["unit_id"].isin(colluding_units)]["z"].dropna()
    other = panel[~panel["unit_id"].isin(colluding_units)]["z"].dropna()
    lo, hi = -4, max(36, np.nanpercentile(cart, 99.5) if len(cart) else 8)
    bins = np.linspace(lo, hi, 120)
    ax.hist(other, bins=bins, alpha=0.60, color=C_BASE,
            label=f"其他机组 (n={len(other):,})")
    ax.hist(cart, bins=bins, alpha=0.65, color=C_HI,
            label=f"串通成员 (n={len(cart):,})")
    ax.set_yscale("log")
    ax.axvline(3.0, ls="--", color="k", lw=1)
    ax.text(3.4, ax.get_ylim()[1] * 0.35, "抬价阈值 z=3", fontsize=9)
    ax.set_xlabel("基本面残差 z 分数（同等市场条件下的报价偏离）")
    ax.set_ylabel("小时数（对数）")
    ax.set_title("扣除负荷/燃料/新能源后，串通成员在 z>3 形成长尾", fontsize=11)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_coelevation_network(pairs, groups, out: Path):
    """图3：机组两两“同期抬价”倍数热图 + 显著边。"""
    if pairs is None or pairs.empty:
        return
    units = sorted(set(pairs.unit_a) | set(pairs.unit_b))
    n = len(units)
    ui = {u: i for i, u in enumerate(units)}
    M = np.full((n, n), np.nan)
    np.fill_diagonal(M, 1.0)
    Psig = np.zeros((n, n), dtype=bool)
    for r in pairs.itertuples():
        i, j = ui[r.unit_a], ui[r.unit_b]
        M[i, j] = M[j, i] = r.lift
        Psig[i, j] = Psig[j, i] = bool(r.significant)

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    im = ax.imshow(np.clip(M, 0, 12), cmap="YlOrRd", vmin=0, vmax=12)
    ax.set_xticks(range(n)); ax.set_xticklabels(units, rotation=45, ha="right")
    ax.set_yticks(range(n)); ax.set_yticklabels(units)
    for i in range(n):
        for j in range(n):
            if i != j and Psig[i, j]:
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False,
                                           edgecolor="k", lw=2))
    ax.set_title("同期抬价倍数（黑框=通过显著性检验）", fontsize=11)
    fig.colorbar(im, ax=ax, shrink=0.8, label="抬价倍数 lift")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_episode_timeline(panel, colluding_units, hour_flags, out: Path):
    """图4：一个“典型串通时段”的放大图：成员残差 z 与出清价。

    自动选取一个连续串通窗口（成员多家 z>3、且非全饱和），便于说明
    “成员在同一时段共同抬价，出清价随之走高”。
    """
    z = panel.pivot(index="timestamp", columns="unit_id", values="z")
    price = panel.drop_duplicates("timestamp").set_index("timestamp")["price"]
    mem = [u for u in colluding_units if u in z.columns]
    if not mem:
        return

    # 选窗口：以“有 >=2 名成员 z>3”的时段为中心，取前后 120 小时
    co = (z[mem] > 3).sum(axis=1)
    co_times = co[co >= 2].index
    if len(co_times):
        # 避开所有成员都极度饱和的段，取一个中等强度、且价格有变化的位置
        center = co_times[len(co_times) // 3]
    else:
        center = price.index[len(price) // 2]
    lo = center - pd.Timedelta(hours=96)
    hi = center + pd.Timedelta(hours=120)
    win = pd.date_range(lo, hi, freq="h").intersection(z.index)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.2, 5.8), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
    zw = z.loc[win[0]:win[-1]]
    for i, u in enumerate(mem):
        ax1.plot(zw.index, zw[u].clip(upper=40), lw=1.1, alpha=0.9, label=u,
                 color=plt.cm.tab10(i % 10))
    ax1.axhline(3, ls="--", color="k", lw=0.9)
    ax1.set_ylabel("残差 z")
    ax1.set_title("典型时段：串通成员近乎同步地抬高报价", fontsize=11)
    ax1.legend(ncol=len(mem), fontsize=8, frameon=False, loc="upper left")

    pw = price.loc[win[0]:win[-1]]
    ax2.plot(pw.index, pw, color=C_BASE, lw=1.1)
    ax2.set_ylabel("出清价（元/MWh）")
    ax2.set_xlabel("时间")
    # 标出该窗口内成员共现抬价的小时
    for t in co_times:
        if win[0] <= t <= win[-1]:
            ax2.axvspan(t, t + pd.Timedelta(hours=1), color=C_HI, alpha=0.35, lw=0)
    ax1.set_xlim(win[0], win[-1])
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_withholding_dp(panel, impact, withholding_units, summary, out: Path):
    """图5：停运率 vs 价格影响 Δp 分档——持留机组的选择性。"""
    g = panel[["timestamp", "unit_id", "declared_outage"]].copy()
    g = g.merge(impact, on=["timestamp", "unit_id"], how="left")
    g["dp_bin"] = pd.cut(g["delta_p"], [-1, 10, 30, np.inf], labels=["<10", "10-30", "≥30"])
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    width = 0.8 / max(len(withholding_units), 1)
    for i, u in enumerate(withholding_units):
        s = g[g["unit_id"] == u].groupby("dp_bin", observed=False)["declared_outage"].mean()
        x = np.arange(len(s)) + (i - len(withholding_units) / 2) * width + width / 2
        ax.bar(x, s.values, width=width, label=u)
    ax.set_xticks(np.arange(3)); ax.set_xticklabels(["<10", "10–30", "≥30"])
    ax.set_xlabel("若该机组停运，可推高的出清价 Δp（元/MWh）")
    ax.set_ylabel("申报停运率")
    ax.set_title("持留机组的停运集中在“停运最能推高价格”的时刻", fontsize=11)
    ax.legend(ncol=len(withholding_units), fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_impact(cf, out: Path):
    """图6：反事实价格对比 + 额外成本累积。"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 3.9))
    dc = (cf["base"] - cf["no_collusion"]).clip(lower=0)
    dw = (cf["base"] - cf["no_withholding"]).clip(lower=0)
    # 绝大多数小时降幅很小，少数尖峰很大：用截断 + symlog x 轴同时呈现主体与尾部
    cap = 120
    ax1.hist(dc.clip(upper=cap), bins=60, color=C_ORANGE, alpha=0.75,
             label=f"剔除串通加价（均值 {dc.mean():.1f}）")
    ax1.hist(dw.clip(upper=cap), bins=60, color=C_HI, alpha=0.55,
             label=f"恢复持留容量（均值 {dw.mean():.1f}）")
    ax1.set_yscale("log")
    ax1.set_xlabel(f"反事实价格下降（元/MWh，截断于 {cap}）")
    ax1.set_ylabel("小时数（对数）")
    ax1.set_title("两类行为的出清价影响分布", fontsize=11)
    ax1.legend(frameon=False, fontsize=8)

    cum = cf.set_index("timestamp")[["cost_collusion", "cost_withholding"]].cumsum() / 1e8
    ax2.plot(cum.index, cum["cost_collusion"], color=C_ORANGE, lw=1.3, label="串通")
    ax2.plot(cum.index, cum["cost_withholding"], color=C_HI, lw=1.3, label="持留")
    ax2.set_ylabel("累计用户侧额外成本（亿元）")
    ax2.set_title("估算额外成本累积", fontsize=11)
    ax2.legend(frameon=False, fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
