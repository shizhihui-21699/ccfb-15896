"""主流程：一键跑通“数据 → 基本面残差 → 串通/持留识别 → 影响量化 → 报告”。

用法：
    python -m power_monitor.run               # 用已有 data/，如需重跑仿真加 --simulate
    python -m power_monitor.run --simulate    # 重新仿真并分析
"""
from __future__ import annotations

import argparse
import json
import numpy as np
import pandas as pd

from .config import DEFAULT, DATA_DIR, OUT_DIR
from .data import load_panel
from .baseline import fit_baseline
from .collusion import detect_collusion
from .withholding import detect_withholding, compute_price_impact
from .impact import counterfactual_price, summarize_impact
from . import plots


def run(simulate_first: bool = False, cfg=DEFAULT, verbose: bool = True):
    OUT_DIR.mkdir(exist_ok=True)
    if simulate_first:
        from .simulate import simulate
        simulate(cfg, verbose=verbose)

    # ---------- 1. 数据与基本面模型 ----------
    panel = load_panel()
    base = fit_baseline(panel, cfg)
    panel = base.panel
    if verbose:
        print("\n=== 1. 基本面残差模型 ===")
        print(f"共同市场条件可解释的报价波动（池化伪R²）: {base.pooled_r2:.3f}")
        print(f"各机组残差稳健尺度: 中位数 {base.fit_quality['resid_scale'].median():.2f} 元/MWh")

    # ---------- 2. 串通识别 ----------
    pairs, groups, hour_flags = detect_collusion(panel, cfg, verbose=verbose)
    colluding = [u for g in groups for u in g["members"]]
    pairs.to_csv(OUT_DIR / "collusion_pairs.csv", index=False)

    # ---------- 3. 持留识别 ----------
    if verbose:
        print("\n=== 3. 策略性持留识别 ===")
    impact = compute_price_impact(panel, cfg)
    with_summary, with_episodes = detect_withholding(panel, impact, cfg, verbose=verbose)
    with_summary.to_csv(OUT_DIR / "withholding_summary.csv", index=False)
    withholding = list(with_summary.loc[with_summary["significant"], "unit_id"]) \
        if "significant" in with_summary else []

    # ---------- 4. 影响量化 ----------
    if verbose:
        print("\n=== 4. 反事实影响量化 ===")
    cf = counterfactual_price(panel, colluding, withholding, cfg)
    impact_summary = summarize_impact(cf)
    cf.to_csv(OUT_DIR / "counterfactual_hourly.csv", index=False)
    impact_summary.to_csv(OUT_DIR / "impact_summary.csv", index=False)
    if verbose:
        print(impact_summary.round(2).to_string(index=False))

    # ---------- 5. 评估（仅方法验证，不进入监管报告）----------
    from .evaluate import evaluate
    metrics = {}
    try:
        metrics = evaluate(DATA_DIR, groups, with_summary,
                           collusion_pairs=pairs, hour_flags=hour_flags,
                           withholding_episodes=with_episodes)
        (OUT_DIR / "evaluation.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        if verbose:
            _print_metrics(metrics)
    except FileNotFoundError:
        if verbose:
            print("（无 truth.json，跳过真值评估）")

    # ---------- 6. 图表 ----------
    try:
        plots.fig_price_and_tightness(panel, OUT_DIR / "fig1_price_tightness.png", colluding)
        plots.fig_residual_distribution(panel, colluding, OUT_DIR / "fig2_residual_z.png")
        plots.fig_coelevation_network(pairs, groups, OUT_DIR / "fig3_network.png")
        plots.fig_episode_timeline(panel, colluding, hour_flags, OUT_DIR / "fig4_timeline.png")
        if withholding:
            plots.fig_withholding_dp(panel, impact, withholding, with_summary,
                                     OUT_DIR / "fig5_withholding.png")
        plots.fig_impact(cf, OUT_DIR / "fig6_impact.png")
        if verbose:
            print(f"\n[plots] 已生成图表至 {OUT_DIR}")
    except Exception as e:
        if verbose:
            print(f"[plots] 绘图失败: {e}")

    # ---------- 7. Markdown 报告 ----------
    report = build_report(panel, base, pairs, groups, hour_flags, with_summary,
                          with_episodes, impact, impact_summary, metrics, cfg)
    (OUT_DIR / "监管分析报告.md").write_text(report, encoding="utf-8")
    if verbose:
        print(f"[report] 已生成 {OUT_DIR / '监管分析报告.md'}")

    return dict(panel=panel, base=base, pairs=pairs, groups=groups,
                hour_flags=hour_flags, with_summary=with_summary,
                with_episodes=with_episodes, impact=impact,
                impact_summary=impact_summary, cf=cf, metrics=metrics)


def _print_metrics(m):
    cu = m.get("collusion_unit", {})
    print("\n=== 5. 真值评估（方法验证）===")
    print(f"[串通-机组级] P={cu.get('precision'):.2f} R={cu.get('recall'):.2f} "
          f"F1={cu.get('f1'):.2f}  命中={cu.get('tp')} 误报={cu.get('fp')}")
    ce = m.get("collusion_edge")
    if ce:
        print(f"[串通-机组对] P={ce['precision']:.2f} R={ce['recall']:.2f} F1={ce['f1']:.2f}")
    ch = m.get("collusion_hour")
    if ch:
        print(f"[串通-时段]   P={ch['precision']:.2f} R={ch['recall']:.2f} "
              f"(标记 {ch['flagged_hours']}h / 真值 {ch['true_hours']}h)")
    wu = m.get("withholding_unit", {})
    print(f"[持留-机组级] P={wu.get('precision'):.2f} R={wu.get('recall'):.2f} "
          f"F1={wu.get('f1'):.2f}  命中={wu.get('tp')} 误报={wu.get('fp')}")
    we = m.get("withholding_event")
    if we:
        print(f"[持留-事件级] P={we['precision']:.2f} R={we['recall']:.2f} "
              f"(标记 {we['flagged']} / 真值 {we['true_events']})")


def build_report(panel, base, pairs, groups, hour_flags, with_summary,
                 with_episodes, impact, impact_summary, metrics, cfg) -> str:
    from datetime import datetime
    L = []
    A = L.append
    A("# 省级现货市场报价串通与持留行为识别分析报告\n")
    A(f"> 分析区间：{panel['timestamp'].min():%Y-%m-%d} 至 {panel['timestamp'].max():%Y-%m-%d}"
      f"（{panel['timestamp'].nunique():,} 小时，{panel['unit_id'].nunique()} 台机组）  ")
    A(f"> 生成时间：{datetime.now():%Y-%m-%d %H:%M}  ")
    A("> **本报告为数据监测结论，不构成违法认定。**\n")

    A("## 一、方法与为什么不能只看“同步”\n")
    A("电价会随负荷、燃料成本、新能源出力剧烈波动，所有机组对相同的市场条件做出同向反应。")
    A(f"本样本中，仅用负荷、燃料、新能源、时段与季节等**共同因素**，即可解释机组报价波动的 "
      f"**{base.pooled_r2*100:.1f}%**（池化伪R²）。")
    A("因此“两家报价同涨同跌”本身几乎不含信息量——真正的证据必须在**扣除这些共同条件之后的残差**里寻找。\n")

    # 相关性对照实验（直接从数据算，最有说服力）
    import itertools
    bid = panel.pivot(index="timestamp", columns="unit_id", values="bid_price")
    zmat = panel.pivot(index="timestamp", columns="unit_id", values="z")
    cart = [u for g in groups for u in g["members"]] if groups else []
    others = [u for u in bid.columns if u not in cart]

    def _avg_corr(us, mat):
        vals = [mat.loc[a, b] for a, b in itertools.combinations(us, 2)]
        return float(np.mean(vals)) if vals else float("nan")

    if cart and len(others) >= 2 and len(cart) >= 2:
        A("**实证：为什么直接看相关性会失效。** 在本样本中做对照——\n")
        A("| 相关性口径 | 串通集团内部 | 非串通机组之间 | 跨组 |")
        A("|---|---|---|---|")
        cb = bid.corr()
        cross_raw = float(np.mean([cb.loc[a, b] for a in cart for b in others]))
        A(f"| 原始报价 | {_avg_corr(cart, cb):.3f} | {_avg_corr(others, cb):.3f} | {cross_raw:.3f} |")
        zc = zmat.corr()
        cross_z = float(np.mean([zc.loc[a, b] for a in cart for b in others]))
        A(f"| 基本面残差 z | {_avg_corr(cart, zc):.3f} | {_avg_corr(others, zc):.3f} | {cross_z:.3f} |")
        A("")
        A("原始报价里，串通集团内部的相关系数与非串通机组之间几乎一样高（都是共同市场条件所致）；"
          "而扣除基本面后，串通的“同步”才与普通机组的“巧合”被彻底拉开。**这正是本方法的核心价值。**\n")

    A("三步法：\n")
    A("1. **基本面残差模型**：对每台机组用分位数回归拟合竞争性报价函数，得到残差 z 分数。"
      "分位数回归（而非最小二乘）对少数被抬高的样本不敏感，避免基准线被可疑样本污染。")
    A("2. **串通识别**：按市场条件分层，在层内检验两家机组是否**同期出现无法由基本面解释的抬价**；"
      "置换检验给出零分布，BH-FDR 控制误报，并用“抬价倍数”提供效应量。")
    A("3. **持留识别**：计算“若该机组此刻停运能把价格抬高多少”（外生价格影响 Δp）。"
      "真实故障无选择性，策略持留则会集中发生在 Δp 大的时刻——以此区分两者。\n")

    A("## 二、串通（报价协同）识别结果\n")
    if groups:
        for gi, g in enumerate(groups, 1):
            A(f"### 候选集团 {gi}：{'、'.join(g['members'])}\n")
            A(f"- 内部显著“同期抬价”机组对：{g['n_edges']} 对（两两组合共 "
              f"{len(g['members'])*(len(g['members'])-1)//2} 对）")
            A(f"- 平均抬价倍数 lift：**{g['mean_lift']:.1f} 倍**（观察到的同期抬价小时数 / "
              f"独立随市场波动时的期望）")
            if g.get("co_hours"):
                A(f"- 共现抬价小时数：**{g['co_hours']} 小时**")
            if g.get("leader_rotation") is not None:
                rot = g["leader_rotation"]
                lc = g.get("leader_counts", {})
                A(f"- 轮值领导：领导权分散度 **{rot:.2f}**（0=固定一家，1=完全轮流）；"
                  f"各机组担任价格领导者小时数：{lc}")
                if rot > 0.7:
                    A("  - 提示**轮流坐庄**：成员轮换领涨，符合默契协同的典型特征。")
            A("")
        sig = pairs[pairs["significant"]] if len(pairs) else pairs
        if len(sig):
            A("**通过检验的机组对（前 12）**\n")
            A("| 机组A | 机组B | 同期抬价小时 | 期望(独立) | 抬价倍数 | p值 | FDR q值 |")
            A("|---|---|---|---|---|---|---|")
            for r in sig.head(12).itertuples():
                A(f"| {r.unit_a} | {r.unit_b} | {r.obs_co:.0f} | {r.exp_co:.1f} | "
                  f"{r.lift:.2f} | {r.p:.4f} | {r.q_bh:.4f} |")
            A("")
    else:
        A("未发现通过显著性检验的串通集团。\n")

    A("## 三、策略性持留识别结果\n")
    if with_summary is not None and len(with_summary):
        A("识别逻辑：比较机组在**高价格影响（Δp≥30 元/MWh）**与**低价格影响（Δp<10）**时段的停运率，"
          "并在紧张度分层内加权，确保比较发生在同等物理压力下。"
          "仅在机组**具备市场力**（Δp 经常能达到 ≥30）时才检验——没有市场力的机组停运推不动价格，"
          "不存在持留动机。\n")
        # 仅展示具备市场力的机组（避免罗列一堆 Δp≈0 的气电/水电）
        show = with_summary.copy()
        if "has_market_power" in show.columns:
            show = show[show["has_market_power"].fillna(False)]
        show["租金损失(万元)"] = (show["rent_loss"] / 1e4).round(0)
        cols = ["unit_id", "tech", "rate_low", "rate_high", "lift",
                "excess_hours", "租金损失(万元)", "q_bh", "significant"]
        show = show[[c for c in cols if c in show.columns]]
        A(show.round(3).to_markdown(index=False))
        A("")
        sig = with_summary[with_summary.get("significant", False) == True] \
            if "significant" in with_summary else with_summary.iloc[0:0]
        if len(sig):
            A(f"**显著持留机组：{'、'.join(sig['unit_id'])}**\n")
            A("要点：这些机组在“停运最能推高价格”的时刻，停运率显著高于其在低 Δp 时段的水平；"
              "而未达门槛的机组（如实为真实故障率随负荷上升者）不具备这种选择性。\n")
        # 说明“看起来可疑但被排除”的机组
        excl = with_summary[(~with_summary.get("significant", False).fillna(False))]
        if "has_market_power" in excl.columns:
            excl = excl[excl["has_market_power"].fillna(False)]
        note = []
        for r in excl.itertuples():
            if pd.notna(getattr(r, "lift", None)) and r.lift >= 1.3:
                note.append(f"- {r.unit_id}：抬价倍数 {r.lift:.2f} 看似偏高，但超额停运仅 "
                            f"{r.excess_hours:.1f} 小时，未达实质门槛（≥{cfg.min_excess_hours:.0f} 小时），"
                            f"按“证据不足”处理，建议列为观察名单。")
        if note:
            A("**被排除但值得留意的机组（证据不足，非清白认定）**\n")
            A("\n".join(note) + "\n")
    else:
        A("未发现通过门槛的持留机组。\n")

    A("## 四、影响量化（反事实）\n")
    A("基于 merit-order 再出清的反事实推断，**假设无网络阻塞、机组按报价诚实出清**。\n")
    s = impact_summary.copy()
    s["total_cost_yi"] = s["total_cost_yi"].round(2)
    A(s.rename(columns={
        "scenario": "情景", "mean_price_base": "基准均价",
        "mean_price_cf": "反事实均价", "mean_price_drop": "均价降幅",
        "p95_price_drop": "P95降幅", "max_price_drop": "最大降幅",
        "hours_affected": "受影响小时", "total_cost_yi": "用户侧额外成本(亿元)",
    }).round(2).to_markdown(index=False))
    A("")

    A("## 五、给监管人员的证据链与使用说明\n")
    A("每条结论都附带可复核的中间量（残差、抬价倍数、Δp、FDR q 值、事件清单），"
      "便于逐条回溯到原始报价与检修申报。建议监管流程：\n")
    A("1. **先看效应量再看显著性**：3.5 万小时样本下微小差异也会“统计显著”，"
      "报告已设抬价倍数、超额停运小时、租金损失等实质门槛。")
    A("2. **串通需协调性证据**：统计上的“同期抬价”是异常线索，认定还需结合通讯记录、"
      "共同行为模式、人员流动、结构性关联等。")
    A("3. **持留需检修真实性核查**：调取工单、备件领用、修复时长、人员到岗、"
      "同类机组同期状态等，验证“缺陷检修”申报是否真实。")
    A("4. **反事实结论标注假设**：价格影响依赖出清模型；应做敏感性分析"
      "（如仅扣除高置信成员、改变 Δp 阈值）后再对外引用。\n")

    if metrics:
        A("## 附录：方法验证（基于仿真真值，非监管结论）\n")
        A("本数据集为可控仿真，内置已知的串通与持留真值，用于检验方法本身的检出与误报：\n")
        cu = metrics.get("collusion_unit", {})
        wu = metrics.get("withholding_unit", {})
        A(f"- 串通机组级：精确率 {cu.get('precision',float('nan')):.2f}，"
          f"召回率 {cu.get('recall',float('nan')):.2f}，F1 {cu.get('f1',float('nan')):.2f}")
        if metrics.get("collusion_hour"):
            ch = metrics["collusion_hour"]
            A(f"- 串通时段级：精确率 {ch['precision']:.2f}，召回率 {ch['recall']:.2f}")
        A(f"- 持留机组级：精确率 {wu.get('precision',float('nan')):.2f}，"
          f"召回率 {wu.get('recall',float('nan')):.2f}，F1 {wu.get('f1',float('nan')):.2f}")
        if metrics.get("withholding_event"):
            we = metrics["withholding_event"]
            A(f"- 持留事件级：精确率 {we['precision']:.2f}，召回率 {we['recall']:.2f}")
        A("")

    A("## 附：图表\n")
    for f, cap in [("fig1_price_tightness.png", "价格与紧张度：共同条件主导"),
                   ("fig2_residual_z.png", "残差分布：串通成员的右尾"),
                   ("fig3_network.png", "同期抬价网络"),
                   ("fig4_timeline.png", "同时抬价时间线"),
                   ("fig5_withholding.png", "停运率 vs 价格影响"),
                   ("fig6_impact.png", "反事实价格与成本")]:
        A(f"![{cap}]({f})\n")

    A("---\n")
    A("**方法学局限**：出清模型简化（无网络约束、无机组组合/爬坡）；"
      "反事实报价用中位数回归预测，可能低估/高估真实竞争水平；"
      "仿真数据用于方法验证，真实数据需重新校准阈值。")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--simulate", action="store_true", help="先重新生成仿真数据")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    run(simulate_first=args.simulate, verbose=not args.quiet)


if __name__ == "__main__":
    main()
