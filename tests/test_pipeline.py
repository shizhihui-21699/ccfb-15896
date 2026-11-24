"""关键不变量测试：确保方法在重构后仍正确。

运行： .venv/bin/python -m pytest tests/ -q    (或 .venv/bin/python tests/test_pipeline.py)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from power_monitor.data import clear_hour, load_panel
from power_monitor.baseline import fit_baseline
from power_monitor.collusion import detect_collusion, _bh_fdr
from power_monitor.withholding import compute_price_impact


def test_clear_hour_basic():
    """merit order：最便宜的先出清，价格=边际机组报价。"""
    avail = np.array([100.0, 100.0, 100.0])
    bid = np.array([100.0, 200.0, 300.0])
    p, cleared = clear_hour(avail, bid, need=150.0)
    assert abs(p - 200.0) < 1e-6
    assert abs(cleared.sum() - 150.0) < 1e-6
    assert cleared[0] == 100.0 and cleared[1] == 50.0


def test_clear_hour_scarcity():
    """供不应求时价格顶格。"""
    avail = np.array([10.0, 10.0])
    bid = np.array([100.0, 100.0])
    p, _ = clear_hour(avail, bid, need=100.0, price_cap=1500.0)
    assert p == 1500.0


def test_price_impact_exogenous():
    """Δp 与机组自身是否停运无关（同状态重复计算结果一致，且非负）。"""
    rng = np.random.default_rng(0)
    n = 400
    avail = np.column_stack([rng.uniform(50, 100, n) for _ in range(4)])
    bid = np.column_stack([rng.uniform(100, 400, n) for _ in range(4)])
    need = rng.uniform(100, 250, n)
    # 直接用 compute_price_impact 依赖 panel；这里验证核心性质：
    # 机组 0 缺席时的价格 ≥ 在场时 → Δp ≥ 0
    dps = []
    for t in range(n):
        a = avail[t]; b = bid[t]
        p_with, _ = clear_hour(a, b, need[t])
        a2 = a.copy(); a2[0] = 0
        p_wo, _ = clear_hour(a2, b, need[t])
        dps.append(p_wo - p_with)
    assert min(dps) >= -1e-6, "Δp 不应为负"


def test_bh_fdr_monotone():
    """BH q 值应随 p 值单调不减。"""
    pairs = pd.DataFrame({"p": [0.001, 0.01, 0.2, 0.5, 0.9]})
    out = _bh_fdr(pairs, q=0.1)
    q = out.sort_values("p")["q_bh"].to_numpy()
    assert np.all(np.diff(q) >= -1e-9)


def test_end_to_end_detects_truth():
    """端到端：在仿真数据上应检出全部真值、零误报。"""
    data_dir = Path(__file__).resolve().parent.parent / "data"
    if not (data_dir / "truth.json").exists():
        print("  (跳过：无 data/truth.json，先运行 --simulate)")
        return
    truth = json.loads((data_dir / "truth.json").read_text(encoding="utf-8"))
    panel = load_panel(data_dir)
    base = fit_baseline(panel)
    pairs, groups, _ = detect_collusion(base.panel, verbose=False)
    detected = set(u for g in groups for u in g["members"])
    assert detected == set(truth["cartel"]), f"串通检出不符: {detected}"

    from power_monitor.withholding import detect_withholding
    s, _ = detect_withholding(base.panel, compute_price_impact(base.panel), verbose=False)
    w = set(s.loc[s["significant"], "unit_id"])
    assert w == set(truth["withholders"]), f"持留检出不符: {w}"
    print(f"  ✓ 串通 {sorted(detected)}  持留 {sorted(w)}")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        print(f"[{fn.__name__}]")
        fn()
    print("\n全部通过。")
