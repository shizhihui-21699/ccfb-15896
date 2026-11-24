"""现货市场仿真器：生成带“已知真值”的小时级市场数据。

设计原则
--------
1. 分析模块只能看到可观测数据（报价、出清价、负荷、新能源、燃料指数、检修申报），
   真值单独落盘，用于事后评估误报/漏报。
2. 刻意制造**混淆**：所有机组都对同一负荷/燃料冲击做出反应（正常同步），
   且串通与持留都集中发生在高负荷高电价时段——这正是现实中监管最难的地方。
3. 真值包含三类：collusion（报价串通）、withholding（策略性持留）、以及
   legitimate_maintenance（合法检修，不应被识别为异常）。

单位口径（关键，避免量纲错误）
----------------------------
* 煤电 heat_rate：g 标煤/kWh，燃料成本 = heat_rate * 煤价(元/吨) / 1000  元/MWh
* 气电 heat_rate：m3 天然气/kWh，燃料成本 = heat_rate * 气价(元/m3) * 1000 元/MWh
* 水电 heat_rate：忽略，按机会成本（来水越枯越高）计

输出（data/）
-----------
market_hourly.csv   每小时：负荷、净负荷、新能源、燃料指数、出清价、RSI
offers.csv          每机组每小时：可用容量、申报价、中标量、是否边际机组
outages.csv         每机组每小时：申报停运、停运类型、申报原因
units.csv           机组静态参数
truth.json          真值（分析模块不得读取）
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

from .config import DEFAULT, DATA_DIR


@dataclass
class Unit:
    uid: str
    tech: str          # coal / gas / hydro
    cap: float         # MW
    heat_rate: float   # 煤:g/kWh标煤；气:m3/kWh
    vom: float         # 变动运维成本 元/MWh
    owner: str
    markup_a: float    # 竞争性加价对紧张度的斜率 元/MWh
    markup_b: float    # 基础加价 元/MWh


UNITS: List[Unit] = [
    # 6 台煤电（基准负荷，通常为边际机组）
    Unit("C01", "coal", 700, 300, 18, "甲集团", 0.28, 10),
    Unit("C02", "coal", 700, 302, 18, "甲集团", 0.27, 10),
    Unit("C03", "coal", 700, 296, 17, "乙集团", 0.30, 12),
    Unit("C04", "coal", 700, 305, 19, "乙集团", 0.29, 11),
    Unit("C05", "coal", 600, 320, 20, "丙集团", 0.32, 13),
    Unit("C06", "coal", 600, 325, 21, "丙集团", 0.31, 12),
    # 6 台燃气（调峰，边际价格常由它决定）
    Unit("G01", "gas", 400, 0.190, 25, "丁集团", 0.55, 18),
    Unit("G02", "gas", 400, 0.195, 26, "丁集团", 0.53, 17),
    Unit("G03", "gas", 400, 0.198, 26, "戊集团", 0.56, 19),
    Unit("G04", "gas", 350, 0.205, 27, "戊集团", 0.54, 18),
    Unit("G05", "gas", 350, 0.200, 26, "己集团", 0.57, 20),
    Unit("G06", "gas", 350, 0.210, 28, "己集团", 0.55, 19),
    # 4 台水电（低成本、受来水影响）
    Unit("H01", "hydro", 250, 0.0, 8, "庚集团", 0.12, 4),
    Unit("H02", "hydro", 250, 0.0, 8, "庚集团", 0.13, 4),
    Unit("H03", "hydro", 200, 0.0, 9, "辛集团", 0.11, 3),
    Unit("H04", "hydro", 200, 0.0, 9, "辛集团", 0.12, 3),
]

# ---- 真值定义（分析模块不可见）----
CARTEL = ["C03", "C04", "G02", "G03"]          # 串通集团（跨所有者、覆盖边际带）
WITHHOLDERS = ["C01", "C02", "C05", "C06"]    # 策略性持留机组（丙/甲集团，具备市场力的煤电）
PLANNED_MAINT_UNITS = ["C01", "C06", "H03"]    # 做合法计划检修的机组

COAL_PRICE_BASE = 700.0     # 元/吨
GAS_PRICE_BASE = 2.6        # 元/m3


def _ar1(n, rho, sigma, rng, mean=0.0):
    e = rng.normal(0, sigma, n)
    x = np.empty(n)
    x[0] = e[0]
    for t in range(1, n):
        x[t] = rho * x[t - 1] + e[t]
    return mean + x


def marginal_costs(m: dict) -> dict:
    """各机组小时边际成本（元/MWh）。"""
    mc = {}
    for u in UNITS:
        if u.tech == "coal":
            fuel = m["coal_price"] * u.heat_rate / 1000.0          # 元/MWh
        elif u.tech == "gas":
            fuel = m["gas_price"] * u.heat_rate * 1000.0           # 元/MWh
        else:  # hydro：机会成本，来水越枯越高
            fuel = 25.0 + (1.0 - m["hydro_inflow"]) * 55.0
        mc[u.uid] = fuel + u.vom
    return mc


def build_market(cfg=DEFAULT):
    rng = np.random.default_rng(cfg.seed)
    idx = pd.date_range(cfg.start, periods=cfg.periods, freq="h")
    n = len(idx)
    hour = idx.hour.to_numpy()
    doy = idx.day_of_year.to_numpy()
    dow = idx.day_of_week.to_numpy()
    year = idx.year.to_numpy()

    # ---------- 负荷 ----------
    annual = 1150 * np.sin(2 * np.pi * (doy - 200) / 365.25) + 1150      # 夏高冬次高
    winter = 520 * np.exp(-0.5 * ((doy - 20) / 25) ** 2)                  # 采暖次峰
    daily = 780 * np.exp(-0.5 * ((hour - 10) / 3.2) ** 2) \
        + 850 * np.exp(-0.5 * ((hour - 19) / 3.0) ** 2)                   # 双峰
    weekend = np.where(dow >= 5, -300, 0)
    weather = _ar1(n, 0.90, 120, rng)
    load = 2600 + annual + winter + daily + weekend + weather
    load = np.clip(load, 1900, None)
    for _ in range(8):                                                  # 极端寒潮/高温
        s = int(rng.integers(0, n - 48))
        load[s:s + int(rng.integers(18, 60))] += rng.uniform(500, 950)

    # ---------- 新能源 ----------
    solar_season = 0.55 + 0.45 * np.sin(2 * np.pi * (doy - 80) / 365.25)
    solar_shape = np.exp(-0.5 * ((hour - 12.5) / 3.0) ** 2)
    solar = 850 * solar_season * solar_shape * (0.7 + 0.3 * rng.random(n)) * rng.uniform(0.85, 1.0, n)
    wind_season = 0.8 + 0.35 * np.cos(2 * np.pi * (doy - 30) / 365.25)
    wind = np.clip(650 * wind_season + _ar1(n, 0.85, 150, rng), 0, None)
    renewable = solar + wind
    net_load = np.clip(load - renewable, 900, None)

    # ---------- 燃料与来水 ----------
    coal_price = COAL_PRICE_BASE + 110 * np.sin(2 * np.pi * (doy - 330) / 365.25) \
        + _ar1(n, 0.98, 16, rng) + 35 * (year - year.min())
    gas_price = GAS_PRICE_BASE + 0.65 * np.sin(2 * np.pi * (doy - 350) / 365.25) \
        + _ar1(n, 0.97, 0.11, rng) + 0.22 * (year - year.min())
    hydro_inflow = np.clip(
        0.6 + 0.4 * np.sin(2 * np.pi * (doy - 200) / 365.25) + 0.1 * rng.normal(size=n),
        0.15, 1.0,
    )

    total_cap = sum(u.cap for u in UNITS)
    rank_net = pd.Series(net_load).rank(pct=True).to_numpy()

    # ---------- 真值：串通时段（集中在高负荷季，制造混淆）----------
    hot = np.where(rank_net > 0.78)[0]
    episodes = []
    used = np.zeros(n, dtype=bool)
    attempt = 0
    while len(episodes) < 28 and attempt < 8000:
        attempt += 1
        s = int(rng.choice(hot))
        e = min(s + int(rng.integers(4, 36)), n)
        if used[max(0, s - 24):e + 24].any():
            continue
        used[s:e] = True
        episodes.append((s, e))
    collude_mask = np.zeros(n, dtype=bool)
    for s, e in episodes:
        collude_mask[s:e] = True
    leader_per_hour = np.full(n, -1, dtype=int)
    for k, (s, e) in enumerate(episodes):
        leader_per_hour[s:e] = k % len(CARTEL)

    stakes = pd.Series(net_load).rank(pct=True).to_numpy()

    return dict(
        idx=idx, n=n, hour=hour, doy=doy, dow=dow, year=year,
        load=load, renewable=renewable, net_load=net_load,
        coal_price=coal_price, gas_price=gas_price,
        hydro_inflow=hydro_inflow, total_cap=total_cap, rng=rng,
        collude_mask=collude_mask, leader_per_hour=leader_per_hour,
        episodes=episodes, stakes=stakes,
    )


def _clearing_with_deltas(avail, bid, net_load, uids, cfg):
    """对每个机组计算“其缺席时出清价的增量 Δp”，用于持留的经济动机。

    返回 (price, {uid: dp_array})。逐小时出清，代价可接受（n_units 很小）。
    """
    n = len(net_load)
    A = np.column_stack([avail[uid] for uid in uids])   # (n, nu)
    B = np.column_stack([bid[uid] for uid in uids])

    price = np.zeros(n)
    dp = {uid: np.zeros(n) for uid in uids}

    for t in range(n):
        a = A[t].copy()
        b = B[t].copy()
        live = a > 1e-6
        if not live.any():
            price[t] = cfg.price_cap
            continue
        need = net_load[t]
        order = np.argsort(np.where(live, b, np.inf))
        cum = 0.0
        marginal = -1
        for j in order:
            if not live[j] or cum >= need - 1e-9:
                break
            cum += min(a[j], need - cum)
            marginal = j
        if marginal < 0 or cum < need - 1e-6:
            base_p = cfg.price_cap
        else:
            base_p = b[marginal]
        price[t] = min(cfg.price_cap, base_p)

        # 每台机组缺席后的价格
        for k, uid in enumerate(uids):
            if not live[k]:
                dp[uid][t] = 0.0
                continue
            a2 = a.copy(); a2[k] = 0.0
            live2 = a2 > 1e-6
            if not live2.any():
                dp[uid][t] = cfg.price_cap - base_p
                continue
            o2 = np.argsort(np.where(live2, b, np.inf))
            cum2 = 0.0; m2 = -1
            for j in o2:
                if not live2[j] or cum2 >= need - 1e-9:
                    break
                cum2 += min(a2[j], need - cum2)
                m2 = j
            p2 = cfg.price_cap if (m2 < 0 or cum2 < need - 1e-6) else b[m2]
            dp[uid][t] = max(0.0, min(cfg.price_cap, p2) - min(cfg.price_cap, base_p))
    return price, dp


def simulate(cfg=DEFAULT, verbose=True):
    m = build_market(cfg)
    rng = m["rng"]
    n = m["n"]
    idx = m["idx"]
    net_load = m["net_load"]
    total_cap = m["total_cap"]
    mc = marginal_costs(m)

    # 紧张度：净负荷相对可调度容量的压力（0~1+）
    tightness = np.clip((net_load - 0.55 * total_cap) / (0.45 * total_cap), 0, 1.2)

    # ---------- 机组可用性 ----------
    avail = {u.uid: np.full(n, u.cap) for u in UNITS}
    outage_declared = {u.uid: np.zeros(n, dtype=bool) for u in UNITS}
    outage_type = {u.uid: np.array([""] * n, dtype=object) for u in UNITS}
    outage_reason = {u.uid: np.array([""] * n, dtype=object) for u in UNITS}

    def declare(uid, s, e, typ, reason):
        avail[uid][s:e] = 0.0
        outage_declared[uid][s:e] = True
        outage_type[uid][s:e] = typ
        outage_reason[uid][s:e] = reason

    # 1) 合法计划检修：安排在春/秋低负荷季，成段 3~7 天
    for uid in PLANNED_MAINT_UNITS:
        for lo, hi in [(80, 130), (290, 335)]:
            cand = np.where((m["doy"] >= lo) & (m["doy"] <= hi))[0]
            s = int(rng.choice(cand))
            declare(uid, s, min(s + int(rng.integers(72, 168)), n),
                    "planned", "计划检修（年度检修计划已报备）")

    # 2) 强制停运（外生）：按技术的常数风险；高负荷时略增（真实相关性）
    forced_rate = {"coal": 0.006, "gas": 0.005, "hydro": 0.003}
    for u in UNITS:
        rate = forced_rate[u.tech] * (1.0 + 0.8 * tightness)
        t = 0
        while t < n:
            if (not outage_declared[u.uid][t]) and rng.random() < rate[t]:
                e = min(t + int(rng.integers(1, 6)), n)
                declare(u.uid, t, e, "forced", "设备故障（非计划）")
                t = e
            else:
                t += 1

    # ---------- 报价 ----------
    bid = {}
    for u in UNITS:
        noise = _ar1(n, 0.55, 4.5, rng)
        b = mc[u.uid] + u.markup_b + u.markup_a * 220 * tightness + noise
        bid[u.uid] = b

    # 串通：成员共同抬价，领导者额外加码（轮换），紧张时幅度更大
    cartel_pos = {uid: i for i, uid in enumerate(CARTEL)}
    for uid in CARTEL:
        add = np.zeros(n)
        seg = m["collude_mask"]
        base_lift = rng.normal(105, 15)
        lead_extra = np.zeros(n)
        for k, (s, e) in enumerate(m["episodes"]):
            if m["leader_per_hour"][s] == cartel_pos[uid]:
                lead_extra[s:e] = rng.normal(70, 10)
        add[seg] = base_lift * (0.8 + 0.5 * tightness[seg]) + lead_extra[seg]
        bid[uid] = bid[uid] + add

    for u in UNITS:
        bid[u.uid] = np.clip(bid[u.uid], mc[u.uid] * 0.98, cfg.price_cap)

    # 3) 策略性持留（真值）：由“市场力”驱动，而非单纯高负荷。
    #    经济逻辑：机组的缺席若能把出清价抬高，且同集团还有别的机组在发电，
    #    则停运可让集团整体获利（放弃自身少量电量，换取兄弟机组高价售电）。
    #    因此停运概率随“反事实价格抬升 Δp”上升——这正是真实故障不具备的选择性。
    #    先做一次“策略前”出清，计算各机组缺席的价格影响。
    pre_price, pre_dp = _clearing_with_deltas(avail, bid, net_load, [u.uid for u in UNITS], cfg)
    owner_units = {}
    for u in UNITS:
        owner_units.setdefault(u.owner, []).append(u.uid)

    for uid in WITHHOLDERS:
        own = UNITS[[u.uid for u in UNITS].index(uid)].owner
        siblings = [s for s in owner_units[own] if s != uid]
        dp = pre_dp[uid]
        # 持留有固定成本（伪造检修理由、承担检查风险），只有 Δp 足够大才值得。
        # Δp<15 时不持留；超过后概率随 Δp 平滑上升。
        p_strat = np.where(dp >= 15.0, 0.85 / (1.0 + np.exp(-(dp - 30.0) / 8.0)), 0.0)
        cand = np.where((~outage_declared[uid]) & (rng.random(n) < p_strat))[0]
        rng.shuffle(cand)
        taken = 0
        for t in cand:
            if taken >= 70:
                break
            if outage_declared[uid][max(0, t - 8):t + 1].any():
                continue
            # 兄弟机组需可发电才能捕获租金
            if not any(avail[s][t] > 1e-6 for s in siblings):
                continue
            e = min(t + int(rng.integers(2, 7)), n)
            declare(uid, t, e, "strategic", "机组缺陷检修")
            taken += 1


    # ---------- 逐小时出清 ----------
    price = np.zeros(n)
    rsi = np.zeros(n)
    dispatched = {u.uid: np.zeros(n) for u in UNITS}
    is_marginal = {u.uid: np.zeros(n, dtype=bool) for u in UNITS}
    order = [u.uid for u in UNITS]

    for t in range(n):
        avail_t = np.array([avail[uid][t] for uid in order])
        bid_t = np.array([bid[uid][t] for uid in order])
        live = avail_t > 1e-6
        if not live.any():
            price[t] = cfg.price_cap
            continue
        need = net_load[t]
        total_live = avail_t[live].sum()
        rsi[t] = (total_live - need) / max(need, 1.0)

        o = np.argsort(np.where(live, bid_t, np.inf))
        cum = 0.0
        marginal = -1
        cleared = np.zeros(len(order))
        for j in o:
            if cum >= need - 1e-9:
                break
            take = min(avail_t[j], need - cum)
            cleared[j] = take
            cum += take
            marginal = j

        if marginal < 0 or cum < need - 1e-6:
            price[t] = cfg.price_cap                     # 供不应求，价格顶格
        else:
            p = bid_t[marginal]
            # 残余供给极低 → 稀缺加价（真实市场的凸性，非串通）
            reserve_ratio = (total_live - need) / max(total_live, 1.0)
            if reserve_ratio < 0.04:
                p = p * (1.0 + 0.6 * (0.04 - reserve_ratio) / 0.04)
            price[t] = min(cfg.price_cap, p)

        for j in range(len(order)):
            dispatched[order[j]][t] = cleared[j]
        if marginal >= 0:
            is_marginal[order[marginal]][t] = True

    price = np.clip(price, 0, cfg.price_cap)

    # ---------- 组装 ----------
    market = pd.DataFrame({
        "timestamp": idx,
        "load_mw": m["load"],
        "renewable_mw": m["renewable"],
        "net_load_mw": net_load,
        "coal_price": m["coal_price"],
        "gas_price": m["gas_price"],
        "hydro_inflow": m["hydro_inflow"],
        "tightness": tightness,
        "rsi": rsi,
        "price": price,
    })

    offers_rows, outages_rows = [], []
    for u in UNITS:
        offers_rows.append(pd.DataFrame({
            "timestamp": idx, "unit_id": u.uid, "tech": u.tech, "owner": u.owner,
            "capacity_mw": u.cap, "available_mw": avail[u.uid],
            "bid_price": bid[u.uid], "dispatched_mw": dispatched[u.uid],
            "is_marginal": is_marginal[u.uid].astype(int),
        }))
        outages_rows.append(pd.DataFrame({
            "timestamp": idx, "unit_id": u.uid,
            "declared_outage": outage_declared[u.uid].astype(int),
            "outage_type": outage_type[u.uid], "outage_reason": outage_reason[u.uid],
        }))

    offers = pd.concat(offers_rows, ignore_index=True)
    outages = pd.concat(outages_rows, ignore_index=True)
    units = pd.DataFrame([{
        "unit_id": u.uid, "tech": u.tech, "capacity_mw": u.cap,
        "heat_rate": u.heat_rate, "owner": u.owner,
    } for u in UNITS])

    # ---------- 落盘 ----------
    DATA_DIR.mkdir(exist_ok=True)
    market.to_csv(DATA_DIR / "market_hourly.csv", index=False)
    offers.to_csv(DATA_DIR / "offers.csv", index=False)
    outages.to_csv(DATA_DIR / "outages.csv", index=False)
    units.to_csv(DATA_DIR / "units.csv", index=False)

    truth = {
        "cartel": CARTEL,
        "withholders": WITHHOLDERS,
        "planned_maintenance_units": PLANNED_MAINT_UNITS,
        "collusion_episodes": [
            {"start": str(idx[s]), "end": str(idx[min(e, n) - 1])} for s, e in m["episodes"]
        ],
        "notes": "真值仅供评估，分析模块不得读取本文件。",
    }
    (DATA_DIR / "truth.json").write_text(
        json.dumps(truth, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if verbose:
        print(f"[simulate] 小时数={n}  机组数={len(UNITS)}  报价行={len(offers):,}")
        print(f"[simulate] 出清价 均值={price.mean():.1f} P50={np.percentile(price,50):.1f} "
              f"P95={np.percentile(price,95):.1f} 顶格占比={(price >= cfg.price_cap-1).mean():.3%}")
        print(f"[simulate] 串通时段={len(m['episodes'])} 段，共 {int(m['collude_mask'].sum())} 小时")
        print(f"[simulate] 策略性持留小时数={int((outages.outage_type=='strategic').sum())}")
        print(f"[simulate] 输出目录: {DATA_DIR}")

    return market, offers, outages, units


if __name__ == "__main__":
    simulate()
