"""数据装载与反事实再出清引擎。

* load_panel(): 把 market/offers/outages/units 合成 unit-hour 面板，并计算
  监管方“可观测”的边际成本 mc_obs（用报备热耗 × 公开燃料价格，不含任何真值）。
* clear_hour(): 给定可用容量与报价，按 merit order 出清单小时，返回价格与各机组中标量。
  用于反事实分析（移除异常加价/恢复停运后的价格）。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATA_DIR


def _mc_observable(tech, heat_rate, coal_price, gas_price, hydro_inflow):
    """监管方可计算的边际成本（元/MWh），不含加价。"""
    tech = np.asarray(tech)
    mc = np.zeros_like(np.asarray(coal_price, dtype=float))
    coal = tech == "coal"
    gas = tech == "gas"
    hyd = tech == "hydro"
    mc[coal] = coal_price[coal] * heat_rate[coal] / 1000.0
    mc[gas] = gas_price[gas] * heat_rate[gas] * 1000.0
    # 水电机会成本：来水越枯越高（可用水库水位/来水指数估计）
    mc[hyd] = 25.0 + (1.0 - hydro_inflow[hyd]) * 55.0
    return mc


def load_panel(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    market = pd.read_csv(data_dir / "market_hourly.csv", parse_dates=["timestamp"])
    offers = pd.read_csv(data_dir / "offers.csv", parse_dates=["timestamp"])
    outages = pd.read_csv(data_dir / "outages.csv", parse_dates=["timestamp"])
    units = pd.read_csv(data_dir / "units.csv")

    df = offers.merge(outages, on=["timestamp", "unit_id"], how="left")
    df = df.merge(market, on="timestamp", how="left")
    df = df.merge(units[["unit_id", "heat_rate"]], on="unit_id", how="left")

    df["hour"] = df["timestamp"].dt.hour
    df["month"] = df["timestamp"].dt.month
    df["dow"] = df["timestamp"].dt.day_of_week

    # mc_obs 需要按行取对应燃料价格
    df["mc_obs"] = np.nan
    coal = df["tech"] == "coal"
    gas = df["tech"] == "gas"
    hyd = df["tech"] == "hydro"
    df.loc[coal, "mc_obs"] = df.loc[coal, "coal_price"] * df.loc[coal, "heat_rate"] / 1000.0
    df.loc[gas, "mc_obs"] = df.loc[gas, "gas_price"] * df.loc[gas, "heat_rate"] * 1000.0
    df.loc[hyd, "mc_obs"] = 25.0 + (1.0 - df.loc[hyd, "hydro_inflow"]) * 55.0

    # 系统层面的价格/紧张度分位（用于“高利害时段”定义）
    hr = market.set_index("timestamp")
    price_rank = hr["price"].rank(pct=True)
    tight_rank = hr["tightness"].rank(pct=True)
    df["price_rank"] = df["timestamp"].map(price_rank)
    df["tight_rank"] = df["timestamp"].map(tight_rank)
    return df.sort_values(["timestamp", "unit_id"]).reset_index(drop=True)


def pivot_matrices(df: pd.DataFrame):
    """把面板转成 (unit x hour) 矩阵，供出清与反事实使用。"""
    units = df["unit_id"].drop_duplicates().tolist()
    times = df["timestamp"].drop_duplicates().sort_values().tolist()
    ui = {u: i for i, u in enumerate(units)}
    ti = {t: i for i, t in enumerate(times)}

    shape = (len(units), len(times))
    avail = np.zeros(shape)
    bid = np.zeros(shape)
    cap = np.zeros(shape)
    row = df["unit_id"].map(ui).to_numpy()
    col = df["timestamp"].map(ti).to_numpy()
    avail[row, col] = df["available_mw"].to_numpy()
    bid[row, col] = df["bid_price"].to_numpy()
    cap[row, col] = df["capacity_mw"].to_numpy()

    need = df.drop_duplicates("timestamp").sort_values("timestamp")["net_load_mw"].to_numpy()
    return dict(units=units, times=times, avail=avail, bid=bid, cap=cap, need=need,
                ui=ui, ti=ti)


def clear_hour(avail: np.ndarray, bid: np.ndarray, need: float, price_cap: float = 1500.0):
    """单小时 merit-order 出清。avail/bid 为长度 n_units 的数组。

    返回 (price, cleared[n_units])。规则与仿真器一致：按报价升序，累加至满足净负荷；
    价格取边际机组报价；供不应求则取价格上限。
    """
    n = len(avail)
    live = avail > 1e-6
    cleared = np.zeros(n)
    if not live.any():
        return price_cap, cleared
    order = np.argsort(np.where(live, bid, np.inf))
    cum = 0.0
    marginal = -1
    for j in order:
        if not live[j]:
            break
        if cum >= need - 1e-9:
            break
        take = min(avail[j], need - cum)
        cleared[j] = take
        cum += take
        marginal = j
    if marginal < 0 or cum < need - 1e-6:
        return price_cap, cleared
    return float(bid[marginal]), cleared
