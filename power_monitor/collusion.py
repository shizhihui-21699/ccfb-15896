"""报价串通识别：残差空间里的“分层同时抬价”检验。

为什么不能直接看同步
--------------------
负荷、燃料、新能源对所有机组是同向冲击，任意两家报价的小时相关系数都可能很高，
“同步”因此几乎没有证据力。真正的证据是：

  在**同等市场条件**下，两家机组在同一小时**同时**出现无法由基本面解释的抬价。

方法
----
1. 用 baseline 的稳健残差 z 定义“抬价”：e_{u,t} = 1[ z_{u,t} > τ ]。
2. 按市场条件分层（紧张度十分位 × 季节），在每个层内分别统计两家机组的抬价率。
   分层的作用是把“共同市场条件导致的同涨”吸收掉——层内比较才公平。
3. 对每一对机组 (u,v)，计算观察到的同期抬价小时数 O_{uv}；
   在层内对 v 的抬价标签做随机置换（保持每层抬价率不变，只打乱“哪一小时”），
   得到零假设下 O 的分布（这正是“两只机组各自独立地随市场波动”的情形）。
4. 置换 p 值 + Benjamini–Hochberg 控制误发现率；同时报告“抬价倍数”
   lift = O / E[O|null]，作为效应量（监管人员看倍数比看 p 值直观）。
5. 通过检验的边构成“同时性网络”，再找连通分量/团 → 候选串通集团；
   并结合“时段内谁抬价最多”识别**轮值领导者**（轮流坐庄）。

输出中的一切都是**统计异常**，需结合协调性证据（通讯、共同行为模式、
结构性关联）形成监管结论，不能单独作为违法认定。
"""
from __future__ import annotations


import numpy as np
import pandas as pd

from .config import DEFAULT, StudyConfig


# ----------------------------------------------------------------------------
# 分层
# ----------------------------------------------------------------------------
def build_strata(panel_hours: pd.DataFrame, n_strata: int = 8) -> np.ndarray:
    """按小时构造分层标签：紧张度分位 × 季节。"""
    h = panel_hours.drop_duplicates("timestamp").sort_values("timestamp").copy()
    h["tight_bin"] = pd.qcut(h["tight_rank"].rank(method="first"), n_strata, labels=False)
    h["season"] = ((h["month"] % 12) // 3)  # 0..3
    key = h["tight_bin"].astype(int) * 4 + h["season"].astype(int)
    return key.to_numpy(), h["timestamp"].to_numpy()


# ----------------------------------------------------------------------------
# 置换检验（层内打乱）
# ----------------------------------------------------------------------------
def _co_elevation_perm(Eu: np.ndarray, Ev: np.ndarray, strata: np.ndarray,
                       strata_idx: dict, n_perm: int, rng: np.random.Generator):
    """观察到的同期抬价数，以及置换零分布。

    在每一层内独立打乱 Ev，保持层内总数不变。
    """
    obs = float(np.sum(Eu & Ev))
    # 预处理每层的 Eu 计数与 Ev 索引
    blocks = []
    for s, idx in strata_idx.items():
        if len(idx) == 0:
            continue
        eu = Eu[idx]
        ev = Ev[idx]
        m = len(idx)
        # 若某机组在该层全为 0 或全为 1，打乱无意义
        if ev.sum() == 0 or ev.sum() == m:
            const = float(eu.sum() * ev.sum() / m) if m else 0.0
            blocks.append(("const", const * 1.0, m, None))
            continue
        blocks.append(("perm", None, m, (idx, eu, ev)))

    null = np.empty(n_perm)
    for b in range(n_perm):
        tot = 0.0
        for kind, const, m, payload in blocks:
            if kind == "const":
                tot += const
                continue
            idx, eu, ev = payload
            perm = rng.permutation(m)
            tot += float(np.sum(eu * ev[perm]))
        null[b] = tot
    return obs, null


def detect_collusion(panel: pd.DataFrame, cfg: StudyConfig = DEFAULT, verbose: bool = True):
    """返回 (pair_table, groups, hour_flags)。"""
    rng = np.random.default_rng(cfg.seed + 7)

    hours = panel.drop_duplicates("timestamp")[["timestamp", "tight_rank", "month"]]
    strata, times = build_strata(hours)
    strata_idx = {s: np.where(strata == s)[0] for s in np.unique(strata)}

    # 抬价指示矩阵（机组 × 小时）
    piv = panel.pivot(index="unit_id", columns="timestamp", values="z")
    piv = piv.reindex(columns=pd.to_datetime(times))
    units = list(piv.index)
    Z = piv.to_numpy()
    E = (Z > cfg.robust_z)
    E = np.nan_to_num(E, nan=False).astype(bool)

    elev_hours = E.sum(1)
    cand = [u for u, k in zip(units, elev_hours) if k >= cfg.min_elevated_hours]

    if verbose:
        print(f"[collusion] 候选机组={len(cand)}/{len(units)} "
              f"(抬价小时 >= {cfg.min_elevated_hours})")
        for u, k in zip(units, elev_hours):
            mark = " ✓候选" if u in cand else ""
            print(f"    {u}: 抬价小时 {int(k):5d}{mark}")

    rows = []
    for i in range(len(cand)):
        for j in range(i + 1, len(cand)):
            u, v = cand[i], cand[j]
            Eu, Ev = E[units.index(u)], E[units.index(v)]
            obs, null = _co_elevation_perm(Eu, Ev, strata, strata_idx, cfg.n_perm, rng)
            # 经验 p 值（单侧：观察值过大）
            p = (1 + np.sum(null >= obs)) / (len(null) + 1)
            exp = float(null.mean())
            lift = obs / exp if exp > 1e-9 else np.inf
            rows.append({"unit_a": u, "unit_b": v, "obs_co": obs,
                         "exp_co": exp, "lift": lift, "p": p})

    pairs = pd.DataFrame(rows)
    if pairs.empty:
        return pairs, pd.DataFrame(), pd.DataFrame()

    # BH-FDR
    pairs = _bh_fdr(pairs, q=cfg.fdr_q)
    pairs["significant"] = pairs["sig_bh"] & (pairs["lift"] >= 1.5)
    pairs = pairs.sort_values(["significant", "lift", "p"], ascending=[False, False, True])

    # ---- 同时性网络 → 连通分量 ----
    groups = _components(pairs[pairs["significant"]])

    # ---- 轮值领导者 ----
    hour_flags = _leader_analysis(panel, groups, cand, units, cfg)

    if verbose:
        sig = pairs[pairs["significant"]]
        print(f"\n[collusion] 检验对数={len(pairs)}  显著边={len(sig)} "
              f"(BH q={cfg.fdr_q}, lift>=1.5)")
        if len(sig):
            print(sig[["unit_a", "unit_b", "obs_co", "exp_co", "lift", "p", "q_bh"]]
                  .head(20).round(4).to_string(index=False))
        print(f"\n[collusion] 识别到 {len(groups)} 个候选集团:")
        for g in groups:
            print(f"    成员={g['members']}  内部边={g['n_edges']}  "
                  f"平均lift={g['mean_lift']:.2f}  共现小时={g['co_hours']}")

    return pairs, groups, hour_flags


def _bh_fdr(pairs: pd.DataFrame, q: float) -> pd.DataFrame:
    p = pairs["p"].to_numpy()
    order = np.argsort(p)
    m = len(p)
    ranked = p[order]
    thresh = q * (np.arange(1, m + 1) / m)
    passed = ranked <= thresh
    kmax = np.max(np.where(passed)[0]) if passed.any() else -1
    sig = np.zeros(m, dtype=bool)
    if kmax >= 0:
        sig[order[:kmax + 1]] = True
    out = pairs.copy()
    out["q_bh"] = np.minimum.accumulate((ranked * m / np.arange(1, m + 1))[::-1])[::-1]
    # 把 q 值放回原顺序
    qv = np.empty(m)
    qv[order] = out["q_bh"].to_numpy()
    out["q_bh"] = qv
    out["sig_bh"] = False
    out.loc[out.index[order[:kmax + 1]], "sig_bh"] = True
    return out


def _components(sig_pairs: pd.DataFrame):
    if sig_pairs.empty:
        return []
    # union-find
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for _, r in sig_pairs.iterrows():
        union(r.unit_a, r.unit_b)

    comp = {}
    for u in list(parent.keys()):
        comp.setdefault(find(u), set()).add(u)

    groups = []
    for root, members in comp.items():
        if len(members) < 2:
            continue
        m = sorted(members)
        sub = sig_pairs[sig_pairs.unit_a.isin(m) & sig_pairs.unit_b.isin(m)]
        groups.append({
            "members": m,
            "n_edges": int(len(sub)),
            "mean_lift": float(sub["lift"].mean()) if len(sub) else np.nan,
            "max_lift": float(sub["lift"].max()) if len(sub) else np.nan,
        })
    groups.sort(key=lambda g: g["mean_lift"], reverse=True)
    return groups


def _leader_analysis(panel, groups, cand, units, cfg):
    """在每个同时抬价小时里，找出抬价幅度最大的成员（轮值领导者）。"""
    if not groups:
        return pd.DataFrame()
    ztab = panel.pivot(index="timestamp", columns="unit_id", values="z")
    rows = []
    for g in groups:
        mem = [u for u in g["members"] if u in ztab.columns]
        sub = ztab[mem]
        # 该集团“共同抬价”的小时：至少 2 名成员 z>τ
        co = (sub > cfg.robust_z).sum(axis=1) >= 2
        if co.sum() == 0:
            continue
        lead = sub[co].idxmax(axis=1)
        counts = lead.value_counts()
        n = counts.sum()
        # 领导权分散度（归一化熵）：越接近 1 越“轮流坐庄”
        p = counts / n
        entropy = float(-(p * np.log(p)).sum() / np.log(len(mem))) if len(mem) > 1 else 0.0
        for t, u in lead.items():
            rows.append({"timestamp": t, "group": "+".join(mem), "leader": u,
                         "max_z": float(sub.loc[t, u])})
        g["leader_counts"] = counts.to_dict()
        g["leader_rotation"] = entropy
        g["co_hours"] = int(co.sum())
    return pd.DataFrame(rows)


if __name__ == "__main__":
    from .data import load_panel
    from .baseline import fit_baseline
    p = load_panel()
    b = fit_baseline(p)
    pairs, groups, hf = detect_collusion(b.panel)
    pairs.to_csv("outputs/collusion_pairs.csv", index=False)
