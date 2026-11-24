"""用仿真真值评估识别效果（仅用于方法验证，不进入监管报告）。

给出：
* 串通：机组级 precision / recall / F1；边级（哪些机组对）同样评估；
  以及“时段命中”——识别出的共现小时中有多少落在真值串通时段内。
* 持留：机组级 precision / recall / F1；停运事件级（被标记事件中
  实际为策略性停运的比例，以及真值策略停运被标记的比例）。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def _prf(tp, fp, fn):
    prec = tp / (tp + fp) if (tp + fp) else np.nan
    rec = tp / (tp + fn) if (tp + fn) else np.nan
    f1 = 2 * prec * rec / (prec + rec) if (prec and rec) else np.nan
    return prec, rec, f1


def evaluate(data_dir: Path, collusion_groups, withholding_summary,
             collusion_pairs=None, hour_flags=None, withholding_episodes=None) -> dict:
    truth = json.loads((data_dir / "truth.json").read_text(encoding="utf-8"))
    truth_cartel = set(truth["cartel"])
    truth_with = set(truth["withholders"])

    # ---------- 串通：机组级 ----------
    pred_cartel = set(u for g in collusion_groups for u in g["members"])
    tp = len(pred_cartel & truth_cartel)
    fp = len(pred_cartel - truth_cartel)
    fn = len(truth_cartel - pred_cartel)
    prec, rec, f1 = _prf(tp, fp, fn)

    # ---------- 串通：边级 ----------
    edge_metrics = None
    if collusion_pairs is not None and len(collusion_pairs):
        sig = collusion_pairs[collusion_pairs["significant"]]
        pred_edges = {frozenset((r.unit_a, r.unit_b)) for r in sig.itertuples()}
        true_edges = {frozenset((a, b)) for a in truth_cartel for b in truth_cartel if a < b}
        etp = len(pred_edges & true_edges)
        efp = len(pred_edges - true_edges)
        efn = len(true_edges - pred_edges)
        ep, er, ef = _prf(etp, efp, efn)
        edge_metrics = {"precision": ep, "recall": er, "f1": ef,
                        "n_pred": len(pred_edges), "n_true": len(true_edges)}

    # ---------- 串通：时段命中 ----------
    hour_metrics = None
    if hour_flags is not None and len(hour_flags):
        ep = pd.DataFrame(truth["collusion_episodes"])
        mask = pd.Series(False, index=pd.to_datetime(hour_flags["timestamp"]).unique())
        flagged_hours = pd.to_datetime(hour_flags["timestamp"]).unique()
        # 真值串通小时集合
        idx = pd.date_range(mask.index.min(), mask.index.max(), freq="h")
        true_hours = pd.Series(False, index=idx)
        for _, r in ep.iterrows():
            true_hours.loc[pd.to_datetime(r["start"]):pd.to_datetime(r["end"])] = True
        fl = pd.Series(False, index=idx)
        fl.loc[flagged_hours] = True
        tp_h = int((fl & true_hours).sum())
        fp_h = int((fl & ~true_hours).sum())
        fn_h = int((~fl & true_hours).sum())
        hp, hr, hf = _prf(tp_h, fp_h, fn_h)
        hour_metrics = {"precision": hp, "recall": hr, "f1": hf,
                        "flagged_hours": int(fl.sum()), "true_hours": int(true_hours.sum())}

    # ---------- 持留：机组级 ----------
    pred_with = set(withholding_summary.loc[withholding_summary["significant"], "unit_id"]) \
        if "significant" in withholding_summary else set()
    wtp = len(pred_with & truth_with)
    wfp = len(pred_with - truth_with)
    wfn = len(truth_with - pred_with)
    wp, wr, wf1 = _prf(wtp, wfp, wfn)

    # ---------- 持留：事件级 ----------
    event_metrics = None
    if withholding_episodes is not None and len(withholding_episodes):
        flagged_keys = set(zip(withholding_episodes["unit_id"],
                               pd.to_datetime(withholding_episodes["timestamp"])))
        # 真值策略停运小时
        import csv
        strat = {}
        with open(data_dir / "outages.csv", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["outage_type"] == "strategic":
                    strat[(row["unit_id"], pd.to_datetime(row["timestamp"]))] = True
        true_keys = set(strat.keys())
        etp = len(flagged_keys & true_keys)
        efp = len(flagged_keys - true_keys)
        efn = len(true_keys - flagged_keys)
        ep, er, ef = _prf(etp, efp, efn)
        event_metrics = {"precision": ep, "recall": er, "f1": ef,
                         "flagged": len(flagged_keys), "true_events": len(true_keys)}

    return {
        "collusion_unit": {"precision": prec, "recall": rec, "f1": f1,
                           "tp": sorted(pred_cartel & truth_cartel),
                           "fp": sorted(pred_cartel - truth_cartel),
                           "fn": sorted(truth_cartel - pred_cartel)},
        "collusion_edge": edge_metrics,
        "collusion_hour": hour_metrics,
        "withholding_unit": {"precision": wp, "recall": wr, "f1": wf1,
                             "tp": sorted(pred_with & truth_with),
                             "fp": sorted(pred_with - truth_with),
                             "fn": sorted(truth_with - pred_with)},
        "withholding_event": event_metrics,
    }
