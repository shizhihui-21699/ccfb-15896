"""全局配置：市场参数、识别阈值、路径。

所有阈值集中在此，便于监管项目的可复现与审计（修改阈值必须留痕）。
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"


@dataclass
class StudyConfig:
    # ---- 仿真/样本 ----
    seed: int = 20240101
    start: str = "2023-01-01"
    periods: int = 8760 * 2          # 小时数（2 年）
    n_units: int = 16

    # ---- 市场与出清 ----
    price_cap: float = 1500.0        # 现货价格上限（元/MWh）
    reserve_target: float = 0.15     # 运行备用率目标（相对负荷）
    scarcity_slope: float = 0.9      # 竞争性报价对紧张度的反应系数（正常市场条件）

    # ---- 基本面残差模型 ----
    huber_epsilon: float = 1.35      # Huber 鲁棒回归阈值
    robust_z: float = 3.0            # 残差“抬价”判定：稳健 z 阈值

    # ---- 串通识别 ----
    coexist_min_edges: int = 2       # 同时性簇最少边数
    n_perm: int = 400                # 置换检验次数
    fdr_q: float = 0.10              # Benjamini-Hochberg 误发现率
    min_elevated_hours: int = 30     # 一个实体至少被抬价多少小时才纳入检验

    # ---- 持留识别 ----
    pivotal_rsi: float = 1.05        # 剩余供给指数低于该值视为 pivotal
    strategic_q: float = 0.10        # 持留 FDR 阈值
    min_excess_hours: float = 40.0   # 超额停运小时下限（实质效应量门槛）
    min_rent_loss: float = 1.0e7     # 租金损失下限（元），约 1000 万元
    high_stakes_pct: float = 0.75    # “高利害”时段定义（价格分位）
    baseline_pct: float = 0.50       # 估计强制停运基线所用的低利害分位

    # ---- 报告 ----
    top_n_cases: int = 12
    font: str = "PingFang SC"

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


DEFAULT = StudyConfig()
DATA_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)
