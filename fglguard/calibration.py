"""过拒答预算下的运行点校准 (论文 Sec. 4.3, Eq. 7)。

::

    tau* = argmax_{tau in [0,1]} Recall(tau; V)   s.t.  FPR(tau; V) <= rho

其中 :math:`V = \\{(\\hat s_m, y_m)\\}` 是**各客户端在本地上计算后上传的标量分数池**
——原始图始终留在 silo 内，服务端只看到 (分数, 标签) 对，这与论文的隐私约束一致。

论文的两个理论性质 (supplementary Proposition 1-2) 在本实现中都可被验证:

1. 只有良性流量时，recall 项为空，则 Eq. (7) 退化为良性分数的经验
   :math:`(1-\\rho)` 分位数，因此在良性子集上预算**恒成立**；
2. 佐证分数 :math:`\\hat s` 会同时抬高检测率与误报率，在 :math:`\\hat s` 上重新校准
   即可吸收这一交换 (Proposition 3)。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import CalibConfig
from .metrics import confusion_at


@dataclass
class CalibrationResult:
    """校准结果与诊断信息。"""

    tau: float
    recall: float
    fpr: float
    n_val: int
    n_pos: int
    budget: float

    def to_dict(self) -> Dict[str, float]:
        """展开成可 JSON 序列化的字典 (用于把校准诊断写入结果文件)。"""
        return {
            "tau": self.tau, "recall": self.recall, "fpr": self.fpr,
            "n_val": self.n_val, "n_pos": self.n_pos, "budget": self.budget,
        }


def calibrate_threshold(scores: Sequence[float], labels: Sequence[int],
                        cfg: Optional[CalibConfig] = None,
                        rho: Optional[float] = None) -> CalibrationResult:
    """求解 Eq. (7)。

    实现为一维网格搜索 (候选点取唯一分数值)。并列时优先 **更低的 FPR**，
    再优先 **更高的阈值** (在相同 FPR 下给良性流量更多余量)。
    """
    cfg = cfg or CalibConfig()
    rho = cfg.over_refusal_budget if rho is None else rho

    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    if s.size == 0:
        return CalibrationResult(tau=1.0, recall=float("nan"), fpr=float("nan"),
                                 n_val=0, n_pos=0, budget=rho)

    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))

    # 候选阈值: 所有唯一分数 + 边界。加入 max+eps 保证一定能取到 FPR=0
    cands = np.unique(np.concatenate([s, [0.0, min(1.0, float(s.max()) + 1e-6)]]))

    best_tau = 1.0
    best_key: Optional[Tuple[float, float, float]] = None
    for tau in cands:
        pred = s >= tau
        tp = float(np.sum(pred & (y == 1)))
        fp = float(np.sum(pred & (y == 0)))
        recall = tp / n_pos if n_pos > 0 else 0.0
        fpr = fp / n_neg if n_neg > 0 else 0.0
        if fpr > rho + 1e-12:
            continue
        # 目标: max Recall, 且在这一最优召回下取**最紧的可行运行点**
        # (即尽量用满 rho 预算、阈值最小)。这等价于 "argmax Recall s.t. FPR <= rho"。
        #
        # 注意: 论文正文写 "breaking recall ties toward lower FPR"。但那与它的
        # Proposition 2 (只有良性流量时退化为 (1-rho) 分位数) 不相容 —— 若并列时
        # 一味偏向更低 FPR, 阈值会被推到 1.0 而不是分位数。因此这里采用与
        # Proposition 2 相容的规则; 需要论文正文行为时把 tie_break 设为 "lower_fpr"。
        if cfg.tie_break == "lower_fpr":
            key = (recall, -fpr, float(tau))
        else:
            key = (recall, fpr, -float(tau))
        if best_key is None or key > best_key:
            best_key, best_tau = key, float(tau)

    diag = confusion_at(s, y, best_tau)
    return CalibrationResult(
        tau=best_tau,
        recall=float(diag["recall"]) if n_pos > 0 else float("nan"),
        fpr=float(diag["fpr"]) if n_neg > 0 else float("nan"),
        n_val=int(s.size), n_pos=n_pos, budget=rho,
    )


def benign_quantile_threshold(scores: Sequence[float], labels: Sequence[int],
                              rho: float) -> float:
    """良性分数经验 :math:`(1-\\rho)` 分位数 —— 当验证集只有良性流量时 Eq. (7) 的闭式解。

    用于单元测试验证 Proposition 2。
    """
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    benign = s[y == 0]
    if benign.size == 0:
        return 1.0
    return float(np.quantile(benign, 1.0 - rho, method="higher"))


def pool_client_validation_scores(client_scores: Sequence[np.ndarray],
                                  client_labels: Sequence[np.ndarray]
                                  ) -> Tuple[np.ndarray, np.ndarray]:
    """把各客户端**本地算好**的标量分数池化 (不涉及任何原始数据交换)。"""
    if not client_scores:
        return np.zeros(0), np.zeros(0)
    return (np.concatenate(list(client_scores)),
            np.concatenate(list(client_labels)))
