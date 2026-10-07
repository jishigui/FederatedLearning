"""评估指标。

阅读顺序: 第 10 步 / 共 18 步 —— 见 READING_ORDER.md

对应论文 Metrics 段落: Episode AUROC (organic 语料在 final-answer agent 上读取;
planted attacker 语料逐节点读取)、ASR、utility、capability。
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np


def auroc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """计算 AUROC (rank 统计量，自动处理并列值)。

    等价于 Mann-Whitney U 统计量:

    .. math:: AUROC = \\frac{1}{P N}\\sum_{i: y_i=1}\\sum_{j: y_j=0} 1[s_i > s_j]

    当某一类为空时返回 ``float('nan')`` (论文中 unguarded 的 detection 定义为 undefined)。
    """
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    pos = s[y == 1]
    neg = s[y == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")

    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(s.size, dtype=float)
    sorted_s = s[order]
    i = 0
    while i < s.size:
        j = i
        while j + 1 < s.size and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        avg_rank = 0.5 * (i + j) + 1.0  # 1-based 平均秩
        ranks[order[i:j + 1]] = avg_rank
        i = j + 1

    rank_sum_pos = ranks[y == 1].sum()
    n_pos, n_neg = pos.size, neg.size
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def confusion_at(scores: Sequence[float], labels: Sequence[int],
                 tau: float) -> Dict[str, float]:
    """给定阈值下的混淆统计。"""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    pred = s >= tau
    tp = float(np.sum(pred & (y == 1)))
    fp = float(np.sum(pred & (y == 0)))
    tn = float(np.sum(~pred & (y == 0)))
    fn = float(np.sum(~pred & (y == 1)))
    n_pos = tp + fn
    n_neg = fp + tn
    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "recall": tp / n_pos if n_pos > 0 else float("nan"),
        "fpr": fp / n_neg if n_neg > 0 else float("nan"),
        "precision": tp / (tp + fp) if (tp + fp) > 0 else float("nan"),
    }


def wilson_interval(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score interval (论文 Fig.4 用它对 ASR 做 95% CI)。"""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return float(centre - margin), float(centre + margin)


def expected_calibration_error(scores: Sequence[float], labels: Sequence[int],
                               n_bins: int = 10) -> float:
    """ECE，用于检查打分器是否过度自信 (可选诊断)。"""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=float)
    if s.size == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        mask = (s >= lo) & (s < hi if b < n_bins - 1 else s <= hi)
        if not np.any(mask):
            continue
        conf = s[mask].mean()
        acc = y[mask].mean()
        ece += (mask.sum() / s.size) * abs(conf - acc)
    return float(ece)
