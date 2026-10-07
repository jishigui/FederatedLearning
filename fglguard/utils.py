"""通用工具: 随机种子、设备选择、结果落盘、绘图。

阅读顺序: 第 11 步 / 共 18 步 —— 见 READING_ORDER.md
"""

from __future__ import annotations

import json
import os
import random
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np


def set_seed(seed: int) -> None:
    """统一设置 numpy / random / torch 的随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(False)
    except ImportError:  # pragma: no cover - torch 一定存在，这里只是防御
        pass


def get_device(name: str = "cpu"):
    """解析设备名；``"auto"`` 表示有 CUDA 就用 CUDA，否则回退 CPU。"""
    import torch

    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def ensure_dir(path: str) -> str:
    """递归创建目录 (已存在则忽略)，返回原路径以便链式调用。"""
    os.makedirs(path, exist_ok=True)
    return path


def save_json(obj: Any, path: str) -> None:
    """把结果写成 UTF-8 JSON (自动建父目录，保留中文不转义)。"""
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False)


def mean_std(values: Sequence[float]) -> Dict[str, float]:
    """返回 ``{"mean": ..., "std": ...}`` (总体标准差，ddof=0)。"""
    arr = np.asarray(list(values), dtype=float)
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=0))}


def format_table(header: Sequence[str], rows: Iterable[Sequence[Any]],
                 floatfmt: str = ".4f") -> str:
    """把二维结果渲染成等宽文本表格。"""
    rows = [[_fmt(c, floatfmt) for c in row] for row in rows]
    widths = [len(h) for h in header]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    sep = "  "
    lines = [sep.join(h.ljust(widths[i]) for i, h in enumerate(header))]
    lines.append(sep.join("-" * w for w in widths))
    for row in rows:
        lines.append(sep.join(c.ljust(widths[i]) for i, c in enumerate(row)))
    return "\n".join(lines)


def _fmt(cell: Any, floatfmt: str) -> str:
    if isinstance(cell, float):
        return format(cell, floatfmt)
    return str(cell)


# --------------------------------------------------------------------------- #
# 绘图
# --------------------------------------------------------------------------- #
def plot_curves(series: Dict[str, List[float]], title: str, xlabel: str,
                ylabel: str, save_path: str, x: Sequence[float] | None = None) -> None:
    """绘制多条曲线 (自动跳过 matplotlib 不可用的情况)。"""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover
        return

    fig, ax = plt.subplots(figsize=(6.0, 4.0), dpi=140)
    for label, ys in series.items():
        xs = list(x) if x is not None else list(range(1, len(ys) + 1))
        ax.plot(xs, ys, marker="o", markersize=3, linewidth=1.6, label=label)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3, linestyle="--", linewidth=0.5)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    ensure_dir(os.path.dirname(os.path.abspath(save_path)))
    fig.savefig(save_path)
    plt.close(fig)


def plot_bars(groups: Sequence[str], series: Dict[str, Sequence[float]],
              title: str, ylabel: str, save_path: str,
              rotate: float = 20.0, figsize: tuple = (7.4, 4.2)) -> None:
    """绘制分组柱状图 (自动旋转 x 轴标签避免重叠)。"""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover
        return

    import numpy as _np

    n_series = len(series)
    x = _np.arange(len(groups), dtype=float)
    width = 0.8 / max(n_series, 1)
    fig, ax = plt.subplots(figsize=figsize, dpi=140)
    for i, (label, vals) in enumerate(series.items()):
        ax.bar(x + i * width - 0.4 + width / 2, list(vals), width=width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=8.5, rotation=rotate, ha="right")
    ax.set_title(title, fontsize=11)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.5)
    if n_series > 1:
        ax.legend(fontsize=8, frameon=False)
    # 给柱顶留白，避免标注被裁掉
    top = max((max(v) for v in series.values() if len(v)), default=1.0)
    ax.set_ylim(0, top * 1.15 if top > 0 else 1.0)
    fig.tight_layout()
    ensure_dir(os.path.dirname(os.path.abspath(save_path)))
    fig.savefig(save_path)
    plt.close(fig)
