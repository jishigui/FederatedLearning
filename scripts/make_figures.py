"""从 ``results/*.json`` 重新生成结果图 (不需要重新训练)。

用法::

    python scripts/make_figures.py
    python scripts/make_figures.py --results results
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fglguard.utils import ensure_dir, plot_bars, plot_curves


def _load(path: str):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------- #
def figures_from_federation(out_dir: str) -> List[str]:
    made = []
    for path in sorted(glob.glob(os.path.join(out_dir, "federation_detection*.json"))):
        data = _load(path)
        if not data:
            continue
        made += _federation_one(data, out_dir, _tag_suffix(path))
    return made


def _tag_suffix(path: str) -> str:
    """``federation_detection_cross_domain.json`` -> ``_cross_domain``。"""
    stem = os.path.splitext(os.path.basename(path))[0]
    return stem[len("federation_detection"):]


def _federation_one(data, out_dir: str, suffix: str) -> List[str]:
    made = []
    results = data["results"]

    arm_labels = [("off_the_shelf", "off-the-shelf"),
                  ("unsupervised", "unsupervised"),
                  ("local_only", "local-only"),
                  ("fedavg", "FedAvg"),
                  ("fglguard", "FGLGuard"),
                  ("centralized", "centralized")]

    # 训练曲线 (优先用 JSON 里持久化的曲线)
    series: Dict[str, List[float]] = {}
    for key, label in (("fedavg", "FedAvg"), ("fglguard", "FGLGuard")):
        curves = (data.get("curves") or {}).get(key)
        if not curves:
            hist = [r[f"{key}_history"]["val_auroc"] for r in results
                    if f"{key}_history" in r]
            curves = np.array(hist, dtype=float).mean(axis=0).tolist() if hist else None
        if curves:
            series[label] = list(curves)
    if series:
        p = os.path.join(out_dir, f"fig_training_curve{suffix}.png")
        plot_curves(series, "Federated training: macro validation AUROC",
                    "communication round", "AUROC", p)
        made.append(p)

    # 检测柱状图
    groups, values = [], []
    for key, label in arm_labels:
        vals = [[r[key]] if not isinstance(r[key], list) else r[key] for r in results]
        groups.append(label)
        values.append(float(np.nanmean(np.array(vals, dtype=float))))
    p = os.path.join(out_dir, f"fig_detection_bars{suffix}.png")
    plot_bars(groups, {"macro AUROC": values},
              "Detection quality (macro AUROC over seeds)", "AUROC", p)
    made.append(p)
    return made


def figures_from_runtime(out_dir: str) -> List[str]:
    made = []
    for path in sorted(glob.glob(os.path.join(out_dir, "runtime_deployment*.json"))):
        data = _load(path)
        if data:
            made += _runtime_one(data, out_dir,
                                 os.path.splitext(os.path.basename(path))[0]
                                 [len("runtime_deployment"):])
    return made


def _runtime_one(data, out_dir: str, suffix: str) -> List[str]:
    agg = data["aggregate"]
    order = ["fglguard", "fedavg", "centralized", "local_only", "off_the_shelf",
             "fglguard_no_corroborate", "fglguard_no_rewrite",
             "fglguard_guard_tools", "unguarded"]
    order = [a for a in order if a in agg]
    labels = {
        "fglguard": "FGLGuard", "fedavg": "FedAvg",
        "centralized": "centralized", "local_only": "local-only",
        "off_the_shelf": "off-the-shelf",
        "fglguard_no_corroborate": "abl: no corroborate",
        "fglguard_no_rewrite": "abl: no rewrite",
        "fglguard_guard_tools": "abl: guard tools",
        "unguarded": "unguarded",
    }
    groups = [labels.get(a, a) for a in order]
    made = []

    def mean(key: str) -> List[float]:
        out = []
        for a in order:
            vals = [v for v in agg[a][key] if v == v]  # 过滤 nan
            out.append(float(np.mean(vals)) if vals else 0.0)
        return out

    if all("asr" in agg[a] for a in order):
        p = os.path.join(out_dir, f"fig_runtime_asr{suffix}.png")
        plot_bars(groups, {"ASR (lower better)": mean("asr"),
                           "over-refusal": mean("over_refusal")},
                  "Runtime: attack success vs over-refusal", "rate", p)
        made.append(p)

    p = os.path.join(out_dir, f"fig_runtime_utility{suffix}.png")
    plot_bars(groups, {"utility (higher better)": mean("utility")},
              "Runtime: clean utility", "rate", p)
    made.append(p)
    return made


def figures_from_demo(out_dir: str) -> List[str]:
    data = _load(os.path.join(out_dir, "main_demo.json"))
    if not data:
        return []
    made = []
    det = data.get("detection", {})
    if det:
        groups = list(det.keys())
        vals = [float(str(v)) for v in det.values()]
        p = os.path.join(out_dir, "fig_main_detection.png")
        plot_bars(groups, {"macro AUROC": vals},
                  "Detection quality (macro AUROC)", "AUROC", p)
        made.append(p)
    rt = data.get("runtime", {})
    if rt:
        groups = list(rt.keys())
        p = os.path.join(out_dir, "fig_main_runtime.png")
        plot_bars(groups, {"ASR (lower better)": [rt[k]["asr"] for k in groups],
                           "over-refusal": [rt[k]["over_refusal"] for k in groups]},
                  "Runtime deployment", "rate", p)
        made.append(p)
    return made


def main() -> int:
    ap = argparse.ArgumentParser(description="从 JSON 结果重绘图表")
    ap.add_argument("--results", default="results")
    args = ap.parse_args()
    ensure_dir(args.results)

    made: List[str] = []
    made += figures_from_federation(args.results)
    made += figures_from_runtime(args.results)
    made += figures_from_demo(args.results)

    if not made:
        print(f"{args.results}/ 下没有可用的结果 JSON，先跑实验。")
        return 1
    for p in made:
        print(f"已生成 {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
