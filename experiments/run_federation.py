"""实验一: 联邦检测 vs 集中式 / 本地 / off-the-shelf (论文 Table 1、Fig.3)。

运行::

    python -m experiments.run_federation                # 默认 3 个种子
    python -m experiments.run_federation --seeds 42     # 快速跑一个种子
    python -m experiments.run_federation --quick        # 缩小规模冒烟测试
    python -m experiments.run_federation --purity-sweep # 追加 non-IID 纯度扫描

对应论文要回答的三个问题:

- **Q1** 联邦 FGLGuard 能否在不汇聚任何原始 trace 的前提下追平/超过"域内集中式"上限，
  而 off-the-shelf 迁移与 local-only 都做不到？
- **Q2** 有免费 judge 标签时，监督式联邦是否优于无监督 / 免训练方案？(本项目用
  "无监督 deviation 分数" 这一免疫对照来体现，见 ``deviation_arm``)
- **Q3** 跨域联邦能否逼近多域池化？
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Sequence

import numpy as np

# 允许从项目根目录直接以 `python experiments/run_federation.py` 运行
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.common import World, macro, train_off_the_shelf
from fglguard.config import Config, default_config, small_config
from fglguard.federated import (train_centralized, train_federated,
                                train_local_only)
from fglguard.metrics import auroc
from fglguard.utils import ensure_dir, format_table, plot_bars, plot_curves, save_json


# --------------------------------------------------------------------------- #
# 单个种子的完整评测
# --------------------------------------------------------------------------- #
def run_single_seed(cfg: Config, seed: int, verbose: bool = False,
                    same_domain: bool = True) -> Dict[str, object]:
    """跑完一个随机种子下的全部对照臂，返回逐域 AUROC 与训练曲线。"""
    w = World(cfg, seed=seed, same_domain=same_domain)
    clients = w.clients
    out: Dict[str, object] = {"seed": seed, "domains": w.domains}

    # --- off-the-shelf (异域合成注入语料上训练，零适配) ---
    ots = train_off_the_shelf(cfg, w.sim, w.encoder, seed=seed)
    out["off_the_shelf"] = w.per_domain_auroc(ots.model)

    # --- local-only: 每个 operator 各自训练，完全不通信 ---
    locs = train_local_only(clients, cfg, w.node_dim, w.edge_dim, seed=seed)
    out["local_only"] = w.local_only_auroc(locs)

    # --- FedAvg: 无近端项 + 纯体量加权 (论文中 "size-only FedAvg") ---
    fedavg = train_federated(clients, cfg, w.node_dim, w.edge_dim, seed=seed,
                             mu=0.0, domain_balanced=False, verbose=verbose)
    out["fedavg"] = w.per_domain_auroc(fedavg.model)
    out["fedavg_history"] = fedavg.history

    # --- FGLGuard: 近端目标 + 域均衡聚合 (论文完整方法) ---
    fgl = train_federated(clients, cfg, w.node_dim, w.edge_dim, seed=seed,
                          mu=cfg.fed.proximal_mu, domain_balanced=True,
                          verbose=verbose)
    out["fglguard"] = w.per_domain_auroc(fgl.model)
    out["fglguard_history"] = fgl.history

    # --- 无监督/免训练对照: deviation-from-theme 打分 ---
    out["unsupervised"] = unsupervised_deviation_arm(w)

    # --- Centralized: 汇聚所有原始图 (隐私上不可行，作为上限参照) ---
    cen = train_centralized(clients, cfg, w.node_dim, w.edge_dim, seed=seed)
    out["centralized"] = w.per_domain_auroc(cen.model)

    return out


def unsupervised_deviation_arm(w: World) -> List[float]:
    """一个"免训练/无监督"对照臂 (类比 BlindGuard / XG-Guard)。

    打分规则忠实于论文描述的 *deviation-from-theme*: 在**图内**比较行动 agent 的
    自身表征与全体 agent 的平均表征，偏离越大越"可疑"。

    为什么它应当接近随机: 论文 Q2 指出 "organically unsafe collaborations drift
    together, so 'deviant agent' / deviation-from-theme scoring has no signal"。
    在我们的仿真里，风险来自**上游**注入，行动 agent 自己的话语与同侪无异，
    因此这个免标签信号几乎不携带信息。
    """
    scores_per_domain: List[float] = []
    for test in w.data.test_by_domain:
        s_list, y_list = [], []
        for g in test:
            theme = g.x.mean(axis=0)
            s_list.append(float(np.linalg.norm(g.x[g.final_agent] - theme)))
            y_list.append(g.label)
        scores_per_domain.append(auroc(s_list, y_list))
    return scores_per_domain


# --------------------------------------------------------------------------- #
# non-IID 纯度扫描 (论文 Fig.3(b))
# --------------------------------------------------------------------------- #
def run_purity_sweep(cfg: Config, purities: Sequence[float], seeds: Sequence[int],
                     same_domain: bool = True,
                     verbose: bool = False) -> Dict[str, List[float]]:
    """扫描标签偏斜纯度 :math:`p`，返回每个臂在各 p 下的宏平均 AUROC。"""
    results: Dict[str, List[float]] = {
        "fedavg": [], "fglguard": [], "local_only": [], "centralized": []
    }
    for p in purities:
        cfg_p = cfg.clone()
        cfg_p.fed.label_skew_purity = float(p)
        acc = {k: [] for k in results}
        for seed in seeds:
            w = World(cfg_p, seed=seed, same_domain=same_domain)
            clients = w.clients

            fedavg = train_federated(clients, cfg_p, w.node_dim, w.edge_dim,
                                     seed=seed, mu=0.0, domain_balanced=False)
            fgl = train_federated(clients, cfg_p, w.node_dim, w.edge_dim,
                                  seed=seed, mu=cfg_p.fed.proximal_mu,
                                  domain_balanced=True)
            locs = train_local_only(clients, cfg_p, w.node_dim, w.edge_dim, seed=seed)
            cen = train_centralized(clients, cfg_p, w.node_dim, w.edge_dim, seed=seed)

            acc["fedavg"].append(macro(w.per_domain_auroc(fedavg.model)))
            acc["fglguard"].append(macro(w.per_domain_auroc(fgl.model)))
            acc["local_only"].append(macro(w.local_only_auroc(locs)))
            acc["centralized"].append(macro(w.per_domain_auroc(cen.model)))
            if verbose:
                print(f"  p={p:.1f} seed={seed}: " +
                      "  ".join(f"{k}={np.mean(v):.3f}" for k, v in acc.items()))
        for k in results:
            results[k].append(float(np.nanmean(acc[k])))
    return results


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
ARM_LABELS = [
    ("off_the_shelf", "off-the-shelf (O-S)"),
    ("unsupervised", "unsupervised (deviant)"),
    ("local_only", "local-only"),
    ("fedavg", "FedAvg"),
    ("fglguard", "FGLGuard (ours)"),
    ("centralized", "centralized (upper bound)"),
]


def main() -> None:
    ap = argparse.ArgumentParser(description="FGLGuard 联邦检测对比实验")
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--quick", action="store_true", help="缩小规模的冒烟测试")
    ap.add_argument("--cross-domain", action="store_true",
                    help="切换到论文 Q3 的跨域设定 (一人一域) 而非同域 non-IID 切片")
    ap.add_argument("--purity-sweep", action="store_true")
    ap.add_argument("--tag", default="", help="产物文件名后缀，避免不同设定互相覆盖")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    cfg = small_config() if args.quick else default_config()
    same_domain = not args.cross_domain
    seeds = args.seeds if args.seeds else (
        (42,) if args.quick else list(cfg.seeds))
    out_dir = ensure_dir(cfg.output_dir)

    mode = "同域 non-IID 切片 (论文 Table 1 / Q1)" if same_domain else \
        "跨域联邦 (论文 Q3)"
    print("=" * 78)
    print(f"FGLGuard :: 联邦拓扑守卫 —— 检测对比 [{mode}]")
    print(f"K={cfg.fed.n_clients} clients, rounds={cfg.fed.n_rounds}, "
          f"local_epochs={cfg.fed.local_epochs}, mu={cfg.fed.proximal_mu}, "
          f"purity={cfg.fed.label_skew_purity}, lr={cfg.fed.lr}, seeds={seeds}")
    print(f"客户端体量乘子 = {cfg.sim.client_size_multipliers} (制造 silo 体量不均)")
    print("=" * 78)

    per_seed: List[Dict[str, object]] = []
    for seed in seeds:
        print(f"\n>>> seed {seed}")
        res = run_single_seed(cfg, seed, verbose=args.verbose, same_domain=same_domain)
        per_seed.append(res)
        for key, label in ARM_LABELS:
            vals = res[key]  # type: ignore[index]
            print(f"    {label:<26} per-domain AUROC = " +
                  " ".join(f"{v:.3f}" for v in vals) + f"   | macro {macro(vals):.4f}")

    print("\n" + "=" * 78)
    print("汇总 (mean +/- std over seeds)")
    print("=" * 78)
    domains = per_seed[0]["domains"]  # type: ignore[index]
    header = (["arm", "macro AUROC"] + [f"domain {d}" for d in domains]
              if len(domains) > 1 else ["arm", "AUROC (shared test set)"])
    rows = []
    for key, label in ARM_LABELS:
        stack = np.array([r[key] for r in per_seed], dtype=float)  # type: ignore[index]
        macro_stack = np.nanmean(stack, axis=1)
        if len(domains) > 1:
            rows.append([label, f"{macro_stack.mean():.4f} +/- {macro_stack.std():.4f}"]
                        + [f"{stack[:, i].mean():.3f}" for i in range(stack.shape[1])])
        else:
            rows.append([label, f"{macro_stack.mean():.4f} +/- {macro_stack.std():.4f}"])
    print(format_table(header, rows))

    def tag(name: str) -> str:
        return f"{name}_{args.tag}" if args.tag else name

    # ---- 落盘 ---- #
    curves: Dict[str, List[float]] = {}
    for key in ("fedavg", "fglguard"):
        stack = np.array([r[f"{key}_history"]["val_auroc"] for r in per_seed
                          if f"{key}_history" in r], dtype=float)
        if stack.size:
            curves[key] = stack.mean(axis=0).tolist()
    curves["rounds"] = [int(x) for x in
                        (per_seed[0].get("fedavg_history", {}).get("round", []))]

    save_json({"config": cfg.to_dict(), "seeds": seeds,
               "same_domain": same_domain, "curves": curves,
               "results": [{k: v for k, v in r.items()
                            if not k.endswith("_history")} for r in per_seed]},
              os.path.join(out_dir, tag("federation_detection") + ".json"))

    hist_series: Dict[str, List[float]] = {}
    for key, label in (("fedavg", "FedAvg"), ("fglguard", "FGLGuard")):
        stack = np.array([r[f"{key}_history"]["val_auroc"] for r in per_seed
                          if f"{key}_history" in r], dtype=float)
        if stack.size:
            hist_series[label] = stack.mean(axis=0).tolist()
    if hist_series:
        plot_curves(hist_series, "Federated training: macro validation AUROC",
                    "communication round", "AUROC",
                    os.path.join(out_dir, tag("fig_training_curve") + ".png"))

    arms = [label for _, label in ARM_LABELS]
    series = {
        f"domain {d}": [macro(np.array([r[k][d] for r in per_seed]))  # type: ignore[index]
                        for k, _ in ARM_LABELS]
        for d in range(len(per_seed[0]["domains"]))  # type: ignore[index]
    }
    plot_bars(arms, series, "Per-domain detection AUROC (episode-level)",
              "AUROC", os.path.join(out_dir, tag("fig_detection_bars") + ".png"))

    if args.purity_sweep:
        print("\n" + "=" * 78)
        print("non-IID 标签偏斜纯度扫描 (论文 Fig.3(b))")
        print("=" * 78)
        purities = [0.4, 0.6, 0.8, 1.0]
        sweep = run_purity_sweep(cfg, purities, seeds, same_domain=same_domain,
                                 verbose=True)
        print("\n" + format_table(["purity"] + list(sweep.keys()),
                                  [[f"{p:.1f}"] + [f"{sweep[k][i]:.4f}"
                                                   for k in sweep]
                                   for i, p in enumerate(purities)]))
        save_json({"purities": purities, "results": sweep},
                  os.path.join(out_dir, tag("purity_sweep") + ".json"))
        plot_curves(sweep, "Robustness to non-IID clients (label-skew purity)",
                    "label-skew purity p", "macro AUROC",
                    os.path.join(out_dir, tag("fig_purity_sweep") + ".png"),
                    x=purities)

    print(f"\n结果已写入 {os.path.abspath(out_dir)}/")


if __name__ == "__main__":
    main()
