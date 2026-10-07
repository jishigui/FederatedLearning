"""实验二: 运行时部署 —— 攻击成功率 vs 效用 (论文 Fig.4、Sec. 5.3 运行时消融)。

阅读顺序: 第 15 步 / 共 18 步 —— 见 READING_ORDER.md

运行::

    python -m experiments.run_runtime
    python -m experiments.run_runtime --quick

流程:

1. 按各臂训练出检测器 (off-the-shelf / local-only / centralized / FGLGuard);
2. 用 **Eq. (7)** 在 ``rho`` 预算下校准阈值 :math:`\\tau^*`
   (统计量取佐证分数 :math:`\\hat s`，与运行时使用的规则一致);
3. 在一批**全新**的 episode 上执行 **Eq. (9)** 的处置流程，统计:

   - **ASR** (攻击成功率, 越低越好)
   - **utility** (良性 episode 的交付率, 越高越好)
   - **over-refusal** (良性 episode 被拒答的比例, 应受 :math:`\\rho` 约束)

4. 附带三组运行时消融: 关闭佐证打分 / 关闭重写通道 / 同时守卫工具调用。
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.common import World, train_off_the_shelf
from fglguard.calibration import calibrate_threshold
from fglguard.config import Config, default_config, small_config
from fglguard.federated import (collect_validation, train_centralized,
                                train_federated, train_local_only)
from fglguard.runtime import RuntimeGuard, evaluate_runtime, sanitizing_rewrite
from fglguard.utils import ensure_dir, format_table, plot_bars, save_json


# --------------------------------------------------------------------------- #
def runtime_episodes(sim, clients, cfg: Config, per_domain: int, seed: int
                     ) -> Dict[int, list]:
    """为每个客户端生成全新的 (攻击集, 良性集) episode，key 为 client_id。"""
    rng = np.random.default_rng(seed + 991)
    out: Dict[int, list] = {}
    for c in clients:
        d = c.domain
        attacks = sim.generate_split(d, per_domain, rng, force_intent=True)
        benign = sim.generate_split(d, per_domain, rng, force_intent=False)
        out[c.client_id] = attacks + benign
    return out


def fmt_pm(values: Sequence[float]) -> str:
    """把多个种子的结果格式化成一行文本 (NaN 会被忽略)。"""
    arr = np.asarray(list(values), dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return "n/a"
    return f"{arr.mean():.4f}"


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="FGLGuard 运行时部署评测")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--cross-domain", action="store_true",
                    help="切换到跨域设定 (一人一域)")
    ap.add_argument("--episodes-per-domain", type=int, default=None,
                    help="每个客户端生成多少条攻击/良性 episode")
    ap.add_argument("--tag", default="", help="产物文件名后缀，避免不同设定互相覆盖")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    cfg = small_config() if args.quick else default_config()
    same_domain = not args.cross_domain
    seeds = args.seeds if args.seeds else ((42,) if args.quick else [42, 43])
    per_domain = args.episodes_per_domain or (40 if args.quick else 200)
    out_dir = ensure_dir(cfg.output_dir)
    rho = cfg.calib.over_refusal_budget

    def suffix(name: str) -> str:
        return f"{name}_{args.tag}" if args.tag else name

    print("=" * 78)
    print("FGLGuard :: 运行时部署 (论文 Fig.4 / Q4)")
    print(f"rho = {rho}, episodes/domain = {per_domain}, seeds = {seeds}")
    print("=" * 78)

    rows: List[List[str]] = []
    arm_order = ["fglguard", "fedavg", "centralized", "local_only",
                 "off_the_shelf", "fglguard_no_corroborate",
                 "fglguard_no_rewrite", "fglguard_guard_tools", "unguarded"]
    arm_label = {
        "fglguard": "FGLGuard (ours)",
        "fedavg": "FedAvg",
        "centralized": "centralized (pooled)",
        "local_only": "local-only (4 silos)",
        "off_the_shelf": "off-the-shelf (O-S)",
        "fglguard_no_corroborate": "  abl: no corroboration",
        "fglguard_no_rewrite": "  abl: no guarded rewrite",
        "fglguard_guard_tools": "  abl: guard tool calls too",
        "unguarded": "unguarded (reference)",
    }
    agg: Dict[str, Dict[str, List[float]]] = {
        a: {"asr": [], "utility": [], "over_refusal": [], "rewrite_rate": []}
        for a in arm_order}
    tau_log: Dict[str, List[float]] = {a: [] for a in arm_order}
    calib_log: Dict[str, List[float]] = {}

    for seed in seeds:
        print(f"\n>>> seed {seed}")
        w = World(cfg, seed=seed, same_domain=same_domain)
        enc, sim, clients = w.encoder, w.sim, w.clients
        eps_by_client = runtime_episodes(sim, clients, cfg, per_domain, seed)

        # ---------------- 训练各臂 ---------------- #
        fgl = train_federated(clients, cfg, w.node_dim, w.edge_dim, seed=seed,
                              mu=cfg.fed.proximal_mu, domain_balanced=True,
                              verbose=args.verbose)
        fedavg = train_federated(clients, cfg, w.node_dim, w.edge_dim, seed=seed,
                                 mu=0.0, domain_balanced=False)
        cen = train_centralized(clients, cfg, w.node_dim, w.edge_dim, seed=seed)
        locs = train_local_only(clients, cfg, w.node_dim, w.edge_dim, seed=seed)
        ots = train_off_the_shelf(cfg, sim, enc, seed=seed)

        # ---------------- 每臂的 (模型, tau) 列表 ---------------- #
        def pooled_tau(model, corroborate: bool = True) -> float:
            vs, vl = collect_validation(model, clients, corroborate=corroborate)
            return calibrate_threshold(vs, vl, cfg.calib, rho=rho).tau

        def make_arms() -> Dict[str, List[Tuple[int, RuntimeGuard]]]:
            arms: Dict[str, List[Tuple[int, RuntimeGuard]]] = {}

            def spread(model, tau: float, rt_cfg) -> List[Tuple[int, RuntimeGuard]]:
                return [(c.client_id, RuntimeGuard(model, tau, rt_cfg, enc, cfg.device))
                        for c in clients]

            tau_fgl = pooled_tau(fgl.model, corroborate=True)
            arms["fglguard"] = spread(fgl.model, tau_fgl, cfg)

            tau_fedavg = pooled_tau(fedavg.model, corroborate=True)
            arms["fedavg"] = spread(fedavg.model, tau_fedavg, cfg)

            tau_cen = pooled_tau(cen.model, corroborate=True)
            arms["centralized"] = spread(cen.model, tau_cen, cfg)

            # local-only: 每个 silo 只能用自己的 val 分数校准 -> 阈值会"饱和或乱开火"
            arms["local_only"] = []
            for m, c in zip(locs, clients):
                t = calibrate_threshold(m.val_scores, m.val_labels, cfg.calib,
                                        rho=rho).tau
                arms["local_only"].append(
                    (c.client_id, RuntimeGuard(m.model, t, cfg, enc, cfg.device)))

            tau_ots = pooled_tau(ots.model, corroborate=True)
            arms["off_the_shelf"] = spread(ots.model, tau_ots, cfg)

            # --- 运行时消融 (论文 Sec. 5.3) --- #
            cfg_nc = cfg.clone()
            cfg_nc.runtime.corroborate = False
            arms["fglguard_no_corroborate"] = spread(fgl.model, tau_fgl, cfg_nc)

            cfg_nr = cfg.clone()
            cfg_nr.runtime.allow_rewrite = False
            arms["fglguard_no_rewrite"] = spread(fgl.model, tau_fgl, cfg_nr)

            cfg_gt = cfg.clone()
            cfg_gt.runtime.guard_final_answer_only = False
            arms["fglguard_guard_tools"] = spread(fgl.model, tau_fgl, cfg_gt)

            return arms

        arms = make_arms()
        # unguarded 参照: 阈值恒不可达 -> 永远原样放行
        arms["unguarded"] = [(c.client_id,
                              RuntimeGuard(fgl.model, 1.01, cfg, enc, cfg.device))
                             for c in clients]

        # ---------------- 评测 ---------------- #
        for arm in arm_order:
            rng = np.random.default_rng(seed + 17)
            all_res: List[Dict[str, float]] = []
            for (cid, guard) in arms[arm]:
                episodes = eps_by_client[cid]
                res = evaluate_runtime(
                    guard, episodes, rng=rng,
                    rewrite_fn=sanitizing_rewrite,
                    unguarded=(arm == "unguarded"))
                all_res.append(res)

            def wavg(key: str) -> float:
                num = sum(r[key] * r["n_attack" if key == "asr" else "n_benign"]
                          for r in all_res)
                den = sum(r["n_attack" if key == "asr" else "n_benign"]
                          for r in all_res)
                return float(num / den) if den else float("nan")

            n_atk = sum(r["n_attack"] for r in all_res)
            asr, util, orf = wavg("asr"), wavg("utility"), wavg("over_refusal")
            rew = (sum(r.get("n_rewrite", 0) for r in all_res) / n_atk
                   if n_atk else float("nan"))
            agg[arm]["asr"].append(asr)
            agg[arm]["utility"].append(util)
            agg[arm]["over_refusal"].append(orf)
            agg[arm]["rewrite_rate"].append(rew)
            if arm not in ("unguarded",):
                tau_log[arm].append(float(np.mean(
                    [g.tau for _, g in arms[arm]])))
            print(f"    {arm_label[arm]:<32} ASR={asr:.4f}  utility={util:.4f}  "
                  f"over-refusal={orf:.4f}")

        # 校准诊断 (FGLGuard)
        vs, vl = collect_validation(fgl.model, clients, corroborate=True)
        cr = calibrate_threshold(vs, vl, cfg.calib, rho=rho)
        calib_log.setdefault("tau", []).append(cr.tau)
        calib_log.setdefault("val_recall", []).append(cr.recall)
        calib_log.setdefault("val_fpr", []).append(cr.fpr)

    # ---------------- 汇总 ---------------- #
    print("\n" + "=" * 78)
    print("运行时汇总 (mean over seeds)")
    print("=" * 78)
    header = ["arm", "ASR down", "utility up", "over-refusal", "rewrite", "tau*"]
    for arm in arm_order:
        taus = tau_log[arm]
        rows.append([
            arm_label[arm],
            fmt_pm(agg[arm]["asr"]),
            fmt_pm(agg[arm]["utility"]),
            fmt_pm(agg[arm]["over_refusal"]),
            fmt_pm(agg[arm]["rewrite_rate"]),
            f"{np.mean(taus):.3f}" if taus else "-",
        ])
    print(format_table(header, rows))
    if calib_log.get("tau"):
        print(f"\nFGLGuard 校准诊断: tau*={np.mean(calib_log['tau']):.3f}  "
              f"val_recall={np.mean(calib_log['val_recall']):.3f}  "
              f"val_FPR={np.mean(calib_log['val_fpr']):.3f}  (budget rho={rho})")

    save_json({"config": cfg.to_dict(), "seeds": seeds, "rho": rho,
               "aggregate": agg, "tau": tau_log, "calibration": calib_log},
              os.path.join(out_dir, suffix("runtime_deployment") + ".json"))

    plot_bars([arm_label[a] for a in arm_order],
              {"ASR ↓": [np.mean(agg[a]["asr"]) for a in arm_order],
               "over-refusal": [np.mean(agg[a]["over_refusal"]) for a in arm_order]},
              "Runtime: attack success vs over-refusal", "rate",
              os.path.join(out_dir, suffix("fig_runtime_asr") + ".png"))
    plot_bars([arm_label[a] for a in arm_order],
              {"utility ↑": [np.mean(agg[a]["utility"]) for a in arm_order]},
              "Runtime: clean utility", "rate",
              os.path.join(out_dir, suffix("fig_runtime_utility") + ".png"))

    print(f"\n结果已写入 {os.path.abspath(out_dir)}/")


if __name__ == "__main__":
    main()
