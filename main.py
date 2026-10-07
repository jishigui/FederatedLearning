"""FGLGuard —— 一键最小可运行示例。

阅读顺序: 第 12 步 / 共 18 步 (把前面所有模块串起来) —— 见 READING_ORDER.md

用法::

    python main.py            # 小型配置，约 1-2 分钟
    python main.py --full     # 论文 Protocol 的缩小版，约 5-10 分钟

流程完全对齐论文 Figure 2:

    各 operator 本地构建 episode 图 G=(X,A,E)
        -> 边特征 GAT 用近端目标本地训练 (Eq. 5)
        -> 服务端只收模型更新, 做域均衡聚合 (Eq. 6)
        -> 用各客户端本地上算的标量分数做预算校准 (Eq. 7)
        -> 部署为 sidecar, 佐证上游风险并至多重写一次 (Eq. 8-9)
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiments.common import World, macro, train_off_the_shelf
from fglguard.calibration import calibrate_threshold
from fglguard.config import default_config, small_config
from fglguard.federated import (collect_validation, train_centralized,
                                train_federated, train_local_only)
from fglguard.runtime import RuntimeGuard, evaluate_runtime, sanitizing_rewrite
from fglguard.utils import ensure_dir, format_table, plot_bars, save_json


def main() -> None:
    ap = argparse.ArgumentParser(description="FGLGuard 最小可运行示例")
    ap.add_argument("--full", action="store_true", help="使用默认(更大)配置")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cross-domain", action="store_true",
                    help="使用跨域联邦设定 (论文 Q3)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    cfg = default_config() if args.full else small_config()
    same_domain = not args.cross_domain
    out_dir = ensure_dir(cfg.output_dir)
    rho = cfg.calib.over_refusal_budget

    print("=" * 74)
    print("FGLGuard :: 隐私保护的拓扑引导多智能体安全防护 (图联邦学习)")
    print(f"配置: K={cfg.fed.n_clients} clients, R={cfg.fed.n_rounds} rounds, "
          f"E={cfg.fed.local_epochs}, mu={cfg.fed.proximal_mu}, rho={rho}")
    print(f"设定: {'同域 non-IID 切片 (论文 Table 1)' if same_domain else '跨域联邦 (论文 Q3)'}")
    print("=" * 74)

    # ------------------------------------------------------------------ #
    # 1) 构造仿真世界 (等价于各 operator 私有的 episode 图)
    # ------------------------------------------------------------------ #
    w = World(cfg, seed=args.seed, same_domain=same_domain)
    clients = w.clients
    test = w.data.test_by_domain[0]
    print(f"\n[1/5] 数据就绪: 每客户端训练图数量 "
          f"{[len(c.train) for c in w.data.clients]}")
    print(f"      标签偏斜纯度 p={cfg.fed.label_skew_purity}, "
          f"主导标签 {[c.dominant_label for c in w.data.clients]}, "
          f"测试集正例率 {np.mean([g.label for g in test]):.3f}")

    # ------------------------------------------------------------------ #
    # 2) 联邦训练 (论文 Eq. 5-6)
    # ------------------------------------------------------------------ #
    print(f"\n[2/5] 联邦近端训练 (FedProx mu={cfg.fed.proximal_mu}) ...")
    fgl = train_federated(clients, cfg, w.node_dim, w.edge_dim, seed=args.seed,
                          mu=cfg.fed.proximal_mu, domain_balanced=True,
                          verbose=not args.quiet)
    print(f"      模型参数量 = {fgl.model.n_parameters:,}, "
          f"耗时 {fgl.wall_time:.1f}s")

    # ------------------------------------------------------------------ #
    # 3) 阈值校准 (论文 Eq. 7)
    # ------------------------------------------------------------------ #
    print(f"\n[3/5] 过拒答预算校准 (rho={rho}) ...")
    vs, vl = collect_validation(fgl.model, clients, corroborate=cfg.runtime.corroborate)
    cal = calibrate_threshold(vs, vl, cfg.calib, rho=rho)
    print(f"      tau* = {cal.tau:.4f}   val_recall = {cal.recall:.4f}   "
          f"val_FPR = {cal.fpr:.4f} (预算 {rho})")

    # ------------------------------------------------------------------ #
    # 4) 与其他方案对比 (AUROC)
    # ------------------------------------------------------------------ #
    print("\n[4/5] 检测质量对比 (逐节点监督, episode 级从 final-answer agent 读出)")
    rows = []
    keys = []
    ots = train_off_the_shelf(cfg, w.sim, w.encoder, seed=args.seed)
    rows.append(["off-the-shelf (异域合成语料)", f"{macro(w.per_domain_auroc(ots.model)):.4f}"])
    keys.append("off-the-shelf")

    locs = train_local_only(clients, cfg, w.node_dim, w.edge_dim, seed=args.seed)
    rows.append(["local-only (各 silo 独立)", f"{macro(w.local_only_auroc(locs)):.4f}"])
    keys.append("local-only")

    fedavg = train_federated(clients, cfg, w.node_dim, w.edge_dim, seed=args.seed,
                             mu=0.0, domain_balanced=False)
    rows.append(["FedAvg (体量加权, 无近端项)", f"{macro(w.per_domain_auroc(fedavg.model)):.4f}"])
    keys.append("FedAvg")
    rows.append(["FGLGuard (ours)", f"{macro(w.per_domain_auroc(fgl.model)):.4f}"])
    keys.append("FGLGuard")

    cen = train_centralized(clients, cfg, w.node_dim, w.edge_dim, seed=args.seed)
    rows.append(["centralized (汇聚原始图, 隐私违规)", f"{macro(w.per_domain_auroc(cen.model)):.4f}"])
    keys.append("centralized")
    print(format_table(["方案", "macro AUROC"], rows))

    # ------------------------------------------------------------------ #
    # 5) 运行时部署评测 (论文 Eq. 8-9 / Fig. 4)
    # ------------------------------------------------------------------ #
    print("\n[5/5] 运行时部署 (sidecar: 佐证打分 + 至多一次 guarded rewrite)")
    rng = np.random.default_rng(args.seed + 991)
    eps_by_client = {}
    for c in w.data.clients:
        d = c.domain
        eps_by_client[c.client_id] = (
            w.sim.generate_split(d, 80, rng, force_intent=True)
            + w.sim.generate_split(d, 80, rng, force_intent=False))

    def pooled_tau(model):
        s, y = collect_validation(model, clients, corroborate=True)
        return calibrate_threshold(s, y, cfg.calib, rho=rho).tau

    taus = {
        "unguarded": 1.01,
        "off-the-shelf": pooled_tau(ots.model),
        "local-only": None,
        "centralized": pooled_tau(cen.model),
        "FGLGuard": cal.tau,
    }
    guards = {
        "unguarded": {c.client_id: RuntimeGuard(fgl.model, taus["unguarded"], cfg, w.encoder)
                      for c in w.data.clients},
        "off-the-shelf": {c.client_id: RuntimeGuard(ots.model, taus["off-the-shelf"], cfg, w.encoder)
                          for c in w.data.clients},
        "local-only": {c.client_id: RuntimeGuard(m.model,
                                                 calibrate_threshold(m.val_scores, m.val_labels,
                                                                     cfg.calib, rho=rho).tau,
                                                 cfg, w.encoder)
                       for m, c in zip(locs, w.data.clients)},
        "centralized": {c.client_id: RuntimeGuard(cen.model, taus["centralized"], cfg, w.encoder)
                        for c in w.data.clients},
        "FGLGuard": {c.client_id: RuntimeGuard(fgl.model, cal.tau, cfg, w.encoder)
                     for c in w.data.clients},
    }

    rt_rows = []
    rt_json = {}
    for arm, gmap in guards.items():
        rr = np.random.default_rng(args.seed + 17)
        results = [evaluate_runtime(gmap[c.client_id], eps_by_client[c.client_id],
                                    rng=rr, rewrite_fn=sanitizing_rewrite,
                                    unguarded=(arm == "unguarded"))
                   for c in w.data.clients]

        def wavg(key, nkey):
            num = sum(r[key] * r[nkey] for r in results)
            den = sum(r[nkey] for r in results)
            return float(num / den) if den else float("nan")

        asr = wavg("asr", "n_attack")
        util = wavg("utility", "n_benign")
        orf = wavg("over_refusal", "n_benign")
        rt_json[arm] = {"asr": asr, "utility": util, "over_refusal": orf}
        rt_rows.append([arm, f"{asr:.4f}", f"{util:.4f}", f"{orf:.4f}"])

    print(format_table(["方案", "ASR ↓", "utility ↑", "over-refusal"], rt_rows))

    base = rt_json["unguarded"]["asr"]
    ours = rt_json["FGLGuard"]["asr"]
    if base and np.isfinite(base) and np.isfinite(ours) and base > 0:
        print(f"\n>>> FGLGuard 把攻击成功率从 {base:.4f} 降到 {ours:.4f} "
              f"(相对削减 {100 * (1 - ours / base):.1f}%)，"
              f"同时保有 {rt_json['FGLGuard']['utility']:.4f} 的良性交付率。")

    # ------------------------------------------------------------------ #
    save_json({"config": cfg.to_dict(),
               "detection": {k: r[1] for k, r in zip(keys, rows)},
               "runtime": rt_json,
               "calibration": cal.to_dict()},
              os.path.join(out_dir, "main_demo.json"))
    plot_bars(keys, {"macro AUROC": [float(r[1]) for r in rows]},
              "Detection quality (macro AUROC)", "AUROC",
              os.path.join(out_dir, "fig_main_detection.png"))
    plot_bars(list(rt_json.keys()),
              {"ASR (lower better)": [rt_json[k]["asr"] for k in rt_json],
               "over-refusal": [rt_json[k]["over_refusal"] for k in rt_json]},
              "Runtime deployment", "rate",
              os.path.join(out_dir, "fig_main_runtime.png"))
    print(f"\n结果已写入 {os.path.abspath(out_dir)}/")


if __name__ == "__main__":
    main()
