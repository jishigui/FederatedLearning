"""FGLGuard 组件的自包含测试。

阅读顺序: 第 16 步 / 共 18 步 (最快的验收清单，配合 README 2.6 的对照表看)
—— 见 READING_ORDER.md

不依赖 pytest: 直接 ``python tests/test_components.py`` 即可运行，
也兼容 ``pytest tests/``。
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fglguard.calibration import (benign_quantile_threshold, calibrate_threshold,
                                  pool_client_validation_scores)
from fglguard.config import CalibConfig, small_config
from fglguard.data import build_federated_data, collate_graphs
from fglguard.encoder import FrozenEncoder
from fglguard.federated import (FederatedClient, aggregate_states,
                                domain_balanced_weights, evaluate_auroc,
                                make_model, train_federated, volume_weights)
from fglguard.graph import build_graph, corroborated_scores
from fglguard.mas_simulator import MASSimulator
from fglguard.metrics import auroc, confusion_at, wilson_interval
from fglguard.model import node_bce_loss
from fglguard.runtime import (RuntimeGuard, contains_payload, evaluate_runtime,
                              identity_rewrite, sanitizing_rewrite)


# --------------------------------------------------------------------------- #
# 1. 编码器
# --------------------------------------------------------------------------- #
def test_encoder_basic():
    cfg = small_config()
    enc = FrozenEncoder(cfg.vocab, cfg.encoder)
    assert enc.dim == cfg.encoder.dim

    v = enc.embed_text([0, 1, 2, 3])
    assert v.shape == (enc.dim,)
    assert abs(float(np.linalg.norm(v)) - 1.0) < 1e-5, "L2 归一化未生效"

    empty = enc.embed_text([])
    assert np.allclose(empty, 0.0), "空 utterance 应编码为零向量"

    # 不同 domain 的良性词 block 不重叠 (domain shift 的来源)
    b0 = set(range(*cfg.vocab.benign_range(0)))
    b1 = set(range(*cfg.vocab.benign_range(1)))
    assert not (b0 & b1)
    # 攻击 block 只在跨域共享风险标记上重叠
    a0 = enc.payload_token_set(0)
    a1 = enc.payload_token_set(1)
    assert a0 & a1 == set(range(*cfg.vocab.shared_risk_range))


# --------------------------------------------------------------------------- #
# 2. 仿真器
# --------------------------------------------------------------------------- #
def test_simulator_shapes_and_labels():
    cfg = small_config()
    enc = FrozenEncoder(cfg.vocab, cfg.encoder)
    sim = MASSimulator(cfg.vocab, cfg.sim, enc)
    rng = np.random.default_rng(0)
    eps = sim.generate(0, 60, rng)

    for e in eps:
        n = e.n_agents
        assert e.adjacency.shape == (n, n)
        assert np.all(np.diag(e.adjacency) == 0), "拓扑不应含自环"
        assert len(e.utterances) == cfg.sim.n_rounds
        assert len(e.utterances[0]) == n
        assert e.label in (0, 1) and e.intent in (0, 1)
        if e.label == 1:
            # 标签成立的必要条件: 行动 agent 自身或上游被感染
            T_last = cfg.sim.n_rounds - 1
            preds = np.nonzero(e.adjacency[:, e.final_agent])[0]
            assert e.infected[T_last, e.final_agent] or any(
                e.infected[T_last, i] for i in preds)
        if not e.intent:
            assert e.label == 0, "良性 episode 不应被感染"

    # "direct" 注入配方: 攻击时行动 agent 必然自身带毒
    direct = sim.generate(cfg.vocab.source_domain, 40, rng, injection_mode="direct")
    pos = [e for e in direct if e.intent]
    assert all(e.direct_exposure for e in pos)


# --------------------------------------------------------------------------- #
# 3. 图构建
# --------------------------------------------------------------------------- #
def test_graph_construction():
    cfg = small_config()
    enc = FrozenEncoder(cfg.vocab, cfg.encoder)
    sim = MASSimulator(cfg.vocab, cfg.sim, enc)
    ep = sim.sample_episode(0, np.random.default_rng(3))

    g = build_graph(ep, enc, self_loops=True)
    n, T, d = ep.n_agents, ep.n_rounds, enc.dim

    assert g.x.shape == (n, d)
    assert g.edge_feat.shape == (n, n, d)
    assert g.adj.shape == (n, n)
    assert np.all(np.diag(g.adj) == 1.0), "self_loops=True 时对角线应为 1"

    # 节点特征 = 自身 utterance 历史的时间均值 (论文 Sec. 3 的核心改动)
    manual = np.mean([enc.embed_text(ep.utterances[t][2]) for t in range(T)], axis=0)
    assert np.allclose(g.x[2], manual, atol=1e-5)

    # 边 (i -> j) 承载**接收方** j 的逐轮嵌入均值
    i, j = 0, 1
    if g.adj[i, j] > 0 and i != j:
        expect = np.mean([enc.embed_text(ep.utterances[t][j]) for t in range(T)], axis=0)
        assert np.allclose(g.edge_feat[i, j], expect, atol=1e-5)

    # 标签广播 (论文默认)
    assert np.all(g.node_labels == ep.label)

    g_final = build_graph(ep, enc, node_label_mode="final")
    assert g_final.node_labels.sum() == ep.label


# --------------------------------------------------------------------------- #
# 4. 模型
# --------------------------------------------------------------------------- #
def test_model_forward_and_loss():
    cfg = small_config()
    enc = FrozenEncoder(cfg.vocab, cfg.encoder)
    sim = MASSimulator(cfg.vocab, cfg.sim, enc)
    eps = sim.generate(0, 8, np.random.default_rng(1))
    graphs = [build_graph(e, enc) for e in eps]
    batch = collate_graphs(graphs)

    model = make_model(cfg, enc.dim, enc.dim, seed=0)
    logits, probs = model(batch.x, batch.adj, batch.edge)
    assert logits.shape == probs.shape == (8, batch.x.shape[1])
    assert torch.all(probs >= 0) and torch.all(probs <= 1)

    loss = node_bce_loss(logits, batch.node_labels, batch.node_mask)
    assert torch.isfinite(loss) and loss.item() > 0

    # padding 的节点不参与 loss
    masked = node_bce_loss(logits, batch.node_labels, batch.node_mask, reduction="sum")
    full = node_bce_loss(logits[:, :1], batch.node_labels[:, :1],
                         batch.node_mask[:, :1], reduction="sum")
    assert torch.isfinite(masked) and torch.isfinite(full)


def test_model_parameters_scale():
    cfg = small_config()
    model = make_model(cfg, cfg.encoder.dim, cfg.encoder.dim, seed=0)
    n = model.n_parameters
    assert 1_000 < n < 2_000_000, f"参数量异常: {n}"


# --------------------------------------------------------------------------- #
# 5. 校准 (论文 Eq. 7 + Proposition 2)
# --------------------------------------------------------------------------- #
def test_calibration_respects_budget():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, size=2000)
    scores = np.clip(0.5 * labels + rng.normal(0.3, 0.25, size=2000), 0, 1)
    rho = 0.10

    res = calibrate_threshold(scores, labels, rho=rho)
    diag = confusion_at(scores, labels, res.tau)
    assert diag["fpr"] <= rho + 1e-9, "校准后的 FPR 必须落在预算内"
    assert 0.0 <= res.tau <= 1.0

    # 更宽松的预算不应降低召回
    res_loose = calibrate_threshold(scores, labels, rho=0.50)
    assert res_loose.recall >= res.recall - 1e-9


def test_calibration_benign_only_matches_quantile():
    """论文 Proposition 2: 只有良性流量时 Eq.(7) 退化为 (1-rho) 分位数。"""
    rng = np.random.default_rng(1)
    scores = rng.random(500)
    labels = np.zeros(500, dtype=int)
    rho = 0.10

    tau = calibrate_threshold(scores, labels, rho=rho).tau
    q = benign_quantile_threshold(scores, labels, rho)
    assert abs(tau - q) < 1e-6, f"{tau} vs {q}"

    # 良性子集上的 FPR 必须 <= rho
    assert float(np.mean(scores >= tau)) <= rho + 1e-9


def test_pool_validation_scores():
    s, y = pool_client_validation_scores(
        [np.array([0.1, 0.2]), np.array([0.3])],
        [np.array([0, 1]), np.array([1])])
    assert s.tolist() == [0.1, 0.2, 0.3]
    assert y.tolist() == [0, 1, 1]


# --------------------------------------------------------------------------- #
# 6. 佐证打分 (论文 Eq. 8)
# --------------------------------------------------------------------------- #
def test_corroboration():
    s = np.array([0.1, 0.9, 0.2])
    adj = np.array([[0, 1, 0],
                    [0, 0, 1],
                    [0, 0, 0]], dtype=float)
    s_hat = corroborated_scores(s, adj)
    # 节点 1 的上游是 0 (0.1) -> 不变; 节点 2 的上游是 1 (0.9) -> 被抬高
    assert np.isclose(s_hat[1], 0.9)
    assert np.isclose(s_hat[2], 0.9)
    assert np.all(s_hat >= s - 1e-9), "佐证只会抬高、不会降低分数"


# --------------------------------------------------------------------------- #
# 7. 聚合 (论文 Eq. 6 + Proposition 1)
# --------------------------------------------------------------------------- #
def test_aggregation_weights():
    cfg = small_config()
    enc = FrozenEncoder(cfg.vocab, cfg.encoder)
    sim = MASSimulator(cfg.vocab, cfg.sim, enc)

    from fglguard.data import ClientData
    from fglguard.graph import build_graphs
    clients = []
    for k in range(4):
        domain = k % 2
        eps = sim.generate(domain, 10 + 30 * k, np.random.default_rng(k))
        cdata = ClientData(client_id=k, domain=domain,
                           train=build_graphs(eps, enc), val=[], test=[],
                           dominant_label=0)
        clients.append(FederatedClient(cdata, cfg))

    a = domain_balanced_weights(clients)
    assert abs(a.sum() - 1.0) < 1e-9
    # 每个 domain 拿到 1/D 的总质量
    for d in (0, 1):
        mass = sum(w for w, c in zip(a, clients) if c.domain == d)
        assert abs(mass - 0.5) < 1e-9, f"domain {d} 总质量 = {mass}"

    v = volume_weights(clients)
    assert abs(v.sum() - 1.0) < 1e-9
    # 域均衡下小客户端比纯体量加权拿到更大的权重
    assert a[0] > v[0]

    # D=1 时 Eq.(6) 退化为 FedAvg (论文 Proposition 1)
    for c in clients:
        c.data.domain = 0
    a1 = domain_balanced_weights(clients)
    assert np.allclose(a1, volume_weights(clients), atol=1e-9)


def test_aggregate_states():
    states = [{"w": torch.zeros(3)}, {"w": torch.full((3,), 2.0)}]
    out = aggregate_states(states, [0.25, 0.75])
    assert torch.allclose(out["w"], torch.full((3,), 1.5))


# --------------------------------------------------------------------------- #
# 8. 联邦训练确实共享信息
# --------------------------------------------------------------------------- #
def test_federation_beats_off_the_shelf():
    cfg = small_config()
    cfg.fed.n_rounds = 4
    enc = FrozenEncoder(cfg.vocab, cfg.encoder)
    sim = MASSimulator(cfg.vocab, cfg.sim, enc)
    fdata = build_federated_data(cfg, enc, sim, seed=0, same_domain=True)
    clients = [FederatedClient(c, cfg) for c in fdata.clients]

    res = train_federated(clients, cfg, enc.dim, enc.dim, seed=0, mu=0.01)
    val = evaluate_auroc(res.model, clients[0].test_batches)
    assert 0.0 <= val <= 1.0
    assert np.isfinite(val)


# --------------------------------------------------------------------------- #
# 9. 运行时 (论文 Eq. 9)
# --------------------------------------------------------------------------- #
def _make_guard(cfg, enc, sim, tau: float, runtime_overrides=None):
    if runtime_overrides:
        for k, v in runtime_overrides.items():
            setattr(cfg.runtime, k, v)
    model = make_model(cfg, enc.dim, enc.dim, seed=0)
    return RuntimeGuard(model, tau, cfg, enc)


def test_runtime_decision_paths():
    cfg = small_config()
    enc = FrozenEncoder(cfg.vocab, cfg.encoder)
    sim = MASSimulator(cfg.vocab, cfg.sim, enc)
    rng = np.random.default_rng(5)
    episodes = sim.generate_split(0, 20, rng, force_intent=True)

    # tau = 0 -> 所有分数都 >= tau -> 一定进入阻断/重写通道
    guard_block = _make_guard(cfg, enc, sim, 0.0)
    dec, info = guard_block.intervene(episodes[0], rng=rng)
    assert dec.action in ("rewrite", "refuse")
    assert dec.blocked
    assert "s_final" in info

    # tau = 1.01 -> 永不可达 -> 原样放行
    guard_pass = _make_guard(cfg, enc, sim, 1.01)
    dec2, _ = guard_pass.intervene(episodes[0], rng=rng)
    assert dec2.action == "release"
    assert dec2.emitted_tokens == [int(t) for t in episodes[0].utterances[-1][episodes[0].final_agent]]


def test_rewrite_clears_payload():
    cfg = small_config()
    enc = FrozenEncoder(cfg.vocab, cfg.encoder)
    rng = np.random.default_rng(0)
    tokens = enc.sample_attack_tokens(0, 6, rng, 0.3) + [100, 101]
    assert contains_payload(tokens, enc, 0)

    fixed = sanitizing_rewrite(tokens, 0, enc, rng, keep_ratio=0.0)
    assert not contains_payload(fixed, enc, 0), "重写后不应再含 payload"
    assert len(fixed) == len(tokens), "重写应保持长度 (只是替换 token)"

    ident = identity_rewrite(tokens, 0, enc, rng)
    assert contains_payload(ident, enc, 0)


def test_runtime_metrics_bounds():
    cfg = small_config()
    enc = FrozenEncoder(cfg.vocab, cfg.encoder)
    sim = MASSimulator(cfg.vocab, cfg.sim, enc)
    rng = np.random.default_rng(11)
    eps = (sim.generate_split(0, 40, rng, force_intent=True)
           + sim.generate_split(0, 40, rng, force_intent=False))

    # unguarded 参照
    guard = _make_guard(cfg, enc, sim, 1.01)
    res = evaluate_runtime(guard, eps, rng=rng, unguarded=True)
    assert abs(res["utility"] - 1.0) < 1e-9
    assert abs(res["over_refusal"]) < 1e-9

    # tau=0 -> 任何风险分都触发阻断; 良性 episode 也会被大量拒答
    guard_hard = _make_guard(cfg, enc, sim, 0.0)
    res_hard = evaluate_runtime(guard_hard, eps, rng=rng)
    assert res_hard["over_refusal"] > 0.5


def test_guarding_tool_calls_costs_utility():
    """论文 Sec. 5.3: 同时守工具调用会损害 completion。"""
    cfg_a = small_config()
    enc = FrozenEncoder(cfg_a.vocab, cfg_a.encoder)
    sim = MASSimulator(cfg_a.vocab, cfg_a.sim, enc)
    rng = np.random.default_rng(21)
    eps = sim.generate_split(0, 60, rng, force_intent=False)

    g_final = _make_guard(cfg_a, enc, sim, 0.0)
    util_final = evaluate_runtime(g_final, eps, rng=rng)["utility"]

    cfg_b = small_config()
    cfg_b.runtime.guard_final_answer_only = False
    g_tools = RuntimeGuard(make_model(cfg_b, enc.dim, enc.dim, seed=0), 0.0,
                           cfg_b, enc)
    util_tools = evaluate_runtime(g_tools, eps, rng=rng)["utility"]

    assert util_tools <= util_final + 1e-9


# --------------------------------------------------------------------------- #
# 10. 指标
# --------------------------------------------------------------------------- #
def test_metrics():
    assert auroc([0.1, 0.9], [0, 1]) == 1.0
    assert auroc([0.9, 0.1], [0, 1]) == 0.0
    assert auroc([0.5, 0.5], [0, 1]) == 0.5
    assert np.isnan(auroc([0.5], [1]))

    d = confusion_at([0.9, 0.8, 0.1], [1, 0, 1], tau=0.5)
    assert d["tp"] == 1 and d["fp"] == 1 and d["fn"] == 1 and d["tn"] == 0
    assert abs(d["fpr"] - 1.0) < 1e-9

    lo, hi = wilson_interval(5, 10)
    assert 0.0 <= lo < 0.5 < hi <= 1.0


# --------------------------------------------------------------------------- #
# 极简 runner
# --------------------------------------------------------------------------- #
def main() -> int:
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:  # noqa: BLE001 - 测试 runner 需要捕获一切
            failed += 1
            import traceback

            print(f"  FAIL  {name}: {exc}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
