"""客户端数据划分与批处理 (论文 Sec. 3 "Federated setting" 与 Sec. 5.1 Protocol)。

阅读顺序: 第 5 步 / 共 18 步 —— 见 READING_ORDER.md

论文设定: :math:`K` 个 operator (client) 各自持有一份**私有的**带标签 episode 图集合
:math:`D_k = \\{(G, y)\\}`，客户端之间互不共享原始图。客户端分布是 non-IID 的，
论文用 **label-skew purity** :math:`p` 建模:

    a fraction :math:`p` of each client's episodes share its dominant label

本模块实现了:

1. 按 ``domain`` 生成数据，每个 domain 一个客户端；
2. 用 **拒绝采样** 精确构造标签偏斜纯度为 :math:`p` 的训练集；
3. 每个客户端额外持有 IID 的 val / test 集 (val 用于联邦校准，test 用于公平评测)；
4. 把 :class:`EpisodeGraph` 列表 padding 成规则张量的 collate 函数。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import torch

from .config import Config
from .encoder import FrozenEncoder
from .graph import EpisodeGraph, build_graphs
from .mas_simulator import Episode, MASSimulator


@dataclass
class ClientData:
    """单个客户端 (operator) 的私有数据。"""

    client_id: int
    domain: int
    train: List[EpisodeGraph]
    val: List[EpisodeGraph]
    test: List[EpisodeGraph]
    dominant_label: int

    @property
    def n_train_nodes(self) -> int:
        """参与聚合加权的节点总数 :math:`N_k` (论文 Eq. 6)。"""
        return int(sum(g.n_agents for g in self.train))

    def positive_rate(self, split: str = "train") -> float:
        """该客户端某个 split 上 unsafe (label=1) 的比例，用于检查标签偏斜。"""
        graphs = getattr(self, split)
        if not graphs:
            return float("nan")
        return float(np.mean([g.label for g in graphs]))


def _select_with_purity(episodes: Sequence[Episode], n: int, dominant: int,
                        purity: float, rng: np.random.Generator) -> List[Episode]:
    """从候选池中选出 ``n`` 个 episode，使 ``purity`` 比例带 ``dominant`` 标签。"""
    dom_pool = [e for e in episodes if e.label == dominant]
    oth_pool = [e for e in episodes if e.label != dominant]

    n_dom = int(round(n * purity))
    n_oth = n - n_dom

    # 池子不够时按可用量回退 (并允许两者混用)，保证函数总能返回 n 个
    if len(dom_pool) < n_dom or len(oth_pool) < n_oth:
        chosen = list(dom_pool) + list(oth_pool)
        rng.shuffle(chosen)
        return chosen[:n]

    idx_dom = rng.choice(len(dom_pool), size=n_dom, replace=False)
    idx_oth = rng.choice(len(oth_pool), size=n_oth, replace=False)
    out = [dom_pool[i] for i in idx_dom] + [oth_pool[i] for i in idx_oth]
    rng.shuffle(out)
    return out


@dataclass
class FederatedData:
    """一次仿真的完整联邦数据视图。"""

    clients: List[ClientData]
    domain_ids: List[int]                       # 参与联邦的目标域
    test_by_domain: List[List[EpisodeGraph]]    # 与 domain_ids 对齐
    same_domain: bool = True


def _take(pool: List[Episode], n: int, rng: np.random.Generator) -> List[Episode]:
    """从池中**无放回**抽取 n 个 episode (保证客户端之间数据不重叠)。"""
    n = min(n, len(pool))
    idx = rng.choice(len(pool), size=n, replace=False)
    chosen = [pool[i] for i in idx]
    for i in sorted(idx, reverse=True):
        pool.pop(i)
    return chosen


def build_federated_data(cfg: Config, encoder: FrozenEncoder,
                         sim: MASSimulator, seed: int | None = None,
                         same_domain: bool = True) -> FederatedData:
    """构造 K 个客户端的数据。

    :param same_domain:
        ``True`` (论文 Table 1 / Q1 的设定) —— K 个 operator 都处于**同一个业务域**，
        各自持有一份 non-IID (标签偏斜 + 体量不均) 的私有切片。此时 :math:`D=1`，
        Eq. (6) 退化为 FedAvg，FGLGuard 的增益来自**近端目标** Eq. (5)。
        ``False`` (论文 Q3 的设定) —— K 个 operator 分属**不同业务域**，
        用于验证"跨域联邦能逼近多域池化"。
    """
    rng = np.random.default_rng(cfg.sim.seed if seed is None else seed)
    purity = cfg.fed.label_skew_purity
    K = cfg.fed.n_clients
    mult = cfg.sim.client_size_multipliers
    dom_labels = cfg.fed.client_dominant_labels

    def size_of(k: int) -> int:
        """第 k 个客户端的训练集大小 = 基准量 x 体量乘子 (循环取用乘子表)。"""
        return max(8, int(cfg.sim.n_train_per_client * mult[k % len(mult)]))

    clients: List[ClientData] = []

    if same_domain:
        # ---------------- 同域: 一个域，K 个私有切片 ---------------- #
        domain = 0
        total_needed = int(sum(size_of(k) for k in range(K)) * 3) + 64
        pool = sim.generate(domain, total_needed, rng)
        pos_pool = [e for e in pool if e.label == 1]
        neg_pool = [e for e in pool if e.label != 1]
        rng.shuffle(pos_pool)
        rng.shuffle(neg_pool)

        for k in range(K):
            n_k = size_of(k)
            dominant = int(dom_labels[k % len(dom_labels)])
            n_dom = int(round(n_k * purity))
            n_oth = n_k - n_dom
            take_dom = _take(pos_pool if dominant == 1 else neg_pool, n_dom, rng)
            take_oth = _take(neg_pool if dominant == 1 else pos_pool, n_oth, rng)
            train_eps = take_dom + take_oth
            rng.shuffle(train_eps)

            clients.append(ClientData(
                client_id=k, domain=domain,
                train=build_graphs(train_eps, encoder, cfg.model.self_loops),
                val=build_graphs(sim.generate(domain, cfg.sim.n_val_per_client, rng),
                                 encoder, cfg.model.self_loops),
                test=[],  # 同域模式下所有客户端共享同一个域的测试集
                dominant_label=dominant,
            ))
        test = build_graphs(sim.generate(domain, cfg.sim.n_test_per_client, rng),
                            encoder, cfg.model.self_loops)
        for c in clients:
            c.test = test
        return FederatedData(clients=clients, domain_ids=[domain],
                             test_by_domain=[test], same_domain=True)

    # ---------------- 跨域: 一人一域 ---------------- #
    D = cfg.vocab.n_target_domains
    domain_ids: List[int] = []
    test_by_domain: List[List[EpisodeGraph]] = []
    for k in range(K):
        domain = k % D
        dominant = int(dom_labels[k % len(dom_labels)])

        n_tr = size_of(k)
        n_va = cfg.sim.n_val_per_client
        n_te = cfg.sim.n_test_per_client

        pool = sim.generate(domain, int(n_tr * 8), rng)
        train_eps = _select_with_purity(pool, n_tr, dominant, purity, rng)
        val_eps = sim.generate(domain, n_va, rng)
        test_eps = sim.generate(domain, n_te, rng)

        clients.append(ClientData(
            client_id=k, domain=domain,
            train=build_graphs(train_eps, encoder, cfg.model.self_loops),
            val=build_graphs(val_eps, encoder, cfg.model.self_loops),
            test=build_graphs(test_eps, encoder, cfg.model.self_loops),
            dominant_label=dominant,
        ))
        domain_ids.append(domain)
        test_by_domain.append(clients[-1].test)

    return FederatedData(clients=clients, domain_ids=domain_ids,
                         test_by_domain=test_by_domain, same_domain=False)


# --------------------------------------------------------------------------- #
# 批处理
# --------------------------------------------------------------------------- #
@dataclass
class GraphBatch:
    """padding 后的规则张量批次 (小图场景下比 block-diagonal 拼接更简单可靠)。"""

    x: torch.Tensor            # (B, n_max, d)
    adj: torch.Tensor          # (B, n_max, n_max)
    edge: torch.Tensor         # (B, n_max, n_max, d)
    node_labels: torch.Tensor  # (B, n_max)
    node_mask: torch.Tensor    # (B, n_max) bool
    final_idx: torch.Tensor    # (B,)
    labels: torch.Tensor       # (B,) episode 级标签
    domains: torch.Tensor      # (B,)

    @property
    def n_graphs(self) -> int:
        """本批包含的 episode 图数量。"""
        return self.x.shape[0]


def collate_graphs(graphs: Sequence[EpisodeGraph], device: str = "cpu") -> GraphBatch:
    """把一批 :class:`EpisodeGraph` 补齐成 :class:`GraphBatch`。"""
    if not graphs:
        raise ValueError("collate_graphs received an empty sequence")
    B = len(graphs)
    n_max = max(g.n_agents for g in graphs)
    d = graphs[0].x.shape[1]

    x = np.zeros((B, n_max, d), dtype=np.float32)
    adj = np.zeros((B, n_max, n_max), dtype=np.float32)
    edge = np.zeros((B, n_max, n_max, d), dtype=np.float32)
    node_labels = np.zeros((B, n_max), dtype=np.float32)
    node_mask = np.zeros((B, n_max), dtype=bool)
    final_idx = np.zeros(B, dtype=np.int64)
    labels = np.zeros(B, dtype=np.float32)
    domains = np.zeros(B, dtype=np.int64)

    for b, g in enumerate(graphs):
        n = g.n_agents
        x[b, :n] = g.x
        adj[b, :n, :n] = g.adj
        edge[b, :n, :n] = g.edge_feat
        node_labels[b, :n] = g.node_labels
        node_mask[b, :n] = True
        final_idx[b] = g.final_agent
        labels[b] = g.label
        domains[b] = g.domain

    to = lambda a, dt=torch.float32: torch.as_tensor(a, dtype=dt, device=device)
    return GraphBatch(
        x=to(x), adj=to(adj), edge=to(edge),
        node_labels=to(node_labels), node_mask=torch.as_tensor(node_mask, device=device),
        final_idx=to(final_idx, torch.int64), labels=to(labels), domains=to(domains, torch.int64),
    )


def iterate_batches(graphs: Sequence[EpisodeGraph], batch_size: int,
                    shuffle: bool, rng: np.random.Generator,
                    device: str = "cpu"):
    """按 batch 迭代。"""
    idx = np.arange(len(graphs))
    if shuffle:
        rng.shuffle(idx)
    for start in range(0, len(idx), batch_size):
        chunk = idx[start:start + batch_size]
        yield collate_graphs([graphs[i] for i in chunk], device=device)
