"""Episode -> 属性图 :math:`G = (X, A, E)` 的构建 (论文 Sec. 3)。

论文原文:

    it derives node features by aggregating incoming edges, leaving nothing for agents
    that sparse topologies isolate, so we anchor each node's feature to the agent's own
    utterance history, :math:`x_i = \\frac{1}{T}\\sum_t \\phi(u_i^t)`.

    edge :math:`(i \\to j)` carries the receiving agent's per-round embedding sequence
    :math:`e_{ij} = [\\phi(u_j^1); \\dots; \\phi(u_j^T)] \\in \\mathbb{R}^{T\\times d}`.

    Labels ... we broadcast the episode label to all nodes, :math:`y_i = y\\ \\forall i`.

本模块严格遵守以上三点，只做一处**工程化**处理: 默认加入自环 (``self_loops=True``)。
原因见 README「与论文的差异」: 论文的注意力只对**入边**归一化
(:math:`\\sum_{i' : A_{i'j}=1}`)，若某节点没有任何入边，其隐藏状态退化为常数，
打分恒为 0.5；而 final-answer agent 恰恰常处于图的下游/近似孤立位置。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from .encoder import FrozenEncoder
from .mas_simulator import Episode


@dataclass
class EpisodeGraph:
    """属性图 :math:`G = (X, A, E)`。

    Attributes
    ----------
    x : np.ndarray
        ``(n, d)`` 节点特征。``x[i]`` 是 agent ``i`` 自身 utterance 历史的时间均值。
    adj : np.ndarray
        ``(n, n)`` 邻接矩阵 (已含自环)。
    edge_feat : np.ndarray
        ``(n, n, d)`` 边特征 :math:`\\psi(e_{ij})`；无边处为 0。
        ``edge_feat[i, j]`` 承载**接收方** :math:`a_j` 的逐轮嵌入均值。
    label : int
        episode 级标签。
    node_labels : np.ndarray
        ``(n,)`` 逐节点标签。按论文默认**广播** episode 标签。
    node_infected : np.ndarray
        ``(n,)`` 真实感染状态 (仅用于诊断 off-the-shelf 检测器，训练时不使用)。
    final_agent : int
    domain : int
    """

    x: np.ndarray
    adj: np.ndarray
    edge_feat: np.ndarray
    label: int
    node_labels: np.ndarray
    node_infected: np.ndarray
    final_agent: int
    domain: int

    @property
    def n_agents(self) -> int:
        """节点数 (= 团队的 agent 数)。"""
        return self.x.shape[0]


def build_graph(episode: Episode, encoder: FrozenEncoder,
                self_loops: bool = True,
                node_label_mode: str = "broadcast") -> EpisodeGraph:
    """把 :class:`Episode` 转成 :class:`EpisodeGraph`。

    :param node_label_mode: ``"broadcast"`` 表示 :math:`y_i = y` (论文默认，
        AUROC 0.708)；``"final"`` 表示只有 final-answer agent 带标签 (论文对比项，
        AUROC 0.705)。两者在论文中几乎无差别。
    """
    T = episode.n_rounds
    n = episode.n_agents
    d = encoder.dim

    x = np.zeros((n, d), dtype=np.float32)
    edge_feat = np.zeros((n, n, d), dtype=np.float32)

    # 先算出每个 agent 每一轮的嵌入，node 与 edge 特征都复用
    per_round = np.zeros((T, n, d), dtype=np.float32)
    for t in range(T):
        for i in range(n):
            per_round[t, i] = encoder.embed_text(episode.utterances[t][i])

    # ---- 节点特征: 自身 utterance 历史的时间均值 (论文核心改动) ---- #
    x = per_round.mean(axis=0)  # (n, d)

    # ---- 边特征: 接收方 j 的逐轮嵌入序列 -> psi = 时间均值 ---- #
    psi = per_round.mean(axis=0)  # (n, d)，psi(e_ij) 只依赖 j
    adj = episode.adjacency.astype(np.float32).copy()
    if self_loops:
        np.fill_diagonal(adj, 1.0)
    for i in range(n):
        for j in range(n):
            if adj[i, j] > 0:
                edge_feat[i, j] = psi[j]

    if node_label_mode == "broadcast":
        node_labels = np.full(n, episode.label, dtype=np.float32)
    elif node_label_mode == "final":
        node_labels = np.zeros(n, dtype=np.float32)
        node_labels[episode.final_agent] = float(episode.label)
    else:  # pragma: no cover
        raise ValueError(f"unknown node_label_mode: {node_label_mode}")

    return EpisodeGraph(
        x=x,
        adj=adj,
        edge_feat=edge_feat,
        label=int(episode.label),
        node_labels=node_labels,
        node_infected=episode.infected[-1].astype(np.float32),
        final_agent=int(episode.final_agent),
        domain=int(episode.domain),
    )


def build_graphs(episodes: List[Episode], encoder: FrozenEncoder,
                 self_loops: bool = True,
                 node_label_mode: str = "broadcast") -> List[EpisodeGraph]:
    """批量构建属性图，参数含义见 :func:`build_graph`。"""
    return [build_graph(e, encoder, self_loops, node_label_mode) for e in episodes]


def corroborated_scores(scores: np.ndarray, adj: np.ndarray) -> np.ndarray:
    """论文 Eq. (8): :math:`\\hat s_j = \\max(s_j, \\max_{i: A_{ij}=1} s_i)`。

    注意这里的 ``adj`` 应当是**原始拓扑** (含自环也无妨，因为 :math:`s_j` 已在 max 内)。
    """
    n = scores.shape[0]
    out = np.array(scores, dtype=np.float64, copy=True)
    for j in range(n):
        incoming = np.nonzero(adj[:, j])[0]
        if incoming.size:
            out[j] = max(out[j], float(np.max(scores[incoming])))
    return out
