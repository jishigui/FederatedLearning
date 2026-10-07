"""联邦近端训练与域均衡聚合 (论文 Sec. 4.2, Eq. 5-6)。

客户端本地目标 (Eq. 5)::

    min_theta  L_k(theta) = sum_{(G,y) in D_k} sum_{i in G} BCE(s_i(theta), y_i)
                            + (mu/2) * || theta - theta^(r) ||_2^2

服务端聚合 (Eq. 6)::

    theta^(r+1) = sum_k alpha_k theta_k,
    alpha_k = (1/D) * N_k / sum_{k' in C_{d(k)}} N_{k'}

即**域内按数据量加权、域间等质量**。当 :math:`D = 1` 时退化为 FedAvg
(论文 supplementary Proposition 1)。

本模块同时实现三个对照臂，且保证**梯度预算匹配** (论文 5.1:
"matched gradient budgets so arms differ only in how updates are shared"):

======================  ====================================================
臂                       说明
======================  ====================================================
``local_only``          各客户端完全不通信，独立训练
``fedavg``              mu=0、纯按体量加权 (体量大者主导)
``fglguard``            mu=0.01 + 域均衡聚合 (论文完整方法)
``centralized``         把所有客户端数据汇聚到一处训练 (论文中的"上限"参照)
``source_only``         在**异域**合成注入语料上训练 (G-Safeguard off-the-shelf 类比)
======================  ====================================================
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .config import Config
from .data import ClientData, GraphBatch, collate_graphs
from .graph import EpisodeGraph
from .metrics import auroc
from .model import FGLGuardScorer, node_bce_loss


# --------------------------------------------------------------------------- #
# 工具
# --------------------------------------------------------------------------- #
def prebatch(graphs: Sequence[EpisodeGraph], batch_size: int,
             device: str = "cpu") -> List[GraphBatch]:
    """把图集合预先 collate 成固定批次，训练时只打乱批次顺序 (小图场景下更快)。"""
    batches: List[GraphBatch] = []
    for start in range(0, len(graphs), batch_size):
        chunk = graphs[start:start + batch_size]
        if chunk:
            batches.append(collate_graphs(chunk, device=device))
    return batches


def episode_scores_of(model: FGLGuardScorer, batches: Sequence[GraphBatch],
                      corroborate: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """对批次逐图打分。

    :returns: ``(episode_scores, episode_labels, node_probs)``
        ``episode_scores`` 在 final-answer agent 上读出；``corroborate=True`` 时
        用论文 Eq. (8) 的佐证分数。
    """
    model.eval()
    all_scores: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    all_nodes: List[np.ndarray] = []
    with torch.no_grad():
        for batch in batches:
            _, probs = model(batch.x, batch.adj, batch.edge)   # (B, n)
            if corroborate:
                # Eq. (8): s_hat_j = max(s_j, max_{i: A_ij=1} s_i)
                # batch.adj 已含自环，因此 max 内天然包含 s_j 本身
                mask = (batch.adj > 0)
                neigh = torch.where(mask, probs.unsqueeze(1), torch.full_like(
                    probs.unsqueeze(1), -1.0))
                probs = torch.maximum(probs, neigh.max(dim=1).values)
            ep = probs.gather(1, batch.final_idx.view(-1, 1)).squeeze(1)
            all_scores.append(ep.cpu().numpy())
            all_labels.append(batch.labels.cpu().numpy())
            all_nodes.append(probs.cpu().numpy())
    return (np.concatenate(all_scores) if all_scores else np.zeros(0),
            np.concatenate(all_labels) if all_labels else np.zeros(0),
            np.concatenate(all_nodes, axis=0) if all_nodes else np.zeros((0, 0)))


# --------------------------------------------------------------------------- #
# 客户端
# --------------------------------------------------------------------------- #
@dataclass
class FederatedClient:
    """一个 operator (组织)。原始 episode 图**永不离开**该对象。"""

    data: ClientData
    cfg: Config
    device: str = "cpu"

    def __post_init__(self) -> None:
        self.train_batches = prebatch(self.data.train, self.cfg.fed.batch_size, self.device)
        self.val_batches = prebatch(self.data.val, self.cfg.fed.batch_size, self.device)
        self.test_batches = prebatch(self.data.test, self.cfg.fed.batch_size, self.device)

    @property
    def client_id(self) -> int:
        """该 operator 的编号 (用于随机种子与日志)。"""
        return self.data.client_id

    @property
    def domain(self) -> int:
        """该 operator 所属业务域，与论文 Eq.(6) 的 :math:`d(k)` 对应。"""
        return self.data.domain

    @property
    def n_nodes(self) -> int:
        """参与聚合加权的训练节点数 :math:`N_k` (论文 Eq. 6)。"""
        return self.data.n_train_nodes

    def local_train(self, model: FGLGuardScorer,
                    global_state: Dict[str, torch.Tensor],
                    mu: float,
                    rng: np.random.Generator,
                    epochs: Optional[int] = None) -> float:
        """执行 Eq. (5) 的本地近端优化，**原地**更新 ``model``。"""
        cen = self.cfg.fed
        epochs = cen.local_epochs if epochs is None else epochs

        model.train()
        optimizer = torch.optim.SGD(
            model.parameters(), lr=cen.lr, momentum=0.9,
            weight_decay=cen.weight_decay)

        # 冻结一份 theta^(r) 用于近端项
        anchor = [global_state[name].detach().clone() for name, _ in model.named_parameters()]

        last_loss = float("nan")
        for _ in range(epochs):
            order = rng.permutation(len(self.train_batches))
            for bi in order:
                batch = self.train_batches[bi]
                optimizer.zero_grad(set_to_none=True)
                logits, _ = model(batch.x, batch.adj, batch.edge)
                loss = node_bce_loss(logits, batch.node_labels, batch.node_mask)
                if mu > 0:
                    prox = torch.zeros((), device=self.device)
                    for p, g in zip(model.parameters(), anchor):
                        prox = prox + ((p - g) ** 2).sum()
                    loss = loss + 0.5 * mu * prox
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                last_loss = float(loss.detach().cpu())
        return last_loss


# --------------------------------------------------------------------------- #
# 服务端
# --------------------------------------------------------------------------- #
def aggregate_states(states: Sequence[Dict[str, torch.Tensor]],
                     weights: Sequence[float]) -> Dict[str, torch.Tensor]:
    """按权重做参数平均。"""
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    out: Dict[str, torch.Tensor] = {}
    for name in states[0]:
        acc = None
        for wi, st in zip(w, states):
            term = st[name].detach().to(torch.float64) * float(wi)
            acc = term if acc is None else acc + term
        out[name] = acc.to(states[0][name].dtype)
    return out


def domain_balanced_weights(clients: Sequence[FederatedClient]) -> np.ndarray:
    """论文 Eq. (6) 的 :math:`\\alpha_k`。"""
    n_nodes = np.array([c.n_nodes for c in clients], dtype=np.float64)
    domains = np.array([c.domain for c in clients], dtype=np.int64)
    uniq = np.unique(domains)
    D = len(uniq)
    alpha = np.zeros(len(clients), dtype=np.float64)
    for d in uniq:
        members = np.nonzero(domains == d)[0]
        tot = n_nodes[members].sum()
        tot = tot if tot > 0 else 1.0
        alpha[members] = (1.0 / D) * n_nodes[members] / tot
    return alpha / alpha.sum()


def volume_weights(clients: Sequence[FederatedClient]) -> np.ndarray:
    """纯体量加权 (FedAvg)。"""
    n_nodes = np.array([c.n_nodes for c in clients], dtype=np.float64)
    if n_nodes.sum() <= 0:
        return np.full(len(clients), 1.0 / len(clients))
    return n_nodes / n_nodes.sum()


# --------------------------------------------------------------------------- #
# 训练主循环
# --------------------------------------------------------------------------- #
@dataclass
class TrainResult:
    """训练产物。"""

    model: FGLGuardScorer
    history: Dict[str, List[float]] = field(default_factory=dict)
    tau: float = 0.5
    wall_time: float = 0.0
    # 服务端可合法持有的校准素材: 各客户端本地上算出的标量分数 + 标签
    val_scores: Optional[np.ndarray] = None
    val_labels: Optional[np.ndarray] = None


def make_model(cfg: Config, node_dim: int, edge_dim: int, seed: int,
               device: str = "cpu") -> FGLGuardScorer:
    """按固定种子新建一个打分器 (联邦每轮/每个客户端都用它从同一初值出发)。"""
    torch.manual_seed(seed)
    return FGLGuardScorer(node_dim, edge_dim, cfg.model).to(device)


@torch.no_grad()
def evaluate_auroc(model: FGLGuardScorer, batches: Sequence[GraphBatch],
                   corroborate: bool = False) -> float:
    """在给定批次上计算 episode 级 AUROC (默认用原始分数，不启用佐证)。"""
    scores, labels, _ = episode_scores_of(model, batches, corroborate=corroborate)
    return auroc(scores, labels)


def collect_validation(model: FGLGuardScorer,
                       clients: Sequence[FederatedClient],
                       corroborate: bool = False
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """把各客户端**本地计算**的验证分数池化，供服务端校准使用 (论文 Eq. 7)。

    这一步只交换标量分数与标签，不涉及任何原始 prompt / 消息 / episode 图。
    """
    scores, labels = [], []
    for c in clients:
        s, y, _ = episode_scores_of(model, c.val_batches, corroborate=corroborate)
        scores.append(s)
        labels.append(y)
    if not scores:
        return np.zeros(0), np.zeros(0)
    return np.concatenate(scores), np.concatenate(labels)


def train_federated(clients: Sequence[FederatedClient], cfg: Config,
                    node_dim: int, edge_dim: int, seed: int,
                    mu: Optional[float] = None,
                    domain_balanced: Optional[bool] = None,
                    verbose: bool = False) -> TrainResult:
    """联邦训练主循环 (论文 Algorithm 1 的对应实现)。

    :param mu: 近端系数；``None`` 时取 ``cfg.fed.proximal_mu``。
    :param domain_balanced: ``None`` 时取 ``cfg.fed.domain_balanced``。
        ``mu=0`` 且 ``domain_balanced=False`` 即标准 FedAvg。
    """
    mu = cfg.fed.proximal_mu if mu is None else mu
    domain_balanced = cfg.fed.domain_balanced if domain_balanced is None else domain_balanced
    device = clients[0].device

    model = make_model(cfg, node_dim, edge_dim, seed, device)
    rng = np.random.default_rng(seed)

    # 服务端持有的全局验证分数池 (只有标量与标签，符合隐私约束)
    pooled_val_scores: List[np.ndarray] = []
    pooled_val_labels: List[np.ndarray] = []

    history: Dict[str, List[float]] = {"round": [], "mean_client_loss": [],
                                       "val_auroc": [], "client_drift": []}
    t0 = time.time()

    for r in range(cfg.fed.n_rounds):
        global_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        client_states: List[Dict[str, torch.Tensor]] = []
        losses: List[float] = []
        drifts: List[float] = []

        for c in clients:
            local_model = make_model(cfg, node_dim, edge_dim, seed, device)
            local_model.load_state_dict(global_state)
            loss = c.local_train(local_model, global_state, mu, rng)
            losses.append(loss)
            with torch.no_grad():
                drift = float(sum(
                    ((lp - gp) ** 2).sum().item()
                    for lp, gp in zip(local_model.parameters(), model.parameters())))
            drifts.append(drift)
            client_states.append({k: v.detach().clone()
                                  for k, v in local_model.state_dict().items()})

        weights = (domain_balanced_weights(clients) if domain_balanced
                   else volume_weights(clients))
        new_state = aggregate_states(client_states, weights)
        model.load_state_dict(new_state)

        # 全局验证: 各客户端在本地上算分，只把标量分数 + 标签交给服务端
        pooled_val_scores, pooled_val_labels = [], []
        per_domain_auroc: List[float] = []
        for c in clients:
            s, y, _ = episode_scores_of(model, c.val_batches, corroborate=False)
            pooled_val_scores.append(s)
            pooled_val_labels.append(y)
            per_domain_auroc.append(auroc(s, y))

        # 报告口径用 **各域 AUROC 的宏平均** (与论文 Table 1 逐域汇报一致);
        # 校准则使用池化后的 (分数, 标签) 对 (论文 Eq. 7 的 V)。
        val_auroc = float(np.nanmean(per_domain_auroc))

        history["round"].append(r + 1)
        history["mean_client_loss"].append(float(np.mean(losses)))
        history["val_auroc"].append(val_auroc)
        history["client_drift"].append(float(np.mean(drifts)))
        if verbose:
            print(f"[round {r + 1:02d}/{cfg.fed.n_rounds}] "
                  f"loss={np.mean(losses):.4f}  macro_val_auroc={val_auroc:.4f}  "
                  f"drift={np.mean(drifts):.2e}")

    return TrainResult(
        model=model, history=history, wall_time=time.time() - t0,
        val_scores=np.concatenate(pooled_val_scores) if pooled_val_scores else None,
        val_labels=np.concatenate(pooled_val_labels) if pooled_val_labels else None,
    )


def train_local_only(clients: Sequence[FederatedClient], cfg: Config,
                     node_dim: int, edge_dim: int, seed: int
                     ) -> List[TrainResult]:
    """Local-only 基线: 各客户端完全独立训练，梯度预算与联邦一致。

    联邦中每个客户端累计 :math:`R \\times E` 个 local epoch，因此这里直接给
    ``cfg.fed.local_only_epochs`` (默认 ``R*E``) 个 epoch。
    """
    device = clients[0].device
    out: List[TrainResult] = []
    epochs = cfg.fed.local_only_epochs or cfg.fed.n_rounds * cfg.fed.local_epochs
    for c in clients:
        rng = np.random.default_rng(seed + c.client_id)
        model = make_model(cfg, node_dim, edge_dim, seed, device)
        state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        t0 = time.time()
        c.local_train(model, state, mu=0.0, rng=rng, epochs=epochs)
        vs, vl, _ = episode_scores_of(model, c.val_batches)
        out.append(TrainResult(model=model, wall_time=time.time() - t0,
                               val_scores=vs, val_labels=vl))
    return out


def train_supervised(graphs: Sequence[EpisodeGraph], cfg: Config,
                     node_dim: int, edge_dim: int, seed: int,
                     epochs: int, device: str = "cpu",
                     model: Optional[FGLGuardScorer] = None) -> TrainResult:
    """在**任意图集合**上做集中式监督训练 (用于 centralized / off-the-shelf 基线)。"""
    batches = prebatch(graphs, cfg.fed.batch_size, device)
    if model is None:
        model = make_model(cfg, node_dim, edge_dim, seed, device)
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.fed.lr, momentum=0.9)
    rng = np.random.default_rng(seed)
    history: Dict[str, List[float]] = {"epoch": [], "loss": []}

    t0 = time.time()
    for ep in range(epochs):
        model.train()
        order = rng.permutation(len(batches))
        tot, cnt = 0.0, 0
        for bi in order:
            batch = batches[bi]
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(batch.x, batch.adj, batch.edge)
            loss = node_bce_loss(logits, batch.node_labels, batch.node_mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            tot += float(loss.detach().cpu())
            cnt += 1
        history["epoch"].append(ep + 1)
        history["loss"].append(tot / max(cnt, 1))
    return TrainResult(model=model, history=history, wall_time=time.time() - t0)


def train_centralized(clients: Sequence[FederatedClient], cfg: Config,
                      node_dim: int, edge_dim: int, seed: int,
                      extra_graphs: Optional[Sequence[EpisodeGraph]] = None
                      ) -> TrainResult:
    """Centralized 参照: 把（本不该汇聚的）原始图全部池化后训练。

    训练 epoch 数与联邦的累计本地 epoch 数一致，使对比公平
    (论文 5.1: "matched gradient budgets so arms differ only in how updates are shared")。
    """
    device = clients[0].device
    pooled: List[EpisodeGraph] = []
    for c in clients:
        pooled.extend(c.data.train)
    if extra_graphs:
        pooled.extend(extra_graphs)

    res = train_supervised(pooled, cfg, node_dim, edge_dim, seed,
                           epochs=cfg.fed.n_rounds * cfg.fed.local_epochs,
                           device=device)
    vs, vl = collect_validation(res.model, clients)
    res.val_scores, res.val_labels = vs, vl
    return res
