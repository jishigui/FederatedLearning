"""实验脚手架: 构造仿真世界、训练 off-the-shelf 基线、通用评测。"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from fglguard.config import Config
from fglguard.data import FederatedData, build_federated_data
from fglguard.encoder import FrozenEncoder
from fglguard.federated import (FederatedClient, TrainResult, evaluate_auroc,
                                prebatch, train_supervised)
from fglguard.graph import EpisodeGraph, build_graphs
from fglguard.mas_simulator import MASSimulator


class World:
    """一次仿真的全部对象: 编码器 / 仿真器 / 客户端 / 测试批次。"""

    def __init__(self, cfg: Config, seed: Optional[int] = None,
                 same_domain: bool = True) -> None:
        self.cfg = cfg
        self.encoder = FrozenEncoder(cfg.vocab, cfg.encoder)
        self.sim = MASSimulator(cfg.vocab, cfg.sim, self.encoder)
        self.data: FederatedData = build_federated_data(
            cfg, self.encoder, self.sim, seed=seed, same_domain=same_domain)
        self.clients: List[FederatedClient] = [
            FederatedClient(c, cfg, cfg.device) for c in self.data.clients]
        self.test_batches = [prebatch(t, cfg.fed.batch_size, cfg.device)
                             for t in self.data.test_by_domain]

    @property
    def node_dim(self) -> int:
        """送入 GAT 的节点特征维度 (= 冻结编码器维度 :math:`d`)。"""
        return self.encoder.dim

    @property
    def edge_dim(self) -> int:
        """送入 GAT 的边特征维度 (:math:`\\psi(e_{ij})` 的维度)。"""
        return self.encoder.dim

    @property
    def domains(self) -> List[int]:
        """参与联邦的目标域 id 列表。"""
        return list(self.data.domain_ids)

    def clients_of(self, domain: int) -> List[FederatedClient]:
        """返回属于指定业务域的全部 operator。"""
        return [c for c in self.clients if c.domain == domain]

    def per_domain_auroc(self, model, corroborate: bool = False) -> List[float]:
        """对**单个全局模型**逐域评测 (论文 Table 1 的 "Organic episodes" 列)。"""
        return [evaluate_auroc(model, tb, corroborate=corroborate)
                for tb in self.test_batches]

    def local_only_auroc(self, models: Sequence) -> List[float]:
        """Local-only: 每个域的分数取该域内各客户端本地模型的平均。"""
        out: List[float] = []
        for di, domain in enumerate(self.data.domain_ids):
            vals = [evaluate_auroc(m.model, self.test_batches[di])
                    for m, c in zip(models, self.clients) if c.domain == domain]
            out.append(float(np.mean(vals)) if vals else float("nan"))
        return out


def build_source_corpus(cfg: Config, sim: MASSimulator, enc: FrozenEncoder,
                        n_episodes: int = 1200, seed: int = 7) -> List[EpisodeGraph]:
    """构造 off-the-shelf 基线的训练语料。

    刻意做成**三重不匹配**，对应论文中 "an off-the-shelf G-Safeguard detector,
    trained per its official recipe on synthetic tool-injection data" 的设定:

    1. **异域**: 使用保留的 source domain id，其良性/攻击词 block 与目标域完全不重合
       -> 直接产生 distribution shift;
    2. **异构注入配方**: payload 直接打在 final-answer agent 上 (目标域的真实攻击是
       从上游沿边传播过来的) -> 学到的判别方向在目标域上失效;
    3. **词表完全不共享**: 源域的攻击 token 一个都不来自跨域共享风险 block
       (``shared_risk_ratio = 0``) -> 连"泛化的风险标记"这一弱信号也拿不到。
    """
    src_sim = MASSimulator(cfg.vocab, _with_shared_ratio(cfg.sim, 0.0), enc)
    rng = np.random.default_rng(seed)
    eps = src_sim.generate(cfg.vocab.source_domain, n_episodes, rng,
                           injection_mode="direct")
    return build_graphs(eps, enc, cfg.model.self_loops, node_label_mode="broadcast")


def _with_shared_ratio(sim_cfg, ratio: float):
    """返回一个把 ``shared_risk_ratio`` 覆盖掉的 SimConfig 副本。"""
    import copy

    cfg = copy.deepcopy(sim_cfg)
    cfg.shared_risk_ratio = float(ratio)
    return cfg


def train_off_the_shelf(cfg: Config, sim: MASSimulator, enc: FrozenEncoder,
                        seed: int = 0) -> TrainResult:
    """在异域合成注入语料上训练一个检测器 (不做任何目标域适配)。"""
    graphs = build_source_corpus(cfg, sim, enc, seed=seed + 7)
    return train_supervised(graphs, cfg, enc.dim, enc.dim, seed=seed,
                            epochs=cfg.fed.n_rounds * cfg.fed.local_epochs,
                            device=cfg.device)


def per_domain_auroc(model, clients: Sequence[FederatedClient],
                     corroborate: bool = False) -> List[float]:
    """逐域 AUROC (论文 Table 1 的 "Organic episodes" 列在各域上的展开)。"""
    return [evaluate_auroc(model, c.test_batches, corroborate=corroborate)
            for c in clients]


def macro(values: Sequence[float]) -> float:
    """忽略 NaN 的均值 (论文 "Avg." 列的口径)。"""
    arr = np.asarray(list(values), dtype=float)
    return float(np.nanmean(arr)) if arr.size else float("nan")


def make_world(cfg: Config, seed: Optional[int] = None, same_domain: bool = True):
    """向后兼容的便捷构造器 (返回 World)。"""
    return World(cfg, seed=seed, same_domain=same_domain)
