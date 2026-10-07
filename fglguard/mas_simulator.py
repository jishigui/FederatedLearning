"""多智能体系统 (MAS) episode 仿真器 + 免费 judge 标签。

阅读顺序: 第 3 步 / 共 18 步 —— 见 READING_ORDER.md

论文 (Sec. 3) 的场景:
    n 个 LLM agent 在有向拓扑 :math:`A \\in \\{0,1\\}^{n\\times n}` 上协作最多 R 轮;
    每个 episode 产出一组 utterance; 一个 **judge** 对整条交互记录打出二分类标签。

真实语料 (Agent-SafetyBench / R-Judge / AgentDojo) 需要跑 LLM，本项目用**结构化仿真**
复现其中的关键机制:

1. **蠕虫式传播** (prompt infection): 被感染的 agent 在其 utterance 中带上攻击 payload，
   其下游邻居以 ``propagation_prob`` 的概率在下一轮被感染。
2. **免费 judge 标签**: episode 标签 :math:`y=1` 当且仅当 payload **抵达 final-answer agent**
   (即攻击成功)。这类标签在论文的语境里由 runtime shield model 免费提供。
3. **域 (domain)**: 不同组织持有不同业务域的数据，良性/攻击词表都不重合。

这样生成的数据天然具备论文强调的三个性质: 私有的、non-IID 的、攻击分布各异的。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import SimConfig, VocabConfig
from .encoder import FrozenEncoder


@dataclass
class Episode:
    """一次完整任务执行产生的通信图。

    Attributes
    ----------
    domain : int
        该 episode 所属业务域 (等价于持有它的客户端/组织)。
    adjacency : np.ndarray
        ``(n, n)`` 的 0/1 矩阵，``A[i, j] = 1`` 表示 :math:`a_i \\to a_j` 有消息投递。
    utterances : list[list[list[int]]]
        形状 ``(T, n)`` 的 token id 嵌套列表。
    infected : np.ndarray
        ``(T, n)`` bool，标记某 agent 在某轮是否携带 payload。
    final_agent : int
        提交最终答案的 agent (episode 级指标从此节点读出)。
    label : int
        episode 标签 (1 = unsafe)，由 judge 对完整记录判定。
    intent : int
        该 episode 是否**预先注入**了攻击意图 (用于统计 ASR)。
    direct_exposure : bool
        final-answer agent **自身**是否在末轮被感染。为 False 时表示风险完全来自
        上游邻居 (即"自身话语看起来干净"的情形)。
    """

    domain: int
    adjacency: np.ndarray
    utterances: List[List[List[int]]]
    infected: np.ndarray
    final_agent: int
    label: int
    intent: int
    direct_exposure: bool = False

    @property
    def n_agents(self) -> int:
        """团队规模 :math:`n`。"""
        return self.adjacency.shape[0]

    @property
    def n_rounds(self) -> int:
        """实际执行的对话轮数 :math:`T` (论文最多 R 轮)。"""
        return len(self.utterances)

    def final_utterance(self) -> List[int]:
        """final-answer agent 在最后一轮的话语 (即被守卫的对象)。"""
        return self.utterances[-1][self.final_agent]

    def upstream_of(self, node: int) -> List[int]:
        """返回指向 ``node`` 的入边邻居 (上游)。"""
        return np.nonzero(self.adjacency[:, node])[0].tolist()

    def downstream_of(self, node: int) -> List[int]:
        """返回 ``node`` 指向的出边邻居 (下游)。"""
        return np.nonzero(self.adjacency[node, :])[0].tolist()


class MASSimulator:
    """可配置的 MAS episode 生成器。"""

    def __init__(self, vocab_cfg: VocabConfig, sim_cfg: SimConfig,
                 encoder: FrozenEncoder) -> None:
        self.vocab_cfg = vocab_cfg
        self.cfg = sim_cfg
        self.encoder = encoder

    # ------------------------------------------------------------------ #
    # 拓扑
    # ------------------------------------------------------------------ #
    def _sample_topology(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """随机有向拓扑，边密度 = ``edge_sparsity``。

        额外保证: 除 final-answer agent 外，每个节点至少有一条出边，
        避免出现完全孤立的"哑"节点 (论文中 n=3、稀疏度 0.2 时会出现，是检测最弱的点，
        这里保留一定稀疏性但不让图退化)。
        """
        s = self.cfg.edge_sparsity
        adj = (rng.random((n, n)) < s).astype(np.int8)
        np.fill_diagonal(adj, 0)
        # 消除 1-环，避免自反馈
        adj = np.triu(adj, 1) | np.tril(adj, 1).T
        for i in range(n):
            if adj[i].sum() == 0 and n > 1:
                j = int(rng.integers(0, n - 1))
                j = j if j < i else j + 1
                adj[i, j] = 1
        return adj

    def _final_agent(self, adj: np.ndarray) -> int:
        """final-answer agent = 入度最高的节点 (聚合证据并提交答案的角色)。"""
        in_deg = adj.sum(axis=0)
        return int(np.argmax(in_deg + 1e-6 * np.arange(adj.shape[0])[::-1]))

    # ------------------------------------------------------------------ #
    # Episode
    # ------------------------------------------------------------------ #
    def sample_episode(self, domain: int, rng: np.random.Generator,
                       force_intent: Optional[bool] = None,
                       injection_mode: str = "propagate") -> Episode:
        """生成一个 episode。

        :param force_intent: 若给定，则强制是否注入攻击意图
            (runtime 实验里用它分别构造"攻击集"和"良性集")。
        :param injection_mode:
            ``"propagate"`` (默认) —— 攻击从某个上游 agent 出发沿边蠕虫式传播，
            这正是目标域的真实分布，也是"内容风险常源于行动 agent 的上游"的来源；
            ``"direct"`` —— payload 直接打在最下游的 final-answer agent 上。
            后者用于构造 **off-the-shelf / 异域合成注入语料** (类比 G-Safeguard 的
            tool-injection recipe)，其学到的判别式与目标域的真实机制不匹配。
        """
        cfg = self.cfg
        n_lo, n_hi = cfg.n_agents_range
        n = int(rng.integers(n_lo, n_hi + 1))
        T = cfg.n_rounds

        adj = self._sample_topology(n, rng)
        final_agent = self._final_agent(adj)

        intent = bool(rng.random() < cfg.unsafe_ratio) if force_intent is None else force_intent
        # level[t, i] 是 agent i 在第 t 轮携带的 payload "强度" (源为 1.0，逐跳衰减)。
        # 用它替代纯布尔感染状态，使"下游载体"的自身话语只带微弱信号 —— 这与论文
        # "unsafe content originates upstream ... whose own text may look clean" 一致，
        # 也让纯局部的无监督检测器失去判别力。
        level = np.zeros((T, n), dtype=np.float32)
        utterances: List[List[List[int]]] = []

        source = -1
        if intent:
            if injection_mode == "direct":
                source = final_agent
            else:
                candidates = [i for i in range(n) if i != final_agent] or list(range(n))
                source = int(rng.choice(candidates))
            level[0, source] = 1.0

        for t in range(T):
            if t > 0:
                for i in range(n):
                    if level[t - 1, i] <= 0.0:
                        continue
                    level[t, i] = max(level[t, i], level[t - 1, i])  # 已感染者持续散播
                    for j in self._downstream(adj, i):
                        if rng.random() < cfg.propagation_prob:
                            child = level[t - 1, i] * cfg.propagation_decay
                            level[t, j] = max(level[t, j], child)

            round_utt: List[List[int]] = []
            n_tok = cfg.n_utterance_tokens
            for i in range(n):
                intensity = float(level[t, i])
                if intensity > 0.0:
                    n_attack = max(1, int(round(n_tok * cfg.attack_token_ratio * intensity)))
                    n_attack = min(n_attack, n_tok)
                    payload = self.encoder.sample_attack_tokens(
                        domain, n_attack, rng, cfg.shared_risk_ratio)
                    benign = self.encoder.sample_benign_tokens(
                        domain, n_tok - n_attack, rng)
                    toks = payload + benign
                    rng.shuffle(toks)
                else:
                    toks = self.encoder.sample_benign_tokens(domain, n_tok, rng)
                round_utt.append([int(x) for x in toks])
            utterances.append(round_utt)

        infected = level > 0.0

        # ---- 免费 judge (runtime shield model 的类比) ------------------- #
        # 论文的核心前提是: "Unsafe content originates upstream of the acting agent,
        # whose own text may look clean." 因此 episode 被判为 unsafe 的充要条件是
        # **注入的 payload 在最后一轮抵达了行动 agent 的上下文**: 它自身被感染,
        # 或它的任一上游邻居被感染(内容将被投递给它)。
        # 这使得风险是"交互性"的而非"局部有害文本"的 —— 单纯看行动 agent 自己的
        # 话语往往看不出问题, 必须借助拓扑聚合。
        T_last = T - 1
        preds = np.nonzero(adj[:, final_agent])[0]
        self_infected = bool(infected[T_last, final_agent])
        # 上游带毒并不总能影响行动 agent (它可能选择忽略/覆盖上游内容)，
        # 因此对每条上游投递独立采样一次 "是否真的被采纳"。
        upstream_infected = any(
            infected[T_last, i] and rng.random() < cfg.delivery_prob for i in preds)
        label = int(self_infected or upstream_infected)

        return Episode(domain=domain, adjacency=adj, utterances=utterances,
                       infected=infected, final_agent=final_agent,
                       label=label, intent=int(intent),
                       direct_exposure=self_infected)

    @staticmethod
    def _downstream(adj: np.ndarray, i: int) -> List[int]:
        return np.nonzero(adj[i, :])[0].tolist()

    # ------------------------------------------------------------------ #
    # 数据集
    # ------------------------------------------------------------------ #
    def generate(self, domain: int, n_episodes: int,
                 rng: np.random.Generator,
                 injection_mode: str = "propagate") -> List[Episode]:
        """批量生成 episode (攻击意图按 ``unsafe_ratio`` 随机决定)。"""
        return [self.sample_episode(domain, rng, injection_mode=injection_mode)
                for _ in range(n_episodes)]

    def generate_split(self, domain: int, n_episodes: int, rng: np.random.Generator,
                       force_intent: Optional[bool] = None,
                       injection_mode: str = "propagate") -> List[Episode]:
        """批量生成 episode，可用 ``force_intent`` **强制**全部为攻击集或良性集。"""
        return [self.sample_episode(domain, rng, force_intent=force_intent,
                                    injection_mode=injection_mode)
                for _ in range(n_episodes)]

    # ------------------------------------------------------------------ #
    # 统计
    # ------------------------------------------------------------------ #
    def positive_rate(self, episodes: List[Episode]) -> float:
        """unsafe (label=1) 的比例，用于检查标签分布是否过于稀疏。"""
        if not episodes:
            return float("nan")
        return float(np.mean([e.label for e in episodes]))

    def attack_success_rate(self, episodes: List[Episode]) -> float:
        """未加防护时的 ASR = 行动 agent 的上下文被污染的比例。"""
        attacks = [e for e in episodes if e.intent]
        if not attacks:
            return float("nan")
        return float(np.mean([e.label for e in attacks]))

    def compromise_breakdown(self, episodes: List[Episode]) -> Dict[str, float]:
        """诊断: 正样本中有多少是"自身可见"vs"仅上游可见"。"""
        pos = [e for e in episodes if e.label == 1]
        if not pos:
            return {"n_pos": 0.0, "direct_frac": float("nan")}
        return {
            "n_pos": float(len(pos)),
            "direct_frac": float(np.mean([e.direct_exposure for e in pos])),
        }
