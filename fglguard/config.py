"""全局配置。

阅读顺序: 第 1 步 / 共 18 步 (先读它了解全部旋钮) —— 见 READING_ORDER.md

超参命名尽量与论文保持一致，并在注释中标注对应的公式/章节编号。
论文 Section 5.1 的 Protocol 为::

    两层 edge-featured GAT (256 hid., 4 heads, T=3) over frozen MiniLM;
    K = 4 clients, 40 x 2 local epochs, rho = 0.10, mu = 0.01

本项目是"小型可运行"版本，因此把隐藏维、词表规模、episode 数量等按比例缩小，
但**算法结构完全保留** (近端目标、域均衡聚合、预算校准、佐证打分、单次重写)。
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Tuple


@dataclass
class VocabConfig:
    """仿真词表布局。

    词表被切成若干 block，使得"域偏移"这一论文核心现象可以被真实复现:

    - ``n_shared_risk``   : **共享风险标记** block。所有域的注入攻击都会含有一部分该
      block 的 token，因此跨域迁移能拿到"弱信号"(论文中 off-domain specialist
      AUROC 0.537-0.573，而非完全的 0.5 随机)。
    - ``benign_per_domain``: 每个域各自的**良性** block (域内话题词)。域之间不重叠，
      这是 domain shift 的主要来源。
    - ``attack_per_domain``: 每个域各自的**注入 payload** block。域之间不重叠，
      因此"只在大域上训练过的专家"在本域上近似失效。

    注意: ``n_target_domains`` 是参与联邦的业务域数量 (等于论文的 :math:`K=4` 个
    operator)，而 ``n_domains`` 额外多出 1 个 **源域 (source domain)**，专门用于生成
    "off-the-shelf / G-Safeguard (O-S)" 基线所需要的**异域合成注入语料**。
    两者共用同一张冻结嵌入表，因此该基线可以直接被搬到目标域上评测——这正是论文中
    "off-the-shelf transfer collapses under distribution shift" 的实验设置。
    """

    n_shared_risk: int = 6
    benign_per_domain: int = 48
    attack_per_domain: int = 48
    n_target_domains: int = 4
    n_domains: int = 5

    @property
    def source_domain(self) -> int:
        """保留给 off-the-shelf 基线的异域 id。"""
        return self.n_target_domains

    @property
    def shared_risk_range(self) -> Tuple[int, int]:
        """跨域共享风险标记 block 的 token id 区间。"""
        return 0, self.n_shared_risk

    def benign_range(self, domain: int) -> Tuple[int, int]:
        """域 ``domain`` 的良性词 block 区间 (域间不重叠)。"""
        base = self.n_shared_risk + domain * self.benign_per_domain
        return base, base + self.benign_per_domain

    def attack_range(self, domain: int) -> Tuple[int, int]:
        """域 ``domain`` 的专属攻击 payload block 区间 (域间不重叠)。"""
        base = self.n_shared_risk + self.n_domains * self.benign_per_domain
        base += domain * self.attack_per_domain
        return base, base + self.attack_per_domain

    @property
    def vocab_size(self) -> int:
        """词表总大小 = 共享风险块 + 所有域的良性块与攻击块。"""
        return self.n_shared_risk + self.n_domains * (
            self.benign_per_domain + self.attack_per_domain
        )


@dataclass
class SimConfig:
    """MAS episode 仿真器配置 (论文 3: Communication graphs from MAS episodes)。"""

    seed: int = 42
    n_agents_range: Tuple[int, int] = (4, 4)   # 团队规模 n (论文 Fig.3(a) 扫 3-8)
    n_rounds: int = 3                          # T，最多 R 轮
    edge_sparsity: float = 0.2                 # 随机有向拓扑的边密度
    n_utterance_tokens: int = 8                # 每条 utterance 的 token 数
    attack_token_ratio: float = 0.6            # 被感染 utterance 中 payload token 占比
    propagation_decay: float = 0.5             # payload 逐跳衰减系数 (越下游越弱)
    shared_risk_ratio: float = 0.3             # payload 中来自共享 block 的比例
    propagation_prob: float = 0.20             # 沿边 "蠕虫式" 传播概率
    delivery_prob: float = 0.32                # 上游 payload 真正影响行动 agent 的概率
    unsafe_ratio: float = 0.5                  # 预先注入攻击意图的 episode 比例
    # 每个客户端 (domain) 生成多少 episode
    n_train_per_client: int = 320
    n_val_per_client: int = 150
    n_test_per_client: int = 200
    # 论文 4.2: "Unequal silo sizes: aggregation weights domains equally rather than
    # by volume." 用一组乘子制造**体量不均**，否则 Eq.(6) 的域均衡会退化回 FedAvg。
    client_size_multipliers: Tuple[float, ...] = (1.75, 0.5, 1.75, 0.5)


@dataclass
class EncoderConfig:
    """冻结句编码器 phi(.) 配置 (论文 3)。

    论文使用冻结的 MiniLM (384 维)。这里用"冻结随机 token embedding + 均值池化"
    作为替代: 它同样是一个与训练解耦的固定映射，且具备 sentence encoder 的关键性质
    ——相邻语义(同 block)映射到相近向量，不同 block 映射到不同方向。
    """

    dim: int = 32
    seed: int = 1234
    l2_normalize: bool = True


@dataclass
class ModelConfig:
    """边特征图注意力打分器 (论文 Eq. 2-4)。"""

    hidden_dim: int = 32     # 论文为 256，此处按比例缩小
    n_layers: int = 2        # L = 2
    n_heads: int = 4         # 4 heads
    dropout: float = 0.1
    self_loops: bool = True  # 工程取舍: 见 README "与论文的差异"


@dataclass
class FedConfig:
    """联邦训练配置 (论文 Eq. 5-6)。"""

    n_clients: int = 4
    n_rounds: int = 30           # R 联邦轮
    local_epochs: int = 2        # E local epochs
    batch_size: int = 64
    lr: float = 0.03
    proximal_mu: float = 0.01    # mu，论文取 0.01 (FedProx 近端项)
    weight_decay: float = 0.0
    domain_balanced: bool = True  # True -> Eq.(6) 域均衡; False -> 纯 FedAvg 体量加权
    label_skew_purity: float = 0.8  # non-IID 标签偏斜纯度 p
    # 每个客户端的主导标签 (与体量乘子解耦，避免类别偏斜与规模偏斜互相混淆)
    client_dominant_labels: Tuple[int, ...] = (0, 1, 1, 0)
    # 0 表示自动对齐为 n_rounds * local_epochs (匹配梯度预算)
    local_only_epochs: int = 0


@dataclass
class CalibConfig:
    """运行点校准 (论文 Eq. 7)。

    ``tie_break``:

    - ``"tightest"`` (默认) —— 在最优召回下取**最紧的可行运行点** (阈值最小、用满
      :math:`\\rho` 预算)。这是 "argmax Recall s.t. FPR <= rho" 的字面实现，
      并且与论文 Proposition 2 (良性流量下退化为 :math:`(1-\\rho)` 分位数) 一致。
    - ``"lower_fpr"`` —— 论文正文提到的 "breaking recall ties toward lower FPR"。
      注意它会把良性-only 情形的阈值推到接近 1.0，与 Proposition 2 冲突。
    """

    over_refusal_budget: float = 0.10  # rho
    tie_break: str = "tightest"


@dataclass
class RuntimeConfig:
    """运行时干预算子 (论文 Eq. 8-9)。"""

    corroborate: bool = True        # 是否启用上游佐证 (Eq. 8)
    guard_final_answer_only: bool = True  # 论文消融结论: 只守 final answer
    allow_rewrite: bool = True      # 是否给予一次 guarded rewrite
    max_rewrites: int = 1           # 至多一次重写


@dataclass
class Config:
    """总配置。"""

    vocab: VocabConfig = field(default_factory=VocabConfig)
    sim: SimConfig = field(default_factory=SimConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    fed: FedConfig = field(default_factory=FedConfig)
    calib: CalibConfig = field(default_factory=CalibConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    # 实验设置
    seeds: Tuple[int, ...] = (42, 43, 44)
    device: str = "cpu"
    output_dir: str = "results"

    def to_dict(self) -> Dict[str, Any]:
        """递归展开为普通字典 (结果文件里会连同超参一起存档)。"""
        return asdict(self)

    def clone(self) -> "Config":
        """深拷贝一份配置，便于在 sweep 中独立修改。"""
        import copy

        return copy.deepcopy(self)


def default_config() -> Config:
    """返回默认配置 (论文 5.1 Protocol 的小型化版本)。"""
    return Config()


def small_config() -> Config:
    """更小的配置，用于快速 smoke test / 单元测试。"""
    cfg = Config()
    cfg.sim.n_train_per_client = 80
    cfg.sim.n_val_per_client = 40
    cfg.sim.n_test_per_client = 40
    cfg.fed.n_rounds = 6
    cfg.model.hidden_dim = 16
    return cfg
