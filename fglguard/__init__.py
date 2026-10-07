"""FGLGuard: 隐私保护的拓扑引导多智能体安全防护 (图联邦学习实现)。

本包是论文 *Privacy-Preserving Topology-Guided Safety for LLM-Based Multi-Agent
Systems via Federated Graph Learning* 的一个小型、可运行的复现框架。

**从这里开始读**: 仓库根目录的 ``READING_ORDER.md`` 给出了 18 步阅读路线
(含论文符号 ↔ 代码变量对照表)。最简单的路径是: ``mas_simulator`` -> ``graph``
-> ``data`` -> ``model`` -> ``federated`` -> ``runtime``。

模块总览
--------
- ``config``        : 全部超参数 (对应论文 5.1 Setup / Protocol)
- ``encoder``       : 冻结句编码器 phi(.) 的轻量替代实现
- ``mas_simulator`` : 可配置的多智能体系统 (MAS) episode 仿真器 + 免费 judge 标签
- ``graph``         : episode -> 属性图 G=(X, A, E) 的构建 (论文 3)
- ``data``          : 客户端数据划分 (domain + 标签偏斜 non-IID)
- ``model``         : 边特征图注意力风险打分器 (论文 4.1)
- ``federated``     : 联邦近端训练 + 域均衡聚合 (论文 4.2)
- ``calibration``   : 过拒答预算下的运行点校准 (论文 4.3)
- ``runtime``       : 上游佐证 + 守卫式重写干预算子 (论文 4.4)
- ``metrics``       : AUROC / Recall / FPR / Wilson 区间等指标
"""

from .config import Config, default_config, small_config
from .calibration import calibrate_threshold
from .encoder import FrozenEncoder
from .graph import EpisodeGraph, build_graph, corroborated_scores
from .mas_simulator import Episode, MASSimulator
from .model import FGLGuardScorer
from .runtime import RuntimeGuard, evaluate_runtime, sanitizing_rewrite

__all__ = [
    "Config", "default_config", "small_config",
    "FrozenEncoder",
    "MASSimulator", "Episode",
    "EpisodeGraph", "build_graph", "corroborated_scores",
    "FGLGuardScorer",
    "calibrate_threshold",
    "RuntimeGuard", "evaluate_runtime", "sanitizing_rewrite",
]
__version__ = "0.1.0"
