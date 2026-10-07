"""冻结句编码器 :math:`\\phi(\\cdot)` 的轻量替代实现。

阅读顺序: 第 2 步 / 共 18 步 —— 见 READING_ORDER.md

论文 (Sec. 3) 使用**冻结的 MiniLM** 把 utterance 映射为 :math:`d` 维向量，并强调:

    Exchanging the frozen-encoder embeddings instead would not relax this constraint:
    sentence embeddings are invertible to near-verbatim text.

因此本项目也把编码器设计成**完全冻结、与训练解耦**的模块。为了让论文的核心
现象(domain shift 导致 off-the-shelf 迁移失效、跨域专家无法覆盖)在仿真中真实出现，
词表被划分为若干 **block**:

- 每个 domain 的良性词 block 互不重合  -> 域内良性分布的"质心方向"不同
- 每个 domain 的攻击 payload block 互不重合 -> 域 A 的专家看不懂域 B 的攻击
- 另有一小段**共享风险标记** block，所有域的攻击都含有一部分 -> 跨域迁移存在弱信号

实现上使用一个固定的随机嵌入矩阵 (不以梯度更新)，utterance 的编码 = token 嵌入的
均值池化 + (可选) L2 归一化。
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .config import EncoderConfig, VocabConfig


class FrozenEncoder:
    """冻结的"句编码器"。

    Parameters
    ----------
    vocab_cfg, enc_cfg
        词表与编码器配置。
    """

    def __init__(self, vocab_cfg: VocabConfig, enc_cfg: EncoderConfig) -> None:
        self.vocab_cfg = vocab_cfg
        self.enc_cfg = enc_cfg
        rng = np.random.default_rng(enc_cfg.seed)
        # 正交化初始化: 让不同 block 的质心方向尽量分离，放大 domain shift。
        table = rng.normal(size=(vocab_cfg.vocab_size, enc_cfg.dim)).astype(np.float32)
        table /= np.linalg.norm(table, axis=1, keepdims=True) + 1e-8
        self.table = table  # (V, d)

    @property
    def dim(self) -> int:
        """嵌入维度 :math:`d` (论文用 384，本仿真 32)。"""
        return self.enc_cfg.dim

    def embed_text(self, token_ids: Sequence[int]) -> np.ndarray:
        """把一个 utterance (token id 序列) 编码成 ``(d,)`` 向量。

        空 utterance 返回全零向量 (论文允许 :math:`u_i^t` 为空)。
        """
        ids = [t for t in token_ids if t >= 0]
        if not ids:
            return np.zeros(self.dim, dtype=np.float32)
        vec = self.table[np.asarray(ids, dtype=np.int64)].mean(axis=0)
        if self.enc_cfg.l2_normalize:
            n = float(np.linalg.norm(vec))
            if n > 1e-8:
                vec = vec / n
        return vec.astype(np.float32)

    def embed_batch(self, texts: Sequence[Sequence[int]]) -> np.ndarray:
        """批量编码，返回 ``(len(texts), d)``。"""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([self.embed_text(t) for t in texts], axis=0)

    # -- 便捷采样接口 (供仿真器使用) -------------------------------------- #
    def sample_benign_tokens(self, domain: int, n_tokens: int,
                             rng: np.random.Generator) -> list[int]:
        """从该 domain 专属的**良性**词 block 中采样 n_tokens 个 token。"""
        lo, hi = self.vocab_cfg.benign_range(domain)
        return rng.integers(lo, hi, size=n_tokens).tolist()

    def sample_attack_tokens(self, domain: int, n_tokens: int,
                             rng: np.random.Generator,
                             shared_ratio: float) -> list[int]:
        """采样注入 payload token。

        ``shared_ratio`` 比例的 token 来自**跨域共享**的风险标记 block，
        其余来自该 domain 专属的攻击 block。
        """
        n_shared = int(round(n_tokens * shared_ratio))
        n_specific = n_tokens - n_shared
        shared_lo, shared_hi = self.vocab_cfg.shared_risk_range
        spec_lo, spec_hi = self.vocab_cfg.attack_range(domain)
        toks: list[int] = []
        if n_shared > 0:
            toks += rng.integers(shared_lo, shared_hi, size=n_shared).tolist()
        if n_specific > 0:
            toks += rng.integers(spec_lo, spec_hi, size=n_specific).tolist()
        rng.shuffle(toks)
        return toks

    def payload_token_set(self, domain: int) -> set[int]:
        """该 domain 攻击 payload 的 token 全集 (用于 ASR 判定)。"""
        sl, sh = self.vocab_cfg.shared_risk_range
        al, ah = self.vocab_cfg.attack_range(domain)
        return set(range(sl, sh)) | set(range(al, ah))
