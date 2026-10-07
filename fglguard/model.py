"""边特征图注意力风险打分器 (论文 Sec. 4.1, Eq. 2-4)。

.. math::

    e^{(\\ell)}_{ij} &= \\mathrm{LeakyReLU}\\!\\left(a_\\ell^\\top
        [\\,W_\\ell h^{(\\ell)}_i \\;;\\; W_\\ell h^{(\\ell)}_j \\;;\\;
         U_\\ell \\psi(e_{ij})\\,]\\right) \\\\
    \\alpha^{(\\ell)}_{ij} &= \\frac{\\exp(e^{(\\ell)}_{ij})}
        {\\sum_{i': A_{i'j}=1} \\exp(e^{(\\ell)}_{i'j})} \\\\
    h^{(\\ell+1)}_j &= \\sigma\\!\\left(\\sum_{i: A_{ij}=1}
        \\alpha^{(\\ell)}_{ij} W_\\ell h^{(\\ell)}_i\\right)

最后用一个线性头得到 :math:`s_j = \\sigma(w^\\top h^{(L)}_j + b) \\in [0, 1]`。

实现要点
--------
- 全用**稠密**张量完成 (团队规模 :math:`n \\le 8`，稠密远比稀疏消息传递简单可靠)，
  便于在任意 batch 上向量化。
- 注意力在**入边方向**归一化 (``dim=1``，即 source ``i``)，与论文一致。
- 掩码边在 softmax 之后再次置零，避免"无入边节点"被均匀分配质量。
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class EdgeFeaturedGATLayer(nn.Module):
    """单层边条件注意力 (Edge-Conditioned Attention)。"""

    def __init__(self, in_dim: int, out_dim: int, edge_dim: int,
                 n_heads: int = 4, dropout: float = 0.0,
                 negative_slope: float = 0.2) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_heads = n_heads
        self.dropout = dropout
        self.negative_slope = negative_slope

        self.W = nn.Linear(in_dim, n_heads * out_dim, bias=False)   # 节点变换
        self.U = nn.Linear(edge_dim, n_heads * out_dim, bias=False)  # 边特征变换
        # att = [a_src ; a_dst ; a_edge]，每个 head 一份
        self.att = nn.Parameter(torch.zeros(n_heads, 3 * out_dim))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Xavier 初始化节点变换、边变换与注意力向量。"""
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.U.weight)
        nn.init.xavier_uniform_(self.att)

    def forward(self, h: torch.Tensor, adj: torch.Tensor,
                edge_feat: torch.Tensor) -> torch.Tensor:
        """
        :param h: ``(B, n, in_dim)``
        :param adj: ``(B, n, n)``，``adj[b, i, j] > 0`` 表示 :math:`i \\to j`
        :param edge_feat: ``(B, n, n, edge_dim)``
        :return: ``(B, n, n_heads * out_dim)``
        """
        B, n, _ = h.shape
        H, Fo = self.n_heads, self.out_dim

        Wh = self.W(h).view(B, n, H, Fo)                     # (B, n, H, Fo)
        Ue = self.U(edge_feat).view(B, n, n, H, Fo)          # (B, n, n, H, Fo)

        Wh_i = Wh.unsqueeze(2).expand(B, n, n, H, Fo)        # 源 i 广播到目标维
        Wh_j = Wh.unsqueeze(1).expand(B, n, n, H, Fo)        # 目标 j 广播到源维

        cat = torch.cat([Wh_i, Wh_j, Ue], dim=-1)            # (B, n, n, H, 3Fo)
        logits = F.leaky_relu((cat * self.att).sum(dim=-1), self.negative_slope)

        mask = (adj > 0)                                     # (B, n, n)
        logits = logits.masked_fill(~mask.unsqueeze(-1), -1e9)
        alpha = torch.softmax(logits, dim=1)                 # 沿源 i 归一化
        alpha = alpha * mask.unsqueeze(-1).to(alpha.dtype)   # 无入边 -> 全 0
        if self.training and self.dropout > 0:
            alpha = F.dropout(alpha, p=self.dropout)

        msg = (alpha.unsqueeze(-1) * Wh_i).sum(dim=1)        # (B, n, H, Fo)
        return msg.reshape(B, n, H * Fo)


class FGLGuardScorer(nn.Module):
    """FGLGuard 的风险打分器 :math:`f_\\theta(X, A, E)`。

    ``FGLGuardScorer`` 是**打分器无关**设计中的具体实例；论文强调
    "federation, calibration, and the runtime operator are scorer-agnostic"，
    因此本类可以被任何满足同一接口的模型替换。
    """

    def __init__(self, node_dim: int, edge_dim: int, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # 每层输出的通道数 = hidden_dim * n_heads (多头拼接)，因此第二层起的输入维度
        # 必须与之对齐；只有第一层吃原始节点特征维度。
        out_channels = cfg.hidden_dim * cfg.n_heads
        in_dims = [node_dim] + [out_channels] * (cfg.n_layers - 1)
        self.layers = nn.ModuleList([
            EdgeFeaturedGATLayer(in_dims[i], cfg.hidden_dim, edge_dim,
                                 n_heads=cfg.n_heads, dropout=cfg.dropout)
            for i in range(cfg.n_layers)
        ])
        self.out_dim = out_channels
        self.head = nn.Linear(self.out_dim, 1)

    def forward(self, x: torch.Tensor, adj: torch.Tensor,
                edge_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """返回 ``(logits, probs)``，形状均为 ``(B, n)``。"""
        h = x
        for layer in self.layers:
            h = layer(h, adj, edge_feat)
        logits = self.head(h).squeeze(-1)          # (B, n)
        return logits, torch.sigmoid(logits)

    # ------------------------------------------------------------------ #
    @property
    def n_parameters(self) -> int:
        """可训练参数量 (论文报告约 0.5M，本仿真约 3 万)。"""
        return sum(p.numel() for p in self.parameters())

    def episode_scores(self, x: torch.Tensor, adj: torch.Tensor,
                       edge_feat: torch.Tensor, final_idx: torch.Tensor
                       ) -> torch.Tensor:
        """按 final-answer agent 读出 episode 级分数 (论文 Sec. 3 的 reduction 约定)。"""
        _, probs = self.forward(x, adj, edge_feat)
        return probs.gather(1, final_idx.view(-1, 1)).squeeze(1)


def node_bce_loss(logits: torch.Tensor, node_labels: torch.Tensor,
                  node_mask: torch.Tensor,
                  reduction: str = "mean") -> torch.Tensor:
    """逐节点 BCE (论文 Eq. 5 的第一项)。

    论文明确: "we broadcast the episode label to all nodes, :math:`y_i = y`"
    以及 "We keep the BCE term unweighted: class reweighting is harmful under rare
    unsafe labels"，因此这里**不做类别重加权**。
    """
    loss = F.binary_cross_entropy_with_logits(logits, node_labels, reduction="none")
    loss = loss * node_mask.to(loss.dtype)
    if reduction == "sum":
        return loss.sum()
    denom = node_mask.to(loss.dtype).sum().clamp_min(1.0)
    return loss.sum() / denom


def isolation_stress_test(model: FGLGuardScorer, cfg: ModelConfig,
                          d: int) -> float:
    """诊断: 无入边的孤立节点打分是否退化为 0.5 (论文 n=3 稀疏拓扑的弱点)。"""
    model.eval()
    with torch.no_grad():
        x = torch.zeros(1, 1, d)
        adj = torch.zeros(1, 1, 1)
        e = torch.zeros(1, 1, 1, d)
        _, p = model(x, adj, e)
    return float(p.item())
