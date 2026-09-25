"""★ 核心创新模块：跨模态互证注意力融合（Cross-Modal Mutual-Verification Attention）

设计动机（申报书「创新点」对应原文）：
    生理信号（心率/PPG/加速度）与行为信号（姿态序列）采样率不同、天然异步、信息互补。
    单模态方案要么误报高（固定阈值）、要么漏检（遮挡/光照）。本模块在特征级完成
    「互证」：两路信号在时间轴上的注意力分布越一致，说明两模态互相印证，融合置信度
    越高；分布冲突则视为需复核（抑制误报）。

算法（forward 流程）：
    1. 双向注意力：A_pb = softmax(Q_p · K_bᵀ / √d / τ)，A_bp = softmax(Q_b · K_pᵀ / √d / τ)
    2. 模态一致性：s = sigmoid(1 - KL_avg)，KL_avg 为两方向注意力分布的平均 KL 散度
    3. 互证增强：att_p = A_pb · H_b，att_b = A_bp · H_p
    4. 融合输出：z = Proj([H_p + s·att_p, H_b + s·att_b])，并返回一致性分数 s

输入：H_b (B, T, d)（行为支路序列特征）、H_p (B, T, d)（生理支路序列特征）
输出：z (B, T, d)（融合序列特征）、s (B,)（模态一致性分数，0~1，可解释/可视化）
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalMutualAttention(nn.Module):
    def __init__(self, d_model: int = 128, temperature: float = 1.0, dropout: float = 0.3):
        super().__init__()
        self.tau = temperature
        self.q_p = nn.Linear(d_model, d_model)   # 生理侧 query
        self.k_b = nn.Linear(d_model, d_model)   # 行为侧 key
        self.q_b = nn.Linear(d_model, d_model)   # 行为侧 query
        self.k_p = nn.Linear(d_model, d_model)   # 生理侧 key
        self.out = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def _attn(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """(B,T,d)×(B,T,d) → (B,T,T) 注意力权重（行归一化）。"""
        scores = torch.bmm(q, k.transpose(1, 2)) / ((k.size(-1) ** 0.5) * self.tau)
        return F.softmax(scores, dim=-1)

    @staticmethod
    def _kl_sym(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """两方向注意力分布的平均 KL 散度（对称近似），输入 (B,T,T) → (B,)。"""
        log_a, log_b = a.clamp_min(1e-9).log(), b.clamp_min(1e-9).log()
        kl_pb = (a * (log_a - log_b)).sum(dim=-1).mean(dim=-1)   # (B,)
        kl_bp = (b * (log_b - log_a)).sum(dim=-1).mean(dim=-1)   # (B,)
        return 0.5 * (kl_pb + kl_bp)

    def forward(self, h_b: torch.Tensor, h_p: torch.Tensor):
        """h_b/h_p: (B, T, d) → (z, s)。"""
        a_pb = self._attn(self.q_p(h_p), self.k_b(h_b))   # 生理 query 关注行为
        a_bp = self._attn(self.q_b(h_b), self.k_p(h_p))   # 行为 query 关注生理

        kl = self._kl_sym(a_pb, a_bp)                     # 越小越一致
        s = 1.0 - torch.tanh(kl)                          # (B,) 0~1，高=互证一致（tanh 避免 sigmoid 饱和）

        att_p = torch.bmm(a_pb, h_b)                      # 行为信息加权到生理侧
        att_b = torch.bmm(a_bp, h_p)                      # 生理信息加权到行为侧

        s_exp = s.view(-1, 1, 1)
        z = torch.cat([h_p + s_exp * att_p, h_b + s_exp * att_b], dim=-1)  # (B,T,2d)
        z = self.out(z)                                   # (B,T,d)
        return z, s


if __name__ == "__main__":
    m = CrossModalMutualAttention(d_model=128)
    hb, hp = torch.randn(4, 150, 128), torch.randn(4, 150, 128)
    z, s = m(hb, hp)
    print("融合输出:", tuple(z.shape), "一致性分数范围: [%.2f, %.2f]" % (s.min().item(), s.max().item()))
