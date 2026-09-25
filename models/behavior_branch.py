"""行为支路：姿态关键点序列编码（轻量 GRU）。

输入：behavior (B, T, K*3) —— T 帧 × (17 关键点 × x/y/conf)
输出：H_b (B, T, d_model)   —— 序列特征，供融合模块对齐使用
"""
from __future__ import annotations
import torch
import torch.nn as nn


class BehaviorBranch(nn.Module):
    def __init__(self, input_dim: int = 17 * 3, d_model: int = 128,
                 num_layers: int = 1, dropout: float = 0.3):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.gru = nn.GRU(
            d_model, d_model, num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, K*3)
        h = torch.relu(self.proj(x))       # (B, T, d)
        h, _ = self.gru(h)                 # (B, T, d)
        h = self.norm(h)
        return self.dropout(h)


if __name__ == "__main__":
    net = BehaviorBranch(input_dim=17 * 3, d_model=128)
    x = torch.randn(4, 150, 17 * 3)
    out = net(x)
    print("BehaviorBranch 输出:", tuple(out.shape))  # 期望 (4, 150, 128)
