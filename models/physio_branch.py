"""生理支路：轻量 1D-CNN + 可学习位置编码。

输入：physio (B, C, T)     —— C 个生理通道（心率/PPG/加速度等）的窗口序列
输出：H_p (B, T, d_model)  —— 与行为支路同维的序列特征，供融合模块对齐使用
"""
from __future__ import annotations
import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """可学习位置编码（异质时间对齐的简化实现：统一时间轴后附加位置信息）。"""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.pe = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.normal_(self.pe, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d)
        return self.dropout(x + self.pe[:, : x.size(1)])


class PhysioBranch(nn.Module):
    def __init__(self, in_channels: int = 4, d_model: int = 128, dropout: float = 0.3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Conv1d(64, d_model, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(d_model),
        )
        self.pos = PositionalEncoding(d_model, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        h = self.conv(x)          # (B, d, T)
        h = h.transpose(1, 2)     # (B, T, d)
        return self.pos(h)


if __name__ == "__main__":
    net = PhysioBranch(in_channels=4, d_model=128)
    x = torch.randn(4, 4, 150)  # B=4, C=4, T=150
    out = net(x)
    print("PhysioBranch 输出:", tuple(out.shape))  # 期望 (4, 150, 128)
