"""Focal Loss：应对类别不平衡（正常样本远多于异常样本时模型会「全说正常」）。

参考：Lin et al., Focal Loss for Dense Object Detection (ICCV 2017)。
alpha 提高正类（异常）权重，gamma 压低易分样本的损失贡献。
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, num_classes: int = 2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, reduction="none")   # (B,)
        pt = torch.exp(-ce)                                       # 该样本的预测概率
        # 简化 alpha：正类（异常）权重 alpha，负类 1-alpha
        alpha_t = torch.where(targets == 1, self.alpha, 1.0 - self.alpha)
        loss = alpha_t * (1.0 - pt) ** self.gamma * ce
        return loss.mean()
