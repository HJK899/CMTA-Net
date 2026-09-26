"""CMTA-Net 主模型：异质时间对齐 → 双流编码 → 跨模态互证注意力融合 → 决策头。

输入（与 data/build_dataset.py 的输出一致）：
    behavior (B, Tb, K*3) —— 姿态关键点窗口序列（30fps，Tb=window_sec*fps）
    physio   (B, C, Tp)   —— 生理通道窗口序列（1Hz，Tp 可能 ≠ Tb）
输出：
    logits   (B, num_classes) —— 2 类（正常/异常）或 4 类（四态联合诊断）
    s        (B,)              —— 模态一致性分数（可解释性/答辩用）

模块①「异质时间对齐」在此实现（TemporalAlign）：生理序列特征线性上采样到
行为序列长度，统一到同一时间轴后再做互证融合——这正是解决「1Hz 生理流 vs
30fps 行为流采样率不一致」的核心机制。

说明：
- 初赛版实现「窗口级时间对齐 + 双流 + 互证融合 + 决策头」；
- 个性化基线先验（模块④）为复赛工作项，此处预留接口 user_embed 参数位。
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.physio_branch import PhysioBranch
from models.behavior_branch import BehaviorBranch
from models.fusion import CrossModalMutualAttention


class TemporalAlign(nn.Module):
    """异质时间对齐：将低采样率模态序列上采样到高采样率模态的时间轴。"""

    def forward(self, x: torch.Tensor, target_len: int) -> torch.Tensor:
        # x: (B, T, d) → (B, target_len, d)
        if x.size(1) == target_len:
            return x
        x_t = x.transpose(1, 2)                      # (B, d, T)
        x_t = F.interpolate(x_t, size=target_len, mode="linear", align_corners=False)
        return x_t.transpose(1, 2)                   # (B, target_len, d)


class CMTA_Net(nn.Module):
    def __init__(self, pose_kps: int = 17, d_model: int = 128,
                 physio_channels: int = 4, num_classes: int = 2,
                 num_gru_layers: int = 1, attn_temperature: float = 1.0,
                 dropout: float = 0.3):
        super().__init__()
        self.align = TemporalAlign()                 # 模块①：异质时间对齐编码
        self.behavior_branch = BehaviorBranch(
            input_dim=pose_kps * 3, d_model=d_model,
            num_layers=num_gru_layers, dropout=dropout,
        )
        self.physio_branch = PhysioBranch(
            in_channels=physio_channels, d_model=d_model, dropout=dropout,
        )
        self.fusion = CrossModalMutualAttention(
            d_model=d_model, temperature=attn_temperature, dropout=dropout,
        )
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )
        # 模块④四态决策：单流诊断头（生理/行为各自独立异常预测）
        # 融合头 → 联合判断；单流头 → 模态归因（生理异常/行为异常/联合异常/正常）
        self.physio_head = nn.Linear(d_model, num_classes)
        self.behavior_head = nn.Linear(d_model, num_classes)

    def forward(self, behavior: torch.Tensor | None, physio: torch.Tensor | None):
        """behavior: (B,Tb,K*3); physio: (B,C,Tp) → (logits, s, logits_p, logits_b)
        - 双模态：logits=融合头；logits_p/logits_b=单流诊断头；s=一致性分数
        - 单流  ：logits=对应单流头输出；其余为 None
        支持单流：behavior=None 时仅生理流；physio=None 时仅行为流。
        """
        if behavior is None and physio is None:
            raise ValueError("CMTA_Net 至少需要一路输入")

        if behavior is None:
            h_p = self.physio_branch(physio)                # (B,Tp,d)
            z = h_p.mean(dim=1)
            return self.physio_head(z), None, None, None

        if physio is None:
            h_b = self.behavior_branch(behavior)            # (B,Tb,d)
            z = h_b.mean(dim=1)
            return self.behavior_head(z), None, None, None

        h_b = self.behavior_branch(behavior)            # (B,Tb,d)
        h_p = self.physio_branch(physio)                # (B,Tp,d)
        h_p = self.align(h_p, h_b.size(1))              # 上采样到 Tb（模块①）
        z, s = self.fusion(h_b, h_p)                    # (B,Tb,d), (B,)
        z = z.mean(dim=1)                               # 全局平均池化 → (B,d)
        logits = self.classifier(z)                     # (B,num_classes) 融合头
        logits_p = self.physio_head(h_p.mean(dim=1))    # 生理单流诊断头
        logits_b = self.behavior_head(h_b.mean(dim=1))  # 行为单流诊断头
        return logits, s, logits_p, logits_b


if __name__ == "__main__":
    net = CMTA_Net(pose_kps=17, d_model=128, physio_channels=4, num_classes=2)
    # 模拟真实异质长度：行为 150 帧 @30fps，生理 5 点 @1Hz
    behavior = torch.randn(4, 150, 17 * 3)
    physio = torch.randn(4, 4, 5)
    logits, s, logits_p, logits_b = net(behavior, physio)
    print("CMTA_Net logits:", tuple(logits.shape), "一致性:", tuple(s.shape),
          "生理头:", tuple(logits_p.shape), "行为头:", tuple(logits_b.shape))
    # 单流测试
    lp, *_ = net(None, physio)
    lb, *_ = net(behavior, None)
    print("单流生理头:", tuple(lp.shape), "单流行为头:", tuple(lb.shape))
    n_params = sum(p.numel() for p in net.parameters())
    print(f"参数量: {n_params / 1e6:.2f}M")
