"""CMTA-Net 全局配置：数据路径、窗口参数、模型结构与训练超参。

所有脚本通过 `from config import get_config` 获取配置，避免各处硬编码。
"""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class DataConfig:
    # 目录约定
    raw_dir: str = "data/raw"
    pose_feat_dir: str = "data/features/pose"
    physio_feat_dir: str = "data/features/physio"
    split_dir: str = "data/splits"
    synthetic_physio: str = "data/features/physio/synthetic.npz"

    # 窗口与采样
    behavior_fps: int = 30            # 视频/姿态帧率
    window_sec: float = 5.0           # 分析窗口（秒）
    stride_sec: float = 2.0           # 滑动步长（秒）
    physio_fs: float = 20.0           # 生理序列统一重采样率（Hz）

    # 标签
    num_classes: int = 2              # 2=正常/异常；4=四态（需合成标签，见 build_dataset）
    four_class: bool = False
    fall_label: int = 1
    seed: int = 42


@dataclass
class ModelConfig:
    pose_kps: int = 17                # YOLOv8-Pose 关键点数
    d_model: int = 128                # 融合特征维度
    physio_channels: int = 4          # 生理通道数（默认: hr, ppg, acc_x, acc_y）
    num_gru_layers: int = 1
    attn_temperature: float = 1.0     # 互证注意力温度 tau
    dropout: float = 0.3


@dataclass
class TrainConfig:
    epochs: int = 50
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 8                 # 早停：验证集连续无提升轮数
    warmup_epochs: int = 3
    use_focal: bool = True
    focal_alpha: float = 0.75
    focal_gamma: float = 2.0
    device: str = "auto"              # auto / cuda / cpu
    checkpoint_dir: str = "checkpoints"
    output_dir: str = "outputs"


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def get_config() -> Config:
    return Config()
