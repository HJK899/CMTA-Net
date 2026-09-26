# -*- coding: utf-8 -*-
"""CMTA-Net 端到端推理接口（模块④四态联合诊断 + 可解释输出）。

用途：
1. 竞赛演示：输入一段窗口的生理/行为数据 → 输出四态诊断 + 概率 + 模态一致性分数；
2. 可作为云-边-端部署的推理服务核心（示例代码）。

用法：
    # 命令行（随机数据冒烟测试）
    python inference.py --checkpoint checkpoints/best_urfall_v2s2.pt
    # 作为模块调用
    from inference import CMTA_Inference
    engine = CMTA_Inference("checkpoints/best_urfall_v2s2.pt")
    res = engine.predict(physio=..., behavior=...)   # → dict
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import torch

from config import get_config
from models.cmta_net import CMTA_Net

STATE_NAMES = {0: "正常", 1: "生理异常", 2: "行为异常", 3: "联合异常"}


class CMTA_Inference:
    """推理引擎：加载 checkpoint → 单窗口四态诊断。"""

    def __init__(self, checkpoint: str, device: str = "auto", tau: float = 0.5):
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        cfg = get_config()
        m = cfg.model
        d_model = int(ckpt["config"]["model"]["d_model"])
        physio_channels = int(ckpt["config"]["model"]["physio_channels"])
        num_classes = int(ckpt["config"]["data"]["num_classes"])
        self.num_classes = num_classes
        self.tau = tau
        self.device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else
                                   torch.device(device if device != "auto" else "cpu"))
        self.model = CMTA_Net(
            pose_kps=m.pose_kps, d_model=d_model, physio_channels=physio_channels,
            num_classes=num_classes, num_gru_layers=m.num_gru_layers,
            attn_temperature=m.attn_temperature, dropout=m.dropout,
        )
        self.model.load_state_dict(ckpt["model"], strict=False)   # 兼容旧版 checkpoint
        self.model.to(self.device).eval()
        self.has_heads = "physio_head.weight" in ckpt["model"]
        self.win_beh = int(cfg.data.window_sec * cfg.data.behavior_fps)
        self.win_phys = int(cfg.data.window_sec * cfg.data.physio_fs)
        print(f"[inference] 模型加载完成（d_model={d_model}, 通道={physio_channels}, "
              f"四态头={self.has_heads}）")

    def predict(self, physio: np.ndarray | None = None, behavior: np.ndarray | None = None) -> dict:
        """单窗口推理。

        physio: (C, Tp) 生理通道窗口（Tp 可≠100，内部插值到 5s/20Hz）
        behavior: (Tb, K*3) 姿态关键点窗口（Tb 可≠150，内部插值到 5s/30fps）
        → dict: state/state_name/异常概率/单流概率/一致性/置信度
        """
        if physio is None and behavior is None:
            raise ValueError("至少提供一路输入（physio 或 behavior）")

        p_t = None
        if physio is not None:
            p = np.asarray(physio, dtype=np.float32)
            if p.ndim == 1:
                p = p[None, :]
            if p.shape[1] != self.win_phys:
                xs = np.linspace(0, p.shape[1] - 1, self.win_phys)
                p = np.stack([np.interp(xs, np.arange(p.shape[1]), p[c]) for c in range(p.shape[0])], 0)
            # 通道标准化（与训练一致）
            mu, sd = p.mean(axis=1, keepdims=True), p.std(axis=1, keepdims=True) + 1e-6
            p_t = torch.as_tensor((p - mu) / sd, dtype=torch.float32)[None].to(self.device)

        b_t = None
        if behavior is not None:
            b = np.asarray(behavior, dtype=np.float32)
            if b.ndim == 2 and b.shape[1] != self.win_beh:
                xs = np.linspace(0, b.shape[0] - 1, self.win_beh)
                b = np.stack([np.interp(xs, np.arange(b.shape[0]), b[:, c]) for c in range(b.shape[1])], 1)
            b = (b / 100.0).astype(np.float32)
            b_t = torch.as_tensor(b, dtype=torch.float32)[None].to(self.device)

        with torch.no_grad():
            out = self.model(b_t, p_t)
            logits, s, logits_p, logits_b = out[0], out[1], out[2], out[3]

        prob = torch.softmax(logits, dim=1)[0, 1].item() if logits is not None else None
        prob_p = torch.softmax(logits_p, dim=1)[0, 1].item() if (self.has_heads and logits_p is not None) else None
        prob_b = torch.softmax(logits_b, dim=1)[0, 1].item() if (self.has_heads and logits_b is not None) else None
        s_val = s.item() if s is not None else None

        # 四态决策（融合头概率作为整体异常置信度；单流头做模态归因）
        if self.has_heads and prob_p is not None and prob_b is not None:
            p1, b1 = prob_p, prob_b
            state = 3 if (p1 >= self.tau and b1 >= self.tau) else \
                    (1 if p1 >= self.tau else (2 if b1 >= self.tau else 0))
            conf = max(p1, b1) if state in (1, 2) else (min(p1, b1) if state == 0 else min(p1, b1))
        elif physio is None and behavior is not None:
            # 行为单流（如照片→骨架序列）：行为诊断头
            state = 2 if (prob or 0.0) >= self.tau else 0
            prob_b = prob
            conf = prob
        elif behavior is None and physio is not None:
            # 生理单流（如穿戴传感窗口）：生理诊断头
            state = 1 if (prob or 0.0) >= self.tau else 0
            prob_p = prob
            conf = prob
        else:
            # 无四态头的双模态：融合头退化（异常/正常）
            state = 1 if (prob or 0.0) >= self.tau else 0
            conf = prob

        return {
            "state": state,
            "state_name": STATE_NAMES.get(state, "未知"),
            "异常概率(融合)": round(prob, 4) if prob is not None else None,
            "生理异常概率": round(prob_p, 4) if prob_p is not None else None,
            "行为异常概率": round(prob_b, 4) if prob_b is not None else None,
            "模态一致性s": round(s_val, 4) if s_val is not None else None,
            "置信度": round(conf, 4),
            "阈值tau": self.tau,
        }


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="CMTA-Net 推理冒烟测试")
    ap.add_argument("--checkpoint", default="checkpoints/best_urfall_v2s2.pt")
    ap.add_argument("--tau", type=float, default=0.5)
    args = ap.parse_args()

    engine = CMTA_Inference(args.checkpoint, tau=args.tau)
    cfg = get_config()
    physio_ch = int(torch.load(args.checkpoint, map_location="cpu", weights_only=False)
                    ["config"]["model"]["physio_channels"])

    # 双模态随机窗口（冒烟）
    physio = np.random.randn(physio_ch, int(cfg.data.window_sec * cfg.data.physio_fs)).astype(np.float32)
    behavior = np.random.randn(int(cfg.data.window_sec * cfg.data.behavior_fps), 17 * 3).astype(np.float32)
    res = engine.predict(physio=physio, behavior=behavior)
    print("\n===== 双模态推理结果 =====")
    for k, v in res.items():
        print(f"  {k}: {v}")

    # 单流冒烟
    res_p = engine.predict(physio=physio)
    print("\n===== 仅生理流推理结果 =====")
    for k, v in res_p.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
