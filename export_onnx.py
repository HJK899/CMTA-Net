# -*- coding: utf-8 -*-
"""export_onnx.py —— 导出浏览器端推理所需的 ONNX 模型（Windows 一键导出）。

产出（放入 web-onnx/models/）：
1. cmta.onnx            —— CMTA-Net 行为单流四态头（输入 behavior (1,150,51) → logits (1,2)）
2. yolov8n-pose.onnx    —— YOLOv8-Pose 骨架提取（ultralytics 官方导出，imgsz=640）

用法：
    python export_onnx.py            # 用默认 venv 环境（D:\\PyTorch\\venv）
    python export_onnx.py --tag best_urfall_v2mt
依赖：pip install torch onnx onnxruntime ultralytics
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

OUT_DIR = os.path.join(BASE, "web-onnx", "models")


def export_cmta(checkpoint: str, opset: int = 13) -> str:
    import torch
    from config import get_config
    from models.cmta_net import CMTA_Net

    cfg = get_config()
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    mcfg = ckpt["config"]["model"]
    d_model = int(mcfg["d_model"])
    physio_channels = int(mcfg["physio_channels"])
    num_classes = int(ckpt["config"]["data"]["num_classes"])

    net = CMTA_Net(
        pose_kps=cfg.model.pose_kps, d_model=d_model, physio_channels=physio_channels,
        num_classes=num_classes, num_gru_layers=int(mcfg["num_gru_layers"]),
        attn_temperature=float(mcfg["attn_temperature"]), dropout=float(mcfg["dropout"]),
    )
    net.load_state_dict(ckpt["model"], strict=False)
    net.eval()

    # 行为单流子图：behavior (B,150,51) → behavior_head 对数
    class BehaviorOnly(torch.nn.Module):
        def __init__(self, n: CMTA_Net):
            super().__init__()
            self.n = n

        def forward(self, behavior):
            h = self.n.behavior_branch(behavior)      # (B,Tb,d)
            z = h.mean(dim=1)
            return self.n.behavior_head(z)            # (B,num_classes)

    m = BehaviorOnly(net)
    behavior = torch.randn(1, int(cfg.data.window_sec * cfg.data.behavior_fps),
                           cfg.model.pose_kps * 3)
    out_path = os.path.join(OUT_DIR, "cmta.onnx")
    os.makedirs(OUT_DIR, exist_ok=True)
    torch.onnx.export(
        m, behavior, out_path,
        input_names=["behavior"], output_names=["logits"],
        dynamic_axes={"behavior": {0: "batch"}}, opset_version=opset,
    )
    print(f"[ONNX] cmta.onnx 导出完成（d_model={d_model}, 行为窗口={behavior.shape[1]}帧）→ {out_path}")
    return out_path


def export_yolo() -> str:
    from ultralytics import YOLO
    out_path = os.path.join(OUT_DIR, "yolov8n-pose.onnx")
    if os.path.exists(out_path) and os.path.getsize(out_path) > 1_000_000:
        print(f"[ONNX] yolov8n-pose.onnx 已存在，跳过（{os.path.getsize(out_path)//1024}KB）")
        return out_path
    model = YOLO("yolov8n-pose.pt")          # 首次自动下载权重
    model.export(format="onnx", imgsz=640, opset=12, simplify=False)
    # ultralytics 默认导出到 cwd，移动进 web-onnx/models/
    src = os.path.join(BASE, "yolov8n-pose.onnx")
    if os.path.exists(src):
        os.replace(src, out_path)
        print(f"[ONNX] yolov8n-pose.onnx 导出完成 → {out_path}")
    return out_path


def verify(out_cmta: str, out_yolo: str) -> None:
    """用 onnxruntime 做数值冒烟验证（与 PyTorch 输出对比）。"""
    import numpy as np
    import onnxruntime as ort

    # CMTA 冒烟：随机行为窗口 → 与 torch 输出对齐
    import torch
    from config import get_config
    from models.cmta_net import CMTA_Net
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    mcfg = ckpt["config"]["model"]
    cfg = get_config()
    net = CMTA_Net(
        pose_kps=cfg.model.pose_kps, d_model=int(mcfg["d_model"]),
        physio_channels=int(mcfg["physio_channels"]),
        num_classes=int(ckpt["config"]["data"]["num_classes"]),
        num_gru_layers=int(mcfg["num_gru_layers"]),
        attn_temperature=float(mcfg["attn_temperature"]),
        dropout=float(mcfg["dropout"]),
    )
    net.load_state_dict(ckpt["model"], strict=False)
    net.eval()
    with torch.no_grad():
        b = torch.randn(1, 150, 51)
        ref = net.behavior_head(net.behavior_branch(b).mean(dim=1)).numpy()

    sess = ort.InferenceSession(out_cmta, providers=["CPUExecutionProvider"])
    got = sess.run(["logits"], {"behavior": b.numpy()})[0]
    diff = float(np.abs(ref - got).max())
    print(f"[verify] cmta.onnx 与 PyTorch 最大误差={diff:.2e}（<1e-3 通过）")
    assert diff < 1e-3, "ONNX 导出数值不一致"

    # YOLO 冒烟
    sess2 = ort.InferenceSession(out_yolo, providers=["CPUExecutionProvider"])
    import cv2
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    img = img[:, :, ::-1]  # RGB→BGR
    blob = img.astype(np.float32) / 255.0
    out = sess2.run(None, {sess2.get_inputs()[0].name: blob[None].transpose(0, 3, 1, 2)})[0]
    print(f"[verify] yolov8n-pose.onnx 输出形状={out.shape}（期望 (1,56,8400)）")
    assert out.shape[1:] == (56, 8400), "YOLO ONNX 输出形状不符"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="导出 ONNX（浏览器端推理）")
    ap.add_argument("--checkpoint", default="checkpoints/best_urfall_v2mt.pt")
    ap.add_argument("--skip-yolo", action="store_true")
    args = ap.parse_args()

    c = export_cmta(args.checkpoint)
    y = None if args.skip_yolo else export_yolo()
    verify(c, y)
    print("\n完成：web-onnx/models/ 下即为浏览器端所需模型文件。")
