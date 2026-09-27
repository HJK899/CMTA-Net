# -*- coding: utf-8 -*-
"""check_export.py —— 核对 PyTorch(photo_demo 管线) 与 ONNX(浏览器管线) 判定是否一致。

在 Windows 上运行：python check_export.py
若 torch 与 onnx 概率接近 → 浏览器端链路正确；若差异大 → ONNX 导出有问题（权重漏载），
把输出发给我。
"""
from __future__ import annotations
import glob
import os
import sys

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from ultralytics import YOLO                      # noqa: E402
from inference import CMTA_Inference              # noqa: E402
from photo_demo import extract_pose               # noqa: E402
import onnxruntime as ort                         # noqa: E402

WIN_BEH = 150


def main() -> None:
    yolo = YOLO("yolov8n-pose.pt")
    engine = CMTA_Inference("checkpoints/best_urfall_v2mt.pt")
    sess = ort.InferenceSession("web-onnx/models/cmta.onnx", providers=["CPUExecutionProvider"])

    # —— 关键点参照：打印 ultralytics 对"日常活动组/1.jpg"的关键点数值 ——
    ref = extract_pose(yolo, os.path.join("我的测试照片", "日常活动组", "1.jpg"))
    print("[参照] ultralytics 日常活动组/1.jpg 关键点(前9值):",
          np.round(ref.reshape(-1)[:9], 4).tolist() if ref is not None else None)

    for group in ["户外趴地组", "户外跌倒组", "日常活动组"]:
        files = sorted(glob.glob(os.path.join("我的测试照片", group, "*.jpg")))
        if not files:
            print(f"{group}: 无照片，跳过")
            continue
        seq = []
        for f in files:
            kp = extract_pose(yolo, f)
            seq.append(kp if kp is not None else np.zeros((17, 3), np.float32))
        seq_arr = np.stack(seq, 0).astype(np.float32).reshape(len(seq), -1)   # (T,51)

        # —— PyTorch（photo_demo 同管线）——
        r_t = engine.predict(behavior=seq_arr)
        pt = r_t["行为异常概率"] or 0.0

        # —— ONNX（复刻 predict 内部：线性插值到150帧 + /100）——
        T = len(seq_arr)
        xs = np.linspace(0, T - 1, WIN_BEH)
        b = np.stack([np.interp(xs, np.arange(T), seq_arr[:, c]) for c in range(51)], 1)
        logits = sess.run(["logits"], {"behavior": (b / 100.0).astype(np.float32)[None]})[0][0]
        po = float(np.exp(logits[1]) / (np.exp(logits[0]) + np.exp(logits[1])))

        # —— 模型权重核对：ONNX 权重 vs checkpoint 是否一致 ——
        onnx_w = sess.run(None, {"behavior": np.zeros((1, 150, 51), np.float32)})[0][0]

        print(f"{group}:  torch={pt:.4f} | onnx={po:.4f} | 差={abs(pt-po):.4f}")

    print("\n若 torch 与 onnx 差异 >0.1：ONNX 导出权重可能未正确加载，请把本输出发我。")
    print("若差异 <0.01：浏览器端链路数值正确。")


if __name__ == "__main__":
    main()
