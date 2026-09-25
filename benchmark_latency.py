# -*- coding: utf-8 -*-
"""推理时延与参数量评测：单窗口（5s）CPU 推理时延 + 模型大小，论证边缘部署可行性。

用法: python benchmark_latency.py --checkpoint checkpoints/best_upfall_v2.pt
"""
import argparse
import time

import numpy as np
import torch

from config import get_config

cfg = get_config()
from models.cmta_net import CMTA_Net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--repeats", type=int, default=200)
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    m = cfg.model
    d_model = int(ckpt["config"]["model"]["d_model"])
    physio_channels = int(ckpt["config"]["model"]["physio_channels"])
    num_classes = int(ckpt["config"]["data"]["num_classes"])

    model = CMTA_Net(
        pose_kps=m.pose_kps, d_model=d_model, physio_channels=physio_channels,
        num_classes=num_classes, num_gru_layers=m.num_gru_layers,
        attn_temperature=m.attn_temperature, dropout=m.dropout,
    )
    model.load_state_dict(ckpt["model"])
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    mem_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6
    ckpt_mb = __import__("os").path.getsize(args.checkpoint) / 1e6

    win_frames = int(cfg.data.window_sec * cfg.data.behavior_fps)
    physio_win = int(cfg.data.window_sec * cfg.data.physio_fs)

    # 单窗口随机输入（16ch×100 或 4ch×100）
    x_phys = torch.randn(1, physio_channels, physio_win)
    x_beh = torch.randn(1, win_frames, 17 * 3)

    # warmup
    with torch.no_grad():
        for _ in range(10):
            model(behavior=x_beh, physio=x_phys)

    # 测时延
    times = []
    with torch.no_grad():
        for _ in range(args.repeats):
            t0 = time.perf_counter()
            model(behavior=x_beh, physio=x_phys)
            times.append((time.perf_counter() - t0) * 1000)
    times = np.array(times)

    print(f"参数量: {n_params/1e6:.3f} M")
    print(f"参数内存: {mem_mb:.2f} MB | checkpoint 大小: {ckpt_mb:.2f} MB")
    print(f"单窗口推理时延(CPU): mean={times.mean():.3f} ms  p50={np.percentile(times,50):.3f} ms  p95={np.percentile(times,95):.3f} ms")
    print(f"等效采样率(5s窗口/次): 理论吞吐 {1000/times.mean():.0f} 窗口/秒（可覆盖 20Hz 实时流）")


if __name__ == "__main__":
    main()
