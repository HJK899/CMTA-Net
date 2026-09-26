# -*- coding: utf-8 -*-
"""urfall_demo.py —— 双模态四态诊断演示（替代 UR Fall 原始数据缺失的现场评估）。

背景：UR Fall 官方数据服务器迁移（fenix.ur.edu.pl 数据文件 404、旧域名 DNS 注销、
Kaggle 镜像 4.5GB 限速至 0、GitCode 无实体），原始视频/传感不可恢复。
本脚本改用"真实输入"演示四态诊断能力（不依赖 UR Fall 原始数据）：

    行为流 ← 真实视频骨架（默认 fall10k 户外跌倒视频，YOLOv8-Pose 150 帧时序）
    生理流 ← 真实传感窗口（默认 UP-Fall 跌倒段：EEG+腕戴加速度，已恢复）

两者喂入已训练的双模态模型 best_urfall_v2mt.pt（含生理/行为单流头），输出
四态诊断（正常/生理异常/行为异常/联合异常）+ 双流概率 + 模态一致性分数。

注意：
- 该模型为 UR Fall 训练（传感流 4 通道 [x,y,z,mag]），此处生理流为 UP-Fall
  [EEG, wrist_x, wrist_y, wrist_z] 跨域输入，属演示口径；predict 内部做逐通道
  标准化，重点展示四态决策链路与模态一致性机制。
- UR Fall 测试集评估指标以技术报告已记录实验为准。

用法：
    python urfall_demo.py                            # 默认演示
    python urfall_demo.py --video 视频.mp4           # 更换行为视频
    python urfall_demo.py --physio 某段.npy          # 更换生理整段(4通道)
    python urfall_demo.py --tau 0.76                 # 判定阈值(默认0.76)
"""
from __future__ import annotations
import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from photo_demo import extract_pose, video_to_photos            # noqa: E402
from inference import CMTA_Inference                            # noqa: E402

CHECKPOINT = "checkpoints/best_urfall_v2mt.pt"
UPFALL_CH_WRIST = [0, 13, 14, 15]      # full16 通道中的 [EEG, WristAcc xyz]
WIN_PHYS = 100                          # 5s @ 20Hz
WIN_BEH = 150                           # 5s @ 30fps


def load_physio_window(npy_path: str, ch_idx: list[int]) -> np.ndarray:
    """读整段生理特征 (C,T)，取指定通道的 5 秒窗口 → (len(ch_idx), WIN_PHYS)。"""
    arr = np.load(npy_path)                     # (C, T)
    sub = arr[ch_idx].astype(np.float32)        # (4, T)
    t = sub.shape[1]
    if t >= WIN_PHYS:
        start = (t - WIN_PHYS) // 2
        return sub[:, start:start + WIN_PHYS]
    xs = np.linspace(0, t - 1, WIN_PHYS)
    return np.stack([np.interp(xs, np.arange(t), sub[c]) for c in range(sub.shape[0])], 0)


def find_upfall_fall_npy(samples_csv: str) -> str:
    """从样本表找第一个 upfall 跌倒(label=1)段的 physio 路径。"""
    with open(samples_csv, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["dataset"] == "upfall" and row["label"] == "1" and row["physio_path"]:
                return row["physio_path"]
    raise FileNotFoundError("样本表中没有 upfall 跌倒段")


def main() -> None:
    ap = argparse.ArgumentParser(description="CMTA-Net 双模态四态诊断演示")
    ap.add_argument("--video", default=r"data/raw/fall10k/extracted/sample_1.mp4",
                    help="行为视频（默认 fall10k 户外跌倒）")
    ap.add_argument("--physio", default="", help="生理整段 npy（默认自动选 UP-Fall 跌倒段）")
    ap.add_argument("--checkpoint", default=CHECKPOINT)
    ap.add_argument("--tau", type=float, default=0.76)
    ap.add_argument("--frames", type=int, default=150, help="视频抽帧数(默认150)")
    args = ap.parse_args()

    print("=" * 64)
    print("CMTA-Net 双模态四态诊断演示（真实输入，不依赖 UR Fall 原始数据）")
    print("=" * 64)

    # 1) 行为流：视频 → 骨架序列
    from ultralytics import YOLO
    print("\n[1/3] 行为流：视频抽帧 → YOLOv8-Pose 骨架")
    yolo = YOLO("yolov8n-pose.pt")
    gname, photos = video_to_photos(args.video, args.frames)
    if not photos:
        print(f"错误：视频读取失败 {args.video}")
        sys.exit(1)
    seq: list[np.ndarray] = []
    missing = 0
    for p in photos:
        kp = extract_pose(yolo, p)
        if kp is None:
            missing += 1
            kp = np.zeros((17, 3), dtype=np.float32)
        seq.append(kp)
    seq_arr = np.stack(seq, 0).astype(np.float32)          # (T,17,3)
    behavior = seq_arr.reshape(len(seq), -1)               # (T,51)
    if missing:
        print(f"  [注意] {missing}/{len(seq)} 帧未检出人体（零填充保时序）")
    print(f"  视频「{gname}」{len(photos)} 帧 → 行为窗口 ({behavior.shape[0]}, 51)")

    # 2) 生理流：真实传感窗口
    print("\n[2/3] 生理流：真实传感窗口（UP-Fall 跌倒段，通道=EEG+腕戴acc）")
    physio_npy = args.physio or find_upfall_fall_npy(os.path.join(BASE, "data", "splits", "all_samples.csv"))
    physio = load_physio_window(physio_npy, UPFALL_CH_WRIST)
    print(f"  来源 {Path(physio_npy).name} → 窗口 ({physio.shape[0]}, {physio.shape[1]}) @20Hz")

    # 3) 双模态推理 → 四态
    print(f"\n[3/3] 加载模型 {args.checkpoint}（tau={args.tau}）并推理…")
    engine = CMTA_Inference(args.checkpoint, tau=args.tau)
    res = engine.predict(physio=physio, behavior=behavior)

    print("\n" + "=" * 64)
    print("双模态四态诊断结果")
    print("=" * 64)
    for k in ("state", "state_name", "异常概率(融合)", "生理异常概率",
              "行为异常概率", "模态一致性s", "置信度", "阈值tau"):
        print(f"  {k}: {res.get(k)}")
    if res["state"] == 3:
        print("判定：联合异常（生理+行为双流均超阈值）→ 立即告警/家属通知。")
    elif res["state"] in (1, 2):
        print("判定：单模态异常 → 触发对应复核流程。")
    else:
        print("判定：正常，未触发告警。")
    print("\n说明：UR Fall 官方数据服务器迁移致原始数据不可恢复，本演示使用")
    print("真实视频骨架（fall10k）× 真实传感窗口（UP-Fall）展示四态诊断链路；")
    print("UR Fall 测试集评估指标见技术报告已记录实验。")


if __name__ == "__main__":
    main()
