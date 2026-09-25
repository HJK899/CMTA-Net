"""Day1 任务：用 Ultralytics YOLOv8-Pose 从视频提取姿态关键点特征。

输出：每个视频保存为一个 .npy，形状 (T, 17, 3) —— T 帧，17 个关键点，每点 (x, y, conf)。
使用方式：
    python data/extract_pose.py --input data/raw/upfall --output data/features/pose

说明：
- 首次运行会自动下载 yolov8n-pose.pt 预训练权重（约 6MB）。
- 递归遍历 input 下所有视频文件（mp4/avi/mov/mkv）。
- 同一视频的帧数按实际帧率保存；窗口切分在 build_dataset.py 完成。
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
from ultralytics import YOLO

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
NUM_JOINTS = 17  # YOLOv8-Pose 默认 17 关键点（COCO 骨架）


def extract_one(model: YOLO, video_path: Path, out_path: Path, device: str) -> int:
    """对单个视频提取关键点序列并保存 .npy，返回帧数。"""
    frames: list[np.ndarray] = []
    for result in model.predict(str(video_path), stream=True, verbose=False, device=device):
        kps = result.keypoints
        if kps is None or len(kps.data) == 0:
            # 当前帧未检测到人体：零填充，保证序列长度连续
            frames.append(np.zeros((NUM_JOINTS, 3), dtype=np.float32))
        else:
            # 取检测置信度最高的人体
            arr = kps.data[0].cpu().numpy().astype(np.float32)  # (17, 3)
            frames.append(arr)
    if not frames:
        print(f"  [skip] {video_path.name}: 无可提取帧")
        return 0
    seq = np.stack(frames, axis=0)  # (T, 17, 3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, seq)
    return len(seq)


def main() -> None:
    ap = argparse.ArgumentParser(description="YOLOv8-Pose 提取姿态特征")
    ap.add_argument("--input", required=True, help="视频根目录")
    ap.add_argument("--output", default="data/features/pose", help="特征输出目录")
    ap.add_argument("--device", default="auto", help="推理设备：auto/cpu/cuda")
    args = ap.parse_args()

    model = YOLO("yolov8n-pose.pt")  # 首次运行自动下载权重
    device = args.device if args.device != "auto" else ("cuda" if _cuda_available() else "cpu")

    src = Path(args.input)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    videos = sorted(p for p in src.rglob("*") if p.suffix.lower() in VIDEO_EXTS)
    print(f"找到 {len(videos)} 个视频，设备: {device}")
    total_frames = 0
    n_ok = 0
    for v in videos:
        rel = v.relative_to(src).with_suffix(".npy")
        dest = out / rel
        if dest.exists():  # 断点续跑
            continue
        n = extract_one(model, v, dest, device)
        if n > 0:
            n_ok += 1
            total_frames += n
            print(f"  [ok] {v.name}: {n} 帧 → {dest}")
    print(f"完成：{n_ok}/{len(videos)} 个视频，共 {total_frames} 帧。输出目录: {out}")


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


if __name__ == "__main__":
    main()
