"""Day1 任务：生理时序数据 → 窗口特征。

输入约定：生理数据根目录下每个样本一个 CSV/NPY 文件，列为时间通道：
    - 格式 A（推荐，心率为主）：一列 `hr`（心率，Hz 级）
    - 格式 B：多列生理信号（ppg, acc_x, acc_y, acc_z, ...），会先降采样到 physio_fs
输出：data/features/physio/<样本名>.npy，形状 (C, T)（T = window_sec * physio_fs）

使用方式：
    python data/preprocess_physio.py --input data/raw/ppg_dalia --output data/features/physio

若数据集还没下载完，可先用 `python data/make_synthetic_physio.py` 生成合成生理特征，
先跑通全流程，再替换真实数据。
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np


def load_series(path: Path) -> np.ndarray:
    """读取单个样本的时序数据，返回 (T, C) 数组。支持 .npy / .csv。"""
    if path.suffix.lower() == ".npy":
        arr = np.load(path).astype(np.float32)
        return arr[:, :] if arr.ndim == 2 else arr.reshape(-1, 1)
    # CSV：跳过表头，数值列全读入
    raw = np.genfromtxt(path, delimiter=",", skip_header=1, dtype=np.float32)
    if raw.ndim == 1:
        raw = raw.reshape(-1, 1)
    return raw


def main() -> None:
    ap = argparse.ArgumentParser(description="生理时序 → 窗口特征")
    ap.add_argument("--input", required=True, help="生理原始数据根目录")
    ap.add_argument("--output", default="data/features/physio")
    ap.add_argument("--window-sec", type=float, default=5.0, help="窗口长度（秒）")
    ap.add_argument("--fs", type=float, default=1.0, help="目标采样率（Hz）")
    args = ap.parse_args()

    src = Path(args.input)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    files = sorted(
        [p for p in src.rglob("*") if p.suffix.lower() in {".npy", ".csv"} and not p.name.startswith(".")]
    )
    print(f"找到 {len(files)} 个生理样本文件")
    win_len = int(args.window_sec * args.fs)
    for f in files:
        try:
            series = load_series(f)
        except Exception as e:  # 单文件失败不阻塞整体
            print(f"  [skip] {f.name}: {e}")
            continue
        # 统一采样到 fs：按比例线性插值到目标长度
        target_len = int(len(series) / series.shape[0] * series.shape[0])  # 保持原长
        # 简单重采样：对每个通道插值到 (T_win * fs) 长度的整数倍窗口
        n_win = max(1, target_len // win_len)
        usable = n_win * win_len
        xs = np.linspace(0, len(series) - 1, usable)
        resampled = np.stack(
            [np.interp(xs, np.arange(len(series)), series[:, c]) for c in range(series.shape[1])],
            axis=0,  # (C, usable)
        ).astype(np.float32)
        dest = out / f.with_suffix(".npy").name
        np.save(dest, resampled[:, :usable])
        print(f"  [ok] {f.name}: {resampled.shape[0]} 通道 × {usable} 点 → {dest}")

    print(f"完成，输出目录: {out}")


if __name__ == "__main__":
    main()
