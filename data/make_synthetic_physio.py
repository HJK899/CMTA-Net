"""Day1 自测用：生成合成生理特征（心率 + 加速度），用于在真实数据未下载完时跑通全流程。

⚠ 合成数据仅用于代码管线自测（验证 train.py / evaluate.py 能跑通），
   **不是最终实验数据**。真实实验以 PPG-DaLiA / Zenodo 老年数据为准。

输出：data/features/physio/synthetic.npz，结构：
    - features: (N, C=4, T)  生理窗口特征（hr, ppg, acc_x, acc_y）
    - labels:   (N,)         0=正常 / 1=异常
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np

N_SAMPLES = 200
NORMAL_FS = 1.0


def make_heart_rate(base: float, n: int, abnormal: bool, rng: np.random.Generator) -> np.ndarray:
    """心率曲线：基线 + 呼吸变异 + 异常时叠加骤升/骤降模式。"""
    t = np.arange(n)
    hr = base + 2.0 * np.sin(2 * np.pi * 0.02 * t) + rng.normal(0, 1.0, n).cumsum() * 0.02
    if abnormal:
        # 在窗口后 1/3 处注入异常（骤升）
        start = int(n * 0.66)
        hr[start:] += np.linspace(0, 40, n - start) + rng.normal(0, 2, n - start)
    return np.clip(hr, 50, 180)


def main() -> None:
    ap = argparse.ArgumentParser(description="生成合成生理特征（自测用）")
    ap.add_argument("--output", default="data/features/physio/synthetic.npz")
    ap.add_argument("--n", type=int, default=N_SAMPLES, help="样本数")
    args = ap.parse_args()

    rng = np.random.default_rng(42)
    T = 120  # 120 秒窗口 @1Hz
    feats, labels = [], []
    for i in range(args.n):
        abnormal = rng.random() < 0.3
        base = rng.uniform(65, 85)
        hr = make_heart_rate(base, T, abnormal, rng)
        ppg = 100 + 8 * np.sin(2 * np.pi * (hr / 60) * np.arange(T) / 60) + rng.normal(0, 3, T)
        ax = 0.1 * np.sin(0.1 * np.arange(T)) + rng.normal(0, 0.05, T)
        ay = 0.1 * np.cos(0.1 * np.arange(T)) + rng.normal(0, 0.05, T)
        feats.append(np.stack([hr, ppg, ax, ay], axis=0).astype(np.float32))  # (4, T)
        labels.append(1 if abnormal else 0)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, features=np.stack(feats), labels=np.array(labels, dtype=np.int64))
    print(f"生成 {args.n} 个合成生理样本（{int(sum(labels))} 异常 / {args.n - int(sum(labels))} 正常）→ {out}")
    print("⚠ 此数据仅用于管线自测，非最终实验数据。")


if __name__ == "__main__":
    main()
