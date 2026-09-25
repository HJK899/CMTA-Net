"""对齐管线 + PyTorch Dataset（真实数据版）。

输入：data/splits/all_samples.csv（整段级样本表）+ <split>.csv（按受试者划分）
职责：
1. 按 split/dataset/mode 过滤样本；
2. pose（整段）与 physio（整段）按「段内比例」同步取窗口；
3. 训练期窗口起点随机（数据增强），验证/测试取中部；
4. 返回模型输入 dict（behavior 可为 None → 单流生理训练）。

窗口对齐：pose 整段 T_p 帧 ↔ physio 整段 T_q 点（比例对齐），
窗口 physio 长度 = round(win_frames / T_p * T_q)，再插值到 physio_win 固定点数。
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class CMTA_Dataset(Dataset):
    def __init__(self, split: str, cfg, split_dir: str = "data/splits",
                 samples_csv: str = "all_samples.csv",
                 pose_dir: str = "data/features/pose",
                 physio_dir: str = "data/features/physio",
                 mode: str = "both", dataset: str = "urfall",
                 augment: bool = False):
        """
        mode: both=双模态（需 pose）；physio=仅生理流；behavior=仅行为流
        dataset: urfall / upfall / geriatric（对应划分文件前缀）
        """
        self.cfg = cfg
        self.mode = mode
        self.dataset = dataset
        self.augment = augment
        self.win_frames = int(cfg.data.window_sec * cfg.data.behavior_fps)
        self.physio_win = int(cfg.data.window_sec * cfg.data.physio_fs)

        base = Path(split_dir)
        all_csv = base / samples_csv
        if not all_csv.exists():
            raise FileNotFoundError(f"样本表不存在: {all_csv}（先运行 data/build_features.py）")
        self.all = pd.read_csv(all_csv, encoding="utf-8")
        self.all["subject"] = self.all["subject"].astype(str)

        # 选数据集
        df = self.all[self.all["dataset"] == dataset].copy()
        if mode == "both":
            df = df[df["pose_path"].notna() & (df["pose_path"] != "")]
        elif mode == "physio":
            df = df[df["physio_path"].notna() & (df["physio_path"] != "")]
        if len(df) == 0:
            raise ValueError(f"dataset={dataset} mode={mode} 无样本")

        # 划分文件（<dataset>_<split>.csv 或 <split>.csv）
        split_csv = base / f"{dataset}_{split}.csv"
        if not split_csv.exists():
            split_csv = base / f"{split}.csv"
        if not split_csv.exists():
            raise FileNotFoundError(f"划分文件不存在: {split_csv}（先运行 data/split_by_subject.py）")
        sp = pd.read_csv(split_csv, encoding="utf-8")
        self.df = df[df["sample_id"].isin(sp["sample_id"])].reset_index(drop=True)

        self.pose_dir = Path(pose_dir)
        self.physio_dir = Path(physio_dir)
        self._cache: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.df)

    def _load(self, path: str) -> np.ndarray:
        if path not in self._cache:
            self._cache[path] = np.load(path).astype(np.float32)
        return self._cache[path]

    def _crop(self, arr: np.ndarray, win: int, rng: np.random.Generator | None):
        """取长度为 win 的窗口：rng 非空则随机起点，否则中部。"""
        n = len(arr)
        if n <= win:
            return arr, 0
        if rng is not None:
            start = rng.integers(0, n - win + 1)
        else:
            start = (n - win) // 2
        return arr[start:start + win], start

    @staticmethod
    def _time_warp(x: np.ndarray, rng: np.random.Generator, n_seg: int = 3, sigma: float = 0.18):
        """时间扭曲：把 (C,T) 按 n_seg 段独立随机伸缩后回采样到 T（跨通道保持对齐）。"""
        T = x.shape[1]
        seg_pts = np.linspace(0, T, n_seg + 1).astype(int)
        warped, out_len = [], 0
        for i in range(n_seg):
            s, e = seg_pts[i], seg_pts[i + 1]
            k = float(np.clip(1.0 + rng.normal(0, sigma), 0.6, 1.4))
            n_new = max(1, int(round((e - s) * k)))
            xs = np.linspace(s, e - 1, n_new)
            warped.append(np.stack([np.interp(xs, np.arange(s, e), x[c, s:e])
                                    for c in range(x.shape[0])], axis=0))
            out_len += n_new
        total = np.concatenate(warped, axis=1)
        xs = np.linspace(0, total.shape[1] - 1, T)
        return np.stack([np.interp(xs, np.arange(total.shape[1]), total[c])
                         for c in range(total.shape[0])], axis=0).astype(np.float32)

    @staticmethod
    def _gauss_noise(x: np.ndarray, rng: np.random.Generator, std: float = 0.05):
        return (x + rng.normal(0.0, std, size=x.shape)).astype(np.float32)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        label = int(row["label"])
        rng = np.random.default_rng(None) if self.augment else None

        out: dict = {"label": torch.as_tensor(label, dtype=torch.long)}

        # ---- 行为窗口 ----
        if self.mode in ("both", "behavior"):
            pose = self._load(row["pose_path"])            # (T_p, 17, 3)
            if self.mode == "both":
                win = self.win_frames
            else:
                win = min(self.win_frames, len(pose))
            p_crop, p_start = self._crop(pose, win, rng)
            if len(p_crop) < self.win_frames:
                pad = np.zeros((self.win_frames - len(p_crop), 17, 3), dtype=np.float32)
                p_crop = np.concatenate([p_crop, pad], axis=0)
            out["behavior"] = torch.as_tensor(
                (p_crop.reshape(self.win_frames, -1) / 100.0), dtype=torch.float32
            )

        # ---- 生理窗口（与行为窗口按比例对齐）----
        if self.mode in ("both", "physio"):
            phys = self._load(row["physio_path"])          # (C, T_q)
            T_q = phys.shape[1]
            if self.mode == "both":
                # 与行为窗口比例对齐：p_start/pose_T → q_start/T_q
                pose_T = len(self._load(row["pose_path"]))
                q_start = int(round(p_start / max(pose_T, 1) * T_q))
                q_len = max(1, int(round(self.win_frames / max(pose_T, 1) * T_q)))
                win_q = phys[:, q_start:q_start + q_len]
            else:
                q_start = 0
                win_q, _ = self._crop(phys.T, min(self.physio_win, T_q), rng)
                win_q = win_q.T
            # 插值到固定 physio_win 点
            if win_q.shape[1] < self.physio_win:
                pad = np.zeros((win_q.shape[0], self.physio_win - win_q.shape[1]), dtype=np.float32)
                win_q = np.concatenate([win_q, pad], axis=1)
            elif win_q.shape[1] > self.physio_win:
                xs = np.linspace(0, win_q.shape[1] - 1, self.physio_win)
                win_q = np.stack([
                    np.interp(xs, np.arange(win_q.shape[1]), win_q[c])
                    for c in range(win_q.shape[0])
                ], axis=0).astype(np.float32)
            # 数据增强（仅训练期）：时间扭曲 + 高斯噪声（标准化前，保留信号形状）
            if self.augment:
                win_q = self._time_warp(win_q, rng)
                win_q = self._gauss_noise(win_q, rng)
            # 通道标准化
            mu = win_q.mean(axis=1, keepdims=True)
            sd = win_q.std(axis=1, keepdims=True) + 1e-6
            out["physio"] = torch.as_tensor((win_q - mu) / sd, dtype=torch.float32)

        if self.mode == "both":
            out["behavior"] = out["behavior"]  # 保持
        return out


def collate(batch: list[dict]) -> dict:
    """DataLoader collate：支持 behavior 可选。"""
    out: dict = {}
    if "behavior" in batch[0]:
        out["behavior"] = torch.stack([b["behavior"] for b in batch])
    if "physio" in batch[0]:
        out["physio"] = torch.stack([b["physio"] for b in batch])
    out["label"] = torch.stack([b["label"] for b in batch])
    return out
