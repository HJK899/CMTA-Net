"""真实数据特征构建：三类传感数据 → 统一 4 通道生理特征（整段）+ 样本表。

数据来源与通道设计（统一 C=4）：
- UR Fall acc CSV：      [x, y, z, mag]            （fall/adl 双模态配对中的传感流）
- UP-Fall CSV：          [EEG, wrist_x, wrist_y, wrist_z]（脑电+腕戴，17人11类）
- 老年腕戴 CSV：         [x, y, z, a]              （a 即总加速度 magnitude，41人）

输出：
- data/features/physio/<dataset>/<样本名>.npy  整段生理特征 (C, T)
- data/splits/all_samples.csv                  整段级样本表

说明：窗口级切分与增强在 build_dataset.py 完成（按比例对齐 pose/physio）。
"""
from __future__ import annotations
import argparse
import csv
import re
from pathlib import Path

import numpy as np
import pandas as pd

FS = 20.0  # 统一重采样率（与 config.physio_fs 一致）
POSE_ROOT = Path("data/features/pose")  # 姿态特征根目录（UR Fall 配对）

# UP-Fall Activity 编号：1-5 跌倒，6-11 日常
UPFALL_FALL = {1, 2, 3, 4, 5}


def resample_to_fs(series: np.ndarray, target_fs: float = FS) -> np.ndarray:
    """(T, C) → (C, T')，线性插值重采样到 target_fs 对应的固定点数。
    输入序列时长按原始行数/实际采样率估算，此处统一映射到「等比例时间轴」：
    为与窗口对齐逻辑一致，这里不做绝对时间假设，仅按序列长度重采样。
    """
    T = series.shape[0]
    target_len = max(8, int(T))  # 整段保留：由 build_dataset 按比例取窗
    xs = np.linspace(0, T - 1, target_len)
    out = np.stack(
        [np.interp(xs, np.arange(T), series[:, c]) for c in range(series.shape[1])],
        axis=0,
    ).astype(np.float32)
    return out


def safe_mag(xyz: np.ndarray) -> np.ndarray:
    return np.sqrt(np.clip((xyz ** 2).sum(axis=1), 0, None))


# ---------- UR Fall acc ----------
def build_urfall_acc(raw_dir: Path, out_dir: Path, rows: list) -> None:
    acc_files = sorted(raw_dir.glob("*-acc.csv"))
    n_ok = 0
    for f in acc_files:
        stem = f.name.replace("-acc.csv", "")  # fall-01 / adl-01
        # CSV 无表头：列0=时间戳，列1-4=acc x,y,z,w(方向余弦相关)
        df = pd.read_csv(f, header=None)
        vals = df.iloc[:, 1:].to_numpy(dtype=np.float64)
        if vals.shape[1] < 3:
            print(f"  [skip] {f.name}: 列数不足 {vals.shape[1]}")
            continue
        xyz = vals[:, :3]
        mag = safe_mag(xyz)
        phys = np.stack([xyz[:, 0], xyz[:, 1], xyz[:, 2], mag], axis=1)  # (T,4)
        feat = resample_to_fs(phys)
        dest = out_dir / f"{stem}.npy"
        np.save(dest, feat)
        label = 1 if stem.startswith("fall") else 0
        subject = ("F" if stem.startswith("fall") else "A") + stem.split("-")[1]
        # 配对姿态特征：fall-XX → fall-XX-cam1.npy；adl-XX → adl-XX-cam0.npy（本机下载的视角）
        cam = "cam1" if stem.startswith("fall") else "cam0"
        pose_path = str(POSE_ROOT / f"{stem}-{cam}.npy") if POSE_ROOT.exists() else ""
        rows.append({
            "sample_id": f"urfall_{stem}",
            "dataset": "urfall", "subject": subject, "label": label,
            "pose_path": pose_path, "physio_path": str(dest),
            "seg_points": feat.shape[1], "channels": feat.shape[0],
        })
        n_ok += 1
    print(f"UR Fall acc: {n_ok} 段")


# ---------- UP-Fall ----------
SENSOR_ACC = [("AnkleAccelerometer", 1), ("RightPocketAccelerometer", 8),
              ("BeltAccelerometer", 15), ("NeckAccelerometer", 22), ("WristAccelerometer", 29)]


def build_upfall(raw_path: Path, out_dir: Path, rows: list, channels: str = "wrist") -> None:
    """channels: wrist=原4通道(EEG+腕戴acc3)；full=16通道(EEG+5处acc15)；full31=31通道(+5处gyro15)。"""
    def to_num(s):
        return pd.to_numeric(s, errors="coerce").to_numpy(dtype=np.float64)

    df = pd.read_csv(raw_path, low_memory=False)
    cols = list(df.columns)
    brain = to_num(df["BrainSensor"])

    if channels == "full":
        ch = [brain]
        for name, start in SENSOR_ACC:
            ch += [to_num(df[cols[start + 1]]), to_num(df[cols[start + 2]]), to_num(df[cols[start + 3]])]
        phys_all = np.stack(ch, axis=1)  # (N, 16)
    elif channels == "full31":
        ch = [brain]
        for name, start in SENSOR_ACC:
            ch += [to_num(df[cols[start + 1]]), to_num(df[cols[start + 2]]), to_num(df[cols[start + 3]])]
            ch += [to_num(df[cols[start + 5]]), to_num(df[cols[start + 6]]), to_num(df[cols[start + 7]])]
        phys_all = np.stack(ch, axis=1)  # (N, 31)
    else:
        wi = cols.index("WristAccelerometer")
        wx = to_num(df[cols[wi + 1]])
        wy = to_num(df[cols[wi + 2]])
        wz = to_num(df[cols[wi + 3]])
        phys_all = np.stack([brain, wx, wy, wz], axis=1)  # (N, 4)

    subj = to_num(df["Subject"])
    act = to_num(df["Activity"])
    trial = to_num(df["Trial"])

    # 过滤无效行（表头残留/空行），再按段切分
    valid = ~(np.isnan(subj) | np.isnan(act) | np.isnan(trial))
    idx = np.where(valid)[0]
    segs: list[tuple[int, int, int, int, int]] = []  # (start, end, sub, act, trial)
    start_i = 0
    for k in range(1, len(idx)):
        if (subj[idx[k]], act[idx[k]], trial[idx[k]]) != (subj[idx[start_i]], act[idx[start_i]], trial[idx[start_i]]):
            segs.append((idx[start_i], idx[k], int(subj[idx[start_i]]), int(act[idx[start_i]]), int(trial[idx[start_i]])))
            start_i = k
    segs.append((idx[start_i], idx[-1] + 1, int(subj[idx[start_i]]), int(act[idx[start_i]]), int(trial[idx[start_i]])))

    n_ok = 0
    for (s, e, sub, act, tr) in segs:
        phys = phys_all[s:e].astype(np.float64)
        # 逐通道前向/线性插值填充 NaN（保留整段，避免丢受试者）
        for c in range(phys.shape[1]):
            col = phys[:, c]
            mask = np.isnan(col)
            if mask.all():
                col[:] = 0.0
            elif mask.any():
                idx = np.where(~mask)[0]
                phys[:, c] = np.interp(np.arange(len(col)), idx, col[idx])
        if len(phys) < 16:
            continue
        feat = resample_to_fs(phys)
        suffix = {"full": "v2", "full31": "v3"}.get(channels, "")
        name = f"sub{sub:02d}_a{act}_t{tr}" + (f"_{suffix}" if suffix else "")
        dest = out_dir / f"{name}.npy"
        np.save(dest, feat)
        label = 1 if act in UPFALL_FALL else 0
        rows.append({
            "sample_id": f"upfall_{name}", "dataset": "upfall", "subject": str(sub),
            "label": label, "pose_path": "", "physio_path": str(dest),
            "seg_points": feat.shape[1], "channels": feat.shape[0],
        })
        n_ok += 1
    print(f"UP-Fall({channels}): {n_ok} 段")


# ---------- 老年腕戴 ----------
def build_geriatric(root: Path, out_dir: Path, rows: list) -> None:
    n_ok = 0
    # 结构：<root>/<adl|fall>/<userN>/<userN_adlK.csv>
    # 文件含两类行：acc/acg 行 [t,x,y,z,a,kind] 与 hrt 行 [t,HR,?,hrt,,]（1Hz 真实心率）
    for f in sorted(root.rglob("*.csv")):
        parts = f.parts
        user = ""
        for p in parts:
            if re.match(r"^user\d+$", p):
                user = p
                break
        if not user:
            print(f"  [skip] {f.name}: 未识别 user 目录")
            continue
        uid = int(re.search(r"(\d+)", user).group(1))
        set_dir = parts[parts.index(user) - 1] if parts.index(user) >= 1 else ""
        label = 1 if set_dir.lower().startswith("fall") else 0
        try:
            df = pd.read_csv(f, header=None)
        except Exception as e:
            print(f"  [skip] {f.name}: {e}")
            continue
        # 按行类型分离：acc/acg 行末列有 kind；hrt 行第4列='hrt'（整行检测更稳）
        arr = df.to_numpy(dtype=str)
        t = pd.to_numeric(pd.Series(arr[:, 0]), errors="coerce").to_numpy(dtype=np.float64)
        is_hrt = np.array([any(str(v).strip().lower().startswith("hrt") for v in row) for row in arr])
        acc_rows = ~is_hrt & ~np.isnan(t)
        if acc_rows.sum() < 16:
            continue
        ta = t[acc_rows]
        raw_xyz = arr[acc_rows, 1:4]  # (n,3) 字符串
        gx = pd.to_numeric(pd.Series(raw_xyz[:, 0]), errors="coerce").to_numpy(dtype=np.float64)
        gy = pd.to_numeric(pd.Series(raw_xyz[:, 1]), errors="coerce").to_numpy(dtype=np.float64)
        gz = pd.to_numeric(pd.Series(raw_xyz[:, 2]), errors="coerce").to_numpy(dtype=np.float64)
        xyz = np.stack([gx, gy, gz], axis=1)
        keep = ~np.isnan(xyz).any(axis=1)
        ta, xyz = ta[keep], xyz[keep]
        if len(xyz) < 16:
            continue
        # 心率：hrt 行 [t, HR, ...]，插值到加速度时间轴
        hr = np.zeros(len(xyz), dtype=np.float32)
        hr_t = t[is_hrt]
        if hr_t.size >= 2:
            hr_raw = pd.to_numeric(pd.Series(arr[is_hrt, 1]), errors="coerce").to_numpy(dtype=np.float64)
            ok = ~np.isnan(hr_raw)
            if ok.sum() >= 2:
                order = np.argsort(hr_t[ok])
                ht, hv = hr_t[ok][order], hr_raw[ok][order]
                lo, hi = max(ht[0], ta.min()), min(ht[-1], ta.max())
                m = (ta >= lo) & (ta <= hi)
                hr[m] = np.interp(ta[m], ht, hv)
        phys = np.stack([xyz[:, 0], xyz[:, 1], xyz[:, 2], hr], axis=1)
        feat = resample_to_fs(phys)
        name = f"{user}_{Path(f.name).stem}"
        dest = out_dir / f"{name}.npy"
        np.save(dest, feat)
        rows.append({
            "sample_id": f"ger_{name}", "dataset": "geriatric", "subject": str(uid),
            "label": label, "pose_path": "", "physio_path": str(dest),
            "seg_points": feat.shape[1], "channels": feat.shape[0],
        })
        n_ok += 1
    print(f"老年腕戴(含心率): {n_ok} 段")


def main() -> None:
    ap = argparse.ArgumentParser(description="构建统一传感生理特征 + 样本表")
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/features/physio")
    ap.add_argument("--splits", default="data/splits")
    ap.add_argument("--upfall-channels", default="wrist", choices=["wrist", "full", "full31"])
    args = ap.parse_args()

    raw = Path(args.raw)
    out = Path(args.out)
    splits = Path(args.splits)
    splits.mkdir(parents=True, exist_ok=True)
    for sub in ("urfall", "upfall", "geriatric"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    build_urfall_acc(raw / "urfall", out / "urfall", rows)
    build_upfall(raw / "upfall" / "CompleteDataSet (1).csv", out / "upfall", rows, channels=args.upfall_channels)
    build_geriatric(raw / "zenodo_ger", out / "geriatric", rows)

    with open(splits / "all_samples.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["sample_id", "dataset", "subject", "label",
                                          "pose_path", "physio_path", "seg_points", "channels"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    # 汇总
    import collections
    cnt = collections.Counter(r["dataset"] for r in rows)
    lab = collections.Counter((r["dataset"], r["label"]) for r in rows)
    print("样本表:", dict(cnt))
    print("标签分布:", dict(lab))
    print(f"样本表 → {splits / 'all_samples.csv'}")


if __name__ == "__main__":
    main()
