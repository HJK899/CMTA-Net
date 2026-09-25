"""按「受试者」划分训练/验证/测试集（6:2:2），防止数据泄漏。

输入：data/splits/all_samples.csv（build_features.py 生成，含 subject/label/pose_path/physio_path）
输出：data/splits/<dataset>_<train|val|test>.csv

各数据集受试者规则：
- urfall：无受试者标注，样本键 F01..F30 / A01..A40 作为「段级」划分键
  （UR Fall 公开论文亦采用段级划分；双视角已保证同段同侧，每段仅一个视角）
- upfall：Subject 1-17（真实受试者，按人划分）
- geriatric：user1..user41（真实老人，按人划分）
"""
from __future__ import annotations
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


def write_split(name: str, rows: list[dict], split_dir: Path) -> None:
    rows.sort(key=lambda r: r["subject"])
    csv_path = split_dir / f"{name}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["sample_id", "dataset", "subject", "label",
                                          "pose_path", "physio_path", "seg_points", "channels"])
        w.writeheader()
        for r in rows:  # 保留原始 sample_id（与样本表一致）
            w.writerow(r)
    print(f"  {name}: {len(rows)} 样本（受试者 {len(set(r['subject'] for r in rows))} 人）→ {csv_path}")


def split_dataset(df: pd.DataFrame, seed: int, split_dir: Path) -> None:
    ds = df["dataset"].iloc[0]
    by_subject: dict[str, list] = defaultdict(list)
    for _, r in df.iterrows():
        by_subject[str(r["subject"])].append(r.to_dict())
    subjects = sorted(by_subject.keys())
    rng = np.random.default_rng(seed)
    rng.shuffle(subjects)

    n = len(subjects)
    n_train = max(1, int(round(n * 0.6)))
    n_val = max(1, int(round(n * 0.2)))
    tr, va, te = set(subjects[:n_train]), set(subjects[n_train:n_train + n_val]), set(subjects[n_train + n_val:])

    print(f"[{ds}] 受试者 {n} → train {len(tr)} / val {len(va)} / test {len(te)}")
    write_split(f"{ds}_train", [r for s in tr for r in by_subject[s]], split_dir)
    write_split(f"{ds}_val", [r for s in va for r in by_subject[s]], split_dir)
    write_split(f"{ds}_test", [r for s in te for r in by_subject[s]], split_dir)


def main() -> None:
    ap = argparse.ArgumentParser(description="按受试者划分数据集")
    ap.add_argument("--samples", default="data/splits/all_samples.csv")
    ap.add_argument("--split-dir", default="data/splits")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--datasets", nargs="+", default=["urfall", "upfall", "geriatric"])
    args = ap.parse_args()

    split_dir = Path(args.split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.samples, encoding="utf-8")
    df["subject"] = df["subject"].astype(str)

    for ds in args.datasets:
        sub = df[df["dataset"] == ds]
        if len(sub) == 0:
            print(f"[{ds}] 无样本，跳过")
            continue
        split_dataset(sub, args.seed, split_dir)


if __name__ == "__main__":
    main()
