"""评估脚本：加载最佳模型，在测试集上输出指标 + 图表。

用法：
    python evaluate.py --checkpoint checkpoints/best.pt
    python evaluate.py --checkpoint checkpoints/best.pt --split test --out outputs/eval

输出：
- 文本：混淆矩阵、精确率/召回率/F1、准确率、AUC、最优阈值
- 图表（需 matplotlib/sklearn）：outputs/eval_cm.png、outputs/eval_roc.png
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import get_config
from data.build_dataset import CMTA_Dataset, collate
from models.cmta_net import CMTA_Net
from utils.metrics import classification_report, save_figures


def main() -> None:
    ap = argparse.ArgumentParser(description="CMTA-Net 评估")
    ap.add_argument("--checkpoint", default="checkpoints/best.pt")
    ap.add_argument("--split", default="test", help="train/val/test")
    ap.add_argument("--out", default="outputs/eval")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dataset", default="urfall", choices=["urfall", "upfall", "geriatric"])
    ap.add_argument("--stream", default="both", choices=["both", "physio", "behavior"])
    args = ap.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"检查点不存在: {ckpt_path}（先运行 train.py）")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    cfg = get_config()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    num_classes = int(ckpt["config"]["data"]["num_classes"])
    m = cfg.model
    physio_channels = int(ckpt["config"]["model"]["physio_channels"])
    d_model = int(ckpt["config"]["model"]["d_model"])
    model = CMTA_Net(
        pose_kps=m.pose_kps, d_model=d_model, physio_channels=physio_channels,
        num_classes=num_classes, num_gru_layers=m.num_gru_layers,
        attn_temperature=m.attn_temperature, dropout=m.dropout,
    )
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    print(f"[model] 已加载 {ckpt_path}（epoch {ckpt['config']['epochs']}）")

    ds = CMTA_Dataset(args.split, cfg, dataset=args.dataset, mode=args.stream, augment=False)
    loader = DataLoader(ds, batch_size=cfg.train.batch_size, shuffle=False,
                        num_workers=0, collate_fn=collate)

    y_true, y_prob, s_scores = [], [], []
    with torch.no_grad():
        for batch in loader:
            behav = batch["behavior"].to(device) if "behavior" in batch and args.stream in ("both", "behavior") else None
            physio = batch["physio"].to(device) if "physio" in batch and args.stream in ("both", "physio") else None
            logits, s = model(behav, physio)
            y_true.extend(batch["label"].cpu().numpy().tolist())
            y_prob.extend(torch.softmax(logits, dim=1).cpu().numpy().tolist())
            if s is not None:
                s_scores.extend(s.cpu().numpy().tolist())

    y_true, y_prob = np.array(y_true), np.array(y_prob)
    rep = classification_report(y_true, y_prob)
    print("\n===== 测试集指标 =====")
    print(f"混淆矩阵: TP={rep['confusion']['tp']} FP={rep['confusion']['fp']} "
          f"FN={rep['confusion']['fn']} TN={rep['confusion']['tn']}")
    print(f"准确率  = {rep['accuracy']:.4f}")
    print(f"精确率  = {rep['precision']:.4f}")
    print(f"召回率  = {rep['recall']:.4f}")
    print(f"F1      = {rep['f1']:.4f}")
    print(f"AUC     = {rep['auc']}")
    print(f"最优阈值= {rep['threshold']}")
    if s_scores:
        s_arr = np.array(s_scores)
        print(f"模态一致性 s: mean={s_arr.mean():.3f} std={s_arr.std():.3f} "
              f"(正常样本 {s_arr[y_true == 0].mean():.3f} vs 异常样本 {s_arr[y_true == 1].mean():.3f})")
    else:
        print("模态一致性 s: 单流模式不计算")

    out_prefix = str(Path(args.out))
    ok = save_figures(y_true, y_prob, out_prefix)
    print(f"图表: {out_prefix}_cm.png / {out_prefix}_roc.png" if ok else "图表生成失败（已降级为文本指标）")


if __name__ == "__main__":
    main()
