# -*- coding: utf-8 -*-
"""诊断：单流头在 UR Fall 上的概率分布与 AUC（四态阈值校准问题的根因）。"""
import numpy as np
import torch
from torch.utils.data import DataLoader

from config import get_config
from data.build_dataset import CMTA_Dataset, collate
from models.cmta_net import CMTA_Net

ckpt_path = r"C:\Users\H\Desktop\算法模型\checkpoints\best_urfall_v2mt.pt"
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
cfg = get_config()
m = cfg.model
d_model = int(ckpt["config"]["model"]["d_model"])
physio_channels = int(ckpt["config"]["model"]["physio_channels"])
num_classes = int(ckpt["config"]["data"]["num_classes"])
model = CMTA_Net(pose_kps=m.pose_kps, d_model=d_model, physio_channels=physio_channels,
                 num_classes=num_classes, num_gru_layers=m.num_gru_layers,
                 attn_temperature=m.attn_temperature, dropout=m.dropout)
model.load_state_dict(ckpt["model"])
model.eval()

for split in ["val", "test"]:
    ds = CMTA_Dataset(split, cfg, dataset="urfall", mode="both", augment=False)
    loader = DataLoader(ds, batch_size=cfg.train.batch_size, shuffle=False, num_workers=0, collate_fn=collate)
    y, pp, pb = [], [], []
    with torch.no_grad():
        for batch in loader:
            b = batch["behavior"]
            p = batch["physio"]
            _, _, lp, lb = model(b, p)
            y.extend(batch["label"].numpy().tolist())
            pp.extend(torch.softmax(lp, dim=1)[:, 1].numpy().tolist())
            pb.extend(torch.softmax(lb, dim=1)[:, 1].numpy().tolist())
    y, pp, pb = np.array(y), np.array(pp), np.array(pb)
    from sklearn.metrics import roc_auc_score, roc_curve
    auc_p = roc_auc_score(y, pp)
    auc_b = roc_auc_score(y, pb)
    fpr_p, tpr_p, ths_p = roc_curve(y, pp)
    fpr_b, tpr_b, ths_b = roc_curve(y, pb)
    tp = ths_p[np.argmax(tpr_p - fpr_p)]
    tb = ths_b[np.argmax(tpr_b - fpr_b)]
    print(f"--- {split} (n={len(y)}, 异常={y.sum()}) ---")
    print(f"生理头: AUC={auc_p:.3f}  Youden阈值={tp:.3f}  正常p均值={pp[y==0].mean():.3f} 异常p均值={pp[y==1].mean():.3f}")
    print(f"行为头: AUC={auc_b:.3f}  Youden阈值={tb:.3f}  正常p均值={pb[y==0].mean():.3f} 异常p均值={pb[y==1].mean():.3f}")
    print(f"  生理p分布(正常): {np.round(pp[y==0],2)}")
    print(f"  生理p分布(异常): {np.round(pp[y==1],2)}")
    print(f"  行为p分布(正常): {np.round(pb[y==0],2)}")
    print(f"  行为p分布(异常): {np.round(pb[y==1],2)}")
