"""训练脚本：CMTA-Net 端到端训练。

用法：
    python train.py --epochs 50 --batch-size 32 --lr 1e-3
    python train.py --device cuda --no-focal        # 有 GPU 时指定设备/关闭 Focal

训练策略（与调试改进方案一致）：
- warmup + cosine 学习率调度；早停（验证集 F1 连续 patience 轮无提升即停）；
- Focal Loss 处理类别不平衡（默认开启）；CUDA 下自动启用混合精度。
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import get_config
from data.build_dataset import CMTA_Dataset, collate
from models.cmta_net import CMTA_Net
from utils.losses import FocalLoss
from utils.metrics import classification_report


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def build_model(cfg, num_classes: int) -> nn.Module:
    m = cfg.model
    return CMTA_Net(
        pose_kps=m.pose_kps, d_model=m.d_model, physio_channels=m.physio_channels,
        num_classes=num_classes, num_gru_layers=m.num_gru_layers,
        attn_temperature=m.attn_temperature, dropout=m.dropout,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="CMTA-Net 训练")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--device", default=None, help="auto/cuda/cpu")
    ap.add_argument("--no-focal", action="store_true", help="关闭 Focal Loss，改用 CE")
    ap.add_argument("--checkpoint-dir", default=None)
    ap.add_argument("--dataset", default="urfall", choices=["urfall", "upfall", "geriatric"],
                    help="数据集（决定划分文件前缀）")
    ap.add_argument("--stream", default="both", choices=["both", "physio", "behavior"],
                    help="输入流：both=双模态；physio=仅生理流；behavior=仅行为流")
    ap.add_argument("--tag", default="", help="实验标签，用于 checkpoint/output 命名区分")
    ap.add_argument("--physio-channels", type=int, default=None,
                    help="覆盖生理通道数（UP-Fall full=16；UR Fall/老年=4）")
    ap.add_argument("--d-model", type=int, default=None, help="覆盖模型宽度 d_model")
    ap.add_argument("--aux-lambda", type=float, default=0.5,
                    help="多任务：单流诊断头损失的权重（仅双模态 both 时启用）")
    args = ap.parse_args()

    cfg = get_config()
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    if args.batch_size is not None:
        cfg.train.batch_size = args.batch_size
    if args.lr is not None:
        cfg.train.lr = args.lr
    if args.device is not None:
        cfg.train.device = args.device
    if args.checkpoint_dir is not None:
        cfg.train.checkpoint_dir = args.checkpoint_dir
    if args.physio_channels is not None:
        cfg.model.physio_channels = args.physio_channels
    if args.d_model is not None:
        cfg.model.d_model = args.d_model
    use_focal = cfg.train.use_focal and not args.no_focal

    device = resolve_device(cfg.train.device)
    torch.manual_seed(cfg.data.seed)
    np.random.seed(cfg.data.seed)

    print(f"[env] device={device}, epochs={cfg.train.epochs}, batch={cfg.train.batch_size}, "
          f"lr={cfg.train.lr}, focal={use_focal}, dataset={args.dataset}, stream={args.stream}")

    # 数据
    train_ds = CMTA_Dataset("train", cfg, dataset=args.dataset, mode=args.stream, augment=True)
    val_ds = CMTA_Dataset("val", cfg, dataset=args.dataset, mode=args.stream, augment=False)
    num_classes = cfg.data.num_classes
    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True,
                              num_workers=0, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=cfg.train.batch_size, shuffle=False,
                            num_workers=0, collate_fn=collate)
    print(f"[data] train={len(train_ds)}, val={len(val_ds)}, classes={num_classes}")

    # 模型（单流时只加载对应分支参数，避免缺失参数报错——两分支均保留，推理时只走一路）
    model = build_model(cfg, num_classes).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] 参数量 {n_params / 1e6:.2f}M")

    criterion = FocalLoss(alpha=cfg.train.focal_alpha, gamma=cfg.train.focal_gamma,
                          num_classes=num_classes) if use_focal else nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr,
                                  weight_decay=cfg.train.weight_decay)

    # warmup + cosine 调度
    warmup_ep = min(cfg.train.warmup_epochs, cfg.train.epochs)
    if warmup_ep > 0 and cfg.train.epochs > warmup_ep:
        from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
        warmup = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_ep)
        cosine = CosineAnnealingLR(optimizer, T_max=cfg.train.epochs - warmup_ep)
        scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_ep])
    else:
        from torch.optim.lr_scheduler import CosineAnnealingLR
        scheduler = CosineAnnealingLR(optimizer, T_max=cfg.train.epochs)

    # 早停
    best_f1, best_epoch, patience_counter = 0.0, 0, 0
    tag = f"_{args.tag}" if args.tag else ""
    best_path = Path(cfg.train.checkpoint_dir) / f"best{tag}.pt"
    best_path.parent.mkdir(parents=True, exist_ok=True)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    def model_inputs(batch):
        """按 stream 构造模型输入。"""
        behav = batch["behavior"].to(device) if "behavior" in batch and args.stream in ("both", "behavior") else None
        physio = batch["physio"].to(device) if "physio" in batch and args.stream in ("both", "physio") else None
        return behav, physio

    for epoch in range(1, cfg.train.epochs + 1):
        model.train()
        total_loss, n_batch = 0.0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.train.epochs}", leave=False)
        for batch in pbar:
            behav, physio = model_inputs(batch)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            with torch.autocast("cuda", enabled=use_amp):
                logits, s, logits_p, logits_b = model(behav, physio)
                loss = criterion(logits, labels)
                # 多任务：双模态时叠加单流诊断头损失（模块④：四态归因学习）
                if args.stream == "both" and args.aux_lambda > 0:
                    loss = loss + args.aux_lambda * criterion(logits_p, labels)
                    loss = loss + args.aux_lambda * criterion(logits_b, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            n_batch += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        scheduler.step()

        # 验证
        model.eval()
        y_true, y_prob = [], []
        with torch.no_grad():
            for batch in val_loader:
                behav, physio = model_inputs(batch)
                logits, _, _, _ = model(behav, physio)
                y_true.extend(batch["label"].cpu().numpy().tolist())
                y_prob.extend(torch.softmax(logits, dim=1).cpu().numpy().tolist())
        rep = classification_report(np.array(y_true), np.array(y_prob))
        f1, acc = rep["f1"], rep["accuracy"]
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch}: loss={total_loss / max(1, n_batch):.4f} | "
              f"val acc={acc:.4f} f1={f1:.4f} auc={rep['auc']} | lr={lr_now:.2e}")

        if f1 > best_f1:
            best_f1, best_epoch, patience_counter = f1, epoch, 0
            torch.save({
                "model": model.state_dict(),
                "config": {
                    "model": vars(cfg.model),
                    "data": vars(cfg.data),
                    "epochs": epoch,
                    "threshold": rep["threshold"],
                },
            }, best_path)
            print(f"  ✓ 保存最佳模型（F1={f1:.4f}）→ {best_path}")
        else:
            patience_counter += 1
            if patience_counter >= cfg.train.patience:
                print(f"早停：{cfg.train.patience} 轮无提升，停止训练（最佳 epoch={best_epoch}, F1={best_f1:.4f}）")
                break

    print(f"训练完成。最佳验证 F1={best_f1:.4f}（epoch {best_epoch}），模型: {best_path}")


if __name__ == "__main__":
    main()
