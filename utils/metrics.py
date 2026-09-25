"""评估指标：混淆矩阵、精确率/召回率/F1、ROC+AUC 与最优阈值选择。

依赖 scikit-learn / matplotlib；缺失时自动降级（仅文本指标，阈值取 0.5）。
"""
from __future__ import annotations
from typing import Optional

import numpy as np


def classification_report(y_true: np.ndarray, y_prob: np.ndarray,
                          threshold: Optional[float] = None) -> dict:
    """给定标签与概率，返回指标字典（含最优阈值）。"""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    pos = y_prob[:, 1] if y_prob.ndim == 2 else y_prob

    try:
        from sklearn.metrics import roc_curve, auc, confusion_matrix
        fpr, tpr, ths = roc_curve(y_true, pos)
        best_idx = np.argmax(tpr - fpr)          # Youden 指数：最大化 TPR - FPR
        threshold = float(ths[best_idx]) if threshold is None else float(threshold)
        roc_auc = float(auc(fpr, tpr))
    except Exception:
        threshold = 0.5 if threshold is None else float(threshold)
        roc_auc = float("nan")

    preds = (pos >= threshold).astype(int)
    tp = int(((preds == 1) & (y_true == 1)).sum())
    fp = int(((preds == 1) & (y_true == 0)).sum())
    fn = int(((preds == 0) & (y_true == 1)).sum())
    tn = int(((preds == 0) & (y_true == 0)).sum())

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    acc = (tp + tn) / max(1, tp + tn + fp + fn)

    return {
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(acc, 4),
        "auc": round(roc_auc, 4) if not np.isnan(roc_auc) else None,
        "threshold": round(threshold, 4),
    }


def save_figures(y_true: np.ndarray, y_prob: np.ndarray, out_prefix: str) -> bool:
    """保存混淆矩阵与 ROC 曲线图（matplotlib），成功返回 True。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve, auc, confusion_matrix
        from sklearn.metrics import ConfusionMatrixDisplay

        y_true = np.asarray(y_true)
        pos = y_prob[:, 1] if np.asarray(y_prob).ndim == 2 else np.asarray(y_prob)

        # 混淆矩阵（用英文标签，避免中文字体缺失显示方块）
        th = classification_report(y_true, np.asarray(y_prob))["threshold"]
        preds = (pos >= th).astype(int)
        cm = confusion_matrix(y_true, preds)
        disp = ConfusionMatrixDisplay(cm, display_labels=["Normal", "Abnormal"])
        disp.plot(cmap="Blues")
        plt.title(f"Confusion Matrix (threshold={th:.3f})")
        plt.savefig(f"{out_prefix}_cm.png", dpi=150, bbox_inches="tight")
        plt.close()

        # ROC 曲线
        fpr, tpr, _ = roc_curve(y_true, pos)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f"CMTA-Net (AUC={roc_auc:.3f})")
        plt.plot([0, 1], [0, 1], "k--", label="Random")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(f"{out_prefix}_roc.png", dpi=150, bbox_inches="tight")
        plt.close()
        return True
    except Exception as e:
        print(f"  [warn] 图表生成失败（降级为文本指标）: {e}")
        return False
