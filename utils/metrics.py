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


def four_state_report(y_true: np.ndarray, prob_p: np.ndarray, prob_b: np.ndarray,
                      s_scores: Optional[np.ndarray] = None,
                      tau_p: float = 0.5, tau_b: float = 0.5) -> dict:
    """模块④四态联合诊断评估。

    状态规则（单流头阈值 tau_p / tau_b，建议由验证集 Youden 指数选出）：
        正常    : 生理<tau_p 且 行为<tau_b
        生理异常: 生理>=tau_p 且 行为<tau_b
        行为异常: 生理<tau_p 且 行为>=tau_b
        联合异常: 生理>=tau_p 且 行为>=tau_b

    y_true 为二分类标签（0=正常, 1=异常/跌倒）：
        - 真实"联合异常"样本 = 标签 1（UR Fall 的跌倒段双模态均异常）
        - 真实"正常"样本 = 标签 0
    中间两态（生理异常/行为异常）为开放诊断能力，在只有"双模态同异常/同正常"标注的
    数据集上无 ground-truth，如实报告其预测分布，不宣称正确率。
    """
    y_true = np.asarray(y_true)
    prob_p = np.asarray(prob_p)
    prob_b = np.asarray(prob_b)
    p1 = prob_p[:, 1] if prob_p.ndim == 2 else prob_p
    b1 = prob_b[:, 1] if prob_b.ndim == 2 else prob_b

    state = np.where((p1 >= tau_p) & (b1 >= tau_b), 3,
             np.where((p1 >= tau_p) & (b1 < tau_b), 1,
              np.where((p1 < tau_p) & (b1 >= tau_b), 2, 0)))  # 0正常 1生理 2行为 3联合
    names = {0: "正常", 1: "生理异常", 2: "行为异常", 3: "联合异常"}

    n = len(y_true)
    joint_recall = float((state[y_true == 1] == 3).mean()) if (y_true == 1).any() else float("nan")
    normal_rec = float((state[y_true == 0] == 0).mean()) if (y_true == 0).any() else float("nan")
    overall = float((((state == 3) & (y_true == 1)) | ((state == 0) & (y_true == 0))).sum() / n) if n else 0.0

    lines = [
        f"决策规则：生理阈值={tau_p:.3f} / 行为阈值={tau_b:.3f} → 四态",
        "状态分布（预测）：" + ", ".join(
            f"{names[k]}={int((state == k).sum())}" for k in range(4) if (state == k).any()),
    ]
    if n:
        lines.append(f"联合异常召回（真实跌倒→联合异常）: {joint_recall:.4f}")
        lines.append(f"正常识别率（真实日常→正常）      : {normal_rec:.4f}")
        lines.append(f"四态判定综合正确率               : {overall:.4f}")
        lines.append("注：中间两态（生理/行为异常）在双模态同异常/同正常的标注下无 ground-truth，")
        lines.append("    作为开放诊断能力如实报告，不宣称其正确率。")
    if s_scores is not None:
        s = np.asarray(s_scores)
        for k in range(4):
            if (state == k).sum() > 0:
                lines.append(f"  {names[k]}：一致性分数均值={s[state == k].mean():.3f}")

    return {"lines": lines, "state": state, "overall": round(overall, 4)}
