# -*- coding: utf-8 -*-
"""photo_demo.py —— 用真实照片/图片序列测试 CMTA-Net 行为流判定。

用法：
    # 单张照片（静态姿态演示，仅作参考）
    python photo_demo.py -p 照片1.jpg
    # 连续照片序列（推荐 >=6 张且须覆盖完整动作过程：如"站立→倒地"全过程）
    python photo_demo.py -p 帧1.jpg 帧2.jpg 帧3.jpg 帧4.jpg 帧5.jpg 帧6.jpg
    # 文件夹内所有图片（按文件名排序）
    python photo_demo.py -d 照片文件夹
    # 多组文件夹：目录下有子文件夹时，每个子文件夹=一组，一次全测
    #   我的测试照片/
    #     ├─ 跌倒组/ 1.jpg 2.jpg ...   （该组按文件名排序）
    #     └─ 日常组/ 1.jpg 2.jpg ...
    python photo_demo.py -d 我的测试照片

判定标准：模型识别的是"异常行为=动态事件"（如站立→倒地的过程），
不是"静态姿态"（人已躺在地上≠可判定跌倒）。因此照片序列必须覆盖完整动作过程。
默认阈值 tau=0.76（验证集 Youden 校准值）；日常活动应低于阈值，完整跌倒过程应高于阈值。

流程：照片 → YOLOv8-Pose 人体骨架(17关键点) → 时序窗口 → CMTA-Net 行为流 → 判定。
输出：每组的行为异常概率 / 四态诊断 / 置信度 + 骨架可视化图（outputs/ 下）。
"""
from __future__ import annotations
import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from inference import CMTA_Inference  # noqa: E402

CHECKPOINT = "checkpoints/best_urfall_v2mt.pt"

# YOLOv8-Pose COCO 17 关键点骨架连线（画可视化用）
COCO_LINES = [[0, 1], [0, 2], [1, 3], [2, 4], [5, 6], [5, 7], [7, 9],
              [6, 8], [8, 10], [5, 11], [6, 12], [11, 12], [11, 13],
              [13, 15], [12, 14], [14, 16]]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_groups(d: str):
    """目录 → [(组名, [照片路径])]。

    有子文件夹：每个子文件夹=一组（组内按文件名排序）；
    无子文件夹：根目录全部图片=单组（组名"默认"）。
    """
    d = Path(d)
    if not d.is_dir():
        return []
    subdirs = sorted([p for p in d.iterdir() if p.is_dir()])
    if subdirs:
        groups = []
        for sd in subdirs:
            photos = sorted([str(p) for p in sd.iterdir() if p.suffix.lower() in IMG_EXTS])
            if photos:
                groups.append((sd.name, photos))
        return groups
    photos = sorted([str(p) for p in d.iterdir() if p.suffix.lower() in IMG_EXTS])
    return [("默认", photos)] if photos else []


def extract_pose(model, img_path: str):
    """单张照片 → (17,3) 归一化骨架；无人返回 None。

    用 PIL 读取以兼容中文路径（cv2.imread 不支持非 ASCII 路径）。
    """
    from PIL import Image
    img = Image.open(img_path).convert("RGB")
    arr = np.asarray(img)[:, :, ::-1].copy()   # RGB → BGR（YOLO 期望 BGR）
    res = model.predict(arr, verbose=False, device="cpu")[0]
    kps = res.keypoints
    if kps is None or len(kps.data) == 0:
        return None
    return kps.data[0].cpu().numpy().astype(np.float32)


def visualize(img_path: str, seq: np.ndarray, out_path: str, note: str = ""):
    """在最后一张照片上画骨架并保存。seq: (T,17,3)。"""
    try:
        from PIL import Image, ImageDraw
    except Exception as e:  # noqa: BLE001
        print(f"[可视化] PIL 不可用，跳过画图（{e}）")
        return None
    img = Image.open(img_path).convert("RGB")
    W, H = img.size
    draw = ImageDraw.Draw(img)
    kp = seq[-1]  # 最后一帧骨架
    for (x, y, c) in kp:
        if c < 0.3:
            continue
        r = max(3, int(0.008 * W))
        draw.ellipse([x * W - r, y * H - r, x * W + r, y * H + r],
                     fill=(255, 60, 60), outline=(255, 255, 255))
    for a, b in COCO_LINES:
        ka, kb = kp[a], kp[b]
        if ka[2] < 0.3 or kb[2] < 0.3:
            continue
        draw.line([ka[0] * W, ka[1] * H, kb[0] * W, kb[1] * H],
                  fill=(80, 200, 255), width=2)
    if note:
        draw.rectangle([0, H - 34, W, H], fill=(20, 20, 20, 160))
        draw.text((8, H - 26), note, fill=(255, 255, 255))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img.save(out_path)
    return out_path


def run_group(engine, yolo, group_name: str, photos: list[str], args, is_multi: bool):
    """单组判定：照片→骨架→模型→输出（含可视化）。"""
    print(f"\n════ 组「{group_name}」：{len(photos)} 张照片 ════")
    if len(photos) < 5:
        print("⚠ 少于5张：行为判定需覆盖完整动作过程（推荐 ≥6 张）")

    seq: list[np.ndarray] = []
    missing = 0
    for p in photos:
        kp = extract_pose(yolo, p)
        if kp is None:
            missing += 1
            kp = np.zeros((17, 3), dtype=np.float32)   # 无人 → 零填充，保时序
            print(f"  [无人] {Path(p).name}")
        else:
            print(f"  [ok]   {Path(p).name}  骨架置信度={kp[:, 2].max():.2f}")
        seq.append(kp)
    if missing == len(photos):
        print(f"组「{group_name}」：所有照片均未检测到人体，跳过")
        return None

    seq_arr = np.stack(seq, axis=0).astype(np.float32)   # (T, 17, 3)
    behavior = seq_arr.reshape(len(photos), -1)          # (T, 51)，与训练格式一致

    res = engine.predict(behavior=behavior)

    print(f"\n—— 组「{group_name}」判定结果 ——")
    for k in ("state", "state_name", "异常概率(融合)", "行为异常概率", "置信度", "阈值tau"):
        print(f"  {k}: {res.get(k)}")
    if res["state"] in (2, 3):
        print(f"判定：存在行为异常（{res['state_name']}），建议复核/告警。")
    else:
        print(f"判定：{res['state_name']}，未触发告警。")

    # 骨架可视化
    if is_multi:
        safe = re.sub(r"[^\w\u4e00-\u9fff-]", "_", group_name) or "group"
        out = os.path.join(BASE, "outputs", f"demo_group_{safe}.png")
    else:
        out = os.path.join(BASE, args.out)
    note = f"{group_name}·行为异常概率={res['行为异常概率'] or 0:.2f} · {res['state_name']}"
    vis = visualize(photos[-1], seq_arr, out, note)
    if vis:
        print(f"骨架可视化已保存: {vis}")
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description="CMTA-Net 照片行为判定演示")
    ap.add_argument("-p", "--photos", nargs="*", help="照片路径列表（按时间顺序）")
    ap.add_argument("-d", "--dir", help="照片文件夹（有子文件夹=多组，每个子文件夹一组）")
    ap.add_argument("--checkpoint", default=CHECKPOINT)
    ap.add_argument("--tau", type=float, default=0.76, help="行为异常判定阈值（默认0.76=验证集Youden校准值）")
    ap.add_argument("--out", default="outputs/photo_demo.png", help="单组时骨架可视化输出路径")
    args = ap.parse_args()

    groups: list[tuple[str, list[str]]] = []
    if args.dir:
        groups = collect_groups(args.dir)
        if not groups:
            print(f"错误：文件夹无照片（或子文件夹内无照片）：{args.dir}")
            sys.exit(1)
    if args.photos:
        photos = [p for p in args.photos if Path(p).suffix.lower() in IMG_EXTS]
        if photos:
            groups.append(("默认", photos))
    if not groups:
        print("错误：未提供照片。用 -p 照片1.jpg 照片2.jpg ... 或 -d 文件夹")
        sys.exit(1)

    is_multi = len(groups) > 1 or (args.dir and collect_groups(args.dir))
    total_n = sum(len(photos) for _, photos in groups)
    print(f"共 {len(groups)} 组 / {total_n} 张照片" + ("（多组模式）" if is_multi else ""))

    from ultralytics import YOLO
    print("加载 YOLOv8-Pose（首次运行自动下载权重）…")
    yolo = YOLO("yolov8n-pose.pt")

    print(f"加载 CMTA-Net 双模态模型（{args.checkpoint}）…")
    engine = CMTA_Inference(args.checkpoint, tau=args.tau)

    results = []
    for gname, photos in groups:
        r = run_group(engine, yolo, gname, photos, args, is_multi)
        if r is not None:
            results.append((gname, r))

    if len(results) > 1:
        print("\n════ 多组汇总 ════")
        for gname, r in results:
            p = r["行为异常概率"] or 0
            mark = "异常" if r["state"] in (2, 3) else "正常"
            print(f"  {gname:<12} 概率={p:.3f}  判定={mark}")


if __name__ == "__main__":
    main()
