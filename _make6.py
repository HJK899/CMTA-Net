# -*- coding: utf-8 -*-
"""实测6张照片判定：fall/adl 各抽6张，跑 photo_demo 完整流程。"""
import os
import cv2
import subprocess
import sys
from PIL import Image

BASE = r"C:\Users\H\Desktop\算法模型"
UR = os.path.join(BASE, "data/raw/urfall")
OUT = os.path.join(BASE, "outputs/demo_photos6")
PY = r"D:\PyTorch\venv\Scripts\python.exe"


def grab_pil(src, sub, a, b, n=6):
    cap = cv2.VideoCapture(src)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = [int((a + (b - a) * i / (n - 1)) * total) for i in range(n)]
    d = os.path.join(OUT, sub)
    os.makedirs(d, exist_ok=True)
    ok_n = 0
    for i, idx in enumerate(idxs):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, f = cap.read()
        if ok:
            Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)).save(
                os.path.join(d, f"f{i + 1:02d}.jpg"))
            ok_n += 1
    cap.release()
    return ok_n


def run_demo(sub, tag):
    r = subprocess.run([PY, "-u", os.path.join(BASE, "photo_demo.py"),
                        "-d", os.path.join(OUT, sub), "--out", f"outputs/demo6_{tag}.png"],
                       cwd=BASE, capture_output=True, text=True, encoding="utf-8")
    return r.stdout


print("fall6张:", grab_pil(os.path.join(UR, "fall-01-cam1.mp4"), "fall", 0.25, 0.90, 6))
print("adl6张:", grab_pil(os.path.join(UR, "adl-01-cam0.mp4"), "adl", 0.15, 0.85, 6))
