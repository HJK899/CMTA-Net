# -*- coding: utf-8 -*-
"""web_demo.py —— CMTA-Net Web 推理演示（产品形态）。

浏览器打开 http://localhost:8000 （手机同一 WiFi 访问 http://<电脑IP>:8000）：
- 上传连续照片（按动作顺序，推荐 ≥6 张，如"站立→倒地"全过程）
- 或上传视频（自动抽 150 帧 = 5 秒窗口）
→ 实时输出四态诊断（正常/生理异常/行为异常/联合异常）+ 双流概率 + 模态一致性
  + 骨架可视化图。

复用 photo_demo.py 的行为流管线（YOLOv8-Pose → CMTA-Net）。
模型在启动时加载一次（YOLO + CMTA-Net），每次请求仅推理。

用法：
    python web_demo.py                # 默认 0.0.0.0:8000
    python web_demo.py --port 9000    # 指定端口
"""
from __future__ import annotations
import argparse
import base64
import io
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from photo_demo import extract_pose, video_to_photos, visualize, IMG_EXTS  # noqa: E402
from inference import CMTA_Inference  # noqa: E402

CHECKPOINT = "checkpoints/best_urfall_v2mt.pt"
DEFAULT_TAU = 0.76
TMP = tempfile.mkdtemp(prefix="cmta_web_")

PAGE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CMTA-Net 老年健康监测 · 在线推理演示</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, "PingFang SC", "Microsoft YaHei", sans-serif;
         background: #f2f5f9; color: #1a2233; line-height: 1.6; }
  .wrap { max-width: 780px; margin: 0 auto; padding: 24px 16px 48px; }
  header { text-align: center; padding: 28px 0 8px; }
  h1 { font-size: 22px; letter-spacing: 1px; }
  .sub { color: #5b6b81; font-size: 13px; margin-top: 6px; }
  .card { background: #fff; border-radius: 12px; padding: 20px;
          box-shadow: 0 2px 8px rgba(20,40,80,.06); margin-top: 18px; }
  h2 { font-size: 15px; margin-bottom: 12px; color: #22304a; }
  .drop { border: 2px dashed #b9c6d8; border-radius: 10px; padding: 26px;
          text-align: center; color: #5b6b81; background: #fafbfd; cursor: pointer; }
  .drop.on { border-color: #2f6fdd; background: #eef4ff; }
  input[type=file] { display: none; }
  .btn { display: inline-block; margin-top: 14px; padding: 11px 34px; border: 0;
         border-radius: 8px; background: #2f6fdd; color: #fff; font-size: 15px;
         cursor: pointer; }
  .btn:disabled { background: #9db4d8; cursor: not-allowed; }
  .hint { font-size: 12px; color: #7c8aa0; margin-top: 10px; }
  .res { display: none; }
  .state { font-size: 20px; font-weight: 700; padding: 12px 0 4px; }
  .st0 { color: #1f9d55; } .st1 { color: #b7791f; }
  .st2 { color: #2f6fdd; } .st3 { color: #c53030; }
  .bar { height: 10px; border-radius: 5px; background: #e5eaf1; margin: 8px 0 4px; overflow: hidden; }
  .bar i { display: block; height: 100%; border-radius: 5px; }
  .probe { font-size: 13px; color: #3d4c63; display: flex; justify-content: space-between; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-top: 16px; }
  .cell { background: #f6f8fb; border-radius: 8px; padding: 10px 12px; }
  .cell b { display: block; font-size: 18px; }
  .cell span { font-size: 12px; color: #7c8aa0; }
  .img { margin-top: 16px; text-align: center; }
  .img img { max-width: 100%; border-radius: 8px; }
  .log { margin-top: 16px; font-size: 12px; color: #5b6b81; background: #f6f8fb;
         border-radius: 8px; padding: 10px; white-space: pre-wrap; max-height: 180px; overflow: auto; }
  @media (max-width: 560px) { .grid { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>CMTA-Net · 老年健康监测在线演示</h1>
    <div class="sub">跨模态时空注意力融合网络 — 生理 × 行为联合异常识别</div>
  </header>

  <div class="card">
    <h2>1. 选择输入</h2>
    <div class="drop" id="drop">
      <div id="dropText">点击选择：连续照片（按动作顺序 ≥6 张）或一段视频</div>
    </div>
    <input type="file" id="file" multiple accept=".jpg,.jpeg,.png,.bmp,.webp,.mp4,.avi,.mov,.mkv">
    <div style="text-align:center"><button class="btn" id="go" disabled>开始检测</button></div>
    <div class="hint">判定的是"动态事件"（如站立→倒地的过程），请上传完整动作过程；
      视频推荐 5-15 秒。照片请按时间顺序命名（如 1.jpg, 2.jpg…）。</div>
  </div>

  <div class="card res" id="resCard">
    <h2>2. 检测结果</h2>
    <div class="state" id="state">…</div>
    <div class="bar"><i id="bar" style="width:0%"></i></div>
    <div class="probe"><span>行为异常概率</span><b id="prob">–</b></div>
    <div class="grid">
      <div class="cell"><b id="physio">–</b><span>生理异常概率</span></div>
      <div class="cell"><b id="consist">–</b><span>模态一致性 s</span></div>
      <div class="cell"><b id="fuse">–</b><span>融合异常概率</span></div>
      <div class="cell"><b id="conf">–</b><span>置信度</span></div>
    </div>
    <div class="img" id="imgBox"></div>
    <div class="log" id="log"></div>
  </div>
</div>

<script>
const drop = document.getElementById('drop'), file = document.getElementById('file'),
      go = document.getElementById('go'), dropText = document.getElementById('dropText');
let chosen = [];
drop.onclick = () => file.click();
drop.ondragover = e => { e.preventDefault(); drop.classList.add('on'); };
drop.ondragleave = () => drop.classList.remove('on');
drop.ondrop = e => { e.preventDefault(); drop.classList.remove('on');
  if (e.dataTransfer.files.length) file.files = e.dataTransfer.files; onChange(); };
file.onchange = onChange;
function onChange() {
  chosen = Array.from(file.files);
  dropText.textContent = chosen.length
    ? '已选择 ' + chosen.length + ' 个文件：' + chosen.map(f => f.name).join(', ')
    : '点击选择文件';
  go.disabled = !chosen.length;
}
go.onclick = async () => {
  if (!chosen.length) return;
  go.disabled = true; go.textContent = '检测中（约 5-30 秒）…';
  const fd = new FormData();
  chosen.forEach(f => fd.append('files', f));
  try {
    const r = await fetch('/predict', { method: 'POST', body: fd });
    const j = await r.json();
    render(j);
  } catch (err) { log('请求失败：' + err); }
  go.disabled = false; go.textContent = '开始检测';
};
function render(j) {
  const cls = 'st' + j.state;
  document.getElementById('state').className = 'state ' + cls;
  document.getElementById('state').textContent = j.state_name + (j.note ? '（' + j.note + '）' : '');
  document.getElementById('bar').style.width = Math.round((j.behavior_prob || 0) * 100) + '%';
  document.getElementById('bar').style.background = j.state === 3 ? '#c53030' : (j.state === 2 ? '#2f6fdd' : '#1f9d55');
  document.getElementById('prob').textContent = (j.behavior_prob || 0).toFixed(3);
  document.getElementById('physio').textContent = j.physio_prob != null ? j.physio_prob.toFixed(3) : '–';
  document.getElementById('consist').textContent = j.consist != null ? j.consist.toFixed(3) : '–';
  document.getElementById('fuse').textContent = j.fuse_prob != null ? j.fuse_prob.toFixed(3) : '–';
  document.getElementById('conf').textContent = j.conf != null ? j.conf.toFixed(3) : '–';
  document.getElementById('resCard').style.display = 'block';
  const imgBox = document.getElementById('imgBox');
  imgBox.innerHTML = j.image ? '<img src="data:image/png;base64,' + j.image + '">' : '';
  log(j.log || '');
}
function log(t) { document.getElementById('log').textContent = t; }
</script>
</body>
</html>
"""


def _save_uploads(files) -> tuple[list[str], str | None]:
    """保存上传文件到临时目录。返回 ([照片路径], 视频路径或None)。"""
    photos, video = [], None
    for f in files:
        name = f.filename or ""
        suf = Path(name).suffix.lower()
        safe = f"u{len(os.listdir(TMP))}_{suf.lstrip('.') or 'bin'}"
        dst = os.path.join(TMP, safe)
        f.save(dst)
        if suf in IMG_EXTS:
            photos.append(dst)
        elif suf in (".mp4", ".avi", ".mov", ".mkv"):
            video = dst
    return photos, video


def predict(engine, yolo, photos, video, tau):
    """核心推理：照片序列或视频 → 行为流 → 四态。返回 (dict, seq, last_img)。"""
    seq: list[np.ndarray] = []
    log_lines = []
    if video:
        gname, vp = video_to_photos(video, 150)
        if not vp:
            return {"error": "视频读取失败"}, None, None
        log_lines.append(f"视频「{gname}」抽 {len(vp)} 帧")
        photos = vp
    if len(photos) < 5:
        log_lines.append("警告：少于5帧，行为判定需覆盖完整动作过程")
    missing = 0
    for p in photos:
        kp = extract_pose(yolo, p)
        if kp is None:
            missing += 1
            kp = np.zeros((17, 3), dtype=np.float32)
        seq.append(kp)
    if missing:
        log_lines.append(f"{missing}/{len(photos)} 帧未检出人体（零填充保时序）")
    if missing == len(photos):
        return {"error": "所有帧均未检测到人体"}, None, None
    seq_arr = np.stack(seq, 0).astype(np.float32)
    behavior = seq_arr.reshape(len(seq), -1)
    res = engine.predict(behavior=behavior)
    res.pop("异常概率(融合)", None)
    out = {
        "state": res["state"],
        "state_name": res["state_name"],
        "behavior_prob": res["行为异常概率"],
        "physio_prob": res["生理异常概率"],
        "fuse_prob": res["异常概率(融合)"],
        "consist": res["模态一致性s"],
        "conf": res["置信度"],
        "tau": res["阈值tau"],
        "note": "行为流判定" if video is None else "视频行为流判定",
        "log": "\n".join(log_lines),
    }
    return out, seq_arr, photos[-1]


def main() -> None:
    from flask import Flask, jsonify, request
    ap = argparse.ArgumentParser(description="CMTA-Net Web 推理演示")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--checkpoint", default=CHECKPOINT)
    ap.add_argument("--tau", type=float, default=DEFAULT_TAU)
    args = ap.parse_args()

    from ultralytics import YOLO
    print("加载 YOLOv8-Pose（首次运行自动下载权重）…")
    yolo = YOLO("yolov8n-pose.pt")
    print(f"加载 CMTA-Net 模型（{args.checkpoint}, tau={args.tau}）…")
    engine = CMTA_Inference(args.checkpoint, tau=args.tau)

    app = Flask(__name__)

    @app.route("/")
    def index():
        return PAGE

    @app.route("/predict", methods=["POST"])
    def do_predict():
        files = request.files.getlist("files")
        if not files:
            return jsonify({"error": "未收到文件"})
        photos, video = _save_uploads(files)
        out, seq, last_img = predict(engine, yolo, photos, video, args.tau)
        if "error" in out:
            return jsonify(out)
        img_path = os.path.join(BASE, "outputs", "web_last.png")
        note = f"{out['state_name']} · 行为异常概率={out['behavior_prob'] or 0:.2f}"
        vis = visualize(last_img, seq, img_path, note)
        out["image"] = None
        if vis:
            with open(vis, "rb") as fh:
                out["image"] = base64.b64encode(fh.read()).decode()
        return jsonify(out)

    print(f"Web 演示已启动：http://localhost:{args.port}  （手机同 WiFi 访问 http://<电脑IP>:{args.port}）")
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
