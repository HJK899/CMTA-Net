# -*- coding: utf-8 -*-
"""CMTA-Net 可视化验证平台（GUI）。

双击「一键验证.bat」或运行本文件即可打开图形界面：
点按钮 → 实时日志 → 指标卡片 → 混淆矩阵/ROC 图表直接预览。

任务：
  1. 评估主模型（快，约1分钟）——UP-Fall 16通道传感流
  2. 从头重训+评估（完整复现，约5分钟）
  3. 双模态四态诊断（UR Fall：骨架×加速度）
  4. 推理接口演示（单窗口→四态+一致性）
  5. 边缘部署时延评测（CPU ms）
"""
import os
import re
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import ttk, messagebox

try:
    from PIL import Image, ImageTk
    HAS_PIL = True
except Exception:
    HAS_PIL = False

BASE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

TASKS = {
    "1": {
        "name": "评估主模型（快）",
        "desc": "UP-Fall 16通道 · 测试集=4名未见受试者",
        "cmds": [["evaluate.py", "--checkpoint", "checkpoints/best_upfall_v2h.pt",
                  "--dataset", "upfall", "--stream", "physio",
                  "--split", "test", "--out", "outputs/verify_gui"]],
    },
    "2": {
        "name": "完整复现（重训+评估）",
        "desc": "从头训练新随机种子，验证全链路真实（约5分钟）",
        "cmds": [["train.py", "--dataset", "upfall", "--stream", "physio",
                  "--physio-channels", "16", "--epochs", "60", "--tag", "gui_r1"],
                 ["evaluate.py", "--checkpoint", "checkpoints/best_gui_r1.pt",
                  "--dataset", "upfall", "--stream", "physio",
                  "--split", "test", "--out", "outputs/verify_gui_r2"]],
    },
    "3": {
        "name": "双模态四态诊断（UR Fall）",
        "desc": "骨架×加速度 · 四态 + 一致性分数",
        "cmds": [["evaluate.py", "--checkpoint", "checkpoints/best_urfall_v2mt.pt",
                  "--dataset", "urfall", "--stream", "both",
                  "--split", "test", "--out", "outputs/verify_gui_mt"]],
    },
    "4": {
        "name": "推理接口演示",
        "desc": "单窗口 → 四态诊断 + 各模态概率 + 一致性",
        "cmds": [["inference.py", "--checkpoint", "checkpoints/best_urfall_v2mt.pt"]],
    },
    "5": {
        "name": "边缘部署时延评测",
        "desc": "CPU 单窗口推理 ms · 参数量",
        "cmds": [["benchmark_latency.py", "--checkpoint", "checkpoints/best_upfall_v2h.pt"]],
    },
}

RE_METRICS = {
    "acc": re.compile(r"准确率\s*=\s*([\d.]+)"),
    "f1": re.compile(r"F1\s*=\s*([\d.]+)"),
    "auc": re.compile(r"AUC\s*=\s*([\d.]+)"),
    "cm": re.compile(r"TP=(\d+)\s+FP=(\d+)\s+FN=(\d+)\s+TN=(\d+)"),
}


class Worker(threading.Thread):
    def __init__(self, cmds, on_line, on_done):
        super().__init__(daemon=True)
        self.cmds = cmds
        self.on_line = on_line
        self.on_done = on_done
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def run(self):
        for i, cmd in enumerate(self.cmds):
            if self._stop.is_set():
                break
            self.on_line(f"\n════ 步骤 {i+1}/{len(self.cmds)}：{' '.join(cmd[:2])} ════\n")
            try:
                env = dict(os.environ)
                env["PYTHONIOENCODING"] = "utf-8"
                p = subprocess.Popen([PY] + cmd, cwd=BASE,
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     encoding="utf-8", errors="replace",
                                     bufsize=1, text=True, env=env)
                for line in p.stdout:
                    if self._stop.is_set():
                        p.kill()
                        break
                    self.on_line(line.rstrip())
                p.wait()
            except Exception as e:  # noqa: BLE001
                self.on_line(f"[错误] {e}")
        self.on_done()


class App:
    def __init__(self, root):
        self.root = root
        root.title("CMTA-Net 可视化验证平台")
        root.geometry("980x720")
        try:
            root.configure(bg="#f4f6fa")
        except Exception:
            pass

        self.worker = None
        self.cm_path = None
        self.roc_path = None
        self._photo = None

        self._build_style()
        self._build_layout()

    def _build_style(self):
        s = ttk.Style()
        try:
            s.theme_use("vista")
        except Exception:
            pass
        s.configure("Card.TFrame", background="#ffffff")
        s.configure("Title.TLabel", font=("Microsoft YaHei UI", 14, "bold"), background="#f4f6fa")
        s.configure("Metric.TLabel", font=("Microsoft YaHei UI", 22, "bold"), foreground="#1f6feb",
                    background="#ffffff")
        s.configure("MetricName.TLabel", font=("Microsoft YaHei UI", 10), background="#ffffff",
                    foreground="#57606a")

    def _build_layout(self):
        pad = {"padx": 12, "pady": 6}

        ttk.Label(self.root, text="CMTA-Net 可视化验证平台", style="Title.TLabel").pack(anchor="w", **pad)
        ttk.Label(self.root, text="算法模型创新竞赛作品 · 老年人『生理×行为』联合异常识别",
                  font=("Microsoft YaHei UI", 9), foreground="#57606a",
                  background="#f4f6fa").pack(anchor="w", padx=12, pady=(0, 6))

        top = ttk.Frame(self.root); top.pack(fill="x", padx=12)
        left = ttk.Frame(top); left.pack(side="left", fill="y")

        # ---- 左侧：任务选择 ----
        ttk.Label(left, text="选择验证任务", font=("Microsoft YaHei UI", 11, "bold"),
                  background="#f4f6fa").pack(anchor="w", pady=(0, 4))
        self.task_var = tk.StringVar(value="1")
        for key, t in TASKS.items():
            rb = ttk.Radiobutton(left, text=f"{key}. {t['name']}", value=key,
                                 variable=self.task_var, command=self._on_task_change)
            rb.pack(anchor="w")
            ttk.Label(left, text="    " + t["desc"], font=("Microsoft YaHei UI", 8),
                      foreground="#6e7781", background="#f4f6fa").pack(anchor="w", pady=(0, 3))

        btn_row = ttk.Frame(left); btn_row.pack(anchor="w", pady=10)
        self.start_btn = ttk.Button(btn_row, text="▶ 开始验证", command=self.start)
        self.start_btn.pack(side="left", padx=(0, 6))
        self.stop_btn = ttk.Button(btn_row, text="■ 停止", command=self.stop, state="disabled")
        self.stop_btn.pack(side="left")

        self.status = ttk.Label(left, text="就绪", font=("Microsoft YaHei UI", 9),
                                foreground="#57606a", background="#f4f6fa")
        self.status.pack(anchor="w", pady=(4, 0))

        # ---- 右侧：日志 ----
        right = ttk.Frame(top); right.pack(side="left", fill="both", expand=True, padx=(16, 0))
        ttk.Label(right, text="运行日志", font=("Microsoft YaHei UI", 11, "bold"),
                  background="#f4f6fa").pack(anchor="w", pady=(0, 4))
        self.log = tk.Text(right, height=16, width=58, font=("Microsoft YaHei UI", 9),
                           bg="#0d1117", fg="#c9d1d9", relief="flat", wrap="none")
        self.log.pack(fill="both", expand=True)
        self.log.insert("end", "就绪。选择左侧任务后点击「开始验证」。\n")

        # ---- 中部：指标卡片 ----
        mid = ttk.Frame(self.root, style="Card.TFrame"); mid.pack(fill="x", padx=12, pady=8)
        self.metric_labels = {}
        for i, (k, name) in enumerate([("acc", "准确率"), ("f1", "F1"), ("auc", "AUC"),
                                       ("cm", "混淆矩阵")]):
            cell = ttk.Frame(mid, style="Card.TFrame"); cell.grid(row=0, column=i, padx=10, pady=8)
            ttk.Label(cell, text=name, style="MetricName.TLabel").pack()
            lbl = ttk.Label(cell, text="—", style="Metric.TLabel")
            lbl.pack()
            self.metric_labels[k] = lbl

        # ---- 底部：图表预览 ----
        bot = ttk.Frame(self.root); bot.pack(fill="both", expand=True, padx=12, pady=(0, 10))
        ttk.Label(bot, text="图表预览", font=("Microsoft YaHei UI", 11, "bold"),
                  background="#f4f6fa").pack(anchor="w")
        self.img_canvas = tk.Label(bot, text="运行后自动显示混淆矩阵 / ROC 曲线", bg="#ffffff",
                                   relief="groove", bd=1)
        self.img_canvas.pack(fill="both", expand=True)
        switch_row = ttk.Frame(bot); switch_row.pack(anchor="w", pady=4)
        self.img_var = tk.StringVar(value="cm")
        ttk.Radiobutton(switch_row, text="混淆矩阵", value="cm", variable=self.img_var,
                        command=self._show_img).pack(side="left")
        ttk.Radiobutton(switch_row, text="ROC 曲线", value="roc", variable=self.img_var,
                        command=self._show_img).pack(side="left", padx=8)

    def _on_task_change(self):
        pass

    def _append_log(self, text):
        self.log.insert("end", text + "\n")
        self.log.see("end")
        self.root.update_idletasks()

    def start(self):
        if self.worker and self.worker.is_alive():
            return
        key = self.task_var.get()
        self._reset_metrics()
        self.cm_path = self.roc_path = None
        self.status.config(text=f"运行中：{TASKS[key]['name']} …", foreground="#1a7f37")
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.worker = Worker(TASKS[key]["cmds"], self._append_log, self._on_done)
        self.worker.start()

    def stop(self):
        if self.worker and self.worker.is_alive():
            self.worker.stop()
            self.status.config(text="已停止", foreground="#cf222e")
            self._append_log("[用户] 任务已停止")

    def _on_done(self):
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        text = self.log.get("1.0", "end")
        parsed = self._parse_metrics(text)
        if parsed:
            self._update_metrics(parsed)
            self.status.config(text="完成 ✔", foreground="#1a7f37")
        else:
            self.status.config(text="完成（未解析到指标，请查看日志）", foreground="#9a6700")
        self._find_images()
        self._show_img()

    def _parse_metrics(self, text):
        out = {}
        for k, rx in RE_METRICS.items():
            m = rx.search(text)
            if m:
                out[k] = m.groups()
        return out

    def _update_metrics(self, parsed):
        if "acc" in parsed:
            self.metric_labels["acc"].config(text=f"{float(parsed['acc'][0])*100:.1f}%")
        if "f1" in parsed:
            self.metric_labels["f1"].config(text=f"{float(parsed['f1'][0]):.3f}")
        if "auc" in parsed:
            self.metric_labels["auc"].config(text=f"{float(parsed['auc'][0]):.3f}")
        if "cm" in parsed:
            tp, fp, fn, tn = parsed["cm"]
            total = int(tp) + int(fp) + int(fn) + int(tn)
            self.metric_labels["cm"].config(
                text=f"TP{tp} FP{fp}\nFN{fn} TN{tn} · n={total}", font=("Microsoft YaHei UI", 13, "bold"))

    def _reset_metrics(self):
        for k, lbl in self.metric_labels.items():
            lbl.config(text="—", font=("Microsoft YaHei UI", 22, "bold"))
        self.metric_labels["cm"].config(text="—")

    def _find_images(self):
        out_dirs = [os.path.join(BASE, "outputs")]
        cands = []
        for d in out_dirs:
            if os.path.isdir(d):
                for f in sorted(os.listdir(d)):
                    if f.endswith("_cm.png"):
                        cands.append((f, os.path.join(d, f)))
        if not cands:
            return
        latest = max(cands, key=lambda x: os.path.getmtime(x[1]))
        self.cm_path = latest[1]
        self.roc_path = latest[1].replace("_cm.png", "_roc.png")
        if not os.path.exists(self.roc_path):
            self.roc_path = None

    def _show_img(self):
        if not HAS_PIL:
            return
        path = self.cm_path if self.img_var.get() == "cm" else self.roc_path
        if not path or not os.path.exists(path):
            return
        try:
            img = Image.open(path)
            w, h = img.size
            max_h = 300
            if h > max_h:
                img = img.resize((int(w * max_h / h), max_h), Image.LANCZOS)
            self._photo = ImageTk.PhotoImage(img)
            self.img_canvas.config(image=self._photo, text="")
        except Exception as e:  # noqa: BLE001
            self._append_log(f"[图表] 加载失败: {e}")


def main():
    root = tk.Tk()
    App(root)
    if len(sys.argv) > 1 and sys.argv[1] == "--selftest":
        root.after(2500, root.destroy)
    root.mainloop()


if __name__ == "__main__":
    main()
