# -*- coding: utf-8 -*-
"""CMTA-Net 一键验证（双击运行入口：一键验证.bat 或本文件）。

菜单：
  1. 直接评估主模型（快，约1分钟）→ 准确率/F1/AUC/混淆矩阵/ROC 图
  2. 从头重训 + 评估（完整复现，约5分钟）→ 全链路真实复现
  3. 双模态四态诊断验证（UR Fall）→ 四态输出 + 一致性分数
  4. 推理接口冒烟 + 边缘部署时延评测
  0. 退出
"""
import os
import subprocess
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable


def run(cmd: list, desc: str):
    print(f"\n▶▶▶ {desc}")
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    r = subprocess.run([PY] + cmd, cwd=BASE, env=env)
    if r.returncode != 0:
        print(f"⚠ 执行失败（exit={r.returncode}），请检查上方报错。")
    return r.returncode


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    print("=" * 58)
    print("  CMTA-Net 一键验证（算法模型创新竞赛作品）")
    print("  主实验：UP-Fall 16通道传感流（按受试者划分，132 个未见样本）")
    print("=" * 58)

    while True:
        print("\n请选择要执行的操作：")
        print("  1. 直接评估主模型（快，约1分钟）→ F1/AUC/混淆矩阵/ROC")
        print("  2. 从头重训 + 评估（完整复现，约5分钟）")
        print("  3. 双模态四态诊断验证（UR Fall）")
        print("  4. 推理接口冒烟 + 边缘部署时延评测")
        print("  0. 退出")
        choice = input("\n输入数字后回车: ").strip()

        if choice == "1":
            run(["evaluate.py", "--checkpoint", "checkpoints/best_upfall_v2h.pt",
                 "--dataset", "upfall", "--stream", "physio",
                 "--split", "test", "--out", "outputs/verify_oneclick"],
                "评估主模型（UP-Fall 16通道，测试集=4名未见受试者）…")
            print("\n✔ 完成！结果见上方指标 + outputs/verify_oneclick_cm.png（混淆矩阵）和 _roc.png（ROC）")

        elif choice == "2":
            run(["train.py", "--dataset", "upfall", "--stream", "physio",
                 "--physio-channels", "16", "--epochs", "60", "--tag", "oneclick"],
                "从头重训主模型（新随机种子，完整复现）…")
            run(["evaluate.py", "--checkpoint", "checkpoints/best_oneclick.pt",
                 "--dataset", "upfall", "--stream", "physio",
                 "--split", "test", "--out", "outputs/verify_oneclick_r2"],
                "评估重训模型（验证全链路真实）…")
            print("\n✔ 完成！注意：不同随机种子结果会有 ±0.01 波动，属正常现象。")

        elif choice == "3":
            run(["evaluate.py", "--checkpoint", "checkpoints/best_urfall_v2mt.pt",
                 "--dataset", "urfall", "--stream", "both",
                 "--split", "test", "--out", "outputs/verify_mt"],
                "双模态四态诊断（UR Fall：骨架×加速度）…")
            print("\n✔ 完成！看「四态联合诊断（模块④）」部分：正常识别率/联合异常召回/一致性分数。")

        elif choice == "4":
            run(["inference.py", "--checkpoint", "checkpoints/best_urfall_v2mt.pt"],
                "推理接口冒烟（双模态→四态+一致性）…")
            run(["benchmark_latency.py", "--checkpoint", "checkpoints/best_upfall_v2h.pt"],
                "边缘部署时延评测（CPU 单窗口 ms）…")

        elif choice == "0":
            print("再见！")
            break

        else:
            print("无效输入，请输入 0-4。")

    input("\n按回车键退出…")


if __name__ == "__main__":
    main()
