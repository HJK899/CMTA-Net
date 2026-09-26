@echo off
chcp 65001 >nul
title CMTA-Net 一键验证
cd /d "C:\Users\H\Desktop\算法模型"
"D:\PyTorch\venv\Scripts\python.exe" run_verify.py
pause
