@echo off
chcp 65001 >nul
title CMTA-Net 可视化验证平台
cd /d "C:\Users\H\Desktop\算法模型"
"D:\PyTorch\venv\Scripts\python.exe" gui_verify.py
pause
