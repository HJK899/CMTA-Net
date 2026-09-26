@echo off
chcp 65001 >nul
cd /d "%~dp0"
start "" "D:\PyTorch\venv\Scripts\pythonw.exe" gui_verify.py
