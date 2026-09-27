@echo off
cd /d "%~dp0"
echo ============================================
echo  CMTA-Net Web Model Export
echo  Output folder: web-onnx\models\
echo  Requires: D:\PyTorch\venv (torch + ultralytics)
echo ============================================
if exist "D:\PyTorch\venv\Scripts\python.exe" (
    "D:\PyTorch\venv\Scripts\python.exe" -m pip install --quiet onnx onnxruntime
    "D:\PyTorch\venv\Scripts\python.exe" export_onnx.py
) else (
    echo [ERROR] Python venv not found: D:\PyTorch\venv
    pause
    exit /b 1
)
echo.
echo Done! Check web-onnx\models\ for cmta.onnx and yolov8n-pose.onnx
pause
