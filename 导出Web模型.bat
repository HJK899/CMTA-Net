@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ============================================
echo  CMTA-Net Web 模型导出（生成 web-onnx\models\）
echo  需要：D:\PyTorch\venv（已装 torch/ultralytics）
echo ============================================
if exist "D:\PyTorch\venv\Scripts\python.exe" (
    "D:\PyTorch\venv\Scripts\python.exe" -m pip install --quiet onnx onnxruntime
    "D:\PyTorch\venv\Scripts\python.exe" export_onnx.py
) else (
    echo [错误] 未找到 D:\PyTorch\venv\Scripts\python.exe
    pause
    exit /b 1
)
echo.
echo 完成！web-onnx\models\ 下应有 cmta.onnx 和 yolov8n-pose.onnx
pause
