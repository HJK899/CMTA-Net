@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ============================================
echo  CMTA-Net Web 推理演示启动中...
echo  浏览器将自动打开 http://localhost:8000
echo  手机同 WiFi 可访问 http://本机IP:8000
echo  关闭此窗口即停止服务
echo ============================================
if exist "D:\PyTorch\venv\Scripts\python.exe" (
    start "" http://localhost:8000
    "D:\PyTorch\venv\Scripts\python.exe" web_demo.py
) else (
    python -c "import flask" 2>nul && (
        start "" http://localhost:8000
        python web_demo.py
    ) || (
        echo [错误] 未找到 Python 或 Flask。请先运行 pip install -r requirements.txt
        pause
    )
)
pause
