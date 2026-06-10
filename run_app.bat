@echo off
chcp 65001 >nul
echo 启动 YOLOv8 数字识别检测应用...
echo 浏览器将自动打开 http://localhost:8501
echo 按 Ctrl+C 停止服务
streamlit run digit_detect_app.py --server.headless true
