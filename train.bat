@echo off
chcp 65001 >nul
echo ========================================
echo  YOLOv8 数字识别 - 优化训练
echo ========================================
echo.
echo 选择训练模式:
echo   1. 仅准备数据集（划分 train/val）
echo   2. 水表数字微调（推荐，使用已有 MNIST 权重）
echo   3. 水表微调 + 全网络精调（效果更好，耗时更长）
echo   4. 完整训练（MNIST 预训练 + 微调 + 精调）
echo   5. 重新训练 MNIST
echo.
set /p choice=请输入选项 [1-5，默认 2]: 
if "%choice%"=="" set choice=2

if "%choice%"=="1" (
    python prepare_digit_dataset.py
    python prepare_mnist_val.py
    goto end
)
if "%choice%"=="2" (
    python train_digit.py --stage digit
    goto end
)
if "%choice%"=="3" (
    python train_digit.py --stage digit --full
    goto end
)
if "%choice%"=="4" (
    python train_digit.py --stage all --full
    goto end
)
if "%choice%"=="5" (
    python train_digit.py --stage mnist
    goto end
)

echo 无效选项
:end
pause
