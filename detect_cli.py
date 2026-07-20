"""命令行检测工具 — 支持单图、批量图片、视频与摄像头。."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ultralytics import YOLO

MODEL_CANDIDATES = [
    "dist/best.pt",
    "runs/detect/digit_finetune_full/weights/best.pt",
    "runs/detect/digit_finetune/weights/best.pt",
    "runs/detect/mnist_pretrain/weights/best.pt",
    "runs/detect/train-14/weights/best.pt",
    "runs/detect/train/weights/best.pt",
    "best.pt",
]


def resolve_model_path(custom: str | None) -> str:
    if custom and Path(custom).exists():
        return custom
    for candidate in MODEL_CANDIDATES:
        if Path(candidate).exists():
            return candidate
    raise FileNotFoundError(f"未找到模型文件，请通过 --model 指定路径。已尝试: {', '.join(MODEL_CANDIDATES)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 数字识别命令行检测")
    parser.add_argument("--model", type=str, default=None, help="模型 .pt 路径")
    parser.add_argument("--source", type=str, required=True, help="输入源: 图片/文件夹/视频路径，或 0 表示摄像头")
    parser.add_argument("--conf", type=float, default=0.12, help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU 阈值")
    parser.add_argument("--save-dir", type=str, default="runs/detect/predict", help="结果保存目录")
    parser.add_argument("--show", action="store_true", help="实时显示（摄像头/视频）")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = resolve_model_path(args.model)
    model = YOLO(model_path)

    source: str | int = args.source
    if args.source.isdigit():
        source = int(args.source)

    print(f"模型: {model_path}")
    print(f"输入: {source}")
    print(f"置信度: {args.conf}")

    results = model.predict(
        source=source,
        conf=args.conf,
        iou=args.iou,
        save=True,
        project=args.save_dir,
        name="",
        exist_ok=True,
        show=args.show,
        verbose=True,
    )

    save_root = Path(args.save_dir)
    print(f"检测完成，结果保存在: {save_root.resolve()}")
    print(f"共处理 {len(results)} 个输入")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        sys.exit(1)
