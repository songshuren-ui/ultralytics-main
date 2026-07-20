"""YOLOv8 数字识别优化训练脚本 — 两阶段：MNIST 预训练 + 水表数字微调。."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import torch

from ultralytics import YOLO

ROOT = Path(__file__).parent

# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------
MNIST_YAML = ROOT / "mnist.yaml"
MNIST_STYLE_YAML = ROOT / "mnist_style.yaml"
DIGIT_YAML = ROOT / "digit_dataset" / "num.yaml"
DIGIT_AUG_YAML = ROOT / "digit_dataset_aug" / "num.yaml"
FAST_DIGIT_IMGSZ = 640
FAST_DIGIT_EPOCHS = 8
MNIST_CHECKPOINTS = [
    ROOT / "runs/detect/mnist_pretrain/weights/best.pt",
    ROOT / "runs/detect/train/weights/best.pt",
    ROOT / "yolov8n.pt",
]
DIGIT_BASE_CHECKPOINTS = MNIST_CHECKPOINTS
DEFAULT_MODEL = "yolov8s.pt"

# ---------------------------------------------------------------------------
# 阶段 1：MNIST 预训练（数字先学会基本笔画与结构）
# ---------------------------------------------------------------------------
MNIST_TRAIN_KWARGS = dict(
    data=str(MNIST_YAML),
    epochs=80,
    patience=20,
    batch=64,
    imgsz=160,
    lr0=0.006,
    lrf=0.05,
    cos_lr=True,
    warmup_epochs=3,
    optimizer="AdamW",
    weight_decay=0.0005,
    fliplr=0.0,
    flipud=0.0,
    degrees=8.0,
    translate=0.08,
    scale=0.2,
    shear=1.0,
    perspective=0.0,
    hsv_h=0.01,
    hsv_s=0.2,
    hsv_v=0.2,
    mosaic=0.3,
    mixup=0.0,
    copy_paste=0.0,
    auto_augment=None,
    erasing=0.0,
    close_mosaic=15,
    cls=1.0,
    box=7.5,
    workers=0,
    amp=True,
    cache=False,
    project=str(ROOT / "runs/detect"),
    name="mnist_pretrain",
    exist_ok=True,
    pretrained=True,
    verbose=True,
)

# ---------------------------------------------------------------------------
# 阶段 2a：水表数字微调 — 小数据集优先稳住分布
# ---------------------------------------------------------------------------
DIGIT_FREEZE_KWARGS = dict(
    data=str(DIGIT_YAML),
    epochs=60,
    patience=24,
    batch=4,
    imgsz=960,
    lr0=0.0015,
    lrf=0.08,
    cos_lr=True,
    warmup_epochs=8,
    optimizer="AdamW",
    weight_decay=0.0008,
    freeze=10,
    fliplr=0.0,
    flipud=0.0,
    degrees=4.0,
    translate=0.05,
    scale=0.18,
    shear=0.5,
    perspective=0.0,
    hsv_h=0.01,
    hsv_s=0.25,
    hsv_v=0.2,
    mosaic=0.2,
    mixup=0.0,
    copy_paste=0.0,
    auto_augment=None,
    erasing=0.0,
    close_mosaic=25,
    cls=1.2,
    box=8.0,
    dfl=1.5,
    workers=0,
    amp=True,
    cache=True,
    rect=True,
    overlap_mask=False,
    project=str(ROOT / "runs/detect"),
    name="digit_finetune",
    exist_ok=True,
    verbose=True,
)

# ---------------------------------------------------------------------------
# 阶段 2b：解冻全网络，低学习率精调
# ---------------------------------------------------------------------------
DIGIT_FULL_KWARGS = dict(
    **{
        k: v
        for k, v in DIGIT_FREEZE_KWARGS.items()
        if k not in ("freeze", "lr0", "epochs", "patience", "name", "mosaic", "close_mosaic")
    },
    freeze=0,
    lr0=0.0003,
    epochs=30,
    patience=12,
    mosaic=0.0,
    close_mosaic=120,
    name="digit_finetune_full",
)


def resolve_device(device: str) -> str:
    """解析训练设备，支持 auto/cpu/cuda/0 等写法。."""
    normalized = device.strip().lower()
    if normalized == "auto":
        return "0" if torch.cuda.is_available() else "cpu"
    if normalized == "cuda":
        return "0" if torch.cuda.is_available() else "cpu"
    if normalized.isdigit():
        return normalized if torch.cuda.is_available() else "cpu"
    return normalized


def find_checkpoint(candidates: list[Path]) -> Path:
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"未找到模型权重，已尝试: {[str(p) for p in candidates]}")


def choose_mnist_yaml(use_style_mnist: bool) -> Path:
    if use_style_mnist and MNIST_STYLE_YAML.exists():
        return MNIST_STYLE_YAML
    return MNIST_YAML


def choose_digit_yaml(use_augmented_digit: bool) -> Path:
    if use_augmented_digit and DIGIT_AUG_YAML.exists():
        return DIGIT_AUG_YAML
    return DIGIT_YAML


def prepare_datasets(
    skip_mnist: bool = False,
    build_style_mnist: bool = False,
    build_augmented_digit: bool = False,
    mnist_style_copies: int = 1,
    digit_aug_copies: int = 3,
    digit_boost_classes: list[int] | None = None,
    digit_boost_copies: int = 2,
) -> None:
    """运行数据集准备脚本。."""
    from augment_digit_dataset import main as build_digit_aug
    from augment_mnist_style import main as build_mnist_style
    from prepare_digit_dataset import main as prep_digit

    prep_digit()
    if not skip_mnist and MNIST_YAML.exists():
        try:
            from prepare_mnist_val import main as prep_mnist

            prep_mnist()
        except Exception as exc:
            print(f"[警告] MNIST 验证集划分跳过: {exc}")

    if build_style_mnist:
        saved_argv = sys.argv[:]
        try:
            sys.argv = [sys.argv[0], "--copies", str(mnist_style_copies)]
            build_mnist_style()
        finally:
            sys.argv = saved_argv

    if build_augmented_digit:
        saved_argv = sys.argv[:]
        try:
            sys.argv = [
                sys.argv[0],
                "--copies",
                str(digit_aug_copies),
                "--boost-copies",
                str(digit_boost_copies),
                *[arg for class_id in (digit_boost_classes or []) for arg in ("--boost-class", str(class_id))],
            ]
            build_digit_aug()
        finally:
            sys.argv = saved_argv


def train_mnist(model_path: Path | None = None, data_yaml: Path | None = None, device: str = "auto") -> Path:
    """阶段 1：MNIST 预训练。."""
    weights = model_path or find_checkpoint([ROOT / DEFAULT_MODEL, ROOT / "yolov8n.pt"])
    resolved_device = resolve_device(device)
    train_kwargs = {**MNIST_TRAIN_KWARGS, "data": str(data_yaml or MNIST_YAML), "device": resolved_device}
    print(
        f"\n{'=' * 60}\n阶段 1: MNIST 预训练\n模型: {weights}\n数据集: {train_kwargs['data']}\n设备: {resolved_device}\n{'=' * 60}"
    )
    model = YOLO(str(weights))
    model.train(**train_kwargs)
    best = ROOT / "runs/detect/mnist_pretrain/weights/best.pt"
    print(f"MNIST 训练完成，最佳权重: {best}")
    return best


def train_digit_finetune(base_weights: Path, data_yaml: Path | None = None, device: str = "auto") -> Path:
    """阶段 2a：冻结骨干微调。."""
    selected_yaml = data_yaml or DIGIT_YAML
    if not selected_yaml.exists():
        raise FileNotFoundError(f"缺少 {selected_yaml}，请先准备对应数据集")

    resolved_device = resolve_device(device)
    train_kwargs = {**DIGIT_FREEZE_KWARGS, "data": str(selected_yaml), "device": resolved_device}
    print(
        f"\n{'=' * 60}\n阶段 2a: 水表数字微调（冻结骨干）\n基础模型: {base_weights}\n数据集: {train_kwargs['data']}\n设备: {resolved_device}\n{'=' * 60}"
    )
    model = YOLO(str(base_weights))
    model.train(**train_kwargs)
    best = ROOT / "runs/detect/digit_finetune/weights/best.pt"
    print(f"微调完成，最佳权重: {best}")
    return best


def train_digit_full(base_weights: Path, data_yaml: Path | None = None, device: str = "auto") -> Path:
    """阶段 2b：解冻全网络精调。."""
    resolved_device = resolve_device(device)
    train_kwargs = {**DIGIT_FULL_KWARGS, "data": str(data_yaml or DIGIT_YAML), "device": resolved_device}
    print(
        f"\n{'=' * 60}\n阶段 2b: 全网络精调（低学习率）\n基础模型: {base_weights}\n数据集: {train_kwargs['data']}\n设备: {resolved_device}\n{'=' * 60}"
    )
    model = YOLO(str(base_weights))
    model.train(**train_kwargs)
    best = ROOT / "runs/detect/digit_finetune_full/weights/best.pt"
    print(f"精调完成，最佳权重: {best}")
    return best


def train_digit_fast_smoke(base_weights: Path, data_yaml: Path | None = None, device: str = "auto") -> Path:
    """快速验证增强是否有效，优先缩短 CPU 训练时间。."""
    selected_yaml = data_yaml or DIGIT_YAML
    if not selected_yaml.exists():
        raise FileNotFoundError(f"缺少 {selected_yaml}，请先准备对应数据集")

    resolved_device = resolve_device(device)
    train_kwargs = {
        **DIGIT_FREEZE_KWARGS,
        "data": str(selected_yaml),
        "device": resolved_device,
        "epochs": FAST_DIGIT_EPOCHS,
        "patience": FAST_DIGIT_EPOCHS,
        "imgsz": FAST_DIGIT_IMGSZ,
        "batch": 8,
        "cache": False,
        "rect": False,
        "mosaic": 0.0,
        "close_mosaic": 0,
        "degrees": 2.0,
        "translate": 0.03,
        "scale": 0.1,
        "shear": 0.0,
        "name": "digit_finetune_fast_smoke",
    }
    print(
        f"\n{'=' * 60}\n快速验证: 水表数字短训\n基础模型: {base_weights}\n数据集: {train_kwargs['data']}\n设备: {resolved_device}\n{'=' * 60}"
    )
    model = YOLO(str(base_weights))
    model.train(**train_kwargs)
    best = ROOT / "runs/detect/digit_finetune_fast_smoke/weights/best.pt"
    print(f"快速验证完成，最佳权重: {best}")
    return best


def deploy_model(best_pt: Path, target: Path | None = None) -> Path:
    """将最佳权重复制到 dist/best.pt 供检测应用使用。."""
    target = target or ROOT / "dist/best.pt"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_pt, target)
    print(f"已部署模型 → {target}")
    return target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="YOLOv8 数字识别优化训练",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python train_digit.py --prepare-only --build-style-mnist --build-augmented-digit
  python train_digit.py --stage all --full --build-style-mnist --build-augmented-digit
  python train_digit.py --stage all --full --mnist-weights yolov8s.pt
        """,
    )
    parser.add_argument(
        "--stage",
        choices=["mnist", "digit", "all"],
        default="digit",
        help="训练阶段: mnist=预训练, digit=水表微调, all=全部",
    )
    parser.add_argument("--full", action="store_true", help="digit 阶段完成后继续全网络精调")
    parser.add_argument("--prepare-only", action="store_true", help="仅划分数据集，不训练")
    parser.add_argument("--skip-mnist-prep", action="store_true", help="跳过 MNIST 验证集划分")
    parser.add_argument("--skip-prepare", action="store_true", help="跳过数据集准备")
    parser.add_argument("--mnist-weights", type=str, default=None, help="指定初始权重路径，例如 yolov8s.pt")
    parser.add_argument("--build-style-mnist", action="store_true", help="生成多风格 MNIST 预训练数据集")
    parser.add_argument("--build-augmented-digit", action="store_true", help="生成增强后的真实数字数据集")
    parser.add_argument("--mnist-style-copies", type=int, default=1, help="每张 MNIST 原图生成多少张风格增强图")
    parser.add_argument("--digit-aug-copies", type=int, default=3, help="每张真实训练图生成多少张增强图")
    parser.add_argument(
        "--digit-boost-class",
        type=int,
        action="append",
        default=[1, 2, 4, 6, 7],
        help="对包含指定类别的训练图追加增强，可多次传入；默认补强弱识别数字",
    )
    parser.add_argument("--digit-boost-copies", type=int, default=2, help="命中 boost 类别的训练图额外生成多少张增强图")
    parser.add_argument("--deploy", action="store_true", default=True, help="训练完成后复制到 dist/best.pt")
    parser.add_argument("--no-deploy", action="store_false", dest="deploy", help="不自动部署")
    parser.add_argument("--fast-smoke", action="store_true", help="使用更快的短训配置验证增强是否有效")
    parser.add_argument("--device", type=str, default="auto", help="训练设备，如 auto、cpu、cuda、0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected_mnist_yaml = choose_mnist_yaml(args.build_style_mnist)
    selected_digit_yaml = choose_digit_yaml(args.build_augmented_digit)

    if not args.skip_prepare:
        print("准备数据集…")
        prepare_datasets(
            skip_mnist=args.skip_mnist_prep,
            build_style_mnist=args.build_style_mnist,
            build_augmented_digit=args.build_augmented_digit,
            mnist_style_copies=args.mnist_style_copies,
            digit_aug_copies=args.digit_aug_copies,
            digit_boost_classes=args.digit_boost_class,
            digit_boost_copies=args.digit_boost_copies,
        )

    if args.prepare_only:
        print("数据集准备完成。")
        return

    final_best: Path | None = None

    if args.stage in ("mnist", "all"):
        mnist_base = Path(args.mnist_weights) if args.mnist_weights else None
        final_best = train_mnist(mnist_base, selected_mnist_yaml, args.device)

    if args.stage in ("digit", "all"):
        if args.stage == "digit":
            base = Path(args.mnist_weights) if args.mnist_weights else find_checkpoint(DIGIT_BASE_CHECKPOINTS)
        else:
            base = final_best or find_checkpoint(
                [ROOT / "runs/detect/mnist_pretrain/weights/best.pt", *MNIST_CHECKPOINTS]
            )
        final_best = (
            train_digit_fast_smoke(base, selected_digit_yaml, args.device)
            if args.fast_smoke
            else train_digit_finetune(base, selected_digit_yaml, args.device)
        )

        if args.full and not args.fast_smoke:
            final_best = train_digit_full(final_best, selected_digit_yaml, args.device)

    if final_best and args.deploy:
        deploy_model(final_best)

    print("\n训练流程结束。")
    if final_best:
        print(f"最佳模型: {final_best}")
        print("检测应用会自动优先加载 dist/best.pt")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        sys.exit(1)
