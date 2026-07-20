"""生成更接近真实拍照场景的 MNIST YOLO 预训练数据。."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).parent
SRC_ROOT = ROOT / "mnist_yolo"
DST_ROOT = ROOT / "mnist_yolo_style"
SEED = 42
DEFAULT_TRAIN_LIMIT = 6000
DEFAULT_VAL_LIMIT = 1000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成多风格 MNIST YOLO 数据集")
    parser.add_argument("--copies", type=int, default=1, help="每张原图额外生成多少张增强图")
    parser.add_argument("--limit", type=int, default=0, help="每个 split 最多处理多少张，0 表示全部")
    parser.add_argument("--seed", type=int, default=SEED, help="随机种子")
    return parser.parse_args()


def ensure_dirs() -> None:
    for split in ("train", "val"):
        (DST_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
        (DST_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)


def read_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"无法读取图片: {path}")
    return image


def build_background(height: int, width: int, rng: random.Random) -> np.ndarray:
    base = rng.randint(150, 255)
    gradient = np.tile(np.linspace(base - rng.randint(0, 40), base, width, dtype=np.float32), (height, 1))
    noise = np.random.normal(0, rng.uniform(4, 14), (height, width)).astype(np.float32)
    background = np.clip(gradient + noise, 0, 255).astype(np.uint8)
    if rng.random() < 0.35:
        background = cv2.GaussianBlur(background, (3, 3), 0)
    return background


def stylize_digit(image: np.ndarray, rng: random.Random) -> np.ndarray:
    canvas = image.copy()
    if canvas.ndim == 3 and canvas.shape[2] in (3, 4):
        canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    elif canvas.ndim == 3 and canvas.shape[2] == 1:
        canvas = canvas[:, :, 0]
    canvas = np.squeeze(canvas)
    if canvas.ndim != 2:
        raise ValueError(f"MNIST 图像维度异常: {canvas.shape}")
    if rng.random() < 0.65:
        canvas = 255 - canvas

    h, w = canvas.shape
    scale = rng.uniform(0.9, 1.25)
    angle = rng.uniform(-12, 12)
    tx = rng.uniform(-2.5, 2.5)
    ty = rng.uniform(-2.5, 2.5)
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    matrix[:, 2] += [tx, ty]
    warped = cv2.warpAffine(canvas, matrix, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)

    fg_color = rng.randint(10, 120)
    background = build_background(h, w, rng).astype(np.float32)
    norm = warped.astype(np.float32) / 255.0
    composed = background * (1.0 - norm) + fg_color * norm

    if rng.random() < 0.5:
        composed = cv2.GaussianBlur(composed, (3, 3), rng.uniform(0.2, 1.0))

    if rng.random() < 0.35:
        alpha = rng.uniform(0.7, 1.15)
        beta = rng.uniform(-18, 18)
        composed = composed * alpha + beta

    if rng.random() < 0.25:
        line_y = rng.randint(0, h - 1)
        cv2.line(composed, (0, line_y), (w - 1, line_y), rng.randint(120, 220), 1)

    noise = np.random.normal(0, rng.uniform(2, 8), (h, w)).astype(np.float32)
    composed = np.clip(composed + noise, 0, 255).astype(np.uint8)
    return composed


def copy_label(src: Path, dst: Path) -> None:
    dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")


def export_yaml() -> None:
    yaml_path = ROOT / "mnist_style.yaml"
    lines = [
        "# 多风格 MNIST 数字检测数据集（由 augment_mnist_style.py 生成）",
        "path: ./mnist_yolo_style",
        "train: images/train",
        "val: images/val",
        "names:",
    ]
    for idx in range(10):
        lines.append(f"  {idx}: '{idx}'")
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def process_split(split: str, copies: int, limit: int, rng: random.Random) -> tuple[int, int]:
    src_img_dir = SRC_ROOT / "images" / split
    src_lbl_dir = SRC_ROOT / "labels" / split
    dst_img_dir = DST_ROOT / "images" / split
    dst_lbl_dir = DST_ROOT / "labels" / split

    if not src_img_dir.exists() or not src_lbl_dir.exists():
        return 0, 0

    images = sorted(src_img_dir.glob("*.*"))
    effective_limit = limit
    if effective_limit <= 0:
        effective_limit = DEFAULT_TRAIN_LIMIT if split == "train" else DEFAULT_VAL_LIMIT
    if effective_limit > 0:
        images = images[:effective_limit]

    written = 0
    for img_path in images:
        lbl_path = src_lbl_dir / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            continue

        original = read_gray(img_path)
        out_img = dst_img_dir / img_path.name
        out_lbl = dst_lbl_dir / lbl_path.name
        cv2.imwrite(str(out_img), original)
        copy_label(lbl_path, out_lbl)
        written += 1

        for index in range(copies):
            aug = stylize_digit(original, rng)
            aug_name = f"{img_path.stem}_style{index}{img_path.suffix}"
            aug_lbl_name = f"{lbl_path.stem}_style{index}{lbl_path.suffix}"
            cv2.imwrite(str(dst_img_dir / aug_name), aug)
            copy_label(lbl_path, dst_lbl_dir / aug_lbl_name)
            written += 1

    return len(images), written


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    if not SRC_ROOT.exists():
        raise FileNotFoundError(f"未找到 MNIST 数据集目录: {SRC_ROOT}")

    ensure_dirs()
    total_src = 0
    total_written = 0
    for split in ("train", "val"):
        src_count, written = process_split(split, args.copies, args.limit, rng)
        total_src += src_count
        total_written += written
        print(f"{split}: 原图 {src_count} 张，输出 {written} 张")

    export_yaml()
    print(f"完成：共处理 {total_src} 张，生成 {total_written} 张到 {DST_ROOT}")


if __name__ == "__main__":
    main()
