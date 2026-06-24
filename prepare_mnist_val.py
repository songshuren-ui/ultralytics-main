"""从 MNIST YOLO 训练集中划分验证集，并更新 mnist.yaml。."""

from __future__ import annotations

import random
import shutil
from pathlib import Path

ROOT = Path(__file__).parent
MNIST_ROOT = ROOT / "mnist_yolo"
SEED = 42
VAL_COUNT = 5000


def _has_val_split() -> bool:
    val_dir = MNIST_ROOT / "images" / "val"
    return val_dir.exists() and any(val_dir.iterdir())


def split_mnist() -> tuple[int, int]:
    train_img_dir = MNIST_ROOT / "images" / "train"
    train_lbl_dir = MNIST_ROOT / "labels" / "train"
    val_img_dir = MNIST_ROOT / "images" / "val"
    val_lbl_dir = MNIST_ROOT / "labels" / "val"

    imgs = sorted(p for p in train_img_dir.iterdir() if p.is_file())
    if not imgs:
        raise RuntimeError(f"MNIST 训练集为空: {train_img_dir}")

    random.seed(SEED)
    random.shuffle(imgs)
    val_set = set(imgs[: min(VAL_COUNT, len(imgs) // 5)])

    val_img_dir.mkdir(parents=True, exist_ok=True)
    val_lbl_dir.mkdir(parents=True, exist_ok=True)

    for img in val_set:
        lbl = train_lbl_dir / f"{img.stem}.txt"
        shutil.move(str(img), val_img_dir / img.name)
        if lbl.exists():
            shutil.move(str(lbl), val_lbl_dir / lbl.name)

    train_count = len(list(train_img_dir.iterdir()))
    val_count = len(list(val_img_dir.iterdir()))
    return train_count, val_count


def update_yaml(train_count: int, val_count: int) -> None:
    yaml_path = ROOT / "mnist.yaml"
    content = """# MNIST 数字检测数据集（由 prepare_mnist_val.py 维护）
path: ./mnist_yolo
train: images/train
val: images/val
names:
  0: '0'
  1: '1'
  2: '2'
  3: '3'
  4: '4'
  5: '5'
  6: '6'
  7: '7'
  8: '8'
  9: '9'
"""
    yaml_path.write_text(content, encoding="utf-8")
    print(f"已更新 {yaml_path}  (train={train_count}, val={val_count})")


def main() -> None:
    if not MNIST_ROOT.exists():
        raise FileNotFoundError(f"未找到 MNIST 目录: {MNIST_ROOT}")

    if _has_val_split():
        train_count = len(list((MNIST_ROOT / "images" / "train").iterdir()))
        val_count = len(list((MNIST_ROOT / "images" / "val").iterdir()))
        print(f"MNIST 验证集已存在: train={train_count}, val={val_count}")
    else:
        print("正在从 MNIST 训练集划分 5000 张验证图片…")
        train_count, val_count = split_mnist()

    update_yaml(train_count, val_count)


if __name__ == "__main__":
    main()
