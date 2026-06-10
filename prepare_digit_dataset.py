"""将 digit_dataset 划分为 train/val，并生成 num.yaml 配置文件。"""

from __future__ import annotations

import random
import shutil
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent
DATASET = ROOT / "digit_dataset"
SEED = 42
VAL_RATIO = 0.2
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

NAMES = {i: str(i) for i in range(10)}


def _is_flat_layout() -> bool:
    """images/ 根目录下仍有图片文件时视为未划分。"""
    images_dir = DATASET / "images"
    if not images_dir.exists():
        return False
    return any(p.suffix.lower() in IMAGE_SUFFIXES for p in images_dir.iterdir() if p.is_file())


def _image_files(directory: Path) -> list[Path]:
    return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def split_dataset() -> tuple[int, int]:
    """将扁平 images/labels 移动到 train/val 子目录。"""
    images_dir = DATASET / "images"
    labels_dir = DATASET / "labels"

    imgs = _image_files(images_dir)
    if not imgs:
        train_imgs = list((DATASET / "images" / "train").glob("*.*")) if (DATASET / "images" / "train").exists() else []
        val_imgs = list((DATASET / "images" / "val").glob("*.*")) if (DATASET / "images" / "val").exists() else []
        return len(train_imgs), len(val_imgs)

    random.seed(SEED)
    random.shuffle(imgs)
    n_val = max(1, int(len(imgs) * VAL_RATIO))
    val_set = set(imgs[:n_val])

    for split in ("train", "val"):
        (DATASET / "images" / split).mkdir(parents=True, exist_ok=True)
        (DATASET / "labels" / split).mkdir(parents=True, exist_ok=True)

    for img in imgs:
        split = "val" if img in val_set else "train"
        lbl = labels_dir / f"{img.stem}.txt"
        shutil.move(str(img), DATASET / "images" / split / img.name)
        if lbl.exists():
            shutil.move(str(lbl), DATASET / "labels" / split / lbl.name)

    train_count = len(list((DATASET / "images" / "train").glob("*.*")))
    val_count = len(list((DATASET / "images" / "val").glob("*.*")))
    return train_count, val_count


def collect_dataset_stats() -> dict[str, object]:
    """汇总图片数、标签数、类别分布和缺失标签。"""
    image_root = DATASET / "images"
    label_root = DATASET / "labels"

    stats: dict[str, object] = {
        "images": {},
        "labels": {},
        "boxes": {},
        "missing_labels": {},
        "class_hist": Counter(),
    }

    for split in ("train", "val"):
        img_dir = image_root / split
        lbl_dir = label_root / split
        images = sorted(img_dir.glob("*.*")) if img_dir.exists() else []
        labels = sorted(lbl_dir.glob("*.txt")) if lbl_dir.exists() else []
        missing = [img.name for img in images if not (lbl_dir / f"{img.stem}.txt").exists()]
        box_count = 0

        for label_path in labels:
            for raw in label_path.read_text(encoding="utf-8").splitlines():
                line = raw.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 5:
                    continue
                cls_id = int(parts[0])
                stats["class_hist"][cls_id] += 1
                box_count += 1

        stats["images"][split] = len(images)
        stats["labels"][split] = len(labels)
        stats["boxes"][split] = box_count
        stats["missing_labels"][split] = missing

    return stats


def print_dataset_report(stats: dict[str, object]) -> None:
    """输出数据集摘要，并对小数据集给出告警。"""
    images = stats["images"]
    labels = stats["labels"]
    boxes = stats["boxes"]
    class_hist: Counter = stats["class_hist"]
    missing = stats["missing_labels"]

    print("数据集统计:")
    print(
        f"  train: images={images['train']} labels={labels['train']} boxes={boxes['train']} | "
        f"val: images={images['val']} labels={labels['val']} boxes={boxes['val']}"
    )
    print(f"  classes: {dict(sorted(class_hist.items()))}")

    total_images = int(images["train"]) + int(images["val"])
    if total_images < 200:
        print("[提示] 当前数据集图片数量较少（<200），建议优先补充真实场景样本，否则精度上限会受限。")

    for split in ("train", "val"):
        missing_list = missing[split]
        if missing_list:
            preview = ", ".join(missing_list[:5])
            print(f"[警告] {split} 存在 {len(missing_list)} 张图片缺少标签，例如: {preview}")


def write_yaml(train_count: int, val_count: int) -> Path:
    yaml_path = DATASET / "num.yaml"
    lines = [
        "# YOLO 水表/数字检测数据集配置（由 prepare_digit_dataset.py 生成）",
        "path: ./digit_dataset",
        "train: images/train",
        "val: images/val",
        "names:",
    ]
    for idx, name in NAMES.items():
        lines.append(f"  {idx}: '{name}'")
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"已写入 {yaml_path}  (train={train_count}, val={val_count})")
    return yaml_path


def main() -> None:
    if not DATASET.exists():
        raise FileNotFoundError(f"未找到数据集目录: {DATASET}")

    if _is_flat_layout():
        print("正在划分 digit_dataset → train/val …")
        train_count, val_count = split_dataset()
    else:
        train_count = len(list((DATASET / "images" / "train").glob("*.*")))
        val_count = len(list((DATASET / "images" / "val").glob("*.*")))
        print(f"数据集已划分: train={train_count}, val={val_count}")

    if train_count == 0:
        raise RuntimeError("训练集为空，请检查 digit_dataset/images 是否有图片。")

    write_yaml(train_count, val_count)
    print_dataset_report(collect_dataset_stats())


if __name__ == "__main__":
    main()
