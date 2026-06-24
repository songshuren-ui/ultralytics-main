"""对真实数字检测数据集进行离线扩增，生成更丰富的小样本训练集。."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).parent
SRC_ROOT = ROOT / "digit_dataset"
DST_ROOT = ROOT / "digit_dataset_aug"
SEED = 42
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成真实场景数字检测增强数据集")
    parser.add_argument("--copies", type=int, default=3, help="每张训练图额外生成多少张增强图")
    parser.add_argument("--seed", type=int, default=SEED, help="随机种子")
    parser.add_argument(
        "--boost-class",
        type=int,
        action="append",
        default=[1, 2, 4, 6, 7],
        help="对包含指定类别的训练图追加更多增强次数，可多次传入；默认补强弱识别数字",
    )
    parser.add_argument(
        "--boost-copies",
        type=int,
        default=2,
        help="命中 boost-class 的训练图额外再生成多少张增强图",
    )
    return parser.parse_args()


def ensure_dirs() -> None:
    for split in ("train", "val"):
        (DST_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
        (DST_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)


def load_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"无法读取图片: {path}")
    return image


def read_boxes(path: Path, width: int, height: int) -> list[list[float]]:
    boxes: list[list[float]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        cls_id, x, y, w, h = line.split()
        cls_value = int(cls_id)
        cx = float(x) * width
        cy = float(y) * height
        bw = float(w) * width
        bh = float(h) * height
        x1 = cx - bw / 2
        y1 = cy - bh / 2
        x2 = cx + bw / 2
        y2 = cy + bh / 2
        boxes.append([cls_value, x1, y1, x2, y2])
    return boxes


def write_boxes(path: Path, boxes: list[list[float]], width: int, height: int) -> None:
    lines: list[str] = []
    for cls_id, x1, y1, x2, y2 in boxes:
        x1 = min(max(x1, 0.0), width - 1.0)
        y1 = min(max(y1, 0.0), height - 1.0)
        x2 = min(max(x2, 1.0), width)
        y2 = min(max(y2, 1.0), height)
        bw = x2 - x1
        bh = y2 - y1
        if bw < 4 or bh < 4:
            continue
        cx = (x1 + x2) / 2 / width
        cy = (y1 + y2) / 2 / height
        nw = bw / width
        nh = bh / height
        lines.append(f"{int(cls_id)} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def transform_boxes(boxes: list[list[float]], matrix: np.ndarray, width: int, height: int) -> list[list[float]]:
    transformed: list[list[float]] = []
    for cls_id, x1, y1, x2, y2 in boxes:
        corners = np.array(
            [[x1, y1, 1.0], [x2, y1, 1.0], [x2, y2, 1.0], [x1, y2, 1.0]],
            dtype=np.float32,
        )
        warped = corners @ matrix.T
        xs = warped[:, 0]
        ys = warped[:, 1]
        nx1 = max(0.0, float(xs.min()))
        ny1 = max(0.0, float(ys.min()))
        nx2 = min(float(width), float(xs.max()))
        ny2 = min(float(height), float(ys.max()))
        if nx2 - nx1 >= 4 and ny2 - ny1 >= 4:
            transformed.append([cls_id, nx1, ny1, nx2, ny2])
    return transformed


def augment_image(
    image: np.ndarray, boxes: list[list[float]], rng: random.Random
) -> tuple[np.ndarray, list[list[float]]]:
    height, width = image.shape[:2]
    angle = rng.uniform(-4.0, 4.0)
    scale = rng.uniform(0.92, 1.08)
    tx = rng.uniform(-0.03, 0.03) * width
    ty = rng.uniform(-0.03, 0.03) * height
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, scale)
    matrix[:, 2] += [tx, ty]

    border = tuple(int(v) for v in np.mean(image.reshape(-1, 3), axis=0))
    augmented = cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_LINEAR, borderValue=border)
    aug_boxes = transform_boxes(boxes, matrix, width, height)

    if rng.random() < 0.35:
        alpha = rng.uniform(0.75, 1.2)
        beta = rng.uniform(-20, 20)
        augmented = cv2.convertScaleAbs(augmented, alpha=alpha, beta=beta)

    if rng.random() < 0.3:
        hsv = cv2.cvtColor(augmented, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 1] *= rng.uniform(0.8, 1.15)
        hsv[..., 2] *= rng.uniform(0.8, 1.15)
        hsv = np.clip(hsv, 0, 255).astype(np.uint8)
        augmented = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    if rng.random() < 0.35:
        augmented = cv2.GaussianBlur(augmented, (3, 3), rng.uniform(0.2, 1.2))

    if rng.random() < 0.2:
        noise = np.random.normal(0, rng.uniform(2, 8), augmented.shape).astype(np.float32)
        augmented = np.clip(augmented.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    if rng.random() < 0.18:
        shade = np.linspace(rng.uniform(0.7, 1.0), rng.uniform(0.9, 1.2), width, dtype=np.float32)
        shade_map = np.tile(shade, (height, 1))[:, :, None]
        augmented = np.clip(augmented.astype(np.float32) * shade_map, 0, 255).astype(np.uint8)

    return augmented, aug_boxes


def copy_originals(split: str) -> int:
    count = 0
    img_dir = SRC_ROOT / "images" / split
    lbl_dir = SRC_ROOT / "labels" / split
    out_img_dir = DST_ROOT / "images" / split
    out_lbl_dir = DST_ROOT / "labels" / split

    for img_path in sorted(img_dir.iterdir()):
        if img_path.suffix.lower() not in IMAGE_SUFFIXES or not img_path.is_file():
            continue
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            continue
        image = load_image(img_path)
        cv2.imwrite(str(out_img_dir / img_path.name), image)
        (out_lbl_dir / lbl_path.name).write_text(lbl_path.read_text(encoding="utf-8"), encoding="utf-8")
        count += 1
    return count


def augment_train(copies: int, rng: random.Random, boost_classes: set[int] | None = None, boost_copies: int = 0) -> int:
    img_dir = SRC_ROOT / "images" / "train"
    lbl_dir = SRC_ROOT / "labels" / "train"
    out_img_dir = DST_ROOT / "images" / "train"
    out_lbl_dir = DST_ROOT / "labels" / "train"

    boost_classes = boost_classes or set()
    written = 0
    for img_path in sorted(img_dir.iterdir()):
        if img_path.suffix.lower() not in IMAGE_SUFFIXES or not img_path.is_file():
            continue
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            continue
        image = load_image(img_path)
        height, width = image.shape[:2]
        boxes = read_boxes(lbl_path, width, height)
        if not boxes:
            continue

        class_ids = {int(box[0]) for box in boxes}
        total_copies = copies + (boost_copies if boost_classes & class_ids else 0)
        for index in range(total_copies):
            augmented, aug_boxes = augment_image(image, boxes, rng)
            if not aug_boxes:
                continue
            out_name = f"{img_path.stem}_aug{index}{img_path.suffix}"
            out_label = f"{lbl_path.stem}_aug{index}{lbl_path.suffix}"
            cv2.imwrite(str(out_img_dir / out_name), augmented)
            write_boxes(out_lbl_dir / out_label, aug_boxes, width, height)
            written += 1
    return written


def export_yaml() -> None:
    yaml_path = DST_ROOT / "num.yaml"
    lines = [
        "# 增强后的真实数字检测数据集（由 augment_digit_dataset.py 生成）",
        "path: ./digit_dataset_aug",
        "train: images/train",
        "val: images/val",
        "names:",
    ]
    for idx in range(10):
        lines.append(f"  {idx}: '{idx}'")
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    if not SRC_ROOT.exists():
        raise FileNotFoundError(f"未找到真实数据集目录: {SRC_ROOT}")

    ensure_dirs()
    train_original = copy_originals("train")
    val_original = copy_originals("val")
    train_augmented = augment_train(
        args.copies,
        rng,
        boost_classes=set(args.boost_class),
        boost_copies=max(args.boost_copies, 0),
    )
    export_yaml()

    print(f"train 原图复制: {train_original} 张")
    print(f"val 原图复制: {val_original} 张")
    print(f"train 增强新增: {train_augmented} 张")
    print(f"输出目录: {DST_ROOT}")


if __name__ == "__main__":
    main()
