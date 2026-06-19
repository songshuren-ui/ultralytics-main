"""YOLOv8 数字识别共享推理逻辑。."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from ultralytics import YOLO
from ultralytics.data.utils import IMG_FORMATS, VID_FORMATS

MODEL_CANDIDATES = [
    "dist/best.pt",
    "runs/detect/digit_finetune_full/weights/best.pt",
    "runs/detect/digit_finetune_fast_smoke/weights/best.pt",
    "runs/detect/digit_finetune/weights/best.pt",
    "runs/detect/mnist_pretrain/weights/best.pt",
    "runs/detect/train-14/weights/best.pt",
    "runs/detect/train/weights/best.pt",
    "best.pt",
]

IMG_TYPES = sorted(IMG_FORMATS)
VID_TYPES = sorted(VID_FORMATS)


def resolve_model_path(custom: str | None = None) -> str:
    """返回第一个存在的模型路径，或用户指定的路径。."""
    if custom and Path(custom).exists():
        return custom
    if hasattr(sys, "_MEIPASS"):
        bundled = os.path.join(sys._MEIPASS, "best.pt")
        if os.path.exists(bundled):
            return bundled
    for candidate in MODEL_CANDIDATES:
        if Path(candidate).exists():
            return candidate
    return custom or MODEL_CANDIDATES[0]


def load_model(model_path: str) -> YOLO:
    """加载 YOLO 模型。."""
    return YOLO(model_path)


def render_detection_summary(result: Any) -> tuple[str, str, int, float, list[dict[str, float | str]]]:
    """将检测结果格式化为可读文本和结构化数据。."""
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return "未检测到数字", "--", 0, 0.0, []

    ordered_boxes = sorted(boxes, key=lambda item: float(item.xyxy[0][0]))
    names = result.names
    lines: list[str] = []
    digits: list[str] = []
    confs: list[float] = []
    items: list[dict[str, float | str]] = []

    for box in ordered_boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = names.get(cls_id, str(cls_id))
        digits.append(label)
        confs.append(conf)
        items.append({"label": label, "confidence": conf})
        lines.append(f"数字 {label} - 置信度 {conf:.2%}")

    reading = "".join(digits)
    avg_conf = sum(confs) / len(confs)
    text = "识别结果：" + reading + "\n\n" + "\n".join(lines)
    return text, reading, len(digits), avg_conf, items


def detect_result_to_payload(result: Any, original: np.ndarray) -> dict[str, object]:
    """将 YOLO 单条结果转为统一状态结构。."""
    annotated = result.plot()
    summary_text, reading, count, avg_conf, items = render_detection_summary(result)
    return {
        "original": original,
        "annotated": annotated,
        "summary_text": summary_text,
        "reading": reading,
        "count": count,
        "avg_conf": avg_conf,
        "items": items,
    }


def detect_image(model: YOLO, image: np.ndarray, conf: float, iou: float) -> dict[str, object]:
    """执行单张图像检测并返回统一结果。."""
    results = model.predict(source=image, conf=conf, iou=iou, verbose=False)
    return detect_result_to_payload(results[0], image)


def save_temp_file(data: bytes, suffix: str) -> str:
    """将字节流保存到临时文件。."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data)
        return tmp.name
