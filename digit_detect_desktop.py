"""YOLOv8 数字识别桌面端。"""

from __future__ import annotations

import os
import sys
import tempfile
import zipfile
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QProgressBar,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from digit_detector_core import detect_image, load_model, resolve_model_path


class BatchWorker(QThread):
    progress = Signal(int, str)
    finished_ok = Signal(list, str)
    failed = Signal(str)

    def __init__(self, model, paths: list[str], conf: float, iou: float) -> None:
        super().__init__()
        self.model = model
        self.paths = paths
        self.conf = conf
        self.iou = iou

    def run(self) -> None:
        rows: list[dict[str, object]] = []
        zip_path = ""
        try:
            temp_dir = tempfile.mkdtemp(prefix="digit_batch_")
            zip_path = os.path.join(temp_dir, "batch_detect_results.zip")
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
                total = max(len(self.paths), 1)
                for index, path in enumerate(self.paths, start=1):
                    self.progress.emit(int(index * 100 / total), f"处理中 {index}/{total}: {Path(path).name}")
                    image = cv2.imread(path)
                    if image is None:
                        continue
                    payload = detect_image(self.model, image, self.conf, self.iou)
                    out_name = f"detect_{Path(path).stem}.jpg"
                    out_path = os.path.join(temp_dir, out_name)
                    cv2.imencode(".jpg", payload["annotated"])[1].tofile(out_path)
                    archive.write(out_path, arcname=out_name)
                    rows.append({
                        "path": path,
                        "payload": payload,
                    })
            self.finished_ok.emit(rows, zip_path)
        except Exception as exc:
            self.failed.emit(str(exc))


class VideoWorker(QThread):
    progress = Signal(int, str)
    finished_ok = Signal(dict, str)
    failed = Signal(str)

    def __init__(self, model, video_path: str, conf: float, iou: float) -> None:
        super().__init__()
        self.model = model
        self.video_path = video_path
        self.conf = conf
        self.iou = iou

    def run(self) -> None:
        try:
            out_dir = tempfile.mkdtemp(prefix="digit_video_")
            results = self.model.predict(
                source=self.video_path,
                conf=self.conf,
                iou=self.iou,
                save=True,
                project=out_dir,
                name="output",
                exist_ok=True,
                verbose=False,
            )
            if not results:
                raise RuntimeError("视频未产出任何检测结果")

            result = results[0]
            capture = cv2.VideoCapture(self.video_path)
            frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            middle = max(frames // 2, 0)
            capture.set(cv2.CAP_PROP_POS_FRAMES, middle)
            ok, frame = capture.read()
            capture.release()
            if not ok or frame is None:
                raise RuntimeError("无法读取视频预览帧")

            payload = detect_image(self.model, frame, self.conf, self.iou)
            output_root = Path(out_dir) / "output"
            candidates = [
                output_root / Path(self.video_path).name,
                output_root / f"{Path(self.video_path).stem}.avi",
                output_root / f"{Path(self.video_path).stem}.mp4",
            ]
            out_video = next((str(path) for path in candidates if path.exists()), "")
            if not out_video:
                raise RuntimeError("未找到导出的视频结果")
            self.finished_ok.emit(payload, out_video)
        except Exception as exc:
            self.failed.emit(str(exc))


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("YOLOv8 数字识别软件")
        self.resize(1440, 900)
        self.model_path = resolve_model_path()
        self.model = None
        self.last_result: dict[str, object] | None = None
        self.last_export_path: str | None = None
        self.batch_worker: BatchWorker | None = None
        self.video_worker: VideoWorker | None = None
        self._build_ui()
        self._load_model()

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        layout = QHBoxLayout(root)
        layout.setContentsMargins(14, 14, 14, 14)

        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setSpacing(12)
        splitter.addWidget(left)

        config_box = QGroupBox("系统配置")
        config_layout = QGridLayout(config_box)
        self.model_label = QLabel(self.model_path)
        self.model_label.setWordWrap(True)
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.0, 1.0)
        self.conf_spin.setSingleStep(0.01)
        self.conf_spin.setValue(0.12)
        self.iou_spin = QDoubleSpinBox()
        self.iou_spin.setRange(0.0, 1.0)
        self.iou_spin.setSingleStep(0.01)
        self.iou_spin.setValue(0.45)
        choose_model_btn = QPushButton("选择模型")
        choose_model_btn.clicked.connect(self.choose_model)
        reload_btn = QPushButton("重新加载模型")
        reload_btn.clicked.connect(self._load_model)
        config_layout.addWidget(QLabel("模型路径"), 0, 0)
        config_layout.addWidget(self.model_label, 0, 1, 1, 2)
        config_layout.addWidget(choose_model_btn, 1, 1)
        config_layout.addWidget(reload_btn, 1, 2)
        config_layout.addWidget(QLabel("置信度"), 2, 0)
        config_layout.addWidget(self.conf_spin, 2, 1)
        config_layout.addWidget(QLabel("IoU"), 2, 2)
        config_layout.addWidget(self.iou_spin, 2, 3)
        left_layout.addWidget(config_box)

        action_box = QGroupBox("识别操作")
        action_layout = QGridLayout(action_box)
        single_btn = QPushButton("单张图片识别")
        single_btn.clicked.connect(self.open_single_image)
        batch_btn = QPushButton("批量图片识别")
        batch_btn.clicked.connect(self.open_batch_images)
        video_btn = QPushButton("视频文件识别")
        video_btn.clicked.connect(self.open_video_file)
        camera_btn = QPushButton("摄像头拍照识别")
        camera_btn.clicked.connect(self.capture_camera_frame)
        export_image_btn = QPushButton("导出当前检测图")
        export_image_btn.clicked.connect(self.export_current_image)
        export_video_btn = QPushButton("导出当前视频结果")
        export_video_btn.clicked.connect(self.export_last_file)
        action_layout.addWidget(single_btn, 0, 0)
        action_layout.addWidget(batch_btn, 0, 1)
        action_layout.addWidget(video_btn, 1, 0)
        action_layout.addWidget(camera_btn, 1, 1)
        action_layout.addWidget(export_image_btn, 2, 0)
        action_layout.addWidget(export_video_btn, 2, 1)
        left_layout.addWidget(action_box)

        self.progress = QProgressBar()
        self.progress.setValue(0)
        self.status_label = QLabel("准备就绪")
        left_layout.addWidget(self.progress)
        left_layout.addWidget(self.status_label)

        self.summary_box = QPlainTextEdit()
        self.summary_box.setReadOnly(True)
        self.summary_box.setPlaceholderText("识别结果会显示在这里")
        left_layout.addWidget(self.summary_box, 1)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setSpacing(12)
        splitter.addWidget(right)

        image_row = QHBoxLayout()
        self.original_label = self._make_image_label("原始画面")
        self.annotated_label = self._make_image_label("标注结果")
        image_row.addWidget(self.original_label)
        image_row.addWidget(self.annotated_label)
        right_layout.addLayout(image_row, 1)

        self.detail_label = QLabel("等待识别")
        self.detail_label.setWordWrap(True)
        right_layout.addWidget(self.detail_label)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([420, 1000])

        self.setStyleSheet(
            """
            QWidget { background: #10161d; color: #eef6fb; font-size: 14px; }
            QGroupBox { border: 1px solid #2b3b49; border-radius: 12px; margin-top: 12px; padding-top: 12px; font-weight: 600; }
            QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 4px; }
            QPushButton { background: #1f7ae0; border: none; border-radius: 10px; padding: 10px 14px; color: white; }
            QPushButton:hover { background: #2b88f0; }
            QPushButton:disabled { background: #495766; }
            QPlainTextEdit, QLabel#imagePanel { background: #151d26; border: 1px solid #2b3b49; border-radius: 12px; }
            QDoubleSpinBox { background: #151d26; border: 1px solid #2b3b49; border-radius: 8px; padding: 4px 8px; }
            QProgressBar { border: 1px solid #2b3b49; border-radius: 8px; text-align: center; background: #151d26; }
            QProgressBar::chunk { background: #31c48d; border-radius: 8px; }
            """
        )

    def _make_image_label(self, title: str) -> QLabel:
        label = QLabel(title)
        label.setObjectName("imagePanel")
        label.setAlignment(Qt.AlignCenter)
        label.setMinimumSize(480, 360)
        label.setScaledContents(False)
        return label

    def _load_model(self) -> None:
        if not Path(self.model_path).exists():
            QMessageBox.critical(self, "模型不存在", f"未找到模型文件：\n{self.model_path}")
            return
        try:
            self.model = load_model(self.model_path)
            self.model_label.setText(self.model_path)
            names = ", ".join(self.model.names.values())
            self.status_label.setText("模型已加载")
            self.detail_label.setText(f"检测类别：{names}")
        except Exception as exc:
            QMessageBox.critical(self, "模型加载失败", str(exc))

    def choose_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择模型权重", self.model_path, "PyTorch Model (*.pt)")
        if path:
            self.model_path = path
            self._load_model()

    def current_thresholds(self) -> tuple[float, float]:
        return self.conf_spin.value(), self.iou_spin.value()

    def open_single_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Images (*.jpg *.jpeg *.png *.bmp *.webp)")
        if path:
            self.detect_single_path(path, mode="单张图片")

    def detect_single_path(self, path: str, mode: str) -> None:
        if self.model is None:
            return
        image = cv2.imread(path)
        if image is None:
            QMessageBox.warning(self, "读取失败", f"无法读取图片：\n{path}")
            return
        conf, iou = self.current_thresholds()
        try:
            payload = detect_image(self.model, image, conf, iou)
            self.last_export_path = None
            self.show_payload(payload, mode, source_name=Path(path).name)
        except Exception as exc:
            QMessageBox.critical(self, "识别失败", str(exc))

    def open_batch_images(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "选择多张图片", "", "Images (*.jpg *.jpeg *.png *.bmp *.webp)")
        if not paths or self.model is None:
            return
        conf, iou = self.current_thresholds()
        self.batch_worker = BatchWorker(self.model, paths, conf, iou)
        self.batch_worker.progress.connect(self.update_progress)
        self.batch_worker.finished_ok.connect(self.on_batch_finished)
        self.batch_worker.failed.connect(self.on_worker_failed)
        self.progress.setValue(0)
        self.status_label.setText("开始批量识别")
        self.batch_worker.start()

    def open_video_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择视频", "", "Videos (*.mp4 *.avi *.mov *.mkv)")
        if not path or self.model is None:
            return
        conf, iou = self.current_thresholds()
        self.video_worker = VideoWorker(self.model, path, conf, iou)
        self.video_worker.progress.connect(self.update_progress)
        self.video_worker.finished_ok.connect(self.on_video_finished)
        self.video_worker.failed.connect(self.on_worker_failed)
        self.progress.setValue(15)
        self.status_label.setText("开始视频识别")
        self.video_worker.start()

    def capture_camera_frame(self) -> None:
        if self.model is None:
            return
        capture = cv2.VideoCapture(0)
        if not capture.isOpened():
            QMessageBox.warning(self, "摄像头不可用", "无法打开默认摄像头")
            return
        ok, frame = capture.read()
        capture.release()
        if not ok or frame is None:
            QMessageBox.warning(self, "拍照失败", "未能从摄像头读取画面")
            return
        conf, iou = self.current_thresholds()
        try:
            payload = detect_image(self.model, frame, conf, iou)
            self.last_export_path = None
            self.show_payload(payload, "摄像头拍照", source_name="camera")
        except Exception as exc:
            QMessageBox.critical(self, "识别失败", str(exc))

    def show_payload(self, payload: dict[str, object], mode: str, source_name: str = "") -> None:
        self.last_result = payload
        self.progress.setValue(100)
        reading = payload["reading"] or "未检测到数字"
        self.status_label.setText(f"{mode}完成：{reading}")
        title = f"来源：{mode}"
        if source_name:
            title += f" / {source_name}"
        self.detail_label.setText(
            f"{title}\n识别串：{reading}\n数字个数：{payload['count']}\n平均置信度：{float(payload['avg_conf']):.2%}"
        )
        self.summary_box.setPlainText(str(payload["summary_text"]))
        self.set_image(self.original_label, payload["original"])
        self.set_image(self.annotated_label, payload["annotated"])

    def set_image(self, label: QLabel, image: object) -> None:
        arr = image if isinstance(image, np.ndarray) else None
        if arr is None:
            label.setText("无图像")
            return
        rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape
        qimage = QImage(rgb.data, width, height, channels * width, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)
        scaled = pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(scaled)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self.last_result:
            self.set_image(self.original_label, self.last_result["original"])
            self.set_image(self.annotated_label, self.last_result["annotated"])

    def update_progress(self, value: int, text: str) -> None:
        self.progress.setValue(value)
        self.status_label.setText(text)

    def on_batch_finished(self, rows: list[dict[str, object]], zip_path: str) -> None:
        self.last_export_path = zip_path
        self.progress.setValue(100)
        if not rows:
            self.status_label.setText("批量识别完成，但没有有效图片")
            return
        last = rows[-1]
        self.show_payload(last["payload"], "批量图片", source_name=Path(str(last["path"])).name)
        self.status_label.setText(f"批量识别完成，共 {len(rows)} 张")
        self.summary_box.appendPlainText(f"\n批量结果压缩包：{zip_path}")

    def on_video_finished(self, payload: dict[str, object], out_video: str) -> None:
        self.last_export_path = out_video
        self.show_payload(payload, "视频文件", source_name=Path(out_video).name)
        self.summary_box.appendPlainText(f"\n视频结果文件：{out_video}")

    def on_worker_failed(self, message: str) -> None:
        self.progress.setValue(0)
        self.status_label.setText("执行失败")
        QMessageBox.critical(self, "处理失败", message)

    def export_current_image(self) -> None:
        if not self.last_result:
            QMessageBox.information(self, "无结果", "请先完成一次识别")
            return
        path, _ = QFileDialog.getSaveFileName(self, "保存检测图", "digit_detect_result.jpg", "JPEG Image (*.jpg)")
        if not path:
            return
        ok, buf = cv2.imencode(".jpg", self.last_result["annotated"])
        if not ok:
            QMessageBox.warning(self, "保存失败", "无法编码当前检测图")
            return
        buf.tofile(path)
        self.status_label.setText(f"已保存图片：{path}")

    def export_last_file(self) -> None:
        if not self.last_export_path or not Path(self.last_export_path).exists():
            QMessageBox.information(self, "无可导出文件", "当前没有视频或批量识别导出文件")
            return
        suffix = Path(self.last_export_path).suffix
        filter_text = "ZIP 文件 (*.zip)" if suffix.lower() == ".zip" else "视频文件 (*.mp4 *.avi)"
        path, _ = QFileDialog.getSaveFileName(self, "导出结果文件", Path(self.last_export_path).name, filter_text)
        if not path:
            return
        import shutil
        shutil.copy2(self.last_export_path, path)
        self.status_label.setText(f"已导出文件：{path}")


def main() -> None:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
