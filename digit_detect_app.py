"""YOLOv8 数字识别检测应用 — 桌面软件风格工作台，支持拍照、图片、批量图片与视频检测。."""

from __future__ import annotations

import base64
import io
import os
import sys
import tempfile
import zipfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from ultralytics import YOLO
from ultralytics.data.utils import IMG_FORMATS, VID_FORMATS

# 默认模型搜索路径（按优先级）
MODEL_CANDIDATES = [
    "dist/best.pt",
    "runs/detect/digit_finetune_full/weights/best.pt",
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


@st.cache_resource(show_spinner="正在加载 YOLO 模型…")
def load_model(model_path: str) -> YOLO:
    return YOLO(model_path)


def render_detection_summary(result) -> tuple[str, str, int, float, list[dict[str, float | str]]]:
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
        lines.append(f"数字 **{label}** — 置信度 {conf:.2%}")

    reading = "".join(digits)
    avg_conf = sum(confs) / len(confs)
    text = f"识别结果：`{reading}`\n\n" + "\n\n".join(lines)
    return text, reading, len(digits), avg_conf, items


def save_uploaded_file(uploaded, suffix: str) -> str:
    """将 Streamlit 上传文件保存到临时路径并返回路径。."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.getbuffer())
        return tmp.name


def image_to_data_url(image: np.ndarray) -> str:
    """将 OpenCV 图像转为 data URL 供 HTML 预览。."""
    ok, buf = cv2.imencode(".jpg", image)
    if not ok:
        return ""
    encoded = base64.b64encode(buf.tobytes()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def detect_result_to_payload(result, original: np.ndarray) -> dict[str, object]:
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
    result = results[0]
    return detect_result_to_payload(result, image)


def init_state() -> None:
    """初始化页面状态。."""
    defaults = {
        "last_result": None,
        "last_mode": "拍照识别",
        "camera_last_frame": None,
        "camera_capture_index": 0,
        "camera_saved_photo": None,
        "camera_saved_path": None,
        "camera_saved_name": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def save_camera_snapshot(image: np.ndarray) -> tuple[str, str]:
    """保存摄像头抓拍原图并返回路径与文件名。."""
    save_dir = Path.cwd() / "captures"
    save_dir.mkdir(parents=True, exist_ok=True)
    timestamp = __import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"camera_capture_{timestamp}.jpg"
    save_path = save_dir / filename
    ok = cv2.imwrite(str(save_path), image)
    if not ok:
        raise RuntimeError("拍照保存失败，请检查磁盘写入权限。")
    return str(save_path), filename


def grab_camera_frame(camera_index: int = 0) -> np.ndarray | None:
    """从本地摄像头读取一帧。."""
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        return None

    try:
        for _ in range(2):
            cap.read()
        ok, frame = cap.read()
        if not ok or frame is None:
            return None
        return frame
    finally:
        cap.release()


def inject_theme() -> None:
    """注入桌面软件风格主题。."""
    st.markdown(
        """
        <style>
        :root {
          --bg: #06131a;
          --bg-soft: rgba(11, 31, 42, 0.70);
          --panel: rgba(16, 39, 52, 0.58);
          --panel-strong: rgba(23, 51, 67, 0.78);
          --line: rgba(169, 241, 227, 0.14);
          --text: #ecfffb;
          --muted: #9ec8c1;
          --accent: #6ff7cb;
          --accent-2: #74b6ff;
          --warn: #ffd36f;
          --shadow: 0 24px 80px rgba(0, 0, 0, 0.28);
          --blob-a: radial-gradient(circle at 20% 20%, rgba(111, 247, 203, 0.22), transparent 55%);
          --blob-b: radial-gradient(circle at 80% 30%, rgba(116, 182, 255, 0.18), transparent 42%);
          --blob-c: radial-gradient(circle at 45% 80%, rgba(105, 241, 231, 0.12), transparent 40%);
        }

        .stApp {
          background:
            var(--blob-a),
            var(--blob-b),
            var(--blob-c),
            linear-gradient(145deg, #041018 0%, #081d25 42%, #0a1a1e 100%);
          color: var(--text);
        }

        .stApp::before,
        .stApp::after {
          content: "";
          position: fixed;
          inset: auto;
          width: 380px;
          height: 380px;
          pointer-events: none;
          filter: blur(28px);
          z-index: 0;
          opacity: 0.68;
          animation: floatBlob 14s ease-in-out infinite;
        }

        .stApp::before {
          top: 4%;
          right: 4%;
          background: radial-gradient(circle, rgba(111, 247, 203, 0.18), transparent 62%);
          border-radius: 58% 42% 61% 39% / 46% 60% 40% 54%;
        }

        .stApp::after {
          bottom: 2%;
          left: 2%;
          background: radial-gradient(circle, rgba(116, 182, 255, 0.14), transparent 58%);
          border-radius: 39% 61% 44% 56% / 58% 39% 61% 42%;
          animation-delay: -7s;
        }

        @keyframes floatBlob {
          0%, 100% { transform: translate3d(0, 0, 0) scale(1); }
          50% { transform: translate3d(18px, -20px, 0) scale(1.08); }
        }

        [data-testid="stHeader"],
        [data-testid="stToolbar"] {
          background: transparent !important;
        }

        [data-testid="stSidebar"] {
          background: linear-gradient(180deg, rgba(8, 22, 30, 0.92), rgba(7, 18, 25, 0.88));
          border-right: 1px solid var(--line);
          backdrop-filter: blur(22px);
        }

        [data-testid="stSidebar"] > div {
          padding-top: 1.1rem;
        }

        .block-container {
          padding-top: 1.4rem;
          padding-bottom: 2rem;
          max-width: 1400px;
          position: relative;
          z-index: 1;
        }

        .hero-shell,
        .soft-panel,
        .metric-card,
        .camera-tip,
        .result-shell {
          background: linear-gradient(180deg, rgba(20, 45, 58, 0.56), rgba(11, 28, 38, 0.74));
          border: 1px solid var(--line);
          backdrop-filter: blur(24px);
          box-shadow: var(--shadow);
        }

        .hero-shell {
          position: relative;
          overflow: hidden;
          padding: 1.6rem 1.7rem;
          border-radius: 32px 46px 34px 52px / 36px 30px 54px 34px;
          margin-bottom: 1.05rem;
        }

        .hero-shell::before {
          content: "";
          position: absolute;
          inset: -20% auto auto -8%;
          width: 280px;
          height: 280px;
          background: radial-gradient(circle, rgba(111, 247, 203, 0.20), transparent 65%);
          filter: blur(12px);
          border-radius: 43% 57% 62% 38% / 41% 37% 63% 59%;
          animation: floatBlob 12s ease-in-out infinite;
        }

        .hero-label {
          display: inline-flex;
          align-items: center;
          gap: 8px;
          padding: 0.36rem 0.9rem;
          border-radius: 999px;
          background: rgba(111, 247, 203, 0.10);
          border: 1px solid rgba(111, 247, 203, 0.18);
          color: var(--accent);
          font-size: 0.86rem;
          letter-spacing: 0.04em;
        }

        .hero-title {
          margin: 0.9rem 0 0.5rem;
          font-size: clamp(2rem, 3.4vw, 3.35rem);
          font-weight: 700;
          line-height: 1.08;
          letter-spacing: -0.03em;
          color: var(--text);
        }

        .hero-desc {
          margin: 0;
          max-width: 780px;
          color: var(--muted);
          font-size: 1rem;
          line-height: 1.8;
        }

        .metric-card {
          min-height: 118px;
          border-radius: 34px 26px 38px 24px / 26px 36px 28px 40px;
          padding: 1rem 1.15rem;
        }

        .metric-label {
          color: var(--muted);
          font-size: 0.86rem;
          margin-bottom: 0.35rem;
        }

        .metric-value {
          color: var(--text);
          font-size: 1.7rem;
          font-weight: 700;
          line-height: 1.2;
        }

        .metric-sub {
          margin-top: 0.35rem;
          color: #bde7df;
          font-size: 0.84rem;
        }

        .soft-panel {
          border-radius: 30px 42px 28px 44px / 34px 26px 40px 30px;
          padding: 1rem 1.15rem;
          margin-bottom: 1rem;
        }

        .panel-title {
          font-size: 1.02rem;
          font-weight: 650;
          margin-bottom: 0.3rem;
          color: var(--text);
        }

        .panel-subtitle {
          color: var(--muted);
          margin: 0;
          line-height: 1.7;
        }

        .result-shell {
          border-radius: 36px 28px 42px 30px / 32px 40px 28px 44px;
          padding: 1rem;
        }

        .preview-frame {
          background: rgba(4, 14, 18, 0.58);
          border-radius: 26px;
          padding: 0.75rem;
          border: 1px solid rgba(255, 255, 255, 0.06);
          height: 100%;
        }

        .preview-label {
          color: var(--muted);
          font-size: 0.84rem;
          margin-bottom: 0.55rem;
        }

        .reading-chip {
          display: inline-flex;
          align-items: center;
          gap: 10px;
          margin: 0.4rem 0 0.9rem;
          padding: 0.52rem 1rem;
          border-radius: 999px;
          background: rgba(111, 247, 203, 0.11);
          border: 1px solid rgba(111, 247, 203, 0.22);
          color: #dffef5;
          font-size: 0.95rem;
        }

        .detail-item {
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 12px;
          padding: 0.75rem 0.9rem;
          margin-bottom: 0.6rem;
          background: rgba(255, 255, 255, 0.03);
          border: 1px solid rgba(255, 255, 255, 0.05);
          border-radius: 20px 26px 18px 24px / 18px 22px 24px 20px;
        }

        .digit-pill {
          min-width: 42px;
          height: 42px;
          display: inline-flex;
          align-items: center;
          justify-content: center;
          border-radius: 50% 42% 58% 46% / 44% 58% 42% 56%;
          background: linear-gradient(145deg, rgba(111, 247, 203, 0.26), rgba(116, 182, 255, 0.18));
          color: var(--text);
          font-weight: 700;
        }

        .camera-tip {
          border-radius: 24px 34px 22px 30px / 28px 20px 36px 24px;
          padding: 0.95rem 1rem;
          margin-top: 0.75rem;
        }

        .camera-tip strong {
          color: #edfff7;
        }

        .stTabs [data-baseweb="tab-list"] {
          gap: 12px;
          background: rgba(6, 20, 27, 0.42);
          border: 1px solid var(--line);
          border-radius: 999px;
          padding: 0.35rem;
          width: fit-content;
        }

        .stTabs [data-baseweb="tab"] {
          border-radius: 999px;
          color: var(--muted);
          padding: 0.55rem 1rem;
          height: auto;
          background: transparent;
        }

        .stTabs [aria-selected="true"] {
          background: linear-gradient(145deg, rgba(111, 247, 203, 0.18), rgba(116, 182, 255, 0.18)) !important;
          color: var(--text) !important;
        }

        .stButton > button,
        .stDownloadButton > button {
          border: 1px solid rgba(111, 247, 203, 0.24);
          background: linear-gradient(145deg, rgba(111, 247, 203, 0.15), rgba(116, 182, 255, 0.12));
          color: var(--text);
          border-radius: 999px;
          min-height: 46px;
          font-weight: 650;
          transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
          box-shadow: 0 12px 30px rgba(0, 0, 0, 0.16);
        }

        .stButton > button:hover,
        .stDownloadButton > button:hover {
          transform: translateY(-1px) scale(1.01);
          border-color: rgba(111, 247, 203, 0.42);
          box-shadow: 0 16px 34px rgba(72, 194, 170, 0.14);
        }

        .stTextInput input,
        .stNumberInput input,
        .stSelectbox div[data-baseweb="select"] > div,
        .stSlider,
        .stFileUploader,
        [data-testid="stCameraInput"] {
          border-radius: 22px !important;
        }

        [data-testid="stFileUploader"],
        [data-testid="stCameraInput"] {
          background: rgba(11, 30, 39, 0.56);
          border: 1px dashed rgba(111, 247, 203, 0.22);
          padding: 0.4rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header(model_path: str) -> None:
    """渲染顶部品牌区。."""
    st.markdown(
        f"""
        <section class="hero-shell">
          <div class="hero-label">YOLOv8 · 本地识别工作台 · Computer Vision Desktop UI</div>
          <h1 class="hero-title">数字识别系统 · 拍照即识别</h1>
          <p class="hero-desc">
            这是围绕你的 <strong>基于 YOLOv8 的数字识别系统</strong> 打造的电脑软件风格界面，不走网页营销风。
            核心入口优先服务拍照、图片导入、批量识别和视频检测，让模型结果像本地工作台一样清晰可控。
          </p>
          <p class="hero-desc" style="margin-top:0.85rem; color:#d7fff7;">
            当前模型：<code>{model_path}</code>
          </p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_metrics(model_path: str, class_names: str) -> None:
    """渲染概览指标。."""
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            """
            <div class="metric-card">
              <div class="metric-label">系统</div>
              <div class="metric-value">数字识别</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
              <div class="metric-label">当前权重</div>
              <div class="metric-value">{Path(model_path).name}</div>
              <div class="metric-sub">路径已自动解析，可在左侧配置中切换</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"""
            <div class="metric-card">
              <div class="metric-label">检测类别</div>
              <div class="metric-value">{len(class_names.split(", "))} 类</div>
              <div class="metric-sub">{class_names}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_workspace_intro() -> None:
    """入口区已精简，不再渲染额外卡片。."""
    return


def render_result_dashboard() -> None:
    """渲染最近一次识别结果。."""
    result = st.session_state.get("last_result")
    if not result:
        st.markdown(
            """
            <div class="soft-panel">
              <div class="panel-title">识别结果</div>
              <p class="panel-subtitle">先运行一次识别。</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    left, right = st.columns([1.25, 0.95])
    with left:
        st.markdown('<div class="result-shell">', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="preview-frame"><div class="preview-label">原始画面</div>', unsafe_allow_html=True)
            st.image(result["original"], channels="BGR", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with col2:
            st.markdown(
                '<div class="preview-frame"><div class="preview-label">YOLOv8 标注结果</div>', unsafe_allow_html=True
            )
            st.image(result["annotated"], channels="BGR", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        reading = result["reading"] or "--"
        count = result["count"]
        avg_conf = float(result["avg_conf"])
        mode = st.session_state.get("last_mode", "拍照识别")
        st.markdown(
            f"""
            <div class="soft-panel">
              <div class="panel-title">最近一次识别</div>
              <p class="panel-subtitle">来源模式：{mode}</p>
              <div class="reading-chip">识别数字串 <strong>{reading}</strong></div>
              <div class="metric-label">数字个数</div>
              <div class="metric-value">{count}</div>
              <div class="metric-sub">平均置信度 {avg_conf:.2%}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="soft-panel">
              <div class="panel-title">识别明细</div>
            """,
            unsafe_allow_html=True,
        )
        if result["items"]:
            for item in result["items"]:
                st.markdown(
                    f"""
                    <div class="detail-item">
                      <div style="display:flex; align-items:center; gap:12px;">
                        <span class="digit-pill">{item["label"]}</span>
                        <div>
                          <div style="color:#ecfffb; font-weight:600;">数字 {item["label"]}</div>
                          <div style="color:#9ec8c1; font-size:0.84rem;">来自左到右排序后的检测框</div>
                        </div>
                      </div>
                      <div style="color:#dffef5; font-weight:600;">{float(item["confidence"]):.2%}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.info("当前结果未识别到数字。")
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown(result["summary_text"])

        ok, buf = cv2.imencode(".jpg", result["annotated"])
        if ok:
            st.download_button(
                "下载当前检测图",
                buf.tobytes(),
                file_name="digit_detect_result.jpg",
                mime="image/jpeg",
                use_container_width=True,
            )


def update_last_result(mode: str, result: dict[str, object]) -> None:
    """写入最近一次识别结果。."""
    st.session_state["last_mode"] = mode
    st.session_state["last_result"] = result


def run_single_image(model: YOLO, conf: float, iou: float) -> None:
    uploaded = st.file_uploader("导入一张数字图片", type=IMG_TYPES, key="single_img")
    if not uploaded:
        st.info("请选择一张图片开始检测。")
        return

    bytes_data = uploaded.getvalue()
    image = cv2.imdecode(np.frombuffer(bytes_data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        st.error("无法读取图片，请检查文件格式。")
        return

    st.image(image, channels="BGR", caption="待检测图片", use_container_width=True)

    if st.button("开始单图识别", key="single_run", type="primary", use_container_width=True):
        result = detect_image(model, image, conf, iou)
        update_last_result("单图检测", result)
        st.success(f"识别完成：{result['reading'] or '未检测到数字'}")
        st.rerun()


def run_batch_images(model: YOLO, conf: float, iou: float) -> None:
    uploaded_list = st.file_uploader(
        "导入多张图片",
        type=IMG_TYPES,
        accept_multiple_files=True,
        key="batch_imgs",
    )
    if not uploaded_list:
        st.info("请选择一张或多张图片进行批量检测。")
        return

    st.caption(f"已选择 {len(uploaded_list)} 张图片")

    if st.button("开始批量识别", key="batch_run", type="primary", use_container_width=True):
        tmp_paths: list[str] = []
        try:
            for file in uploaded_list:
                tmp_paths.append(save_uploaded_file(file, Path(file.name).suffix))

            progress = st.progress(0, text="正在批量检测…")
            results = model.predict(source=tmp_paths, conf=conf, iou=iou, save=False, verbose=False)
            progress.progress(100, text="批量检测完成")

            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                for idx, (uploaded, result) in enumerate(zip(uploaded_list, results)):
                    annotated = result.plot()
                    ok, buf = cv2.imencode(".jpg", annotated)
                    if ok:
                        arcname = f"detect_{Path(uploaded.name).stem}.jpg"
                        zf.writestr(arcname, buf.tobytes())

                    summary_text, reading, count, avg_conf, _ = render_detection_summary(result)
                    st.markdown(f"### {uploaded.name}")
                    col1, col2, col3 = st.columns([1.2, 1.2, 0.9])
                    original = cv2.imread(tmp_paths[idx])
                    with col1:
                        st.image(original, channels="BGR", caption="原图", use_container_width=True)
                    with col2:
                        st.image(annotated, channels="BGR", caption="检测结果", use_container_width=True)
                    with col3:
                        st.metric("数字串", reading or "--")
                        st.metric("检测数量", count)
                        st.metric("平均置信度", f"{avg_conf:.2%}")
                    st.markdown(summary_text)

                if results:
                    last_original = cv2.imread(tmp_paths[-1])
                    if last_original is not None:
                        update_last_result("批量处理", detect_result_to_payload(results[-1], last_original))

            zip_buffer.seek(0)
            st.download_button(
                "下载全部结果（ZIP）",
                zip_buffer,
                file_name="batch_detect_results.zip",
                mime="application/zip",
                use_container_width=True,
            )
        finally:
            for path in tmp_paths:
                if os.path.exists(path):
                    os.unlink(path)


def run_video_detection(model: YOLO, conf: float, iou: float) -> None:
    st.markdown(
        """
        <div class="soft-panel">
          <div class="panel-title">视频 / 摄像头</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    source_mode = st.radio(
        "选择输入来源",
        options=["上传视频文件", "实时摄像头识别"],
        horizontal=True,
        key="video_source_mode",
    )

    if source_mode == "实时摄像头识别":
        st.markdown(
            """
            <div class="camera-tip">
              <strong>使用建议：</strong> 先点“刷新实时画面”观察取景，再点“拍照并识别”；系统会在识别同时保留一张原始照片。
            </div>
            """,
            unsafe_allow_html=True,
        )

        control_col1, control_col2 = st.columns(2)
        with control_col1:
            refresh_clicked = st.button("刷新实时画面", key="camera_refresh", use_container_width=True)
        with control_col2:
            detect_clicked = st.button("拍照并识别", key="camera_detect", type="primary", use_container_width=True)

        should_grab_frame = refresh_clicked or detect_clicked or st.session_state["camera_last_frame"] is None
        if should_grab_frame:
            frame = grab_camera_frame(st.session_state["camera_capture_index"])
            if frame is None:
                st.error("无法打开摄像头，请检查设备是否被占用或权限是否允许。")
                return
            st.session_state["camera_last_frame"] = frame

        current_frame = st.session_state.get("camera_last_frame")
        if current_frame is None:
            st.info("点击“刷新实时画面”后可查看当前摄像头取景。")
            return

        st.image(current_frame, channels="BGR", caption="当前实时画面", use_container_width=True)

        if detect_clicked:
            try:
                save_path, file_name = save_camera_snapshot(current_frame)
            except Exception as exc:
                st.error(str(exc))
                return

            result = detect_image(model, current_frame, conf, iou)
            update_last_result("实时摄像头", result)
            st.session_state["camera_saved_photo"] = current_frame.copy()
            st.session_state["camera_saved_path"] = save_path
            st.session_state["camera_saved_name"] = file_name
            st.success(f"拍照识别完成：{result['reading'] or '未检测到数字'}")
            st.rerun()

        saved_photo = st.session_state.get("camera_saved_photo")
        saved_path = st.session_state.get("camera_saved_path")
        saved_name = st.session_state.get("camera_saved_name")
        if saved_photo is not None and saved_path and saved_name:
            st.markdown("### 最近一次保留照片")
            st.image(saved_photo, channels="BGR", caption=saved_name, use_container_width=True)
            ok, buf = cv2.imencode(".jpg", saved_photo)
            if ok:
                st.download_button(
                    "下载最近拍照原图",
                    buf.tobytes(),
                    file_name=saved_name,
                    mime="image/jpeg",
                    use_container_width=True,
                )
            st.caption(f"已保存到：{saved_path}")
        return

    uploaded = st.file_uploader("导入视频文件", type=VID_TYPES, key="video_file")
    show_preview = st.checkbox("显示关键帧预览", value=True, key="video_preview")

    if not uploaded:
        st.info("请上传 mp4 / avi / mov 等格式的视频文件。")
        return

    if st.button("开始视频识别", key="video_run", type="primary", use_container_width=True):
        in_path = save_uploaded_file(uploaded, Path(uploaded.name).suffix)
        out_dir = tempfile.mkdtemp()

        try:
            with st.spinner("YOLO 正在逐帧分析视频，请稍候…"):
                results = model.predict(
                    source=in_path,
                    conf=conf,
                    iou=iou,
                    save=True,
                    project=out_dir,
                    name="output",
                    exist_ok=True,
                    verbose=False,
                )

            saved_dir = Path(out_dir) / "output"
            saved_videos = list(saved_dir.glob("*.*")) if saved_dir.exists() else []
            if not saved_videos:
                st.error("视频处理失败，未生成输出文件。")
                return

            out_video = saved_videos[0]
            st.success(f"处理完成，共分析 {len(results)} 帧")

            if show_preview and results:
                mid = len(results) // 2
                preview_frame = results[mid].plot()
                summary_text, reading, count, avg_conf, items = render_detection_summary(results[mid])
                update_last_result(
                    "视频检测",
                    {
                        "original": cv2.imread(in_path) if Path(in_path).suffix.lower() in IMG_TYPES else preview_frame,
                        "annotated": preview_frame,
                        "summary_text": summary_text,
                        "reading": reading,
                        "count": count,
                        "avg_conf": avg_conf,
                        "items": items,
                    },
                )
                st.image(
                    preview_frame, channels="BGR", caption=f"中间帧预览（第 {mid + 1} 帧）", use_container_width=True
                )

            with open(out_video, "rb") as file:
                st.download_button(
                    "下载带检测框的视频",
                    file,
                    file_name=f"detect_{Path(uploaded.name).stem}{out_video.suffix}",
                    mime="video/mp4",
                    use_container_width=True,
                )
        finally:
            os.unlink(in_path)


def main() -> None:
    st.set_page_config(page_title="YOLOv8 数字识别系统", page_icon="🔢", layout="wide")
    init_state()
    inject_theme()

    with st.sidebar:
        st.markdown("### 系统配置")
        st.caption("本地电脑软件风格工作台 · YOLOv8 Digit Recognition")
        default_model = resolve_model_path()
        model_input = st.text_input("模型路径", value=default_model)
        model_path = resolve_model_path(model_input.strip() or None)

        if not Path(model_path).exists():
            st.error(f"模型文件不存在: {model_path}")
            st.stop()

        conf = st.slider("置信度阈值", 0.0, 1.0, 0.12, 0.01)
        iou = st.slider("IoU 阈值", 0.0, 1.0, 0.45, 0.01)
        st.caption("数字漏检时可优先尝试 0.08 ~ 0.18 的置信度阈值")

        try:
            model = load_model(model_path)
            class_names = ", ".join(model.names.values())
            st.success("模型已加载")
            st.code(model_path)
            st.info(f"检测类别: {class_names}")
        except Exception as exc:
            st.error(f"模型加载失败: {exc}")
            st.stop()

    render_header(model_path)
    render_metrics(model_path, class_names)
    render_workspace_intro()

    work_col, result_col = st.columns([1.25, 1])
    with work_col:
        tabs = st.tabs(["单张图片", "批量图片", "视频 / 摄像头"])
        with tabs[0]:
            run_single_image(model, conf, iou)
        with tabs[1]:
            run_batch_images(model, conf, iou)
        with tabs[2]:
            run_video_detection(model, conf, iou)

    with result_col:
        render_result_dashboard()


if __name__ == "__main__":
    main()
