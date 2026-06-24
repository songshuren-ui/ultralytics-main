import os

# 程序内部强制离线，不用CMD临时变量
os.environ["ULTRALYTICS_OFFLINE"] = "1"
os.environ["NO_PROXY"] = "*"

from ultralytics import YOLO

if __name__ == "__main__":
    # 同目录best.pt
    model = YOLO("./best.pt")
    # 导出onnx
    out = model.export(format="onnx", opset=12, simplify=True, imgsz=640)
    print("导出成功，onnx路径：", out)
