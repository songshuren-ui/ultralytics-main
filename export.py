from ultralytics import YOLO

model = YOLO("dist/best.pt")
model.export(format="onnx", imgsz=640, nms=True, simplify=True)
