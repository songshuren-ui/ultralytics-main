import sys
from ultralytics import YOLO
import os

# 打包动态路径
def get_model_path():
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, "best.pt")
    return "best.pt"

if __name__ == '__main__':
    model = YOLO(get_model_path())
    # 测试digit数据集图片
    res = model.predict(source="digit_dataset/images", save=True, conf=0.03)