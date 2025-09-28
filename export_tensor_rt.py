from ultralytics import YOLO

from ultralytics import YOLO

# 加载训练好的权重
model = YOLO("runs/detect/train/weights/best.pt")

# 导出为不同格式
#model.export(format="onnx")       # 导出 ONNX
#model.export(format="openvino")   # 导出 OpenVINO IR
model.export(format="engine")     # 导出 TensorRT

# 加载 TensorRT 引擎文件
#model = YOLO("best.engine")

# 推理
#results = model("test.jpg", conf=0.5)
#results[0].show()
