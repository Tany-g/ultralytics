from ultralytics import YOLO

# Load a model
# model = YOLO('yolov8x.yaml')  # build a new model from YAML
# model = YOLO('yolov8x.pt')  # load a pretrained model (recommended for training)
model = YOLO("/home/ubuntu/GITHUG/ultralytics/runs/detect/train9/weights/best.pt")  # build from YAML and transfer weights
model.to(device="cpu")
# Train the model
model.predict(source='/home/ubuntu/Downloads/5.测试图片/1月17日测试集/测试集/middle_usual_png',save=True)

# 2560 172ms
# 1280 47ms