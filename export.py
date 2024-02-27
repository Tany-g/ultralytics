from ultralytics import YOLO

# Load a model
# model = YOLO('yolov8n.pt')  # load an official model
model = YOLO('/home/ubuntu/GITHUG/ultralytics/runs/detect/train9/weights/best.pt')  # load a custom trained model

# Export the model
model.export(format='engine', simplify=True, workspace=16)
