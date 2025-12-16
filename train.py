from ultralytics import YOLO
import torch

# Sanity check
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")

# Load model
model = YOLO(
    '/content/yolov12/ultralytics/cfg/models/v12/yolov12.yaml',
    scale='n'
)

# Dataset config
data_yaml_path = "DUO dataset.v2i.yolov12 - Copy/data.yaml"

# Train
results = model.train(
    data=data_yaml_path,
    epochs=30,
    imgsz=840,
    batch=16,
    device=0,
    patience=5,
    save=True,
    project="runs/detect",
    name="yolov12m_underwater_training",
    weight_decay=0.0005
)

print("✅ Training completed!")
