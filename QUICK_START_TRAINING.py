#!/usr/bin/env python3
"""
Quick reference: How to train YOLOv12 with your own dataset
"""

from ultralytics import YOLO

# Basic training example
model = YOLO('yolov12n.pt')  # or yolov12s, yolov12m, yolov12l, yolov12x

results = model.train(
    data='path/to/your/data.yaml',  # Path to dataset YAML
    epochs=100,                      # Number of training epochs
    imgsz=640,                       # Image size
    batch=16,                        # Batch size
    device=0,                        # GPU device ID (or use device='' for auto)
    patience=20,                     # Early stopping patience
    save=True,                       # Save checkpoints
    project='runs/detect',           # Output directory
    name='my_experiment',            # Experiment name
)

# Validation
metrics = model.val()

# Prediction
results = model.predict(source='path/to/image.jpg', conf=0.5)

# Export model
model.export(format='onnx')

print("✅ Training complete!")
