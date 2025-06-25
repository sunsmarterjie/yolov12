from ultralytics import YOLO

model = YOLO('yolov12n.pt')
model.predict('bus.jpg')