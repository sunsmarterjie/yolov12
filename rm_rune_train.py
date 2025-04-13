from ultralytics import YOLO

model = YOLO('ultralytics/cfg/models/v12/yolov12-rune-pose.yaml')

# Train the model
results = model.train(
  data='rm_rune_pose.yaml',
  epochs=600, 
  batch=32, 
  imgsz=640,
  pretrained=False,
  scale=0.5,  # S:0.9; M:0.9; L:0.9; X:0.9
  mosaic=0.2, # 数据集样本太少，开一点mosaic增强
  close_mosaic=20, 
  mixup=0.0,  # S:0.05; M:0.15; L:0.15; X:0.2
  fliplr=0.0,  # 关闭水平翻转（默认0.5）
  flipud=0.0,  # 关闭垂直翻转（默认0.0）
  degrees=0.0, # 关闭旋转（避免隐含翻转）
  perspective=0.0,  # 关闭透视变换（可能含翻转）
  copy_paste=0,  # S:0.15; M:0.4; L:0.5; X:0.6(默认0.1翻转)
  box= 8.0,
  cls= 5.0, # 增强分类损失的权重（ 
  dfl= 1.5,
  pose= 32.0, # 增强关键点损失的权重（适当增强有利于解决关键点漂移
  kobj= 1.0,
  nbs= 64,
  hsv_h= 0.015,
  hsv_s= 0.7,
  translate=0.1,
)

# Evaluate model performance on the validation set
metrics = model.val()

# Perform object detection on an image
results = model("data/images/test/fe818847-2694.jpg")
results[0].show()