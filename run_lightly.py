from ultralytics import settings
import lightly_train

lightly_train.train(
    out="out/my_experiment",  # Output directory
    data="datasets/beans_split/train",    # Your dataset folder with images
    model="ultralytics/yolov12n.yaml",  # YOLOv12 model config
    epochs=5,                 # Number of epochs
    batch_size=32,            # Batch size
    overwrite=True,

)


#pip install lightly lightly_train