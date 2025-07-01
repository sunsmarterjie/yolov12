

<div align="center">
<h1>YOLOv12</h1>
<h3>YOLOv12: Attention-Centric Real-Time Object Detectors</h3>

[Yunjie Tian](https://sunsmarterjie.github.io/)<sup>1</sup>, [Qixiang Ye](https://people.ucas.ac.cn/~qxye?language=en)<sup>2</sup>, [David Doermann](https://cse.buffalo.edu/~doermann/)<sup>1</sup>

<sup>1</sup>  University at Buffalo, SUNY, <sup>2</sup> University of Chinese Academy of Sciences.

</div>

## Main Results (ImageNet-1K)

[**Classification**](https://github.com/sunsmarterjie/yolov12/tree/Cls):
| Model (cls)                                                                              | size<br><sup>(pixels) | Acc.<br><sup>top-1<br> | Acc.<br><sup>top-5<br> | Speed  (ms) <br><sup>T4 TensorRT10<br> | params<br><sup>(M) | FLOPs<br><sup>(B) |
| :----------------------------------------------------------------------------------------| :-------------------: | :------------: | :------------: | :-------------------------------------:| :----------------: | :---------------: |
| [YOLOv12n-cls](https://github.com/sunsmarterjie/yolov12/releases/download/cls/yolov12n-cls.pt) | 224             | 71.7           | 90.5           | 1.27                                   | 2.9                | 0.5               |
| [YOLOv12s-cls](https://github.com/sunsmarterjie/yolov12/releases/download/cls/yolov12s-cls.pt) | 224             | 76.4           | 93.3           | 1.52                                   | 7.2                | 1.5               |
| [YOLOv12m-cls](https://github.com/sunsmarterjie/yolov12/releases/download/cls/yolov12m-cls.pt) | 224             | 78.8           | 94.4           | 2.03                                   | 12.7               | 4.5               |
| [YOLOv12l-cls](https://github.com/sunsmarterjie/yolov12/releases/download/cls/yolov12l-cls.pt) | 224             | 79.5           | 94.5           | 2.73                                   | 16.8               | 6.2               |
| [YOLOv12x-cls](https://github.com/sunsmarterjie/yolov12/releases/download/cls/yolov12x-cls.pt) | 224             | 80.1           | 95.3           | 3.64                                   | 35.5               | 13.7              |


## Installation
```
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu11torch2.2cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
conda create -n yolov12 python=3.11
conda activate yolov12
pip install -r requirements.txt
pip install -e .
```

## Validation
[`yolov12n-cls`](https://github.com/sunsmarterjie/yolov12/releases/download/cls/yolov12n-cls.pt)
[`yolov12s-cls`](https://github.com/sunsmarterjie/yolov12/releases/download/cls/yolov12s-cls.pt)
[`yolov12m-cls`](https://github.com/sunsmarterjie/yolov12/releases/download/cls/yolov12m-cls.pt)
[`yolov12l-cls`](https://github.com/sunsmarterjie/yolov12/releases/download/cls/yolov12l-cls.pt)
[`yolov12x-cls`](https://github.com/sunsmarterjie/yolov12/releases/download/cls/yolov12x-cls.pt)

```python
from ultralytics import YOLO

model = YOLO('yolov12{n/s/m/l/x}-cls.pt')
model.val(data='imagenet', save_json=True)
```

## Training 
```python
from ultralytics import YOLO

model = YOLO('yolov12n-cls.yaml')

# Train the model
results = model.train(
  data='imagenet',
  epochs=200, 
  batch=256, 
  imgsz=224,
  lr0=0.2,
  lrf=0.01,
  nbs=256,
  warmup_epochs=0,
  warmup_bias_lr=0.1,
  weight_decay=0.0001,
  cos_lr=True,
  hsv_s=0.4,
  optimizer='SGD',
  device="0",
)

# Evaluate model performance on the validation set
metrics = model.val()

# Perform object detection on an image
results = model("path/to/image.jpg")
results[0].show()

```

## Prediction
```python
from ultralytics import YOLO

model = YOLO('yolov12{n/s/m/l/x}-cls.pt')
model.predict()
```

## Export
```python
from ultralytics import YOLO

model = YOLO('yolov12{n/s/m/l/x}-cls.pt')
model.export(format="engine", half=True)  # or format="onnx"
```


## Demo

```
python app.py
# Please visit http://127.0.0.1:7860
```


## Acknowledgement

The code is based on [ultralytics](https://github.com/ultralytics/ultralytics). Thanks for their excellent work!

## Citation

```BibTeX
@article{tian2025yolov12,
  title={YOLOv12: Attention-Centric Real-Time Object Detectors},
  author={Tian, Yunjie and Ye, Qixiang and Doermann, David},
  journal={arXiv preprint arXiv:2502.12524},
  year={2025}
}
```

