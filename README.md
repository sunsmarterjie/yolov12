

<div align="center">
<h1>YOLOv12</h1>
<h3>YOLOv12: Attention-Centric Real-Time Object Detectors</h3>

[Yunjie Tian](https://sunsmarterjie.github.io/)<sup>1</sup>, [Qixiang Ye](https://people.ucas.ac.cn/~qxye?language=en)<sup>2</sup>, [David Doermann](https://cse.buffalo.edu/~doermann/)<sup>1</sup>

<sup>1</sup>  University at Buffalo, SUNY, <sup>2</sup> University of Chinese Academy of Sciences.

## Main Results

[**Instance segmentation**](https://github.com/sunsmarterjie/yolov12/tree/Seg):
| Model                                                                               | size<br><sup>(pixels) | mAP<sup>box<br>50-95 | mAP<sup>mask<br>50-95 | Speed<br><sup>T4 TensorRT10<br> | params<br><sup>(M) | FLOPs<br><sup>(B) |
| :------------------------------------------------------------------------------------| :--------------------: | :-------------------: | :---------------------: | :--------------------------------:| :------------------: | :-----------------: |
| [YOLOv12n-seg](https://github.com/sunsmarterjie/yolov12/releases/download/seg/yolov12n-seg.pt) | 640                   | 39.9                 | 32.8                  | 1.84                           | 2.8                | 9.9              |
| [YOLOv12s-seg](https://github.com/sunsmarterjie/yolov12/releases/download/seg/yolov12s-seg.pt) | 640                   | 47.5                 | 38.6                  | 2.84                           | 9.8                | 33.4              |
| [YOLOv12m-seg](https://github.com/sunsmarterjie/yolov12/releases/download/seg/yolov12m-seg.pt) | 640                   | 52.4                 | 42.3                  | 6.27                           | 21.9               | 115.1             |
| [YOLOv12l-seg](https://github.com/sunsmarterjie/yolov12/releases/download/seg/yolov12l-seg.pt) | 640                   | 54.0                 | 43.2                  | 7.61                          | 28.8               | 137.7             |
| [YOLOv12x-seg](https://github.com/sunsmarterjie/yolov12/releases/download/seg/yolov12x-seg.pt) | 640                   | 55.2                 | 44.2                  | 15.43                          | 64.5               | 308.7             |



## Installation
```
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu11torch2.2cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
conda create -n yolov12 python=3.11
conda activate yolov12
pip install -r requirements.txt
pip install -e .
```

## Validation
[`yolov12n-seg`](https://github.com/sunsmarterjie/yolov12/releases/download/seg/yolov12n-seg.pt)
[`yolov12s-seg`](https://github.com/sunsmarterjie/yolov12/releases/download/seg/yolov12s-seg.pt)
[`yolov12m-seg`](https://github.com/sunsmarterjie/yolov12/releases/download/seg/yolov12m-seg.pt)
[`yolov12l-seg`](https://github.com/sunsmarterjie/yolov12/releases/download/seg/yolov12l-seg.pt)
[`yolov12x-seg`](https://github.com/sunsmarterjie/yolov12/releases/download/seg/yolov12x-seg.pt)

```python
from ultralytics import YOLO

model = YOLO('yolov12{n/s/m/l/x}-seg.pt')
model.val(data='coco.yaml', save_json=True)
```

## Training 
```python
from ultralytics import YOLO

model = YOLO('yolov12n-seg.yaml')

# Train the model
results = model.train(
  data='coco.yaml',
  epochs=600, 
  batch=128, 
  imgsz=640,
  scale=0.5,  # S:0.9; M:0.9; L:0.9; X:0.9
  mosaic=1.0,
  mixup=0.0,  # S:0.05; M:0.15; L:0.15; X:0.2
  copy_paste=0.1,  # S:0.15; M:0.4; L:0.5; X:0.6
  device="0,1,2,3",
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

model = YOLO('yolov12{n/s/m/l/x}-seg.pt')
model.predict()
```

## Export
```python
from ultralytics import YOLO

model = YOLO('yolov12{n/s/m/l/x}-seg.pt')
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

