from roboflow import Roboflow
from ultralytics import YOLO
import torch

def main():
 

    model = YOLO('yolov12n.yaml')  # Use correct model config
    
    

    results = model.train(
        data='datasets/GBC_Annotated/data.yaml', # mao ning coffee bean dataset
        epochs=20,
        batch=16,
        imgsz=640,
        scale=0.7,
        mosaic=0.5,
        mixup=0.2,
        copy_paste=0.1,
        device=0 if torch.cuda.is_available() else 'cpu'
    )
    return "✅ Training complete. Check 'runs/train/' for results."





if __name__ == "__main__":
    
    main()
    
    #model = YOLO('yolov12{n/s/m/l/x}.pt')
    #model.val(data='coco.yaml', save_json=True)
    #model = YOLO('runs/detect/train7/weights/last.pt')  # Update path if needed
    #metrics = model.val()
    #print(metrics)