# --------------------------------------------------------
# Based on yolov10
# https://github.com/THU-MIG/yolov10/app.py
# --------------------------------------------------------'

import gradio as gr
import cv2
import tempfile
import numpy as np
from ultralytics import YOLO

def custom_draw(results, image):
    """支持标签显示和关键点连线的绘制函数"""
    # 保持RGB颜色空间
    img = image.copy()
    h, w = img.shape[:2]
    
    # 绘制检测框和标签
    if results.boxes is not None:
        for box, cls, conf in zip(results.boxes.xyxy.cpu().numpy(),
                                results.boxes.cls.cpu().numpy(),
                                results.boxes.conf.cpu().numpy()):
            x1, y1, x2, y2 = map(int, box[:4])
            
            # 绘制边界框（BGR转RGB颜色）
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 2)
            
            # 准备标签文本
            label = f"{results.names[int(cls)]} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            
            # 标签背景
            cv2.rectangle(img, (x1, y1-20), (x1+tw, y1), (180, 180, 255), -1)
            # 标签文字（白色文字）
            cv2.putText(img, label, (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    # 关键点绘制逻辑
    if results.keypoints is not None:
        connections = [(0,2), (1,3), (0,3), (1,2)]  # 四边形对角线连接
        
        for kps in results.keypoints.xy.cpu().numpy():
            # 有效性检查：必须包含至少4个有效点
            if kps.shape[0] < 4 or np.any(kps[0:4] < 0):
                continue  # 跳过不满足条件的实例

            valid_points = []
            # 严格限定处理前4个关键点
            for i in range(4):  # 0-3号点
                x, y = kps[i]
                # 坐标有效性验证
                if 0 <= x < w and 0 <= y < h:
                    color = (0, 255, 0)  # 统一使用绿色
                    cv2.circle(img, (int(x), int(y)), 5, color, -1)
                    # 标注点序号（白色文字）
                    cv2.putText(img, str(i), (int(x)+5, int(y)), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                    valid_points.append((x,y))
                else:
                    valid_points.append(None)

            # 安全绘制连接线
            for (s, e) in connections:
                # 双重有效性检查
                if s >= len(valid_points) or e >= len(valid_points):
                    continue
                start = valid_points[s]
                end = valid_points[e]
                if start and end:  # 两个点都有效
                    # 坐标转换与绘制
                    start = tuple(map(int, start))
                    end = tuple(map(int, end))
                    cv2.line(img, start, end, (255, 0, 255), 2)

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def yolov12_inference(image, video, model_id, image_size, conf_threshold):
    model = YOLO(model_id)
    if image:
        # 处理图片
        results = model.predict(image, imgsz=image_size, conf=conf_threshold)
        annotated = custom_draw(results[0], results[0].orig_img)
        return annotated, None
    else:
        # 处理视频
        video_path = tempfile.mktemp(suffix=".mp4")
        with open(video_path, "wb") as f:
            f.write(open(video, "rb").read())

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_size = (int(cap.get(3)), int(cap.get(4)))
        
        output_path = tempfile.mktemp(suffix=".mp4")
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, frame_size)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            # YOLO预测
            results = model.predict(frame, 
                                  imgsz=image_size, conf=conf_threshold)
            
            # 使用自定义绘制
            annotated =cv2.cvtColor(custom_draw(results[0], results[0].orig_img),cv2.COLOR_BGR2RGB)

            out.write(annotated)

        cap.release()
        out.release()
        return None, output_path


def yolov12_inference_for_examples(image, model_path, image_size, conf_threshold):
    annotated_image, _ = yolov12_inference(image, None, model_path, image_size, conf_threshold)
    return annotated_image


def app():
    with gr.Blocks():
        with gr.Row():
            with gr.Column():
                image = gr.Image(type="pil", label="Image", visible=True)
                video = gr.Video(label="Video", visible=False)
                input_type = gr.Radio(
                    choices=["Image", "Video"],
                    value="Image",
                    label="Input Type",
                )
                model_id = gr.Dropdown(
                    label="Model",
                    choices=[
                        "runs/pose/train38/weights/best.pt",
                        "runs/pose/train39/weights/best.pt",
                        "runs/pose/train39/weights/last.pt",
                        "runs/pose/train38/weights/last.pt",
                    ],
                    value="runs/pose/train39/weights/last.pt",
                )
                image_size = gr.Slider(
                    label="Image Size",
                    minimum=320,
                    maximum=1280,
                    step=32,
                    value=640,
                )
                conf_threshold = gr.Slider(
                    label="Confidence Threshold",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.05,
                    value=0.25,
                )
                yolov12_infer = gr.Button(value="Detect Objects")

            with gr.Column():
                output_image = gr.Image(type="numpy", label="Annotated Image", visible=True)
                output_video = gr.Video(label="Annotated Video", visible=False)

        def update_visibility(input_type):
            image = gr.update(visible=True) if input_type == "Image" else gr.update(visible=False)
            video = gr.update(visible=False) if input_type == "Image" else gr.update(visible=True)
            output_image = gr.update(visible=True) if input_type == "Image" else gr.update(visible=False)
            output_video = gr.update(visible=False) if input_type == "Image" else gr.update(visible=True)

            return image, video, output_image, output_video

        input_type.change(
            fn=update_visibility,
            inputs=[input_type],
            outputs=[image, video, output_image, output_video],
        )

        def run_inference(image, video, model_id, image_size, conf_threshold, input_type):
            if input_type == "Image":
                return yolov12_inference(image, None, model_id, image_size, conf_threshold)
            else:
                return yolov12_inference(None, video, model_id, image_size, conf_threshold)


        yolov12_infer.click(
            fn=run_inference,
            inputs=[image, video, model_id, image_size, conf_threshold, input_type],
            outputs=[output_image, output_video],
        )

        gr.Examples(
            examples=[
                [
                    "ultralytics/assets/bus.jpg",
                    "yolov12s.pt",
                    640,
                    0.25,
                ],
                [
                    "ultralytics/assets/zidane.jpg",
                    "yolov12x.pt",
                    640,
                    0.25,
                ],
            ],
            fn=yolov12_inference_for_examples,
            inputs=[
                image,
                model_id,
                image_size,
                conf_threshold,
            ],
            outputs=[output_image],
            cache_examples='lazy',
        )

gradio_app = gr.Blocks()
with gradio_app:
    gr.HTML(
        """
    <h1 style='text-align: center'>
    YOLOv12: Attention-Centric Real-Time Object Detectors
    </h1>
    """)
    gr.HTML(
        """
        <h3 style='text-align: center'>
        <a href='https://arxiv.org/abs/2502.12524' target='_blank'>arXiv</a> | <a href='https://github.com/sunsmarterjie/yolov12' target='_blank'>github</a>
        </h3>
        """)
    with gr.Row():
        with gr.Column():
            app()
if __name__ == '__main__':
    gradio_app.launch()
