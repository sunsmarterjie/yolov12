import json
import os

# 加载 COCO JSON
with open("/root/rm/dataset/rm_armor_coco/annotations/coco_instances_test.json", "r") as f:
    coco_data = json.load(f)

# 创建 YOLO 格式的标注
for img in coco_data["images"]:
    img_id = img["id"]
    img_name = img["file_name"]
    img_width = img["width"]
    img_height = img["height"]

    # 找到该图像的所有标注
    anns = [a for a in coco_data["annotations"] if a["image_id"] == img_id]

    # 生成 YOLO 格式的 .txt 文件
    with open(f"/root/rm/dataset/rm_armor_yolo/labels/test/{img_name.replace('.jpg', '.txt')}", "w") as f_txt:
        for ann in anns:
            # 边界框 (COCO 格式是 [x,y,width,height], YOLO 需要归一化的 [x_center,y_center,width,height])
            x, y, w, h = ann["bbox"]
            x_center = (x + w / 2) / img_width
            y_center = (y + h / 2) / img_height
            w_norm = w / img_width
            h_norm = h / img_height

            # 关键点 (COCO 格式是 [x1,y1,v1, x2,y2,v2, ...], v=0/1/2)
            kpts = ann["segmentation"][0]
            kpts_yolo = []
            for i in range(0, len(kpts), 2):
                x_kpt = kpts[i] / img_width
                y_kpt = kpts[i+1] / img_height
                # vis = 1 if kpts[i+2] > 0 else 0  # COCO: 0=未标注, 1=标注但不可见, 2=可见
                kpts_yolo.extend([x_kpt, y_kpt, 1])

            # 写入 YOLO 格式
            line = f"0 {x_center} {y_center} {w_norm} {h_norm} " + " ".join(map(str, kpts_yolo)) + "\n"
            f_txt.write(line)