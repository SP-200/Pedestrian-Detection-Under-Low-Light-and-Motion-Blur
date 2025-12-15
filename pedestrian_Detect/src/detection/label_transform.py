import json
import os

# ==============================
JSON_PATH = r"C:\Users\ibrahimovic\Downloads\nightowls_validation\nightowls_validation.json"
IMG_ROOT = r"C:\Users\ibrahimovic\Downloads\nightowls_validation\nightowls_pedestrian"    # 只含行人图片
LABEL_ROOT = r"C:\Users\ibrahimovic\Downloads\nightowls_validation\labels"               # 生成 YOLO 标签
# ==============================

os.makedirs(LABEL_ROOT, exist_ok=True)

# 加载 JSON
with open(JSON_PATH, "r") as f:
    data = json.load(f)

# 建立 image_id → file_name 映射
id_to_filename = {img["id"]: img for img in data["images"]}

# 按 annotation 生成 YOLO 标签
for ann in data["annotations"]:
    if ann["category_id"] != 1:  # pedestrian
        continue
        
    image_id = ann["image_id"]
    img_info = id_to_filename[image_id]
    file_name = img_info["file_name"]

    # 如果该图片存在于你提取的行人图片子集中
    img_path = os.path.join(IMG_ROOT, file_name)
    if not os.path.exists(img_path):
        continue

    # 读取图像大小
    W, H = img_info["width"], img_info["height"]

    # NightOwls bbox = [x, y, w, h]
    x, y, w, h = ann["bbox"]

    # 转 YOLO 格式
    xc = (x + w/2) / W
    yc = (y + h/2) / H
    ww = w / W
    hh = h / H

    # 保存 txt 标签
    label_path = os.path.join(LABEL_ROOT, file_name.replace(".png", ".txt"))
    os.makedirs(os.path.dirname(label_path), exist_ok=True)
    with open(label_path, "a") as f:
        f.write(f"0 {xc} {yc} {ww} {hh}\n")

print("YOLO 标签生成完成。")
