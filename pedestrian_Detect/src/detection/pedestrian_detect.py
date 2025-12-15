import os
import json
import shutil
from tqdm import tqdm

# ==============================
# 替换为你自己的路径
JSON_PATH = r"C:\Users\ibrahimovic\Downloads\nightowls_validation\nightowls_validation.json"
IMG_ROOT = r"C:\Users\ibrahimovic\Downloads\nightowls_validation\nightowls_validation"
SAVE_ROOT = r"C:\Users\ibrahimovic\Downloads\nightowls_validation\nightowls_pedestrian"

# 创建保存目录
os.makedirs(SAVE_ROOT, exist_ok=True)

# 读取标注文件
with open(JSON_PATH, "r") as f:
    data = json.load(f)

# 记录哪些图片包含行人
img_has_person = set()

# 统计 annotation 中所有 pedestrian 对应的 image_id
for ann in data["annotations"]:
    if ann["category_id"] == 1:      # 1 = pedestrian
        img_has_person.add(ann["image_id"])

print(f"共找到 {len(img_has_person)} 张包含行人的图片。")

# 建立 image_id -> file_name 的映射
id_to_filename = {img["id"]: img["file_name"] for img in data["images"]}

# 复制包含行人的图片
count = 0
for img_id in tqdm(img_has_person):
    file_name = id_to_filename[img_id]
    src = os.path.join(IMG_ROOT, file_name)
    dst = os.path.join(SAVE_ROOT, file_name)

    # 创建子目录
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    
    # 复制文件
    if os.path.exists(src):
        shutil.copy(src, dst)
        count += 1

print(f"复制完成，共复制 {count} 张图片到：{SAVE_ROOT}")
