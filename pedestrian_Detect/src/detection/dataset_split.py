import os
import random
import shutil

IMG_ROOT = r"C:\Users\ibrahimovic\Downloads\nightowls_validation\nightowls_pedestrian"
LABEL_ROOT = r"C:\Users\ibrahimovic\Downloads\nightowls_validation\labels"

DATASET_ROOT = r"C:\Users\ibrahimovic\Downloads\nightowls_yolo"

TRAIN_IMG = os.path.join(DATASET_ROOT, "images/train")
VAL_IMG   = os.path.join(DATASET_ROOT, "images/val")
TRAIN_LAB = os.path.join(DATASET_ROOT, "labels/train")
VAL_LAB   = os.path.join(DATASET_ROOT, "labels/val")

os.makedirs(DATASET_ROOT, exist_ok=True)

os.makedirs(TRAIN_IMG, exist_ok=True)
os.makedirs(VAL_IMG, exist_ok=True)
os.makedirs(TRAIN_LAB, exist_ok=True)
os.makedirs(VAL_LAB, exist_ok=True)

# 所有图片路径
all_images = []
for root, _, files in os.walk(IMG_ROOT):
    for f in files:
        if f.endswith(".png"):
            all_images.append(os.path.join(root, f))

random.shuffle(all_images)

# 划分
train_size = int(0.8 * len(all_images))
train_imgs = all_images[:train_size]
val_imgs = all_images[train_size:]

def copy(img_list, img_dst, label_dst):
    for img in img_list:
        rel = os.path.relpath(img, IMG_ROOT)
        lab = os.path.join(LABEL_ROOT, rel.replace(".png", ".txt"))
        shutil.copy(img, os.path.join(img_dst, rel))
        shutil.copy(lab, os.path.join(label_dst, rel.replace(".png", ".txt")))

copy(train_imgs, TRAIN_IMG, TRAIN_LAB)
copy(val_imgs, VAL_IMG, VAL_LAB)

print("数据集划分完成！")
