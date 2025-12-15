import os
import shutil
import cv2
from tqdm import tqdm

ROOT = r"C:\Users\ibrahimovic\Downloads\nightowls_yolo"

HE_ROOT = r"C:\Users\ibrahimovic\Downloads\nightowls_HE"


for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
    os.makedirs(os.path.join(HE_ROOT, sub), exist_ok=True)

def clahe_enhance(img):
    """对输入图像执行 CLAHE 亮度增强"""
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = cv2.split(ycrcb)

    # CLAHE 参数（适合夜间）
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    Y = clahe.apply(Y)

    enhanced = cv2.merge([Y, Cr, Cb])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_YCrCb2BGR)
    return enhanced

def process_split(split):
    print(f"\n=== Processing {split} images with CLAHE ===")
    
    in_dir = os.path.join(ROOT, "images", split)
    out_dir = os.path.join(HE_ROOT, "images", split)

    for img_name in tqdm(os.listdir(in_dir)):
        img_path = os.path.join(in_dir, img_name)
        img = cv2.imread(img_path)

        if img is None:
            print("Error reading:", img_path)
            continue

        enhanced = clahe_enhance(img)
        cv2.imwrite(os.path.join(out_dir, img_name), enhanced)


# 对 train 和 val 分别增强
process_split("train")
process_split("val")


print("\n=== Copying label files ===")

for split in ["train", "val"]:
    src = os.path.join(ROOT, "labels", split)
    dst = os.path.join(HE_ROOT, "labels", split)
    for file in os.listdir(src):
        shutil.copy(os.path.join(src, file), os.path.join(dst, file))

yaml_path = os.path.join(os.getcwd(), "data_HE.yaml")

with open(yaml_path, "w") as f:
    f.write(
"""# HE-enhanced dataset
train: {}/images/train
val: {}/images/val

nc: 1
names: ["person"]
""".format(
    HE_ROOT.replace("\\", "/"),
    HE_ROOT.replace("\\", "/")
)
    )

print("\n=== DONE! ===")
print(f"HE dataset saved to: {HE_ROOT}")
print(f"data_HE.yaml generated at: {yaml_path}")
