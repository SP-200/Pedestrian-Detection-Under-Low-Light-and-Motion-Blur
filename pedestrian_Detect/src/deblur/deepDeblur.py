import os
import cv2
import torch
import numpy as np
import argparse
from torchvision.transforms import ToTensor, ToPILImage
from models import DeblurNet 

def load_model(checkpoint_path, device):
    """
    加载 DeepDeblur 预训练模型
    """
    model = DeblurNet()
    model = model.to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model

def deblur_image(input_img_path, output_img_path, model, device):
    """
    对单张图像进行 DeepDeblur 推理
    """
    # 读取图像
    img = cv2.imread(input_img_path)
    if img is None:
        raise FileNotFoundError(f"cannot read {input_img_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 预处理到 [0,1] tensor
    tensor = ToTensor()(img_rgb).unsqueeze(0).to(device)

    # 推理
    with torch.no_grad():
        out = model(tensor)

    # 转回 numpy
    out_img = out.clamp(0, 1).cpu()[0]
    out_img = ToPILImage()(out_img)

    out_bgr = cv2.cvtColor(np.array(out_img), cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_img_path, out_bgr)
    print(f"Deblurred saved to {output_img_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to blurred input image")
    parser.add_argument("--output", required=True, help="Path to save deblurred output")
    parser.add_argument("--checkpoint", required=True, help="Path to pretrained model .pth")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)
    deblur_image(args.input, args.output, model, device)

if __name__ == "__main__":
    main()
