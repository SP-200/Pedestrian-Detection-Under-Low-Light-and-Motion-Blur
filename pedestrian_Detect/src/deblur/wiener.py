import cv2
import numpy as np
from skimage.restoration import wiener


def wiener_deblur(image, kernel, balance=0.01):
   
    # 转为 float32 0~1
    img_f = image.astype(np.float32) / 255.0

    # 输出图像
    out = np.zeros_like(img_f)

    # 三通道分别做 Wiener
    for c in range(3):
        out[:, :, c] = wiener(img_f[:, :, c], kernel, balance)

    # 截断区间
    out = np.clip(np.real(out), 0, 1)

    return (out * 255).astype(np.uint8)

if __name__ == "__main__":

    img = cv2.imread("linear_blur.png")
    if img is None:
        raise FileNotFoundError("blur.png not found!")

    def linear_motion_kernel(length=20, angle=45):
        kernel = np.zeros((length, length), np.float32)
        kernel[length // 2, :] = 1.0
        kernel /= kernel.sum()
        M = cv2.getRotationMatrix2D((length/2, length/2), angle, 1)
        kernel = cv2.warpAffine(kernel, M, (length, length))
        kernel = np.clip(kernel, 0, None)
        kernel /= kernel.sum()
        return kernel

    kernel = linear_motion_kernel(length=20, angle=45)

    result = wiener_deblur(img, kernel, balance=0.01)

    cv2.imwrite("wiener_deblur.png", result)
    print("Wiener deblur saved as wiener_deblur.png")
