import numpy as np


def stretching(img):
    height = len(img)
    width = len(img[0])
    for k in range(0, 3):
        Max_channel = np.max(img[:, :, k])
        Min_channel = np.min(img[:, :, k])
        for i in range(height):
            for j in range(width):
                # 对图像的所有像素值进行归一化
                img[i, j, k] = (img[i, j, k] - Min_channel) * (255 - 0) / (Max_channel - Min_channel) + 0
    return img
