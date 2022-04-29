# encoding=utf-8
import os
import numpy as np
import cv2
# 高级的列表排序模块
import natsort

from LabStretching import LABStretching
from global_stretching_RGB import stretching
from relativeglobalhistogramstretching import RelativeGHstretching

np.seterr(over='ignore')
if __name__ == '__main__':
    pass

folder = "E:\硕士\研一课程\研一下\（选修）深度学习\作业1\实现\LiverClassfication-main(without_data)"
path = folder + "/enhance_before_data"
files = os.listdir(path)
# 文件名排序
files = natsort.natsorted(files)

for i in range(len(files)):
    file = files[i]
    filepath = path + "/" + file
    # 文件名前缀
    prefix = file.split('.')[0]
    if os.path.isfile(filepath):
        print('********    file   ********', file)
        img = cv2.imread(folder + '/enhance_before_data/' + file)

        height = len(img)
        width = len(img[0])
        print('height', height)
        print('width', width)

        sceneRadiance = img

        # 图像归一化
        sceneRadiance = stretching(sceneRadiance)
        # 相对全局直方图拉伸
        # sceneRadiance = RelativeGHstretching(sceneRadiance, height, width)
        # 颜色均衡化
        sceneRadiance = LABStretching(sceneRadiance)

        # 保存处理后的图像
        cv2.imwrite(folder + '/enhance_after_data/RGHS/' + prefix + '.png', sceneRadiance)
        print('outputpath:', folder + '/enhance_after_data/RGHS/' + prefix + '.png')
