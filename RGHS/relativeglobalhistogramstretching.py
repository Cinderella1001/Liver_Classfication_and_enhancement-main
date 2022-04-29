# ———————————————————————————  左右两边选取参数进行拉伸 mode/std and 0.95%/t_n ——————————————————————————————————

import math
import numpy as np
from stretchRange import stretchrange

pi = math.pi
e = math.e


# lamda和k分别是什么？
def global_stretching(r_array, height, width, lamda, k):

    length = height * width
    R_rray = []
    for i in range(height):
        for j in range(width):
            # 整个图像的像素值按照行先序排成一个数组
            R_rray.append(r_array[i][j])
    # 图像像素值升序排列
    R_rray.sort()

    # 去掉前0.5%
    I_min = R_rray[int(length / 200)]
    # 去掉后0.5%
    I_max = R_rray[-int(length / 200)]

    array_Global_histogram_stretching = np.zeros((height, width))
    d = 4

    # 最常出现的像素值为mode，像素值最初出现位置的0.5%为SR_min，右侧对应的位置为SR_max
    SR_min, SR_max, mode = stretchrange(R_rray, height, width)
    # 自己加的
    # I_max = SR_max
    # 自己加的
    # I_min = SR_min

    # O_lamda_min
    DR_min = (1 - 0.655) * mode
    # t_lamda_x
    t_n = lamda ** d

    # miu_lamda的左右范围
    O_max_left = SR_max * t_n * k / mode
    O_max_right = 255 * t_n * k / mode
    # miu_lamda的左右区间
    Dif = O_max_right - O_max_left
    # if Dif >= 1.526:
    if Dif >= 1:
        sum = 0
        # for i in range(2, int(Dif + 1)):
        for i in range(1, int(Dif + 1)):
            # sum = sum + ((i - 1.526) * 0.665 * mode + mode) / (t_n * k)
            sum = sum + (1.526 + i) * mode / (t_n * k)
        # O_lamda_max
        DR_max = sum / int(Dif)

        for i in range(0, height):
            for j in range(0, width):
                if r_array[i][j] < I_min:

                    p_out = (r_array[i][j] - I_min) * (DR_min / I_min) + I_min
                    array_Global_histogram_stretching[i][j] = p_out
                elif r_array[i][j] > I_max:
                    p_out = (r_array[i][j] - DR_max) * (DR_max / I_max) + I_max
                    array_Global_histogram_stretching[i][j] = p_out
                else:
                    p_out = int((r_array[i][j] - I_min) * ((255 - I_min) / (I_max - I_min))) + I_min
                    array_Global_histogram_stretching[i][j] = p_out
    else:

        if r_array[i][j] < I_min:

            p_out = (r_array[i][j] - np.min(r_array)) * (DR_min / np.min(r_array)) + np.min(r_array)
            array_Global_histogram_stretching[i][j] = p_out
        else:
            p_out = int((r_array[i][j] - I_min) * ((255 - DR_min) / (I_max - I_min))) + DR_min
            array_Global_histogram_stretching[i][j] = p_out

    return array_Global_histogram_stretching


def RelativeGHstretching(sceneRadiance, height, width):
    # blue lamda推荐的范围为0.95——0.99
    sceneRadiance[:, :, 0] = global_stretching(sceneRadiance[:, :, 0], height, width, 0.97, 1.25)
    # green lamda推荐的范围为0.93——0.97
    sceneRadiance[:, :, 1] = global_stretching(sceneRadiance[:, :, 1], height, width, 0.95, 1.25)
    # red lamda推荐的范围为0.8——0.85
    sceneRadiance[:, :, 2] = global_stretching(sceneRadiance[:, :, 2], height, width, 0.83, 0.85)
    return sceneRadiance
