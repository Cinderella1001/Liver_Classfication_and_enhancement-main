from scipy import stats


def stretchrange(r_array, height, width):
    length = height * width
    R_rray = r_array.flatten()
    R_rray.sort()
    print('R_rray', R_rray)
    # 最常出现的像素值
    mode = stats.mode(R_rray).mode[0]
    # 最常出现的像素值所在的最早的位置
    mode_index_before = list(R_rray).index(mode)

    # 最常出现的像素值所在下标的0.5%
    SR_min = R_rray[int(mode_index_before * 0.005)]
    # 从右边数 最常出现的像素值所在下标的0.5%
    SR_max = R_rray[int(-(length - mode_index_before) * 0.005)]

    print('mode', mode)
    print('SR_min', SR_min)
    print('SR_max', SR_max)

    # mode是最常出现的像素值
    return SR_min, SR_max, mode
