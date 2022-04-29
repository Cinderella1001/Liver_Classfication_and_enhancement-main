import numpy as np
from sklearn.metrics import precision_recall_curve


# 参见https://blog.csdn.net/qq_35705332/article/details/109028620
# https://blog.csdn.net/Blateyang/article/details/81054881
def cal_mAP(target, output):
    # 每个分类对应的P-R曲线下的面积
    aps = []
    # 遍历每个分类
    for key in target:
        # 得到一个分类下的以Precision为纵坐标，以Recall为横坐标的P-R曲线
        precision, recall, _ = precision_recall_curve(target[key], output[key])
        # 不知道为什么要进行左右翻转
        # np.fliplr(矩阵)：实现矩阵的左右翻转
        precision = np.fliplr([precision])[0]
        recall = np.fliplr([recall])[0]
        # 计算曲线下面积
        ap = voc_ap(recall, precision)
        aps.append(ap)
    # 对每个分类P-R曲线下面积求和，除以分类数，得到mAP
    mAP = sum(aps) / len(aps)
    return mAP


def voc_ap(rec, prec):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """

    # correct AP calculation
    # first append sentinel values at the end
    # 首尾分别加上初始点和结束点
    # np.concatenate后接多个数组或列表，实现拼接
    mrec = np.concatenate(([0.], rec, [1.]))  # [0.  0.0666, 0.1333, 0.4   , 0.4666,  1.]
    mpre = np.concatenate(([0.], prec, [0.]))  # [0.  1.,     0.6666, 0.4285, 0.3043,  0.]

    # 下面两步进行P-R曲线的平滑

    # compute the precision envelope
    # 计算出precision的各个断点(折线点)
    # 滤除fp增加条件下导致的pre减小的无效值？ 不太理解哎
    # 倒序看数组是否是降序
    # range(start,stop,step)
    for i in range(mpre.size - 1, 0, -1):
        # np.maximum(X,Y)：逐位比较大小，取较大的值
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])  # [1.     1.     0.6666 0.4285 0.3043 0.    ]

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    # 滤除总检测样本数增加导致计算的recall的未增加的量? 不太理解哎
    i = np.where(mrec[1:] != mrec[:-1])[0]  # precision前后两个值不一样的点

    # AP= AP1 + AP2+ AP3+ AP4
    # and sum (\Delta recall) * prec
    # 面积直接用矩形的面积代替？
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap
