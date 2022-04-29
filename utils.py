import random
import numpy as np
import os
import json
import math
import torch


# 传入训练集和测试集的路径
def read_split_data(train_dir: str, val_dir: str):
    # 随机种子
    random.seed(0)

    # 遍历文件夹，一个文件夹对应一个类别
    # liver_class = [4, 58, 67, 123, 1234, 5678]
    liver_class = [cla for cla in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, cla))]

    # 排序，保证顺序一致
    liver_class.sort()

    # 生成类别名称以及对应的数字索引
    # class_indices = {123:0, 1234:1, 4:2, 5678:3, 58:4, 67:5}
    class_indices = dict((k, v) for v, k in enumerate(liver_class))

    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []  # 存储训练集的所有图片路径
    train_images_label = []  # 存储训练集图片的分类对应索引信息
    val_images_path = []  # 存储验证集的所有图片路径
    val_images_label = []  # 存储验证集图片的分类对应索引信息
    every_class_num = []  # 存储每个类别的样本总数

    # 遍历每个文件夹下的文件
    for cla in liver_class:
        cla_path = os.path.join(train_dir, cla)
        # 遍历获取supported支持的所有文件路径
        images = [os.path.join(train_dir, cla, i) for i in os.listdir(cla_path)]
        # 获取该类别对应的索引
        image_class = class_indices[cla]
        # 记录该类别的样本数量
        every_class_num.append(len(images))

        for img_path in images:
            train_images_path.append(img_path)
            train_images_label.append(image_class)

    for cla in liver_class:
        cla_path = os.path.join(val_dir, cla)
        # 遍历获取supported支持的所有文件路径
        images = [os.path.join(val_dir, cla, i) for i in os.listdir(cla_path)]
        # 获取该类别对应的索引
        image_class = class_indices[cla]
        # 记录该类别的样本数量
        every_class_num.append(len(images))

        for img_path in images:
            val_images_path.append(img_path)
            val_images_label.append(image_class)

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))

    # 返回所有训练图片组成的列表[...,...,...,...,...,...]、训练图片对应的类别[0,...,0,1,...,1,2,...,2,3,...,3,4,...,4,5,...,5]
    # 测试图片组成的列表、测试图片对应的类别
    return train_images_path, train_images_label, val_images_path, val_images_label


# 学习率随着迭代次数的增加而变化
def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3,
                        end_factor=1e-6):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            current_step = (x - warmup_epochs * num_step)
            cosine_steps = (epochs - warmup_epochs) * num_step
            # warmup后lr倍率因子从1 -> end_factor
            return ((1 + math.cos(current_step * math.pi / cosine_steps)) / 2) * (1 - end_factor) + end_factor

    # new_lr = lambda * old_lr
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)


# 获得网络模型中训练过程中所有参数名称和值
def get_params_groups(model: torch.nn.Module, weight_decay: float = 1e-5):
    # 记录optimize要训练的权重参数
    parameter_group_vars = {"decay": {"params": [], "weight_decay": weight_decay},
                            "no_decay": {"params": [], "weight_decay": 0.}}

    # 记录对应的权重名称
    parameter_group_names = {"decay": {"params": [], "weight_decay": weight_decay},
                             "no_decay": {"params": [], "weight_decay": 0.}}

    # model.named_parameters()打印每一次迭代参数的名字和param
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights

        if len(param.shape) == 1 or name.endswith(".bias"):
            group_name = "no_decay"
        else:
            group_name = "decay"

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)

    print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    return list(parameter_group_vars.values())


def convert_img(ir_img):
    ir_img = ir_img.astype(np.uint16) * 4

    return ir_img


if __name__ == '__main__':
    # train_dir = 'data/train_set'
    # val_dir = 'data/test_set'
    # read_split_data(train_dir=train_dir, val_dir=val_dir)
    convert_img()
