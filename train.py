import os
import sys
import argparse
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from dataset import LiverDataset
from utils import read_split_data, get_params_groups, create_lr_scheduler
from net.unet import Unet
from net.resnet import resnet34
from net.googlenet import GoogLeNet
from net.alexnet import AlexNet
from eval_map import cal_mAP

import operator
import functools


def net_train_one_epoch(model, optimizer, data_loader, device, epoch, lr_scheduler):
    print("training......")
    # 开启训练模式
    model.train()

    # 训练时使用交叉熵损失函数
    loss_function = torch.nn.CrossEntropyLoss()

    # .to(device)数据传入执行设备
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数

    # 将梯度记录清零
    optimizer.zero_grad()

    # 已经参与训练的样本数
    sample_num = 0

    # tqdm用来显示进度条和每一轮耗费的时间
    data_loader = tqdm(data_loader, file=sys.stdout)

    for step, data in enumerate(data_loader):
        images, labels = data

        sample_num += images.shape[0]

        pred = model(images.to(device))

        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        # 梯度下降
        loss.backward()
        # loss.detach()使这部分loss不可求导？
        accu_loss += loss.detach()

        # tqdm.desc传入进度条的前缀
        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}, lr: {:.5f}".format(
            epoch,
            accu_loss.item() / (step + 1),
            accu_num.item() / sample_num,
            optimizer.param_groups[0]["lr"]
        )

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        # 进行下一步梯度下降
        optimizer.step()
        optimizer.zero_grad()
        # 更新学习率
        lr_scheduler.step()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


@torch.no_grad()
def net_evaluate(model, data_loader, device, epoch):
    print("evaluating......")
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()

    accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失

    sample_num = 0

    target = {'0': [], '1': [], '2': [], '3': [], '4': [], '5': []}
    predi = {'0': [], '1': [], '2': [], '3': [], '4': [], '5': []}

    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        for i in range(0, images.shape[0] - 1):
            predict = torch.softmax(pred[i], dim=0)
            predi['0'].append(predict[0].numpy())
            predi['1'].append(predict[1].numpy())
            predi['2'].append(predict[2].numpy())
            predi['3'].append(predict[3].numpy())
            predi['4'].append(predict[4].numpy())
            predi['5'].append(predict[5].numpy())

            truth_class = os.path.dirname(labels[i]).split('/')[-1]

            if truth_class == '123':
                target['0'].append(1)
            else:
                target['0'].append(0)

            if truth_class == '1234':
                target['1'].append(1)
            else:
                target['1'].append(0)

            if truth_class == '4':
                target['2'].append(1)
            else:
                target['2'].append(0)

            if truth_class == '5678':
                target['3'].append(1)
            else:
                target['3'].append(0)

            if truth_class == '58':
                target['4'].append(1)
            else:
                target['4'].append(0)

            if truth_class == '67':
                target['5'].append(1)
            else:
                target['5'].append(0)


        loss = loss_function(pred, labels.to(device))
        accu_loss += loss

        # 一个step表示一个batch
        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}, mAP: {:.3f}".format(
            epoch,
            accu_loss.item() / (step + 1),
            accu_num.item() / sample_num,
            cal_mAP(target, predi)
        )
    # 判断模型好坏的标准是精确率这个指标，mAP没有回传
    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


def main(args):
    # 使用CPU还是GPU
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"using {device} device.")

    # 是否存在预训练模型文件夹
    if os.path.exists("./weights") is False:
        os.makedirs("./weights")
    # tensorboard存储变量
    tb_writer = SummaryWriter()

    batch_size = args.batch_size
    # number of workers
    # 每个batch本应在单独的处理单元上处理，需要看机器配备多少个并行处理单元以及设定的batch的个数
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])

    # 没有GPU的情况下，nw设为0
    # nw = 0
    print('Using {} dataloader workers every process'.format(nw))

    # 数据导入
    train_images_path, train_images_label, val_images_path, val_images_label = \
        read_split_data(train_dir=args.train_set,
                        val_dir=args.val_set)

    # 数据预处理
    train_dataset = LiverDataset(images_path=train_images_path,
                                 images_class=train_images_label,
                                 mode='train')

    val_dataset = LiverDataset(images_path=val_images_path,
                               images_class=val_images_label,
                               mode='val')

    # torch.utils.data.DataLoader数据加载器，多个线程处理数据集
    # dataset:数据来源
    # batch_size:每个batch加载多少数据
    # shuffle:在每个epoch中重新随机排列batch中的数据
    # collate_fn: 把 list sample 合并成 mini-batch
    # num_workers:多少个线程用于加载数据
    # pin_memory： the data loader will copy tensors into CUDA pinned memory before returning them.
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, pin_memory=True,
                              num_workers=nw, collate_fn=train_dataset.collate_fn)

    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, pin_memory=True,
                            num_workers=nw, collate_fn=val_dataset.collate_fn)

    if args.model == "resnet":
        net = resnet34().to(device)
    elif args.model == "googlenet":
        net = GoogLeNet(num_classes=6, aux_logits=False, init_weights=True).to(device)
    elif args.model == "alexnet":
        net = AlexNet(num_classes=6, init_weights=True).to(device)

    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)["model"]

        # 删除有关分类类别的权重
        for k in list(weights_dict.keys()):
            if "head" in k:
                del weights_dict[k]

        print(net.load_state_dict(weights_dict, strict=False))

    if args.freeze_layers:
        for name, para in net.named_parameters():
            # 除head外，其他权重全部冻结
            if "head" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    # 所有参数的值
    # {"decay": {"params": [], "weight_decay": weight_decay},
    #  "no_decay": {"params": [], "weight_decay": 0.}}
    pg = get_params_groups(net, weight_decay=args.wd)

    # torch.optim
    # optim.AdamW使用L2正则化和权重衰减
    # final_loss = loss + wd * all_weights.pow(2).sum() / 2
    # 对final_loss求导，w = w - lr * w.grad - lr * wd * w
    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=args.wd)

    # 调整学习率
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs,
                                       warmup=True, warmup_epochs=1)

    best_acc = 0.0

    for epoch in range(args.epochs):
        # train
        train_loss, train_acc = net_train_one_epoch(model=net,
                                                    optimizer=optimizer,
                                                    data_loader=train_loader,
                                                    device=device,
                                                    epoch=epoch,
                                                    lr_scheduler=lr_scheduler)

        # validate
        val_loss, val_acc = net_evaluate(model=net,
                                         data_loader=val_loader,
                                         device=device,
                                         epoch=epoch)

        # 记录训练损失、训练精确度、测试损失、测试精确度、学习率
        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        # add_scalar
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

        if best_acc < val_acc:
            print("saving best model......")
            if args.model == "resnet":
                torch.save(net.state_dict(), "./weights/best_resnet_model.pth".format(epoch))
            elif args.model == "googlenet":
                torch.save(net.state_dict(), "./weights/best_googlenet_model.pth".format(epoch))
            elif args.model == "alexnet":
                torch.save(net.state_dict(), "./weights/best_alexnet_model.pth".format(epoch))
            best_acc = val_acc

        print("best_acc=", best_acc)
    # net = Unet(in_ch=1, num_classes=6)

    print('Finished Training')


if __name__ == '__main__':
    # 定义运行命令需要的参数

    parser = argparse.ArgumentParser()

    # 使用什么模型
    parser.add_argument('--model', type=str,
                        default="resnet")
    # 图片类别数
    parser.add_argument('--num_classes', type=int, default=6)
    # 迭代次数
    parser.add_argument('--epochs', type=int, default=50)
    # 数据集分成多个batch，每个batch的大小
    parser.add_argument('--batch-size', type=int, default=8)
    # 学习率
    parser.add_argument('--lr', type=float, default=5e-4)
    # 权重衰减率
    parser.add_argument('--wd', type=float, default=5e-2)

    # 训练集路径
    parser.add_argument('--train-set', type=str,
                        default="data/train_set")
    # 测试集路径
    parser.add_argument('--val-set', type=str,
                        default="data/test_set")

    # 预训练模型路径
    parser.add_argument('--weights', type=str, default='',
                        help='initial weights path')
    # 是否冻结head以外所有权重
    parser.add_argument('--freeze-layers', type=bool, default=False)

    # 设备是什么,并没有起到什么作用，程序会自动判断使用CPU还是GPU
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    # 运行main函数
    main(opt)
