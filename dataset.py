from matplotlib import transforms
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import PIL.Image as Image
from utils import convert_img

# torchvision.transforms是Pytorch中的图像预处理包，包含了很多对图像数据进行变换的函数
# transforms.Compose(transforms)将多种变换组合在一起
# ToPILImage():将shape为(C,H,W)的Tensor或shape为(H,W,C)的numpy.ndarray转换成PIL.Image
# RandomResizedCrop(224):进行随机大小和随机高宽比例的剪裁，然后resize到指定大小224
# CenterCrop():将给定的PIL.Image进行中心切割，得到给定的size，size可以是tuple，(target_height, target_width)。
# size也可以是一个Integer，在这种情况下，切出来的图片的形状是正方形。
# RandomHorizontalFlip():进行图像的随机水平翻转
# ToTensor():将PILImage转变成torch.FloatTensor的形式
# Normalize(mean,std):用给定的均值和标准差分别对每个通道的数据进行正则化
data_transform = {
    "train": transforms.Compose([transforms.ToPILImage(),
                                 transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize(0.5, 0.5)]),

    "val": transforms.Compose([transforms.ToPILImage(),
                               transforms.Resize(int(224 * 1.143)),
                               transforms.CenterCrop(224),
                               transforms.ToTensor(),
                               transforms.Normalize(0.5, 0.5)]),
}


class LiverDataset(Dataset):
    def __init__(self, images_path: list, images_class: list, mode='train'):
        self.images_path = images_path
        self.images_class = images_class
        self.mode = mode
        if self.mode == 'train':
            self.transform = data_transform['train']
        if self.mode == 'val':
            self.transform = data_transform['val']

    def __len__(self):
        return len(self.images_path) - 1

    def __getitem__(self, item):
        original_img = sitk.ReadImage(self.images_path[item])
        # PILImage->ndarray
        img_array = sitk.GetArrayFromImage(original_img)
        # ndarray->tensor
        # img中包含同属于一类的很多图像
        img = torch.from_numpy(img_array)

        if len(img.shape) == 4:
            img = img[:, :, :, 0]

        if self.transform is not None:
            img = self.transform(img)

        label = self.images_class[item]

        return img, label

    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels
