# coding=utf-8
import torch.utils.data as data
import os, random, sys
import numpy as np
import PIL.Image as Image
import gdal
import torch
import rasterio
sys.path.append('/home/zhoudengji/ghx/data/code/mycnn')
from zengqiang import zengqiang

# data.Dataset:
# 所有子类应该override__len__和__getitem__，前者提供了数据集的大小，后者支持整数索引，范围从0到len(self)


def process_lines(full_path, label_path, augment=True):
    if augment:
        fanzhuan = random.randint(0, 3) - 1
        pingyiX, pingyiY = random.randint(-100, 100), random.randint(-100, 100)
        xuanzhuan, suofang = random.randint(0, 180) - 90, random.random() * 0.4 + 0.8
        liangdu = random.random() * 0.1 + 0.95

        panduan = random.randint(0,1)

    with rasterio.open(full_path) as ds:
        x = ds.read(
            out_shape=(4, 512, 512),
            resampling=rasterio.enums.Resampling.bilinear
        )
        x = nomlize(x)
        if augment:
            x = zengqiang(x)
            x.fanzhuan(fanzhuan)
            if panduan:
                x.xuanzhuansuofang(0, suofang)
            # x.liangdu(liangdu)
            else:
                x.pingyi(pingyiX, pingyiY)
            x = x.array
        # x = np.transpose(x, (1, 2, 0))

    with rasterio.open(label_path) as ds:
        y = ds.read(
            out_shape=(1, 512, 512),
            resampling=rasterio.enums.Resampling.bilinear
        )
        # y = np.where(y > 0, 1, y)
        if augment:
            y = zengqiang(y)
            y.fanzhuan(fanzhuan)
            if panduan:
                y.xuanzhuansuofang(0, suofang)
            # y.liangdu(liangdu)
            else:
                y.pingyi(pingyiX, pingyiY)
            y = y.array
        y = np.where(y > 0, 1, y)
        y = np.squeeze(y, 0)

    return x, y


def nomlize(array):
    return np.divide(array, np.max(array))


def onehot(data, n):
    buf = np.zeros(data.shape + (n, ))
    nmsk = np.arange(data.size)*n + data.ravel()
    buf.ravel()[nmsk-1] = 1
    return buf


class Dataset(data.Dataset):
    # 创建Dataset类的实例时，就是在调用init初始化
    def __init__(self, img_path, mask_path='D', transform=None, target_transform=None):  # root表示图片路径
        names = os.listdir(img_path)
        n = len(names) // 2  # os.listdir(path)返回指定路径下的文件和文件夹列表。/是真除法,//对结果取整

        imgs = []
        for i in names:
            img = os.path.join(img_path, i)  # os.path.join(path1[,path2[,......]]):将多个路径组合后返回
            mask = os.path.join(mask_path, i)
            imgs.append([img, mask])  # append只能有一个参数，加上[]变成一个list

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        x_path, y_path = self.imgs[index]
        img_x, img_y = process_lines(x_path, y_path)  #x channl first, y no channl

        img_x = torch.FloatTensor(img_x)

        img_y = torch.FloatTensor(img_y)


        return img_x, img_y  # 返回的是图片

    def __len__(self):
        return len(self.imgs)  # 400,list[i]有两个元素，[img,mask]
