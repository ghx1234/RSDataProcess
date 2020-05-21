# coding=utf-8
import torch.utils.data as data
import os
import numpy as np
import PIL.Image as Image
import gdal
import torch
# data.Dataset:
# 所有子类应该override__len__和__getitem__，前者提供了数据集的大小，后者支持整数索引，范围从0到len(self)


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
        img_x = nomlize(np.transpose(gdal.Open(x_path).ReadAsArray(), (1, 2, 0)).astype(np.float32))
        if os.path.exists(y_path):
            img_y = gdal.Open(y_path).ReadAsArray().astype(np.int32) #onehot
            if self.target_transform is not None:
                img_y = torch.FloatTensor(img_y)
        else:
            img_y = 1
        if self.transform is not None:
            img_x = self.transform(img_x)

        return img_x, img_y  # 返回的是图片

    def __len__(self):
        return len(self.imgs)  # 400,list[i]有两个元素，[img,mask]
