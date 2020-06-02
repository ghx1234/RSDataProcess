import torch
from dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np
import os, sys
from osgeo import gdal_array
import gdal
import argparse
from torchvision.transforms import transforms as T
sys.path.append(r'/home/zhoudengji/ghx/data/code/mycnn')
from model import MyCNN


x_transform = T.Compose([
    T.ToTensor(),
    # 标准化至[-1,1],规定均值和标准差
    # T.Normalize([0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5])  # torchvision.transforms.Normalize(mean, std, inplace=False)
])
# mask只需要转换为tensor
y_transform = T.ToTensor()


def nomlize(array):
    return np.divide(array, np.max(array))
# 测试
def test():
    model = MyCNN(4, 512, 512)
    # model = models.UNet(2)
    model.load_state_dict(torch.load(args.weight, map_location='cpu'))
    model.eval()

    with torch.no_grad():
        ds = gdal.Open(args.image)
        result = gdal.GetDriverByName('GTiff').Create(args.result,
            int(np.floor(ds.RasterXSize / 512)) * 512, int(np.floor(ds.RasterYSize / 512)) * 512, 1, gdal.GDT_Byte)
        for i in range(int(np.floor(ds.RasterXSize / 512))):
            for j in range(int(np.floor(ds.RasterYSize / 512))):
                x = nomlize(ds.ReadAsArray(i * 512, j * 512, 512, 512))
                x = np.broadcast_to(x, (1, 4, 512, 512))
                y = model(torch.FloatTensor(x))
                img_y = torch.squeeze(y).numpy()
                img_y = np.argmin(img_y, 0)

                result.GetRasterBand(1).WriteArray(img_y, xoff=i*512, yoff=j*512)
        result.SetGeoTransform(ds.GetGeoTransform())
        result.SetProjection(ds.GetProjection())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, default=r'/home/zhoudengji/ghx/data/yangben/testdataset/GF1_PMS2_E116.4_N39.9_20191027_L1A0004341376-PAN2_ORTHO_PSH4_4.tif')
    parser.add_argument('--result', type=str, default=r'/home/zhoudengji/ghx/data/yangben/tesperedict/a.tif')
    parser.add_argument('--weight', type=str, default=r'/home/zhoudengji/ghx/data/code/mycnn/weight/weights_5.pth')

    args = parser.parse_args()
    test()