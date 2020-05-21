import torch
from dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np
import os, sys
import cv2
from torchvision.transforms import transforms as T
sys.path.append(r'D:\\project\\pyproject\\water\\cnn')
import FCN

x_transform = T.Compose([
    T.ToTensor(),
    # 标准化至[-1,1],规定均值和标准差
    # T.Normalize([0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5])  # torchvision.transforms.Normalize(mean, std, inplace=False)
])
# mask只需要转换为tensor
y_transform = T.ToTensor()

# 测试
def test():
    # model = MyCNN(4, 512, 512)
    model = FCN.UNet(2)
    model.load_state_dict(torch.load(r"D:\project\water\gf\MyCNN\weights_199.pth", map_location='cpu'))
    liver_dataset = Dataset(r"D:\project\water\gf\MyCNN\t", transform=x_transform, target_transform=y_transform)
    dataloaders = DataLoader(liver_dataset)  # batch_size默认为1
    #img_x = np.transpose(gdal.Open(r"D:\project\water\gf\MyCNN\t\10275_6763_3.tif").ReadAsArray(), (1, 2, 0)).astype(np.float32)
    #img_x = x_transform(img_x)
    model.eval()
    import matplotlib.pyplot as plt
    plt.ion()

    import matplotlib.pyplot as plt
    plt.ion()
    with torch.no_grad():
        i = 0
        for x, _ in dataloaders:
            i = i + 1
            y = model(x)
            img_y = torch.squeeze(y).numpy()
            img_y = np.argmin(img_y, 0)

            # cv2.imwrite(r'D:\project\test_images\%s.tif'%(str(i)), img_y)
            plt.imshow(img_y)
        plt.show()


if __name__ == '__main__':
    test()