import torch
import torch.nn as nn
import os, sys
import gdal
import numpy as np
from torchvision.transforms import transforms as T
import torch.nn.functional as F
import argparse  # argparse模块的作用是用于解析命令行参数，例如python parseTest.py input.txt --port=8080
from torch import optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
sys.path.append(r'D:\\project\\pyproject\\water\\cnn')
# from model import MyCNN
from dataset import Dataset
# import loss
import FCN

# 是否使用current cuda device or torch.device('cuda:0')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

x_transform = T.Compose([
    T.ToTensor(),
    # 标准化至[-1,1],规定均值和标准差
    # T.Normalize([0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5])  # torchvision.transforms.Normalize(mean, std, inplace=False)
])
# mask只需要转换为tensor
y_transform = T.ToTensor()


def accuracy(input, target,):
    # input: (n, c, h, w), target: (n, h, w)
    n, c, h, w = input.size()
    input = torch.argmax(input, dim=1)
    right_number = torch.sum(input == target).float()
    right_number = right_number.cpu()
    return right_number/(h*w)


def cross_entropy2d(input, target, weight=None, size_average=True):
    # input: (n, c, h, w), target: (n, h, w)
    n, c, h, w = input.size()
    # log_p: (n, c, h, w)
    log_p = F.log_softmax(input, dim=1)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous()
    log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
    log_p = log_p.view(-1, c)
    # target: (n*h*w,)
    mask = target >= 0
    target = target[mask]
    loss = F.nll_loss(log_p, target, weight=weight, reduction='sum')
    if size_average:
        loss /= mask.data.sum()
    return loss


def train_model(model, criterion, optimizer, dataload, valdataloader, num_epochs=2):

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        dataset_size = len(dataload.dataset)
        epoch_loss = 0
        step = 0  # minibatch数
        val_loss = 0
        val_epoch_acc = 0
        val_step = 0
        for x, y in dataload:  # 分100次遍历数据集，每次遍历batch_size=4
            optimizer.zero_grad()  # 每次minibatch都要将梯度(dw,db,...)清零
            inputs = x.to(device)
            labels = y.to(device)
            outputs = model(inputs)# 前向传播

            loss = cross_entropy2d(outputs, torch.squeeze(labels, dim=1).long())  # 计算损失
            loss.backward()  # 梯度下降,计算出梯度
            optimizer.step()  # 更新参数一次：所有的优化器Optimizer都实现了step()方法来对所有的参数进行更新
            epoch_loss += loss.item()
            step += 1
            print("%d/%d,train_loss:%0.3f" % (step, dataset_size // dataload.batch_size, loss.item()))
        for x, y in valdataloader:
            inputs = x.to(device)
            labels = y.to(device)
            outputs = model(inputs)# 前向传播

            val_acc = accuracy(outputs, torch.squeeze(labels, dim=1).long())
            loss = cross_entropy2d(outputs, torch.squeeze(labels, dim=1).long())  # 计算损失

            val_epoch_acc += float(val_acc.numpy())
            val_loss += loss.item()
            val_step += 1

        print("epoch %d loss:%0.3f  val_loss:%0.3f, val_acc:%0.3f" % (epoch, epoch_loss, val_loss, (val_acc/val_step)))
    torch.save(model.state_dict(), os.path.join(args.weight, 'weights_%d.pth' % epoch))  # 返回模型的所有内容
    return model


# 训练模型
def train():
    # model = MyCNN(4, 512, 512).to(device)
    model = FCN.UNet(2).to(device)
    # batch_size = 1
    batch_size = args.batch_size
    epoch = args.epoch

    # 损失函数
    criterion = 1
    # 梯度下降
    optimizer = optim.Adam(model.parameters(), lr=args.lr)  # model.parameters():Returns an iterator over module parameters
    # 加载数据集
    dataset = Dataset(args.imgpath, args.maskpath, transform=x_transform, target_transform=y_transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    validdataset = Dataset(args.valpath, args.valmaskpath, transform=x_transform, target_transform=y_transform)
    valdataloader = DataLoader(validdataset , batch_size=1, shuffle=True)
    # DataLoader:该接口主要用来将自定义的数据读取接口的输出或者PyTorch已有的数据读取接口的输入按照batch size封装成Tensor
    # shuffle:每个epoch将数据打乱，这里epoch=10。一般在训练数据中会采用
    # num_workers：表示通过多个进程来导入数据，可以加快数据导入速度
    train_model(model, criterion, optimizer, dataloader,valdataloader, num_epochs=epoch)


if __name__ == '__main__':
    # 参数解析
    # train()
    # test()
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.batch_size = 2
    args.lr = 0.001
    args.epoch = 200
    args.weight = r'D:\project\water\gf\MyCNN'
    args.imgpath = r'D:\project\water\gf\MyCNN\c'
    args.maskpath = r'D:\project\water\gf\MyCNN\s'
    args.valpath = r'D:\project\water\gf\MyCNN\t'
    args.valmaskpath = r'D:\project\water\gf\MyCNN\s'
    train()
    # parser = argparse.ArgumentParser()  # 创建一个ArgumentParser对象
    # parser.add_argument('action', type=str, help='train or test')  # 添加参数
    # parser.add_argument('--batch_size', type=int, default=4)
    # parser.add_argument('--lr', type=float, default=0.001)
    # parser.add_argument('--epoch', type=int, default=200)
    # parser.add_argument('--weight', type=str, help='the path of the mode weight file')
    # parser.add_argument('--imgpath', type=str, help='the path of the mode img path file')
    # parser.add_argument('--maskpath', type=str, help='the path of the mode img path file')
    # args = parser.parse_args()
    # if args.action == 'train':
    #     train()
    # elif args.action == 'test':
    #     test()
