# 导入必要的包
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# 配置训练环境和超参数

# 方案二：使用“device”，后续对要使用GPU的变量用.to(device)即可
device = torch.device("mps")

## 配置其他超参数，如batch_size, num_workers, learning rate, 以及总的epochs
batch_size = 256
num_workers = 0   # 对于Windows用户，这里应设置为0，否则会出现多线程错误
lr = 1e-4
epochs = 20

# 首先设置数据变换
from torchvision import transforms

image_size = 28
data_transform = transforms.Compose([
    # transforms.ToPILImage(),
     # 这一步取决于后续的数据读取方式，如果使用内置数据集读取方式则不需要
    transforms.Resize(image_size),
    transforms.ToTensor()
])

## 读取方式一：使用torchvision自带数据集，下载可能需要一段时间
from torchvision import datasets
# 由于网络问题，只能从github上下载下来
train_data = datasets.FashionMNIST(root='./', train=True, download=True, transform=data_transform)
test_data = datasets.FashionMNIST(root='./', train=False, download=True, transform=data_transform)

# 在构建训练和测试数据集完成后，需要定义DataLoader类，以便在训练和测试时加载数据
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# 数据可视化操作，主要是验证我们读入的数据是否正确
import matplotlib.pyplot as plt
image, label = next(iter(train_loader))
print(image.shape, label.shape)
plt.imshow(image[0][0], cmap="gray")