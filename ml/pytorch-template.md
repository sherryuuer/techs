## Pytorch构架的模板

---
### pytorch数据和模型框架模板

在Github上有很复杂的模板，程序很干净，模块区分很清晰，但是一开始很难上手，不利于初学者理解，下面的模板，比较适合理解和学习。

PS：一想到其实很多的学习内容都可以整理为模板就打了一个激灵，但是要想活学活用还是需要在理解了的基础上。


```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import random
import numpy as np
import matplotlib.pyplot as plt

# from tqdm import tqdm  # 普通命令行执行的时候
from tqdm.notebook import tqdm  # jupyter笔记本执行的时候

# 检查是否可以使用gpu，这里是cuda，但是mac的情况下是mps
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 设置seed
def fix_seed(seed):
    # random
    random.seed(seed)
    # numpy
    np.random.seed(seed)
    # pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed = 42
fix_seed(seed)

# data_loader的seed设置
def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


# Data preprocessing ----------------------------------------------------------

# dataset类
class Mydataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        feature = self.X[index]
        label = self.y[index]
        # preprocessing -----

        # --------------------
        return feature, label

train_dataset = Mydataset(train_X, train_y)
test_dataset = Mydataset(test_X, test_y)


# dataloader
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=16,
                                           shuffle=True,
                                           num_workers=2,  # 高速化
                                           pin_memory=True,  # 高速化
                                           worker_init_fn=worker_init_fn
                                           )
test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size=16,
                                          shuffle=False,
                                          num_workers=2,
                                          pin_memory=True,
                                          worker_init_fn=worker_init_fn
                                          )


# Modeling --------------------------------------------------------------------

# model类
class Mymodel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Sequential(nn.Conv2d(3, 16, 3, 2, 1), # in_channels, out_channels, kernel_size, strides, padding                         
                                         nn.BatchNorm2d(16), # 特征图归一化，有助于减少内部协变量偏移
                                         nn.ReLU())
        self.conv2 = torch.nn.Sequential(nn.Conv2d(16, 64, 3, 2, 1),
                                         nn.BatchNorm2d(64),
                                         nn.ReLU())

        self.fc1 = nn.Linear(2 * 2 * 64, 100) # 全联接层
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(100, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1) # 将张量 x 的形状调整为 (batch_size, -1)，也就是将向量展平
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# 设置模型，损失函数，优化函数
model = Mymodel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)


# 模型训练函数
def train_model(model, train_loader, test_loader):
    # Train loop ----------------------------
    model.train()  # 设定为训练模式
    train_batch_loss = []
    for data, label in train_loader:
        # 将数据送往device（gpu）
        data, label = data.to(device), label.to(device)
        # 1. 梯度重置为零
        optimizer.zero_grad()
        # 2. 正向传播
        output = model(data)
        # 3. loss计算
        loss = criterion(output, label)
        # 4. 误差反向传播
        loss.backward()
        # 5. 更新权重
        optimizer.step()
        # train_loss添加进列表
        train_batch_loss.append(loss.item())

    # Test(val) loop ----------------------------
    model.eval()  # 设定为推理模式
    test_batch_loss = []
    with torch.no_grad():  # 不再计算梯度
        for data, label in test_loader:
            data, label = data.to(device), label.to(device)
            output = model(data)
            loss = criterion(output, label)
            test_batch_loss.append(loss.item())

    return model, np.mean(train_batch_loss), np.mean(test_batch_loss)


# 训练
epoch = 100
train_loss = []
test_loss = []

for epoch in tqdm(range(epoch)):
    model, train_l, test_l = train_model(model)
    train_loss.append(train_l)
    test_loss.append(test_loss)


# 损失计算的可视化
plt.plot(train_loss, label='train_loss')
plt.plot(test_loss, label='test_loss')
plt.legend()

# Evaluation ----------------------------------------------------------------

# 使用训练好的模型，进行推理，取得推理结果和正确标签列表，方便后续进行打分
def retrieve_result(model, dataloader):
    model.eval()
    preds = []
    labels = []
    # Retreive prediction and labels
    with torch.no_grad():
        for data, label in dataloader:
            data, label = data.to(device), label.to(device)
            output = model(data)
            # Collect data
            preds.append(output)
            labels.append(label)
    # Flatten
    preds = torch.cat(preds, axis=0)
    labels = torch.cat(labels, axis=0)
    # Returns as numpy (CPU環境の場合は不要)
    preds = preds.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    return preds, labels


preds, labels = retrieve_result(model, test_loader)


# Other ----------------------------------------------------------------------

# 模型保存路径
path_saved_model = "./saved_model"

# 模型保存，这里只保留参数
torch.save(model.state_dict(), path_saved_model)
# 模型载入
model = Mymodel()
model.load_state_dict(torch.load(path_saved_model))


# Model summary
from torchsummary import summary
model = model().to(device)
summary(model, input_size=(1, 50, 50))
```

什么是协变量偏移？

协变量偏移（Covariate Shift）是指在训练过程中，由于输入数据分布的变化而导致模型在不同的训练批次之间需要重新适应的情况。

在深度学习中，由于网络参数的更新导致模型的输出分布发生变化，因此输入数据的分布也会相应地发生变化。如果训练数据和测试数据的分布不同，模型在测试数据上的性能可能会受到影响，这就是协变量偏移的一种情况。

协变量偏移的存在可能会导致模型在测试数据上的性能下降，因为模型在训练数据上学到的特征可能无法泛化到测试数据上。批量归一化（Batch Normalization）就是一种用来减少协变量偏移的方法，它通过在每个批次的特征之间引入归一化来确保输入数据的分布稳定，从而加速模型的训练并提高其泛化性能。

### 数据增强和数据预处理模板

以下是一个简单的 PyTorch 数据增强模板，使用了 torchvision.transforms 模块中的一些常用的数据增强方法：

```python
import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

# 定义数据增强的转换
transform = transforms.Compose([
    transforms.CenterCrop(224),         # 从中心开始裁剪
    transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转，概率为p
    transforms.RandomVerticalFlip(p=0.5), # 随机垂直翻转，概率为p
    transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1), # 亮度，对比度，饱和度，色相
    transforms.RandomGrayscale(p=0.01), # 概率转换成灰度率，三通道就是R=G=B
    transforms.RandomRotation(10),       # 随机旋转（-10~10度之间）
    transforms.ToTensor(),               # 转换为张量
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 标准化
])

# 加载 CIFAR-10 数据集
train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)

# 定义数据加载器
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 示例用法：
for images, labels in train_loader:
    # 在这里进行模型训练
    pass

for images, labels in test_loader:
    # 在这里进行模型测试
    pass
```

数据预处理也是同样的使用Compose进行变换，比如：

```python
# 定义数据预处理的转换
transform = transforms.Compose([
    transforms.Resize((32, 32)),            # 调整图像大小为 32x32
    transforms.ToTensor(),                   # 转换为张量
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 标准化
])
```

这个模板中使用了以下常见的数据预处理方法：

- `transforms.Resize(size)`：调整图像大小为指定的大小，这里设置为 32x32 像素。
- `transforms.ToTensor()`：将图像转换为张量格式。
- `transforms.Normalize(mean, std)`：对图像进行标准化，`mean` 和 `std` 是图像的均值和标准差，用于将图像像素值缩放到一定范围。

### datasets的ImageFolder模块

PyTorch 中的 `ImageFolder` 模块是用于加载图像数据集的一个方便的工具，特别适用于文件夹结构组织的数据集，其中每个子文件夹代表一个类别，每个子文件夹中包含了对应类别的图像。

这个模块位于 `torchvision.datasets` 包中，通常与 `torchvision.transforms` 结合使用，以实现数据加载和预处理的功能。

下面是一个使用 `ImageFolder` 加载数据集的模板：

```python
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# 定义数据预处理的转换
transform = transforms.Compose([
    transforms.Resize((224, 224)),           # 调整图像大小为 224x224
    transforms.ToTensor(),                   # 转换为张量
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 标准化
])

# 加载 ImageFolder 数据集
train_dataset = ImageFolder(root='./data/train', transform=transform)
test_dataset = ImageFolder(root='./data/test', transform=transform)

# 定义数据加载器
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 示例用法：
for images, labels in train_loader:
    # 在这里进行模型训练
    pass

for images, labels in test_loader:
    # 在这里进行模型测试
    pass
```

在这个模板中，我们首先定义了数据预处理的转换，包括将图像调整为 224x224 大小、转换为张量并进行标准化。然后，我们使用 `ImageFolder` 分别加载训练集和测试集，其中 `root` 参数指定了数据集的根目录，模块会自动根据文件夹结构解析每个类别的图像。最后，我们使用 `DataLoader` 定义了数据加载器，以便进行批量加载和处理。

`ImageFolder` 的使用非常简单，而且适用于很多图像分类任务中，特别是数据集以文件夹结构组织的情况。
