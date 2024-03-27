## CNN：Pytorch图像预处理ImageFolder解析

---
### ImageFolder在干什么

当新手第一次上手 MNIST 或者 FashionMNIST 等数据集的时候，会发现这些数据集已经帮我们编辑好了他的数据，标签，等各种方法，拿到数据立刻就可以进行数据探索，载入dataloader。但是自己的custom数据集如何进行这样的处理。就是使用ImageFolder这个工具接口。

`torchvision.datasets.ImageFolder` 是 PyTorch 中用于处理图像数据集的一个类。它主要用于加载具有以下结构的数据集：
```
root/dog/xxx.png
root/dog/xxy.png
root/dog/xxz.png
...
root/cat/123.png
root/cat/nsdf3.png
root/cat/asd932_.png
...
```
其中，root 是数据集的根目录，dog 和 cat 是类别标签，`xxx.png` 等是相应类别的图像文件。这种数据集结构将不同类别的图像分别放置在不同的子目录下。

`ImageFolder` 类会自动地将这种目录结构的图像数据加载并组织成 PyTorch 的 `Dataset` 对象。当创建了一个 `ImageFolder` 对象后，就可以通过索引的方式来获取每个图像的数据和对应的标签。

使用 ImageFolder 类的主要步骤如下：
1. 导入 `torchvision.datasets.ImageFolder` 类：
```python
from torchvision.datasets import ImageFolder
```
2. 创建 `ImageFolder` 对象，并指定数据集的根目录：
```python
dataset = ImageFolder(root='path/to/dataset')
```
3. 然后就可以通过索引访问数据集中的样本：
```python
image, label = dataset[0]  # 获取第一个样本的图像数据和标签
```
4. 接着就可以使用 `torch.utils.data.DataLoader` 来创建一个数据加载器，用于批量加载数据：
```python
from torch.utils.data import DataLoader

dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
```
这样，就可以通过 `dataloader` 来批量加载数据，用于训练模型。

使用这个接口，就可以轻松的处理我们自己的图像文件，方便地加载图像数据集，不需要编写额外的代码来处理文件读取和标签分配等操作。只需将图像数据按照类别存放在不同的文件夹中，ImageFolder 就能自动识别并加载数据，自动为加载的图像数据分配标签，标签的赋值是根据文件夹的名称来实现的。这使得加载数据时不需要手动为每个图像指定标签，减少了人为出错的可能性。

可以与 torchvision.transforms 模块中提供的图像变换函数结合使用，方便对图像进行预处理、数据增强等操作。这使得数据预处理过程更加灵活和高效。如代码所示：就是结合了transform的用法。使用 ImageFolder 处理过后就可以进行如`class_names = train_dataset.classes`这样方便的操作了。

```python
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

# 定义数据预处理的转换器
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图像大小
    transforms.ToTensor(),           # 将图像转换为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
])

# 加载图像数据集
train_dataset = datasets.ImageFolder(root='train_data', transform=transform)

# 创建 DataLoader 实例
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

# 遍历 DataLoader 加载数据
for batch_idx, (data, target) in enumerate(train_loader):
    # 在这里进行模型训练等操作
    print(f'Batch {batch_idx}, Data shape: {data.shape}, Target shape: {target.shape}')

```

### 手动Implement自己的ImageFolder，以此解构这个类接口

虽然我们不需要手写背后的代码，但是如果明白了背后的工作原理，有助于我们更好的理解我们使用的工具，全都是来源于第一性原理。

这里会尝试在`torch.utils.data.Dataset`的基础上构建自己的ImageFolder类。因为`torch.utils.data.Dataset`是 PyTorch 中定义数据集的抽象基类，所有自定义的数据集都应该继承自这个类，并且实现 `__len__()` 和 `__getitem__()` 方法。

首先，加载所有的库。
```python
import os
import pathlib
import torch

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from typing import Tuple, Dict, List
```

其次，图片数据文件夹的定义。
```python
# setup train and testing paths
train_dir = image_path / "train"
test_dir = image_path / "test"
```

然后我们虽然可以通过如下一行，找到所有的类。但是为了在后面定义类的时候方便，所以定义一个函数。

```python
class_names_found = sorted([entry.name for entry in list(os.scandir(train_dir))])

# make function to find classes in target directory
def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
    """ Find the class folder names in a target directory.
    Assumes target directory is in standard image classification format.
    Returns:
    Tuple(list_of_class_names, dict(class_name: index))"""
    # 1, get the class names by scanning the directory
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    # 2, raise an error if class names not found
    if not classes:
        raise FileNotFoundError(f"Couldn't find any classes in {directory}.")
    # 3, create a directory of index labels
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx

find_classes(train_dir)
```

重新定义 ImageFolderCustom 的流程：
1. 初始化一个torch.utils.data.Dataset的子类
2. 初始化这个子类的参数：目标文件夹，transform，之类的
3. 创建几个初始化属性：paths，transform，classes，class_to_idx
4. 创建一个function用来载入图像，使用PIL或者torchvision.io库
5. 重写父类的__len__方法
6. 重写父类的__getitem__方法

```python
# write a custom dataset class
from torch.utils.data import Dataset

# 1, subclass torch.utils.data.Dataset
class ImageFolderCustom(Dataset):
    # 2, initialize with a targ_dir and transform (optional) parameter
    def __init__(self, targ_dir, transform=None):
        # 3, create class attributes
        # get all image paths
        self.paths = list(pathlib.Path(targ_dir).glob("*/*.jpg"))
        # set up transform
        self.transform = transform
        # create classes and class_to_idx attributes
        self.classes, self.class_to_idx = find_classes(targ_dir)

    # 4, create function to load images
    def load_image(self, index):
        image_path = self.paths[index]
        return Image.open(image_path)

    # 5, overwrite the __len__() method
    def __len__(self):
        return len(self.paths)

    # 6, overwrite the __getitem__() method
    def __getitem__(self, index):
        img = self.load_image(index)
        class_name = self.paths[index].parent.name
        class_idx = self.class_to_idx[class_name]
        # transform if necessary 
        if self.transform:
            return self.transform(img), class_idx
        else:
            return img, class_idx
```

然后就可以使用这个类了，下面是尝试使用的代码。

```python
# Augment train data
train_transforms = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor()
])
# Don't augment test data, only reshape
test_transforms = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])
# try to use the custom class
train_data_custom = ImageFolderCustom(targ_dir=train_dir,
                                      transform=train_transforms)
test_data_custom = ImageFolderCustom(targ_dir=test_dir,
                                     transform=test_transforms)

len(train_data_custom), len(test_data_custom)
train_data_custom.classes, train_data_custom.class_to_idx
```
