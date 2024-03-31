## 从2012到2021年的三个重要的CNN网络构架

---
这里主要介绍AlexNet，VGG，InceptionNet/GoogleNet三种重要的历史CNN网络构架，虽然说是历史，但是学习他们可以深入了解CNN大发展，期待未来的更多的创造发明。

### AlexNet

AlexNet 的名字，来自于当年2012年挑战赛的参赛者科学家的名字。在比赛中它取得了巨大的成功。为深度学习奠定了基础。

深度和参数量：AlexNet 是一个相对较深的神经网络，共有 8 层神经网络（5 个卷积层，11 x 11 的卷积核，和 3 个全连接层），包含了 60 百万个参数。相较于之前的模型，AlexNet 的深度和参数量都是巨大的提升，使其可以更好地捕获图像的特征。

这个网络第一次使用ReLU，我们现在也总是使用ReLU。同时它首次引入了 Dropout 正则化技术，以减少过拟合。Dropout 在训练过程中随机丢弃一部分神经元，从而防止神经网络过度依赖某些特定的神经元，提高了网络的泛化能力。它还首次使用了最大池化。

它还是第一个使用多 GPU 进行训练的深度学习模型。作者使用了两个 NVIDIA GTX 580 GPU 来并行训练网络，大大加快了训练速度。

可见它的成功标志着深度学习在计算机视觉领域的崛起。它的架构和创新点为后续的深度学习模型奠定了基础，并激发了对更深、更复杂神经网络的研究和应用。

整个代码其实只有35行，但是每一行每一个block每一个组合都是一种全新的创造。

```python
import torch
import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 1000) -> None:
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

model = AlexNet(num_classes=10)
inp = torch.rand(1,3,128,128)
print(model(inp).shape)
# torch.Size([1, 10])
```

features 主要可以看到三组处理，最后一组使用了三次卷积加池化，在classifier 中，使用了dropout，敢想敢做的科学家太可怕了。

### VGG

VGG 是由牛津大学计算机视觉组（Visual Geometry Group）提出的一个深度卷积神经网络架构，是深度学习领域中非常重要的模型之一。VGG 网络通过堆叠简单的卷积层和池化层来构建深度网络，具有优秀的性能和可解释性，成为了许多图像分类和目标检测任务的基准模型之一。

VGG 网络的主要创新在于采用了统一的卷积层和池化层的设计。它使用了连续的 3x3 卷积核和 2x2 的最大池化操作，通过不断地堆叠这些卷积层和池化层来构建深度网络。这种简单而统一的设计使得 VGG 网络具有清晰的结构和易于理解的特点。

VGG 网络通过不断增加网络的深度来提高性能。经典的 VGG16 网络有 16 层（13 个卷积层和 3 个全连接层），而 VGG19 网络有 19 层（16 个卷积层和 3 个全连接层）。VGG 网络的深度使得它能够更好地捕获图像的特征，从而在图像分类等任务上取得优异的性能。

VGG 网络采用了较小的 3x3 卷积核，相比于更大的卷积核，小卷积核可以增加网络的深度而不增加参数量，同时提高了网络的非线性能力。这种设计使得 VGG 网络具有较强的特征提取能力。

在卷积层之后连接了几个全连接层，用于实现最终的分类任务。这些全连接层使得网络能够将从图像中提取的高级特征映射到具体的类别上。

这个构架在学习中也手动复制过，给我的感觉就是每一个核都很小，但是层数很深。很精炼的感觉。

当时我在学习第一个深度构架的时候使用了前辈们的[可视化网站](https://poloclub.github.io/)有当今各种技术的可视化解说，同时这个[Tiny VGG](https://poloclub.github.io/cnn-explainer/)网站也很好的可视化了这个过程。

这是一个简单的框架代码。在features部分的层复用度很高，所以直接用内置方法简化了代码，很精妙。

```python
import torch
import torch.nn as nn

# 定义VGG网络的卷积部分
class VGG(nn.Module):
    def __init__(self, num_classes=1000):
        super(VGG, self).__init__()
        self.features = self._make_layers([64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'])
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=3, padding=1),
                           nn.BatchNorm2d(v),
                           nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

# 创建一个VGG16的实例
vgg16 = VGG(num_classes=1000)
print(vgg16)
```

### InceptionNet/GoogleNet

由 Google Brain 团队提出的一种深度卷积神经网络架构，旨在解决深度神经网络的计算复杂性和参数数量过多的问题。它是深度学习领域中非常重要的模型之一，获得了2014年ImageNet图像识别挑战赛的冠军，并且对深度学习的发展产生了深远的影响。

InceptionNet 使用了称为 Inception 模块的基本构建块，它同时使用不同大小的卷积核和池化操作，以并行的方式捕获不同尺度的特征。Inception 模块的并行结构使得网络能够在不同尺度上进行特征提取，从而提高了模型的表达能力。

为了减少计算量和参数数量，Inception 模块引入了 1x1 的卷积核，用于在通道维度上进行特征降维和组合。1x1 卷积可以减少特征图的深度，降低了计算复杂度，同时提高了模型的非线性能力。为了缓解梯度消失的问题，InceptionNet 在网络的中间层添加了多个辅助分类器，用于辅助训练。这些辅助分类器可以在训练过程中提供额外的梯度信号，有助于加速模型的收敛和优化。

InceptionNet 在最后一层使用了**全局平均池化层**，将每个特征图的大小降为1x1，然后连接到全连接层进行分类。这种操作减少了参数数量，避免了过拟合，并且使得网络对输入图像的尺寸变化具有更强的鲁棒性。

这个网络我没构架过，是一种新颖的观点。如果之前的网络都是往深度走，这个网络就是宽度上扩展。我只能说大开眼界。准备在自己的小项目上用一下！

```python
import torch
import torch.nn as nn

class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InceptionModule, self).__init__()
        relu = nn.ReLU()
        self.branch1 = nn.Sequential(
                  nn.Conv2d(in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0),
                  relu)

        conv3_1 = nn.Conv2d(in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
        conv3_3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.branch2 = nn.Sequential(conv3_1, conv3_3,relu)

        conv5_1 = nn.Conv2d(in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
        conv5_5 = nn.Conv2d(out_channels, out_channels, kernel_size=5, stride=1, padding=2)
        self.branch3 = nn.Sequential(conv5_1,conv5_5,relu)

        max_pool_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        conv_max_1 = nn.Conv2d(in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
        self.branch4 = nn.Sequential(max_pool_1, conv_max_1,relu)

    def forward(self, input):
        output1 = self.branch1(input)
        print(output1.shape)
        output2 = self.branch2(input)
        print(output2.shape)
        output3 = self.branch3(input)
        print(output3.shape)
        output4 = self.branch4(input)
        print(output4.shape)
        return torch.cat([output1, output2, output3, output4], dim=1)

model = InceptionModule(in_channels=3,out_channels=32)
inp = torch.rand(1,3,128,128)
print(model(inp).shape)
# torch.Size([1, 32, 128, 128])
# torch.Size([1, 32, 128, 128])
# torch.Size([1, 32, 128, 128])
# torch.Size([1, 32, 128, 128])
# torch.Size([1, 128, 128, 128])
```
