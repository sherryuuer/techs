## 深入卷积深度神经网络构架，空间正规化，防过拟合和跳跃连接

---
### 1 - 再谈一下CNN的构架

卷积神经网络（CNN）在图像处理中的作用和原理，主要是捕获空间中的向量位置。

通过卷积操作，我们将输入空间中的二维矩阵转换到特征空间，但同时保留了输入的二维形式。这样，网络可以捕获图像中仅出现在部分区域的上下文信息，而这些信息在全连接层中可能会丢失。

更加直观地说，CNN能够识别图像中的边缘、角落、圆等模式。从另一个角度来看，CNN可以被视为局部连接的神经网络，因为特征图的每个像素只受到输入的局部区域影响，而不是整个图像的影响。最后将这些局部的信息组合在一起就是一张完整的特征图了。

**重要的注意事项**包括：
- 卷积仍然是一个线性操作。它是对图像进行一行一行的操作而已。这个地方经常成为误区。
- 卷积核中的权重是可训练的，并且通过输入共享。对，在CNN中可以训练的权重就是filter。一样是通过反向传播。
- 每个点积运算给出了相似性（similarity）的概念。
- 卷积层可以在任意数量的维度上执行。
- 我们滑动图像的轴定义了卷积的维度。对于图像来说，这是一个二维卷积。但我们仍然可以在具有某种局部结构的一维序列上应用卷积。这也就是说在处理文字等问题的过程中，也是可以进行卷积操作提取特征的。

```python
import torch

def conv2d(image, kernel):
    H, W = list(image.size())
    M, N = list(kernel.size())  # 假设为 3

    out = torch.zeros(H - M + 1, W - N + 1, dtype=torch.float32)
    for i in range(H - M + 1):
        for j in range(W - N + 1):
            out[i, j] = torch.sum(image[i:i + M, j:j + N] * kernel)
    return out
```
- 这段代码实现的是一个具体的图像的卷积操作。
- 这里因为预设卷积核大小为 3，所以输出就是图像和卷积核边长的差加一。于是就可以计算出output的尺寸了。
- 通过两次嵌套循环计算了行和列的卷积结果。最后输出结果。

**关于通道的概念 Channel 对我的陷阱**：在一开始学习的时候，我理解的通道是颜色的通道RGB，这没有错误，但是在卷积后的特征图会出现很多层，他们也叫做通道。但是这是两种不同的概念。前者是实际世界中的颜色通道，后者是特征通道。另外我通过实际调查，发现其实颜色的世界还有各种不同的通道划分，比如CMYK之类的，但是RGB因为已经可以满足CNN的需要了，但是不要觉得这个世界只有RGB就好。限制自己的思想边界总是不好的。

### 2 - 从很多角度上来看其实CNN的过程和基本的NN模型差别不大

如题，在CNN中，核的具体数值和bias就是要学习的权重，将它当成一个神经元操作，然后进行激活函数的线形变换，在CNN中一般使用ReLU函数进行，想象一下将一张特征图进行非线形变换后，它变的高低起伏的过程，就很美好。

最后将每一个这样的过程堆叠在一起，就是一层隐藏层。每一个隐藏层就是一组特征图（feature map）的堆叠。然后将前一层的特征图作为下一次处理的输入，继续进行卷积操作。所以每一层卷积的输出的`out_channels`的大小就是核的数量，同时也是下一层的输入层数`in_channels`。

如果理解了整个过程，那么除了计算过程不同，其他的构架基本相同。

```python
import torch
import torch.nn as nn

input_img = torch.rand(1, 3, 7, 7) # 1 is batch size ignore it now
layer = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=2, padding=1)
out = layer(input_img)
print(out.shape) 

# output: torch.Size([1, 6, 4, 4])
```

### 3 - 那么池化的过程是在做什么

池化层不会进行权重学习了，它更多的是一种**对特征图进行下采样**。

池化操作是一种用于降低特征图尺寸的操作，通常应用在卷积层之后。在每个矩形的2x2区域内，池化操作会保留区域中的最大元素，并将其作为新的特征图中的一个值。这样可以将原始输入图像的尺寸减半，从而降低计算量和参数数量。

增加不变性的角度：池化操作可以增加对图像中微小空间变化的不变性。例如，如果两个图像具有轻微的平移，那么经过池化操作后的特征图将是相同的。这种不变性对于图像分类任务很有用，因为我们希望网络能够识别物体的关键特征而不受其位置变化的影响。

逐渐降低分辨率：在CNN的前向传播过程中，随着网络的深度增加，我们希望逐渐减小输入图像的分辨率。这是因为深层特征应该具有更广泛的感受野，即对整个图像更加敏感。通过池化操作逐渐减小输入图像的分辨率，可以使得深层特征具有更大的感受野，从而更好地捕获图像中的整体信息。想象一下你眯着眼睛看世界的感受，即使如此，你的世界变得模糊了，但是你依然可以辨别物体，这是一种鲁棒性的表现（我是这么理解的）。

抽象特征表示：池化操作可以使学习到的特征更加抽象化。通过保留区域中的最大值或平均值，池化操作能够提取图像的重要特征并抑制不重要的细节，从而使得特征更具有泛化能力。

```python
import torch
import torch.nn as nn

input_img = torch.rand(1, 3, 8, 8) # 1 is batch size too
layer = nn.MaxPool2d(kernel_size=2, stride=2)
out = layer(input_img)
print(out.shape) 

# torch.Size([1, 3, 4, 4])
```
### 4 - nn.ReLU 和 nn.functional.relu 的区别

在 PyTorch 中，`nn.ReLU` 和 `nn.functional.relu` 都是用于实现 ReLU（修正线性单元）激活函数的方法，但它们有一些细微的区别：

1. **`nn.ReLU`：**
   - `nn.ReLU` 是 `torch.nn` 模块中的一个类，它继承自 `torch.nn.Module`。
   - 可以通过创建 `nn.ReLU` 类的实例来使用 ReLU 激活函数。
   - `nn.ReLU` 类的实例是一个可调用的对象，可以直接作为神经网络模型的一部分。
   - `nn.ReLU` 对象会自动管理内部的状态和参数，并且可以与其他 `torch.nn.Module` 实例组合使用。

```python
import torch
import torch.nn as nn

# 使用 nn.ReLU 类创建 ReLU 激活函数实例
relu = nn.ReLU()

# 将 ReLU 激活函数实例作为神经网络模型的一部分
model = nn.Sequential(
    nn.Linear(10, 20),
    relu
)
```

2. **`nn.functional.relu`：**
   - `nn.functional.relu` 是 `torch.nn.functional` 模块中的一个函数。
   - 可以直接调用 `nn.functional.relu` 函数来对张量应用 ReLU 激活函数。
   - `nn.functional.relu` 是一个普通的函数，不会自动管理内部的状态和参数，也不能作为神经网络模型的一部分。

```python
import torch
import torch.nn.functional as F

# 使用 nn.functional.relu 函数应用 ReLU 激活函数
x = torch.randn(10)
output = F.relu(x)
```

总的来说，`nn.ReLU` 是一个类，用于创建 ReLU 激活函数的实例，并且可以作为神经网络模型的一部分，而 `nn.functional.relu` 是一个函数，用于对张量应用 ReLU 激活函数，它不会维护内部状态，并且不能作为神经网络模型的一部分。两者的功能相同，只是使用方式有所不同。

### 5 - 批正则化（Batch Normalization，简称BN）

批正则化（Batch Normalization，简称BN）是一种用于提高深度神经网络训练稳定性和加速收敛的技术。它通过对每一层的输入进行规范化处理，使得每一层的输出具有稳定的分布。

批正则化能够加速深度神经网络的训练速度，并解决梯度消失问题。每个输入 mini-batch 计算不同的统计信息，因此引入了一种正则化效果，有助于减少过拟合风险。批正则化改善了网络中的梯度流动，减少了梯度对参数规模或初始值的依赖性，使得可以使用更高的学习率。

但是当批量大小较小时，批正则化可能会导致对批量统计量的不准确估计，增加了模型的误差。这在某些任务中，如图像分割中，可能会成为问题。虽然理论上批正则化可以使用饱和非线性（饱和非线性指的是在输入达到一定阈值之后，函数的输出停止增长或减少的现象。在神经网络中，常见的饱和非线性函数包括 sigmoid 函数和 tanh 函数），但实践中通常仍然使用 ReLU 激活函数。

在处理中，空间维度以及图像批次均被平均。通过这种方式，我们将特征集中在一个紧凑的**类高斯空间**中，这通常是有益的。类高斯空间（Gaussian Pyramid）通常被用于构建多尺度图像金字塔，以便在不同尺度下检测和分析图像中的特征。

> 类高斯空间的主要意义包括：
> 1. **多尺度特征提取：** 类高斯空间通过在不同尺度下对图像进行平滑操作，生成具有不同分辨率的图像，从而使得网络可以在多个尺度下提取特征。这有助于网络检测和识别不同尺度下的目标和特征。
> 2. **尺度不变性：** 类高斯空间可以提供尺度不变性，使得网络能够识别不同尺度下的相同对象或特征。通过构建多尺度图像金字塔，网络可以在不同尺度下对图像进行分析，从而提高了网络对尺度变化的适应能力。
> 3. **特征融合：** 类高斯空间生成的多尺度图像可以用于进行特征融合，从而获得更加丰富和全面的特征表示。通过在不同尺度下提取特征并进行融合，可以使得网络能够捕获到更多的图像信息，提高网络的性能和泛化能力。
> 总的来说，类高斯空间在CNN中的意义是通过构建多尺度图像金字塔，提供了多尺度特征提取、尺度不变性和特征融合等功能，从而帮助网络更好地理解和分析图像，并提高网络在不同尺度下的性能表现。

这是一段空间正则化的代码，是对公式的重构，从中可以看到对一个channel进行正则化的过程。

```python
import torch

# Gamma and beta are provided as 1d tensors. 

def batchnorm(X, gamma, beta):
    # 提取维度 extract the dimensions
    N, C, H, W = list(X.size())
    # 小批的均值计算 mini-batch mean 在C维度上
    mean = torch.mean(X, dim=(0, 2, 3))
    # 小批的方差计算 mini-batch mean
    variance = torch.mean((X - mean.reshape((1, C, 1, 1))) ** 2, dim=(0, 2, 3))
    # 正则化 normalize
    X_hat = (X - mean.reshape((1, C, 1, 1))) * 1.0 / torch.sqrt(variance.reshape((1, C, 1, 1)))
    # 缩放 scale 和平移 shift
    out = gamma.reshape((1, C, 1, 1)) * X_hat + beta.reshape((1, C, 1, 1))

    return out
```

方差计算的那一行中计算了小批量输入数据的方差。

1. `X` 是一个包含小批量输入数据的张量，通常是一个四维张量，表示为 `(N, C, H, W)`，其中：
   - `N` 是批量大小（batch size），
   - `C` 是通道数（channels），
   - `H` 是输入数据的高度（height），
   - `W` 是输入数据的宽度（width）。

2. `mean` 是对输入数据的均值（mean）进行计算得到的一个张量，它的形状为 `(C,)`，表示每个通道上的均值。

3. `mean.reshape((1, C, 1, 1))` 是将均值张量转换为与输入数据相同的形状，以便进行计算。具体地，将形状为 `(C,)` 的均值张量转换为形状为 `(1, C, 1, 1)` 的四维张量，其中除了通道维度外的其他维度都是 1。

4. `(X - mean.reshape((1, C, 1, 1)))` 是对输入数据与均值进行逐元素相减，这样可以得到每个通道上的偏差（即每个像素值与对应通道均值的差值）。

5. `(X - mean.reshape((1, C, 1, 1))) ** 2` 是对每个偏差值进行平方操作，得到每个通道上的偏差的平方。

6. `torch.mean(..., dim=(0, 2, 3))` 是对每个通道上的偏差的平方进行平均操作，计算得到每个通道上的方差（variance）。具体地，`dim=(0, 2, 3)` 表示在批量维度、高度维度和宽度维度上进行求平均操作，得到每个通道上的方差。

最后两步代码的作用是对规范化后的输入进行缩放（scale）和平移（shift），以得到最终的批量归一化结果。

1. **缩放（Scale）：**
   - `gamma.reshape((1, C, 1, 1))` 表示将缩放参数 `gamma` 转换为与规范化后的输入相同的形状，其中 `gamma` 是一个 1 维张量，包含了每个通道上的缩放因子。
   - `* X_hat` 表示对规范化后的输入逐元素地与缩放参数相乘，这样可以对每个通道上的规范化结果进行缩放操作。

2. **平移（Shift）：**
   - `beta.reshape((1, C, 1, 1))` 表示将平移参数 `beta` 转换为与规范化后的输入相同的形状，其中 `beta` 是一个 1 维张量，包含了每个通道上的平移量。
   - `+` 操作符表示对缩放后的结果进行平移操作，这样可以对每个通道上的缩放结果进行平移，进而得到最终的批量归一化结果。

综合起来，最后两步代码的作用是对规范化后的输入进行缩放和平移操作，以得到最终的批量归一化结果。缩放参数 `gamma` 控制了规范化后的输入的尺度，而平移参数 `beta` 则控制了规范化后的输入的偏移量。

另外，gamma 和 beta 是批正则化层中的可学习参数。

1. **gamma（缩放参数）：** gamma 是用来缩放规范化后的输入的参数，它控制了规范化后的输入的尺度。通过调整 gamma 的值，可以增加或减少规范化后的输入的尺度，从而影响网络的表示能力。在训练过程中，gamma 的值会随着梯度下降的过程而被学习调整。

2. **beta（平移参数）：** beta 是用来平移规范化后的输入的参数，它控制了规范化后的输入的偏移量。通过调整 beta 的值，可以增加或减少规范化后的输入的偏移，从而影响网络的表示能力。与 gamma 类似，在训练过程中，beta 的值也会随着梯度下降的过程而被学习调整。

gamma 和 beta 的引入使得批正则化层具有更多的灵活性，能够更好地适应不同的数据分布和网络结构。通过学习得到合适的 gamma 和 beta 参数，批正则化层能够提高网络的性能和泛化能力。

这部分的代码，通过在脑中抽象变换，可以对小批量正规化有一个更深入的理解。在我看来，这种正则化技术，是在通道尺度上的一种缩放，让不同的特征图也就是每一层的输出，得到正则化处理，加速训练，稳定分布。如果将所有的特征图的每个固定位置的点，连起来，其实就是空间中的无数条线性上的正则化处理。以小见大。

### 6 - 深度模型技术Dropout

Dropout是另一种重要的深度神经网络技术，它的实质是训练各种不同的神经网络构架。这是因为，在训练过程中每次训练都会随机丢弃一些输出，也就是神经元，这会导致每次都构架都看起来不同，或者说，是若干层和神经元的重新组合进行的训练。当然在丢弃过程中是有一定的概率（通常是0.5）的。哦对了，最终每次的不同构架，我们可以叫它**视图**，也就是view。这个技术有利于减少过拟合的风险，使得最终的模型具有更好的鲁棒性。 Dropout促使网络中的每个神经元都能够独立地学习到有用的特征，避免了神经元之间的共适应（Co-adaptation）现象，提高了网络的稳定性和可靠性。

我觉得这技术很有趣，相当于将一个网络进行打碎重组，看起没有丢弃，但是在过程中，不使用某些神经元，让他们减少对彼此的互相依赖，提高每个神经元的耐性，是一种非常有意思的手段。

代码和输出示例：
```python
import torch
import torch.nn as nn

inp = torch.rand(1,8)
layer = nn.Dropout(0.5)
out1 = layer(inp)
out2 = layer(inp)
print(inp)
print(out1)
print(out2)

# output:
# tensor([[0.0159, 0.4352, 0.0863, 0.7790, 0.9235, 0.2026, 0.3964, 0.2865]])
# tensor([[0.0000, 0.8705, 0.0000, 0.0000, 0.0000, 0.0000, 0.7929, 0.0000]])
# tensor([[0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.4052, 0.0000, 0.0000]])
```

在这个例子中每个神经元都会以50%的概率被清零，但是这不意味着每次都会清零一半，比如这里，数据量只有8个有6个就在第一次丢弃操作中被清零了。

### 7 - Skip Connections 跳跃网络

这是一种为了解决梯度消失问题的方法。当年的什么什么图像识别大赛中出现了无法解决的梯度消失问题，然后这种方法才得以诞生。跳跃网络主要有两种方法，分别是基于加法和基于连接（串联）：

首先是**加法（Addition）**，它通常用于残差架构（如 ResNet），其中输入信号通过恒等映射（Identity Mapping）直接跳过一个或多个层，然后与该层的输出相加。这种方式创建了一个跳跃连接，允许输入信号继续沿着网络传播，并通过网络进行非线性变换。

具体步骤如下：
- 输入信号首先通过一个或多个层的非线性变换，例如卷积层或全连接层。
- 经过变换后的输出与输入信号进行逐元素相加。
- 最终的结果通过激活函数进行处理，例如 ReLU。

这种方法的优点是简单易于实现，而且减少了梯度消失和爆炸的问题。在训练过程中，梯度能够直接通过跳跃连接传播，从而使得更深的网络更容易优化。既然简单，那就可以直接用代码简答的实现出来：

```python
import torch
import torch.nn as nn

torch.manual_seed(42)

class SkipConnection(nn.Module):

    def __init__(self):
        super(SkipConnection, self).__init__()
        self.conv_layer1 = nn.Conv2d(3, 6, 2, stride=2, padding=2)
        self.relu = nn.ReLU(inplace=True)
        self.conv_layer2 = nn.Conv2d(6, 3, 2, stride=2, padding=2)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, input: torch.FloatTensor) -> torch.FloatTensor:
        x = self.conv_layer1(input)
        x = self.relu(x)
        x = self.conv_layer2(x)
        x = self.relu2(x)
        return x + input # 注意这里
```

然后是**连接（Concatenation）**，它通常用于密集连接架构（如 DenseNet），其中输入信号通过多个层后，与每一层的输出进行连接（串联），然后再进行进一步处理。这种连接方式可以增加网络的深度和宽度，从而使得网络更加灵活。

具体步骤如下：
- 输入信号经过一个或多个层的非线性变换。
- 经过变换后的输出与每一层的原始输入信号进行连接（串联）。
- 将连接后的结果传递给下一个层进行进一步处理。

这种方法的优点是允许网络中的每一层都直接访问之前所有层的输出，从而能够更好地捕捉输入信号中的特征。此外，连接操作也有助于减轻梯度消失问题，因为每一层都可以直接从之前的所有层中获取梯度信息。

这里需要注意的是我在学习Tensorflow的时候，定义层的时候也有一个叫做Dense层的东西，但是他们是不同的东西，只是取了密集这个相同含义的单词而已。

**TensorFlow 中的 Dense 层**是指全连接层（Fully Connected Layer），也称为稠密层。在 Dense 层中，每个神经元与前一层的所有神经元都有连接，因此输入层与输出层之间的连接是“密集”的，这也是为什么称为 Dense 层的原因。每个神经元都与上一层的所有神经元相连，每个连接都有一个权重。Dense 层在传统的前馈神经网络中经常使用，用于学习输入数据中的复杂模式。

**DenseNet 中的 Dense 连接**不同于 TensorFlow 中的 Dense 层。DenseNet 使用密集连接（Dense Connection），其基本原理是在网络的每一层之间建立密集的连接。在每一层中，每个子网络（称为 Dense Block）都与之前的所有层有连接。每个 Dense Block 中的每一层都接收来自前面所有层的特征图作为输入，并将其与当前层的输出连接在一起，然后将结果传递给下一层。这种密集连接的设计的重点是增加特征重用，促进信息的流动，减轻梯度消失问题，并可以让网络更加深层。DenseNet 主要应用于图像分类、分割和生成等任务，在一些数据集上取得了很好的效果。

因此，TensorFlow 中的 Dense 层是一种常规的全连接层，而 DenseNet 中的 Dense 连接层是一种特殊的连接方式，用于构建深度网络结构。

之后的文章会总结很重要的历史CNN框架。
