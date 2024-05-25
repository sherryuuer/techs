## ResNet残差网络

---
### 针对梯度消失问题的超大型深度网络

残差网络出现之前，经典的深度学习网络是VGG经典模型，通过不断增加卷积和池化层，提升了模型的精度，但是VGG的精度，在到达16层后，精度就上不去了，深度网络的发展遇到了瓶颈。理论上是越深，精度越高，但是出现了越深反而开始出现精度降低的情况了。这时候出现了残差网络。

残差网络（Residual Network，简称为ResNet）是由微软亚洲研究院的研究员何恺明（Kaiming He）等人于2015年提出的一种深度神经网络结构。它的设计目的是为了解决深度神经网络在训练过程中出现的*梯度消失和梯度爆炸*问题，从而使得训练更深的网络变得更加容易。

ResNet的核心思想是引入了残差块（Residual Block），这是一个包含*跳跃连接（skip connection）*的模块。传统的神经网络通过堆叠层来逐渐学习特征，但随着网络层数的增加，*梯度信号*可能逐渐变小，导致难以训练。残差块通过在网络中引入跳跃连接，使得神经网络可以直接学习残差（即输入与期望输出之间的差异），从而更容易地进行优化。

一个典型的ResNet残差块可以用以下形式表示:

```
-------------------------------
|        Input (X)            |
|            |                |
|            V                |
|        Conv 1x1             | ----------------\
|            |                |                  |
|         Conv 3x3            |                  |
|            |                |                  |
|        Conv 1x1             |                  |
|            V                |                  |
|           Sum               |                  |
|            |                |                  |
|        Activation           |                  |
|            V                |                  |
|        Output (X')          | <----------------/
-------------------------------
```

解释:

1. **Input (X)**: 这是残差块的输入张量。
2. **Conv 1x1**: 这是一个1x1卷积层，通常用于调整维度和通道数。
3. **Conv 3x3**: 这是一个3x3卷积层，执行主要的特征提取。
4. **Conv 1x1**: 另一个1x1卷积层，用于调整输出通道数。
5. **Sum**: 将主分支的输出与捷径分支(输入X)相加，实现残差连接。
6. **Activation**: 通常是ReLU激活函数。
7. **Output (X')**: 残差块的输出张量。

这种残差结构允许网络更容易学习恒等映射，从而缓解了深层网络的梯度消失问题。通过将输入直接与输出相加，网络只需学习残差映射,而不是完整的映射,这简化了优化过程。

ResNet的设计使得它可以轻松地训练数百层甚至上千层的深度网络，同时保持了良好的性能和梯度流通。由于其出色的性能和训练效果，ResNet已经成为深度学习领域中广泛应用的经典模型之一。

### 前置知识

ImageNet是一个超大型图片网络，使用许多许多层的神经网络进行训练，最终还是导致了梯度爆炸和梯度消失问题，因为出现了残差网络模型。

机器学习中层数不断增加，比如几百几百层，当超过50层后，RN（resnet，后面都这样简称）就会开始加一个瓶颈层，然后引入残差网络。

和之前SqueezeNet一样，RN也是使用block类将所有的残差处理打包，然后堆叠block实现多个残差处理。

### 初始化残差网络

以下的初始化`block_layer_sizes`的key是权重层weight layer（卷积+全连接层）的层数，当层数在50以下的时候，使用常规组块，当层数超过了50，则使用瓶颈组块。value部分是当为指定层的时候，使用该size。

```python
import tensorflow as tf

block_layer_sizes = {
    18: [2, 2, 2, 2],
    34: [3, 4, 6, 3],
    50: [3, 4, 6, 3],
    101: [3, 4, 23, 3],
    152: [3, 8, 36, 3],
    200: [3, 24, 36, 3]
}

class ResNetModel(object):
    # Model Initialization
    def __init__(self, min_aspect_dim, resize_dim, num_layers, output_size,
        data_format='channels_last'):
        self.min_aspect_dim = min_aspect_dim
        self.resize_dim = resize_dim
        self.filters_initial = 64 # 初始filter大小
        self.block_strides = [1, 2, 2, 2] # 步长默认为2
        self.data_format = data_format
        self.output_size = output_size
        self.block_layer_sizes = block_layer_sizes[num_layers]
        self.bottleneck = (num_layers >= 50) # 返回布尔值，指示是否使用瓶颈组块
```

### Padding图像扩充

关于**通道位置**，一般采用*NCHW的是Tensorflow的GPU处理器*，这种顺序方式比较适合GPU训练，*而在CPU上更适合NHWC的顺序*。可以在函数中引入option条件判断进行灵活应对。

这里执行padding='same'的填充方式，这部分代码的目的是计算在输入张量的高度和宽度维度上需要填充的零值数量，以确保使用SAME填充模式进行卷积操作后，输出张量的空间维度(高度和宽度)与输入张量保持一致。

具体来说:

1. `pad_total = kernel_size - 1`

这一行计算了在高度和宽度维度上需要填充的总数量。这是因为卷积核在输入张量上滑动时，会在边界处丢失一部分信息。例如，对于3x3的卷积核，它无法完全覆盖输入张量边界处的1个单元，因此需要在边界处填充1个单元，从而保证输出张量的大小不会缩小。所以对于kernel_size为3，需要在每个维度上填充1个单元，总共是`3 - 1 = 2`个单元。

2. `pad_before = pad_total // 2`和`pad_after = pad_total - pad_before`

这两行代码的目的是将总的填充量`pad_total`平均分配到输入张量的前后两侧。由于期望填充是对称的，因此我们将`pad_total`等分为两部分:`pad_before`和`pad_after`。

- `pad_before`是指在输入张量前面填充的零值数量，它取`pad_total`的下半部分(使用整数除法`//`向下取整)。
- `pad_after`是指在输入张量后面填充的零值数量，它等于`pad_total`减去`pad_before`。

这种计算方式可以确保填充是对称的，并且可以最大程度地保留输入张量的有效信息，同时确保经过卷积运算后，输出张量的空间维度与输入相同。

```python
def custom_padding(self, inputs, kernel_size):
    pad_total = kernel_size - 1
    pad_before = pad_total // 2
    pad_after = pad_total - pad_before
    if self.data_format == 'channels_first':
        padded_inputs = tf.pad(
            inputs,
            # 表示在batch和channel维度上不填充,而在height和width维度上前后各填充pad_before和pad_after个零。
            [[0, 0], [0, 0], [pad_before, pad_after], [pad_before, pad_after]]
        )
    else:
        padded_inputs = tf.pad(
            inputs,
            [[0, 0], [pad_before, pad_after], [pad_before, pad_after], [0, 0]]
        )
    return padded_inputs

# Custom convolution function w/ consistent padding
def custom_conv2d(self, inputs, filters, kernel_size, strides, name=None):
    if strides > 1:
        padding = 'valid'
        inputs = self.custom_padding(inputs, kernel_size)
    else:
        padding = 'same'
    return tf.keras.layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=self.data_format,
        name=name
        )(inputs)
```

补充：valid 填充模式的工作方式是:在输入张量的边界处不进行任何填充，卷积核只在有效的数据区域上滑动，多余的部分将被丢弃。这种方式会导致输出张量的空间维度(高度和宽度)比输入张量小。

### 内部协变量偏移

内部协变量偏移(Internal Covariate Shift)是深度神经网络在训练过程中可能出现的一种现象，它会导致网络训练变得更加困难和缓慢。下面我将详细解释什么是内部协变量偏移，并举例说明。

在深度神经网络中，每一层的输入数据都来自于前一层的输出。如果前一层的输出数据分布发生变化(例如均值和方差的变化)，那么这种变化将传递到后面的层，并影响每一层学习到的特征分布。这种层与层之间的分布变化就被称为"内部协变量偏移"。

内部协变量偏移会导致以下问题:

1. 网络需要不断调整参数来适应新的数据分布，从而降低了训练效率。
2. 网络参数可能会陷入饱和区(如ReLU的死亡区)，使得反向传播梯度变小或为零，进而导致权重无法更新。
3. 随着网络深度的增加，内部协变量偏移会越来越严重，从而阻碍网络的收敛。

为了缓解内部协变量偏移的问题，研究人员提出了一些有效的方法，其中最著名的是批归一化(Batch Normalization)。

**批归一化示例:**

批归一化在每一层的输入数据上执行以下操作:

1. 计算当前小批量数据的均值和方差。
2. 将数据归一化(减去均值，除以标准差)。
3. 对归一化后的数据进行缩放和平移(通过可学习的参数γ和β)。

对于某些层，我们可能不希望输入具有标准化分布。也许我们想要具有不同均值或方差的分布。批量归一化有两个可训练变量，γ和β，使我们能够分别改变分布的方差和均值。γ和β是可训练变量，模型将自动微调每个批量归一化层的值。

经过这些操作后，每一层的输入数据将被归一化到近似的均值为0、标准差为1的分布，从而减轻了内部协变量偏移的影响。

下面是一个简单的批归一化层实现示例(使用PyTorch伪代码):

```python
class BatchNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        # 可学习的缩放和平移参数
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        
        # 用于存储运行时均值和方差
        self.running_mean = torch.zeros(num_features)
        self.running_var = torch.ones(num_features)

    def forward(self, x):
        if self.training:
            # 计算小批量数据的均值和方差
            mean = x.mean(dim=0)
            var = x.var(dim=0)
            
            # 更新运行时均值和方差
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
            
            # 执行归一化和缩放/平移
            x_norm = (x - mean) / torch.sqrt(var + self.eps)
            output = self.gamma * x_norm + self.beta
            
        else:
            # 在推理时直接使用运行时均值和方差
            output = (x - self.running_mean) / torch.sqrt(self.running_var + self.eps)
            output = self.gamma * output + self.beta
            
        return output
```

通过使用批归一化等技术，可以有效缓解内部协变量偏移问题，从而提高网络的训练效率和收敛性能。

### 批归一化和预激活函数

在Resnet的2.0版本中，在将输入放入layer之前过了一下激活函数。传统的都是在输出之后进行激活，这里的预激活被实验证明，具有更好的效果。

1. **批归一化（Batch Normalization，BN）：**
   - BN 是一种在深度神经网络中应用的技术，旨在加速训练过程并提高模型的鲁棒性。其基本思想是在每个训练小批次的数据上，对每个输入特征进行*归一化，使其均值接近零，标准差接近一*。这有助于防止网络中的梯度消失或梯度爆炸问题，使得网络更容易收敛。BN 通常被放置在激活函数之前。

2. **预激活（Pre-Activation）：**
   - 预激活是一种与传统激活函数的连接方式不同的神经网络结构。在传统的结构中，激活函数通常紧跟在卷积层或全连接层之后，而预激活则将激活函数移动到了网络层的前面。这样，每个层的计算可以分为两步：首先进行预激活（无激活函数的加权和操作），然后应用激活函数。

相对的使用tf的函数：

```python
# Applies pre-activation to the inputs
def pre_activation(self, inputs, is_training):
    axis = 1 if self.data_format == 'channels_first' else 3
    bn_inputs = tf.keras.layers.BatchNormalization(axis=axis)(inputs, training=is_training)
    pre_activated_inputs = tf.nn.relu(bn_inputs)
    return pre_activated_inputs
```

### 跳跃链接shortcut

在 ResNet（Residual Network）中，shortcut（或称为跳跃连接）是一种通过将输入直接加到层的输出来建立捷径或直通连接的机制。这个连接通过跨越一个或多个层，将输入信号直接传递到较深层的输出。这个设计有助于解决深度神经网络训练过程中的梯度消失和梯度爆炸问题。

残差块（residual block）和恒等块（identity block）是实现深度卷积神经网络的核心组件，它们的主要区别在于输入和输出之间的连接方式，以及是否需要对输入进行变换以匹配输出的维度。

**残差块（Residual Block）**

残差块的设计理念是通过引入残差连接（skip connection）来解决深层网络训练中的梯度消失问题。残差块包括以下部分：

1. **卷积层**：一系列卷积操作，通常包括批量归一化（Batch Normalization）和ReLU激活函数。
2. **残差连接**：直接将输入添加到卷积层的输出上。

在某些情况下，输入的维度与卷积层输出的维度不匹配，这时需要使用卷积操作对输入进行变换（即使用1x1卷积）以匹配维度，然后再进行相加。

- **输入变换**（如果需要）：使用1x1卷积调整输入的维度，使其与输出维度匹配。
- **卷积操作**：通常是两层或三层卷积。
- **相加操作**：将变换后的输入与卷积输出相加。

示意图：

```
Input --> [Conv1] --> [BatchNorm1] --> [ReLU] --> [Conv2] --> [BatchNorm2] --> Add --> Output
  |                                                                                         ^
  |------------------------------------> [1x1 Conv] --> [BatchNorm] --> Add
```

**恒等块（Identity Block）**

恒等块是残差块的一种特殊情况，其中输入的维度与卷积层输出的维度相同，因此不需要对输入进行变换。恒等块也包含残差连接，但由于维度匹配，直接将输入添加到卷积输出即可。

- **卷积操作**：通常是两层或三层卷积。
- **相加操作**：直接将输入与卷积输出相加。

示意图：

```
Input --> [Conv1] --> [BatchNorm1] --> [ReLU] --> [Conv2] --> [BatchNorm2] --> Add --> Output
  |-----------------------------------------------------------------------------------> Add
```

示例代码：

以下是使用PyTorch实现残差块和恒等块的代码示例：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        
        # Adjust input dimension if necessary
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class IdentityBlock(nn.Module):
    def __init__(self, in_channels):
        super(IdentityBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += x
        out = F.relu(out)
        return out
```

在上述代码中，`ResidualBlock`类处理输入和输出维度不同时的情况，而`IdentityBlock`类用于输入和输出维度相同的情况。这两个类分别实现了残差块和恒等块。

使用残差块的这部分代码：

```python
# Returns pre-activated inputs and the shortcut
def pre_activation_with_shortcut(self, inputs, is_training, shortcut_params):
    pre_activated_inputs = self.pre_activation(inputs, is_training)
    shortcut = inputs
    shortcut_filters = shortcut_params[0]
    if shortcut_filters is not None:
        strides = shortcut_params[1]
        kernel_size = 1
        shortcut = self.custom_conv2d(pre_activated_inputs, shortcut_filters, 1, strides)      
    return pre_activated_inputs, shortcut
```

### 残差网络常规层 regular block

这个函数定义了一个带有残差连接的卷积块，典型地用于构建 ResNet 模型中的一个单元。它包含了预激活、两次卷积操作以及一个快捷连接的加法运算。通过这种结构，神经网络可以更有效地训练更深的层数，同时减轻梯度消失的问题。

```python
def regular_block(self, inputs, filters, strides, is_training, index, shortcut_filters=None):
    with tf.compat.v1.variable_scope('regular_block{}'.format(index)):
        shortcut_params = (shortcut_filters, strides)
        pre_activated1, shortcut = self.pre_activation_with_shortcut(inputs, is_training, shortcut_params)
        conv1 = self.custom_conv2d(pre_activated1, filters, 3, strides)
        pre_activated2 = self.pre_activation(conv1, is_training)
        conv2 = self.custom_conv2d(pre_activated2, filters, kernel_size=3, strides=1)
        return conv2 + shortcut
```

### 瓶颈组块 bottleneck block

在残差网络（ResNet）中，瓶颈块（Bottleneck Block）是一种特殊的残差块结构，用于在深度神经网络中降低计算复杂度和提高性能。瓶颈块相对于普通的残差块主要引入了一个 1x1 的卷积层，以减小输入的维度，然后再通过一个 3x3 的卷积层处理，最后再通过一个 1x1 的卷积层将维度还原回去。

一个典型的瓶颈块的结构如下：

1. **1x1 卷积层（降维）：**
   - 使用 1x1 的卷积核，通常将输入的特征维度减小，以减少计算量。这一步被称为降维。

2. **3x3 卷积层：**
   - 使用 3x3 的卷积核进行卷积操作，处理降维后的特征。

3. **1x1 卷积层（升维）：**
   - 再次使用 1x1 的卷积核，将特征维度升回到原始的维度。这一步被称为升维。

4. **Shortcut（跳跃连接）：**
   - 残差网络的核心思想是跳跃连接，将输入直接添加到最后的输出上。

瓶颈块的设计主要是为了降低计算复杂度，因为通过使用 1x1 的卷积降维和升维，可以减少中间层的特征数量。这样，瓶颈块在提供较大的感受野的同时，有效地减小了计算成本。

一个典型的瓶颈块的计算公式可以表示为：

output = F(input) + shortcut

其中，F(input) 是整个瓶颈块的输出，包括了两个 1x1 卷积和一个 3x3 卷积，shortcut 是跳跃连接的输出。

总体而言，瓶颈块是 ResNet 中的一种优化结构，使得深度网络的训练更加有效和稳定。

```python
def bottleneck_block(self, inputs, filters, strides, is_training, index, shortcut_filters=None):
    with tf.compat.v1.variable_scope('bottleneck_block{}'.format(index)):
        shortcut_params = (shortcut_filters, strides)
        pre_activated1, shortcut = self.pre_activation_with_shortcut(inputs, is_training, shortcut_params)
        conv1 = self.custom_conv2d(pre_activated1, filters, 1, 1)
        pre_activated2 = self.pre_activation(conv1, is_training)
        conv2 = self.custom_conv2d(pre_activated2, filters, 3, strides)
        pre_activated3 = self.pre_activation(conv2, is_training)
        conv3 = self.custom_conv2d(pre_activated3, 4 * filters, 1, 1)
        return conv3 + shortcut
```

这里最后的4倍数量的过滤器，相当于将进行特征提取后的特征图，增加了更多通道数，也就是增加到4倍的通道数。

### Block layer

典型的 ResNet 结构可以包括多个 block layer，每个 block layer 都包含多个残差块。这些 block layers 的设计有助于建立层次化的特征提取过程，使得网络可以逐渐学习到越来越复杂、抽象的特征表示。

例如，一个简化的 ResNet 结构可能如下所示：

```plaintext
Input
|
Convolutional Layer
|
Block Layer 1 (多个残差块)
|
Block Layer 2 (多个残差块)
|
...
|
Block Layer N (多个残差块)
|
Global Average Pooling
|
Fully Connected Layer (输出层)
```

在这个结构中，每个 "Block Layer" 包含多个残差块，而每个残差块可能由一个普通的残差块或者瓶颈块组成。整个网络通过层层堆叠这样的结构，逐渐提取输入数据的高层次特征，最终输出用于分类、回归或其他任务的结果。

总的来说，"block layer" 是卷积神经网络中的一层，由多个相同类型的基本块组成，用于建立深层次的特征表示。

```python
# Creates a layer of blocks
def block_layer(self, inputs, filters, strides, num_blocks, is_training, index):
    with tf.compat.v1.variable_scope('block_layer{}'.format(index)):
        shortcut_filters = 4 * filters if self.bottleneck else filters
        block_fn = self.bottleneck_block if self.bottleneck else self.regular_block
        block_output = block_fn(inputs, filters, strides, is_training, 0, 
            shortcut_filters=shortcut_filters)
        # stack the blocks in this layer
        for i in range(1, num_blocks):
            block_output = block_fn(block_output, filters, 1, is_training, i)
        return block_output
```

### 完整的model layer

```python
import tensorflow as tf

class ResNetModel(object):
    # __init__ and other functions omitted

    # Model Layers
    # inputs (channels_last): [batch_size, resize_dim, resize_dim, 3]
    # inputs (channels_first): [batch_size, 3, resize_dim, resize_dim]
    def model_layers(self, inputs, is_training):
        # initial convolution layer
        conv_initial = self.custom_conv2d(
            inputs, self.filters_initial, 7, 2, name='conv_initial')
        # pooling layer
        curr_layer = tf.keras.layers.MaxPool2D(
            3, 2, padding='same',
            data_format=self.data_format,
            name='pool_initial')(conv_initial)
        # stack the block layers
        for i, num_blocks in enumerate(self.block_layer_sizes):
            filters = self.filters_initial * 2**i
            strides = self.block_strides[i]
            # stack this block layer on the previous one
            curr_layer = self.block_layer(
                curr_layer, filters, strides,
                num_blocks, is_training, i)
        # pre-activation
        pre_activated_final = self.pre_activation(curr_layer, is_training)
        filter_size = int(pre_activated_final.shape[2])
        # final pooling layer
        avg_pool = tf.keras.layers.AveragePooling2D(
            filter_size, 1, 
            data_format=self.data_format)(pre_activated_final)
        final_layer = tf.layers.flatten(avg_pool)
        # get logits from final layer
        logits = tf.keras.layers.Dense(self.output_size, name='logits')(final_layer)
        return logits
```
