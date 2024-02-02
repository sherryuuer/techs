## ResNet残差网络

---
### 针对梯度消失问题的超大型深度网络

残差网络（Residual Network，简称为ResNet）是由微软亚洲研究院的研究员何恺明（Kaiming He）等人于2015年提出的一种深度神经网络结构。它的设计目的是为了解决深度神经网络在训练过程中出现的梯度消失和梯度爆炸问题，从而使得训练更深的网络变得更加容易。

ResNet的核心思想是引入了残差块（Residual Block），这是一个包含跳跃连接（skip connection）的模块。传统的神经网络通过堆叠层来逐渐学习特征，但随着网络层数的增加，梯度信号可能逐渐变小，导致难以训练。残差块通过在网络中引入跳跃连接，使得神经网络可以直接学习残差（即输入与期望输出之间的差异），从而更容易地进行优化。

一个基本的残差块的结构如下：

```
Input
  |
  v
[   ]
  |
  v
[   ]
  |
  v
Output
```

在这里，方框表示神经网络的层，箭头表示跳跃连接。残差块的输出是输入与原始输出的和。这种结构使得网络可以学习残差，而不是直接学习映射关系。如果网络认为学习恒等映射是最好的，那么就可以通过将权重调整为零，使得跳跃连接直接将输入传递给输出。

ResNet的设计使得它可以轻松地训练数百层甚至上千层的深度网络，同时保持了良好的性能和梯度流通。由于其出色的性能和训练效果，ResNet已经成为深度学习领域中广泛应用的经典模型之一。

### 前置知识

ImageNet是一个超大型图片网络，使用许多许多层的神经网络进行训练，最终还是导致了梯度爆炸和梯度消失问题，因为出现了残差网络模型。

机器学习中层数不短增加，比如几百几百层，当超过50层后，RN（resnet，后面都这样简称）就会开始加一个瓶颈层，然后引入残差网络。

和之前SqueezeNet一样，RN也是使用black类将所有的残差处理打包，然后堆叠block实现多个残差处理。

### 初始化残差网络

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
        self.filters_initial = 64
        self.block_strides = [1, 2, 2, 2]
        self.data_format = data_format
        self.output_size = output_size
        self.block_layer_sizes = block_layer_sizes[num_layers]
        self.bottleneck = (num_layers >= 50)
```

### Padding图像扩充

关于**通道位置**，一般采用NCHW的是Tensorflow的GPU处理器，这种顺序方式比较适合GPU训练，而在CPU上更适合NHWC的顺序。可以在函数中引入option条件判断进行灵活应对。

```python
def custom_padding(self, inputs, kernel_size):
        pad_total = kernel_size - 1
        pad_before = pad_total // 2
        pad_after = pad_total - pad_before
        if self.data_format == 'channels_first':
            padded_inputs = tf.pad(
                inputs,
                [[0, 0], [0, 0], [pad_before, pad_after], [pad_before, pad_after]])
        else:
            padded_inputs = tf.pad(
                inputs,
                [[0, 0], [pad_before, pad_after], [pad_before, pad_after], [0, 0]])
        return padded_inputs
```

### 预激活

批归一化（Batch Normalization，简称BN）和预激活（Pre-Activation）是深度学习中常用的两种技术，用于改善神经网络的训练和性能。

1. **批归一化（Batch Normalization，BN）：**
   - BN 是一种在深度神经网络中应用的技术，旨在加速训练过程并提高模型的鲁棒性。其基本思想是在每个训练小批次的数据上，对每个输入特征进行归一化，使其均值接近零，标准差接近一。这有助于防止网络中的梯度消失或梯度爆炸问题，使得网络更容易收敛。BN 通常被放置在激活函数之前。

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

在 ResNet 的基本块中，包含了残差块（Residual Block）或恒等块（Identity Block），而其中的 shortcut 就是这个直通连接的一部分。

**残差块（Residual Block）：**

ResNet 中的基本残差块由两个部分组成：

1. **主路径（Main Path）：** 主要执行卷积操作，学习输入的表示。

2. **捷径（Shortcut）：** 直通连接，将输入跳过主路径，直接添加到主路径的输出上。

具体而言，残差块的输出计算公式为：

output=F(input)+input

其中，F(input)代表主路径的输出，而input是残差块的输入。通过这样的设计，网络可以学习一个残差函数 F(input)，而不是直接学习整个映射。这使得梯度更容易传播，有助于训练非常深的网络。

**恒等块（Identity Block）：**

如果主路径的维度和输入的维度相同，那么直接将输入加到主路径输出上。这种块称为恒等块，因为它保持了输入的恒等性。

如果主路径的维度与输入的维度不同，通常使用一个额外的 1x1 卷积来调整输入的维度，使其与主路径的维度一致，然后再将它们相加。

通过这种残差连接，ResNet 成功解决了深度神经网络中的梯度问题，允许训练更深的网络，从而提高了性能。

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
