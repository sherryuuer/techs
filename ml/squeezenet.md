## SqueezeNet

SqueezeNet是一种轻量级的神经网络架构，专注于在模型大小和计算资源方面的高效性。它由DeepScale公司于2016年提出，旨在减小神经网络的模型大小，同时保持良好的性能。

他的设计思想是通过*使用1x1卷积层*（也称为*逐点卷积或逐元素卷积*）来减小网络的参数数量。1x1卷积层的作用是在通道之间进行线性组合，从而降低输入特征图的通道数，减小模型的复杂度。这种结构被称为"Fire Module"，由一个squeeze层和一个expand层组成。

在SqueezeNet中，作者还采用了一种称为“ bypass”（旁路连接）的方法，将某些层的输入直接传递到输出，以帮助保留更多的信息。这有助于减小模型的损失，同时减小了参数数量。

SqueezeNet相对于一些传统的深度神经网络，如AlexNet（200MB的参数量）和VGG，具有*更小的模型大小*（Squeeze只有1MB参数量呐！），同时在一些图像分类任务上仍能取得相近的性能。由于其轻量级的特点，SqueezeNet常被用于在资源受限的环境中，如移动设备和嵌入式系统中进行图像分类等任务。

**总的来说它的大小只有1M比之前的手写模型更小，但是精度却达到了AlexNet的程度**。

一个卷积网络的**参数量计算方法**是：核高 x 核宽 x 核（filter）数量 x 通道channel数量 + 偏置bias数量（一般是filter数量，因为一个核一个偏置）

**重点：减少参数的方法**在于三种，减少核的数量，缩小核尺寸，减少输入通道数量。Squeezenet的fire model的减少参数的策略就应用了一些。比如：

1. 核的大小，可以混合使用大的核和小的核。
2. 加入一个中间层（更小的核和更少的通道）从而将它更小的output重新输入原来的层，达到减少参数的目的。我觉得这个层就相当于一个大型过滤器了。这一层又可以叫做**挤压层**，就是这个net的名字由来吧。

## Fire Module

SqueezeNet中的**Fire Module**是该网络架构的一个核心组成部分，负责提取特征并降低模型参数的数量。Fire Module的设计旨在通过使用1x1卷积层（squeeze层）和3x3卷积层（expand层）来实现这一目标。

Fire Module分为两个阶段：

1. **Squeeze阶段（squeeze layer）：** 在这一阶段，使用1x1卷积核对输入进行通道压缩。1x1卷积的作用是在通道间进行线性组合，减小输入特征图的通道数，从而减少模型参数。这有助于保留输入的主要特征。

2. **Expand阶段（expand layer）：** 在这一阶段，使用两组卷积核，分别是1x1和3x3卷积核，对Squeeze阶段的输出进行通道扩展。1x1卷积核负责将通道数扩展回来，而3x3卷积核则负责捕捉更复杂的特征。这种结构允许网络保持一定的复杂性，同时仍然保持相对较少的参数。

这两个阶段的压缩比率称为压缩比。

整个Fire Module的*计算流程*可以用以下步骤表示：

1. 输入通过Squeeze阶段，经过1x1卷积，通道数减小。
2. Squeeze阶段的输出通过Expand阶段，经过1x1和3x3卷积，通道数再次增加。
3. 最终的输出被用作下一层的输入，传递到整个网络中。

Fire Module的设计使得SqueezeNet在保持相对较小的模型尺寸的同时，仍能在一些图像分类任务上保持较好的性能。这种结构在资源受限的环境中，如移动设备和嵌入式系统中，具有较大的应用潜力。

## SqueezeNet Model Code

```python
import tensorflow as tf

class SqueezeNetModel(object):
    # Model Initialization
    def __init__(self, original_dim, resize_dim, output_size):
        self.original_dim = original_dim
        self.resize_dim = resize_dim
        self.output_size = output_size
    
    # Convolution layer wrapper
    def custom_conv2d(self, inputs, filters, kernel_size, name):
        return tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            padding='same',
            activation='relu',
            name=name
        )(inputs)

    # SqueezeNet fire module
    def fire_module(self, inputs, squeeze_depth, expand_depth, name):
        with tf.compat.v1.variable_scope(name):
            # 压缩通道
            squeezed_inputs = self.custom_conv2d(
                inputs,
                squeeze_depth,
                [1, 1],
                'squeeze')
            # 两次拓展通道
            expand1x1 = self.custom_conv2d(
                squeezed_inputs,
                expand_depth,
                [1, 1],
                'expand1x1')
            expand3x3 = self.custom_conv2d(
                squeezed_inputs,
                expand_depth,
                [3, 3],
                'expand3x3')
            return tf.concat([expand1x1, expand3x3], axis=-1)

    # Stacked fire modules: 使用多层挤压，提高性能，原本的ImageNet使用了8层
    def multi_fire_module(self, layer, params_list):
        for params in params_list:
            layer = self.fire_module(
                layer,
                params[0],
                params[1],
                params[2]
            )
        return layer
```

上面的fire-model方法可能会有些费解，提出来解释一下：

```python
def fire_module(self, inputs, squeeze_depth, expand_depth, name):
    # 定义一个 Fire Module 函数，接收输入、压缩通道深度、拓展通道深度和模块名称作为参数

    with tf.compat.v1.variable_scope(name):
        # 在 TensorFlow 中，使用 variable_scope 来定义变量作用域，确保变量的命名不会冲突

        # 压缩通道阶段
        squeezed_inputs = self.custom_conv2d(
            inputs,
            squeeze_depth,
            [1, 1],
            'squeeze')
        # 使用自定义的 1x1 卷积函数 custom_conv2d 对输入进行压缩通道操作
        # squeeze_depth 指定压缩后的通道深度
        # [1, 1] 表示卷积核的大小是 1x1

        # 拓展通道阶段，使用两个不同尺寸的卷积核
        expand1x1 = self.custom_conv2d(
            squeezed_inputs,
            expand_depth,
            [1, 1],
            'expand1x1')
        # 使用 1x1 卷积核进行通道拓展
        # expand_depth 指定拓展后的通道深度

        expand3x3 = self.custom_conv2d(
            squeezed_inputs,
            expand_depth,
            [3, 3],
            'expand3x3')
        # 使用 3x3 卷积核进行通道拓展
        # expand_depth 指定拓展后的通道深度

        # 将两个阶段的输出在通道维度上进行连接
        return tf.concat([expand1x1, expand3x3], axis=-1)

```

**关于最后一步为什么要进行concat？**

最后一步的两个通过深度的连接（`tf.concat([expand1x1, expand3x3], axis=-1)`）是为了将压缩通道阶段和拓展通道阶段的输出在通道维度上进行合并。这种设计有几个目的：

1. **特征融合（Feature Fusion）：** 通过将不同尺寸卷积核的输出在通道维度上连接，可以将两个阶段提取的不同特征信息融合在一起。1x1 卷积核用于压缩通道，而 3x3 卷积核用于在更大的感受野内捕捉更复杂的特征。连接操作可以将这些不同尺寸卷积核提取的特征有机地结合在一起。

2. **增加网络的非线性：** 由于连接是在通道维度上进行的，这意味着网络中不同通道的信息被交织在一起。这有助于引入非线性，增加网络的表达能力，使其更适应复杂的数据分布。

3. **提高网络的表达能力：** 通过连接不同尺寸卷积核的输出，可以增加网络对多尺度特征的敏感性，从而提高网络的表达能力。这对于处理不同尺寸的对象或结构非常重要。

总体而言，这个连接操作有助于提高模型的表示能力，使其能够更好地捕捉和利用输入数据中的信息，从而提高网络性能。这是 SqueezeNet 架构的一种设计选择，旨在在保持模型轻量级的同时，保持较好的性能。


## 探索CIFAR10数据集

CIFAR-10(Canadian Institute for Advanced Research) 数据集是用于机器视觉领域的图像分类数据集，它有飞机、汽车、鸟类、猫、鹿、狗、青蛙、马、船和卡车共计10 个类别的60000 张彩色图像，尺寸均为32*32，其包含5个训练集和1个测试集，每个数据集有10000 张图像。

- init：初始化模型类：原始维度，输出大小，处理后维度
- image_preprocessing：对图像的前置处理，包括数据增强（裁剪，反转0.5prob）以及图像正规化，将0-255变成0-1区间的张量。
- model_layers：
1. 常规处理，是在开头有一个正常的卷积层，在最后有一个正常的池化层。（采用延迟下采样（delayed downsampling）是比如把最大池化层放在最后层的行为。而在前期拥有更多的特征，有助于提高精度。）
2. 常规操作：挤压层加上池化层。
3. 增加深度的操作：为了提高精度再次加上一个挤压层，这次不需要池化，在输出logits阶段，会使用池化。
4. dropout层：防止过拟合。
5. AveragePool层：输出logits**使用全局平均池化而不是全连接层有两个主要优势。**
   - **CNN结构的本土性：** 全局平均池化更符合CNN结构，通过通道获得logits而不是将数据转换为平坦向量。这使得CNN能够为每个图像类别获取更准确的logits。
   - **无参数化：** 全局平均池化仅是一个池化层，因此不使用任何参数。这意味着在全局平均池化层不存在过拟合的风险。相比之下，全连接层使用许多权重参数。尽管dropout可以帮助减轻问题，但过拟合的风险仍然存在。

**注意**：在结果输出上，这个数据集是稀疏标签，而不是onehot，所以需要tf.nn.sparse_softmax_cross_entropy_with_logits function进行处理。

```python
import tensorflow as tf

class SqueezeNetModel(object):
    # Model Initialization
    def __init__(self, original_dim, resize_dim, output_size):
        self.original_dim = original_dim
        self.resize_dim = resize_dim
        self.output_size = output_size
    
    # Random crop and flip
    def random_crop_and_flip(self, float_image):
        crop_image = tf.compat.v1.random_crop(float_image, [self.resize_dim, self.resize_dim, 3])
        updated_image = tf.image.random_flip_left_right(crop_image)
        return updated_image
    
    # Data Augmentation
    def image_preprocessing(self, data, is_training):
        reshaped_image = tf.reshape(data, [3, self.original_dim, self.original_dim])
        transposed_image = tf.transpose(reshaped_image, [1, 2, 0])
        float_image = tf.cast(transposed_image, tf.float32)
        # 训练模式下进行数据增强（大小调整和flip翻转）
        if is_training:
            updated_image = self.random_crop_and_flip(float_image)
        # 除此之外，进行图像大小调整
        else:
            updated_image = tf.image.resize_image_with_crop_or_pad(float_image, self.resize_dim, self.resize_dim)
        standardized_image = tf.image.per_image_standardization(updated_image)
        return standardized_image

    # Convert final convolution layer to logits
    def get_logits(self, conv_layer):
        avg_pool1 = tf.keras.layers.AveragePooling2D(
            [conv_layer.shape[1], conv_layer.shape[2]],
            1)(conv_layer)
        logits = tf.keras.layers.Flatten(name='logits')(avg_pool1)
        return logits
    
    # Convolution layer wrapper
    def custom_conv2d(self, inputs, filters, kernel_size, name):
        return tf.keras.layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        padding='same',
        activation='relu',
        name=name)(inputs)

    # Max pooling layer wrapper
    def custom_max_pooling2d(self, inputs, name):
        return tf.keras.layers.MaxPool2D(
        pool_size=[2, 2],
        strides=2,
        name=name)(inputs)
    
    # Model Layers
    def model_layers(self, inputs, is_training):
        conv1 = self.custom_conv2d(
            inputs,
            64,
            [3, 3],
            'conv1')
        pool1 = self.custom_max_pooling2d(
            conv1,
            'pool1')
        # add fire model
        fire_params1 = [
            (32, 64, 'fire1'),
            (32, 64, 'fire2')
        ]
        multi_fire1 = self.multi_fire_module(
            pool1,
            fire_params1)
        pool2 = self.custom_max_pooling2d(
            multi_fire1,
            'pool2')
        # add fire model (add depth)
        fire_params2 = [
            (32, 128, 'fire3'),
            (32, 128, 'fire4')
        ]
        multi_fire2 = self.multi_fire_module(
            pool2,
            fire_params2)
        # dropout to prevent overfitting
        dropout1 = tf.keras.layers.Dropout(rate=0.5)(multi_fire2, training=is_training)
        conv_layer = self.custom_conv2d(
            dropout1,
            self.output_size,
            [1, 1],
            'final_conv')
        return self.get_logits(conv_layer)
```

## 模型训练部分代码示例

```python
import tensorflow as tf

class SqueezeNetModel(object):
    # __init__ and other functions omitted

    # Set up and run model training
    def run_model_setup(self, inputs, labels):
      logits = self.model_layers(inputs, is_training)
      self.probs = tf.nn.softmax(logits, name='probs')
      self.predictions = tf.math.argmax(
          self.probs, axis=-1, name='predictions')
      is_correct = tf.math.equal(
          tf.cast(self.predictions, tf.int32),
          labels)
      is_correct_float = tf.cast(
          is_correct,
          tf.float32)
      self.accuracy = tf.math.reduce_mean(
          is_correct_float)
      # calculate cross entropy
      cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
          labels=labels,
          logits=logits)
      self.loss = tf.math.reduce_mean(
          cross_entropy)
      adam = tf.compat.v1.train.AdamOptimizer()
      self.train_op = adam.minimize(
          self.loss, global_step=self.global_step)
```
