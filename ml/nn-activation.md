## 神经网络的激活函数们

---
### 激活函数的作用

激活函数在神经网络中起着非常重要的作用，它能够引入非线性特性，增强模型的表达能力，解决梯度消失问题，以及引入稀疏性和稳定性，从而提高神经网络的性能和效果。

这个世界很多规律无法用线性模型表达，比如XOR问题等，你无法用一条直线来解决，所以需要非线形的模型来做这类事。

这里总结一下遇到的常用的那些激活函数。

### Sigmoid

Sigmoid 的输出是在0到1之间，可以作为二分类算法的输出分类概率使用，同时它单调可导。

但是它的局限性也很明显：

- 当输出远离 0 无限向负延伸，可能会无限小，这会导致梯度消失。
- 输出不是以 0 为中心的，这会导致梯度更新不均匀，增加学习难度。
- Sigmoid 函数的输出范围是 (0, 1)，因此其输出总是正数。这意味着在神经网络的某些层中，神经元的激活值总是正数，这可能会导致一些梯度也总是正数，从而影响到网络参数的更新方向。
- 指数运算很贵。

在torch中表达如下：

```python
import torch
import torch.nn as nn
# init layer
activation_layer = nn.Sigmoid()
# input
x = torch.rand(2, 10)
print(x)
# apply activation
x = activation_layer(x)
print(x)
```

### Tanh

双曲正切曲线。它的区间为 -1 到 1，并且以 0 为中心。虽然 tanh 函数具有零中心和更广的输出范围等优点，但在深层神经网络中仍然存在一些局限性，如梯度消失、梯度爆炸、计算复杂度高等问题。

在torch中表达如下：

```python
import torch
import torch.nn as nn
activation_layer = nn.Tanh()
x = torch.rand(2, 10)
x = activation_layer(x)
```

### Relu

ReLU 激活函数具有计算速度快、抑制梯度消失问题、稀疏激活性和线性特性等优点。这可能是我们最常用的激活函数了，但它也存在 Dead ReLU 问题和不以零为中心的缺点。

Dead ReLU 问题：当神经元的输入小于等于零时，ReLU 激活函数输出为零，这可能会导致一些神经元永远无法被激活，称为“死亡神经元”（Dead Neurons）。这些神经元对于梯度更新不起作用，可能会影响网络的表达能力。

它的表达是好简单：f(x) = max(x, 0)

在torch中的表达如下：

```python
import torch
import torch.nn as nn
activation_layer = nn.ReLU()
x = torch.rand(2, 10)
x = activation_layer(x)
```

### Leaky Relu

为了解决Dead ReLU的问题产生。公式：f(x) = max(0.01x, x)

看公式表达就可以知道，它是通过给负数都乘上一个很小的数字解决的，这样就会避免了很多的死亡神经元。

在torch中的表达是`nn.LeakyReLU(0.01)`。

### Parametric Relu

公式：f(x) = max(ax, x)。

Parametric ReLU（PReLU）是一种改进的 ReLU 激活函数，与标准的 ReLU 不同之处在于，PReLU 具有一个可学习的参数，即它允许神经网络自动学习激活函数的形状，而不是像 ReLU 那样硬编码为 0 或者线性。

其中，a 是一个可学习的参数，可以通过反向传播算法来优化。如果 a 被学习到的值接近于 0，则 PReLU 的行为类似于 ReLU；如果 a 被学习到的值大于 0，则 PReLU 的行为类似于 Leaky ReLU，即允许小于零的部分有一个小的斜率。

PReLU 的优点包括：
- 允许负数斜率：PReLU 允许负数输入拥有一个小的斜率，这有助于减少死亡神经元的数量，并且可以提高网络的表达能力。
- 减少梯度消失：PReLU 允许负数输入拥有一个小的梯度，这有助于减少梯度消失问题，使得神经网络的训练更加稳定。
- 参数化形状：PReLU 具有一个可学习的参数 a，使得神经网络可以自动学习激活函数的形状，从而更好地适应数据的特征。

然而，PReLU 也存在一些缺点：
- 参数化计算成本：PReLU 比标准的 ReLU 多了一个可学习的参数 a，这增加了神经网络的计算成本和模型的复杂度。
- 过拟合风险：PReLU 具有更多的参数，因此存在过拟合的风险，特别是在数据量较小的情况下。

在torch中的表达是`nn.PReLU()`。

### Softmax

softmax大软熊主要在多分类问题的最后一个layer中出现。在torch中的表达如下：`nn.Softmax(dim=-1)`。它所输出的值的综合为1，并增强对最大值的响应。

在训练过程中，Softmax 函数常与交叉熵损失函数（Cross-Entropy Loss）结合使用，用于衡量模型预测值与真实标签之间的差异，并通过反向传播算法进行模型参数的优化。

### 这些激活函数的适用算法

以下是各种激活函数适用的常见算法总结：

1. **Sigmoid 函数**：
   - 适用于二分类问题，常见于 Logistic 回归、MLP（多层感知器）等模型中作为输出层激活函数。

2. **tanh 函数**：
   - 适用于隐藏层，常见于 MLP、RNN（循环神经网络）等模型中，也可用于输出层的多分类问题。
   - 由于其输出范围为 (-1, 1)，tanh 函数在数据已标准化的情况下尤为适用。

3. **ReLU 函数**（Rectified Linear Unit）：
   - 适用于隐藏层，常见于深度神经网络（如 CNN、DNN）中。
   - ReLU 可以引入非线性特性，加速网络的收敛，避免梯度消失问题。

4. **Leaky ReLU 函数**：
   - 类似于 ReLU，但允许负值输入有一个小的斜率，有助于避免“死亡神经元”问题。
   - 适用于深度神经网络中，可以缓解 ReLU 中的一些问题。

5. **PReLU 函数**（Parametric ReLU）：
   - 类似于 Leaky ReLU，但参数化斜率，允许模型自动学习最优的斜率。
   - 适用于深度神经网络中，对于具有不同数据分布的不同层，PReLU 可以根据数据动态调整斜率。

6. **Softmax 函数**：
   - 适用于多分类问题，常见于神经网络的输出层，将原始输出转换为表示各个类别概率的向量。
   - Softmax 函数通常与交叉熵损失函数结合使用，用于多分类问题的模型训练。

### 手写代码

```python
import torch

seed = 172
torch.manual_seed(seed)


def m_sigmoid(x):
    return 1 / (1 + torch.exp(-x))


def m_tanh(x):
    return (torch.exp(x) - torch.exp(-x)) / (torch.exp(x) + torch.exp(-x))


def m_relu(x):
   return torch.mul(x, (x > 0))
   return torch.max(torch.tensor(0, dtype=x.dtype), x)
   # 这两种写法都可以，但是可以看到第一种更短，第二种看到如果不加数据类型会报错

def m_softmax(x):
    return torch.exp(x) / torch.sum(torch.exp(x))
```




