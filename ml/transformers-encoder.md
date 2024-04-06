## Transformers 编码器代码实践

---
在[上一篇](ml/transformers-parts.md)学习的基础上，这里开始，构架一个 Transformer 的编码器。虽然 Pytorch 里面有直接可以用的类，但是这里都是从零开始的代码，通过对代码结构，从而更好的理解模型的内部原理。这是目的。

### recap the architecture

编码器的最终输出是输入序列经过多个编码器层处理后得到的高维表示。这个高维表示包含了输入序列的各个位置的信息，并且经过了多层的特征提取和组合，具有丰富的语义信息。

在 Transformer 模型中，编码器的最终输出通常是一个张量，维度为 [batch_size, seq_length, d_model]，其中 batch_size 表示批量大小，seq_length 表示序列长度，d_model 表示模型的维度大小。这个张量中的每个位置包含了对应输入序列位置的信息。

编码器的最终输出被传递给解码器，用于生成输出序列。解码器使用编码器的输出作为输入，并结合自身的注意力机制来逐步生成输出序列。在解码器的每一步中，它会根据当前的上下文信息和编码器的输出，预测下一个位置的输出，并更新当前的状态。解码器的自注意力机制会确保在生成输出时使用了输入序列的信息，并根据需要调整注意力权重以关注不同位置的信息。

**Transformer 编码器**构架重申：

为了处理一个句子，我们需要执行以下三个步骤：

- 同时计算输入句子的词嵌入。
- 然后对每个嵌入应用位置编码，从而得到包含位置信息的词向量。
- 将词向量传递给第一个编码器块。

每个编码器块包含以下层，按照相同的顺序排列：

- 多头自注意力层，用于找到每个单词之间的相关性。
- 一个归一化层。
- 在前两个子层周围的一个残差连接。
- 一个线性层。
- 第二个归一化层。
- 第二个残差连接。

在实际的构架中，上述块可以多次复制以形成编码器。在原始论文中，编码器由 6 个相同的块组成。

下面的编码主药注重对每个重要模块的学习。

### Linear layers

从最简单的线性层开始：按照下面的简单结构进行coding。这部分是复用简单的 PyTorch 线性层，就不再赘述了。

- Linear Layer
- RELU
- Dropout
- 2nd Linear layer

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        """
        Args:
            `d_model`: model dimension
            `d_ff`: hidden dimension of feed forward layer
            `dropout`: ropout rate, default 0.1
        """
        super(FeedForward, self).__init__() 
        linear = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
       
    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """
        Args:
            `x`: shape (batch_size, max_len, d_model)
        Returns:
            same shape as input x
        """
        ## 2.  RETURN THE FORWARD PASS 
        return self.linear(x)

FeedForward(10, 100, 0.1)
```

### Layer normalization

归一化的计算公式，是 LN(x) = alpha(x - mean/standard_deviation)+beta

```python
class LayerNorm(nn.Module):
    def __init__(self, features: int, eps: float = 1e-6):
        # features = d_model
        super(LayerNorm, self).__init__()
        self.a = nn.Parameter(torch.ones(features))
        self.b = nn.Parameter(torch.zeros(features))
        # eps 是一个很小的常数，避免除零错误
        self.eps = eps

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        # 沿着最后一个-1维度求均值，keepdim=True 表示保持结果张量的维度与输入张量一致
        mean = x.mean(-1, keepdim=True)
        # 标准差
        std = x.std(-1, keepdim=True)
        # 按照公式计算
        return self.a * (x - mean) / (std + self.eps) + self.b
```

### Skip connection

残差连接层。

```python
class SkipConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size: int, dropout: float):
        super(SkipConnection, self).__init__()
        # 复用归一化层，和 nn 的 dropout 层
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, 
                x: torch.FloatTensor, 
                sublayer: Union[MultiHeadAttention, FeedForward]
                ) -> torch.FloatTensor:
        """Apply residual connection to any sublayer with the same size."""
        return x + self.dropout(sublayer(self.norm(x)))
```

### EncoderLayer

编码器单层的代码，这其中用到了多头注意力类，这个类在 [注意力机制](ml/attention.md) 中有详细的代码和解说，这里就不再repeat我自己了。回去参考即可。

其中的sublayer，使用列表推导式创建了一个包含两个 SkipConnection 层的 nn.ModuleList。每个 SkipConnection 层都包含一个子层，分别是自注意力机制和前向传播网络。使用 copy.deepcopy 复制 SkipConnection 层，以便每个层都有独立的参数。

前向传播部分可能会有费解的感觉，解释就是：
- x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))：使用第一个 SkipConnection 层对输入张量 x 进行处理，其中传入的函数是自注意力机制（self.self_attn）对输入张量的计算结果。
- return self.sublayer[1](x, self.feed_forward)：使用第二个 SkipConnection 层对经过自注意力机制处理后的张量 x 进行处理，其中传入的函数是前向传播网络（self.feed_forward）对输入张量的计算结果。
- 因为上面一个模块已经定义了sublayer: Union[MultiHeadAttention, FeedForward]，如此就实现了一次定义两个层然后进行复用的方便手法。

```python
class EncoderLayer(nn.Module):
    """Encoder is made up of self-attn and feed forward"""

    def __init__(self, size: int, self_attn: MultiHeadAttention, feed_forward: FeedForward, dropout: float):
        super(EncoderLayer, self).__init__()
        # 初始化各个块，自注意力机制，前向传播，需要进行重复利用的sub层，以及size
        self.self_attn = self_attn  # 用于计算自注意力权重
        self.feed_forward = feed_forward  # 对自注意力输出进行线性变换
        # 
        self.sublayer = nn.ModuleList([copy.deepcopy(SkipConnection(size, dropout)) for _ in range(2)])
        # 编码器层的大小，输入和输出的维度
        self.size = size

    def forward(self, x: torch.FloatTensor, mask: torch.ByteTensor) -> torch.FloatTensor:
        # 前向传播计算：
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)
```

### Encoder

将单一layer堆叠，加上了mask和归一化。

```python
class Encoder(nn.Module):
    # 核心编码器，layer的类型可以看到是上面的类的实例，将多个层堆叠起来
    """Core encoder is a stack of N layers"""

    def __init__(self, layer: EncoderLayer, N: int):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm = LayerNorm(layer.size)

    def forward(self, x: torch.FloatTensor, mask: torch.ByteTensor) -> torch.FloatTensor:
        """Pass the input (and mask) through each layer in turn."""
        for layer in self.layers:
            # 每一个输入都要加入mask
            x = layer(x, mask)
        return self.norm(x)  # 在输出的时候进行归一化层的处理
```

### TransformerEncoder

这里总结了前面的所有步骤：

- 定义一个多头注意力层
- 定义一个前向传播层
- 使用上面两个层定义单一encoder层
- 使用上面一个单一encoder层定义最后的encoder层

```python
class TransformerEncoder(nn.Module):
    # T编码器主体
    """The encoder of transformer
    Args:
        `n_layers`: number of stacked encoder layers
        `d_model`: model dimension
        `d_ff`: hidden dimension of feed forward layer
        `n_heads`: number of heads of self-attention
        `dropout`: dropout rate, default 0.1
    """

    def __init__(self, d_model: int, d_ff: int, n_heads: int = 1, n_layers: int = 1,
                 dropout: float = 0.1):
        super(TransformerEncoder, self).__init__()
        self.multi_headed_attention = MultiHeadAttention(n_heads, d_model, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.encoder_layer = EncoderLayer(d_model, self.multi_headed_attention, self.feed_forward, dropout)
        self.encoder = Encoder(self.encoder_layer, n_layers)
        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            # 条件判断是否为矩阵参数
            if p.dim() > 1:
                # Xavier 初始化方法是一种常用的参数初始化方法，它可以使得神经网络的权重在初始化时分布在均值为 0、方差适当的范围内，有助于避免梯度消失或爆炸的问题，提高模型的训练效果和收敛速度。
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.FloatTensor, mask: torch.ByteTensor) -> torch.FloatTensor:
        return self.encoder(x, mask)
```
### recap

`TransformerEncoder` 类的作用是搭建 Transformer 模型的编码器部分。总结一下它的几个部分：

1. 多头自注意力层（`MultiHeadAttention`）：用于计算输入序列中各个位置之间的相关性。
2. 前向传播网络（`FeedForward`）：对自注意力层的输出进行线性变换和非线性变换。
3. 编码器层（`EncoderLayer`）：由自注意力层和前向传播网络组成的一个编码器层。
4. 编码器（`Encoder`）：由多个编码器层堆叠而成的完整编码器。

在初始化时，`TransformerEncoder` 类会创建并组合这些部分，形成一个完整的 Transformer 编码器。然后，通过 `forward` 方法，将输入序列和 mask 传入编码器，得到编码器的输出。`TransformerEncoder` 类是实现 Transformer 模型编码器部分的一个封装，用于处理输入序列并生成高维表示。

最后再总结一下整个过程中我想要注意的点：

- QKV三个矩阵其实都是对输入 x 对线形变换。
- 注意力分数的计算，是对于每个头，进行QK都点积运算，然后进行缩放得到的注意力分数。
- 在特殊任务中需要运用mask。
- 加权求和：将每个头的值根据注意力分数进行加权求和。
- 多头自注意力层的输出是线性变换和残差连接的结果。

关于代码中存在 mask 的疑惑：一般来说，在 Transformer 模型中，编码器是不需要使用 mask 的。因为编码器的任务是处理输入序列，它不涉及生成输出序列的过程，也不需要考虑输出的顺序。

在代码中，虽然定义了一个 `mask` 参数，但是在标准的编码器中，通常不会使用到这个参数。这个参数的存在可能是为了保持接口的一致性，或者为了允许用户在某些特殊情况下自定义编码器的行为。

因此，在使用这个编码器时，可以选择不传递 `mask` 参数，或者将其设置为 None。这样编码器就会忽略这个参数，正常进行编码操作。
