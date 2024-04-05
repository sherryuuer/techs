## 注意力，自注意力和多头注意力模型 Attention

---
### Seq2Seq 模型

序列到序列模型是理解注意力机制的前提条件。首先什么是序列，一个最好的例子就是机器翻译，一个句子就是一个序列，也就是说，在机器翻译中，一个序列是一串单词的集合。

从工作原理上，感觉之前两天所看到的不管是自编码器，还是生成模型，都使用了encoder，decoder的组合，在这里其实也不例外，对序列进行encoder，然后再进行decoder对过程，就是机器翻译的全部过程。之前的自编码器和生成模型的篇章中，降到了z，潜在空间，这里也是一样，一个中间表示（intermediate representation），一样称为z。生成模型中的z空间是一种数据分布表示，在文本处理中你一定知道embedding技术，也就是向量嵌入，在机器翻译中，就是使用了上下文向量的嵌入空间，来生成最终的结果。在我看来他们似乎说的不是一种内容，但在我看来都是一样的，embedding也是一种数据在高维度空间中的表示，只不过这里换成了文本的token而已。

在输入和输出上，是我们常说的token，我不知道中文怎么翻译，暂且就用token，他可以是任何东西，可以是文本，像素，或者是视频中的图像。

在前面的RNN篇章中，也讲RNN适合处理序列数据，他们的区别是什么呢，我们说使用循环神经网络，处理时间序列数据，我们似乎在按顺序处理内容，就好像序列这个东西就是要按照顺序进行处理，但是真的如此吗，注意力模型和Transformer就颠覆了这个观点。

其实RNN和LSTM原本就是有局限性的。这种局限性从第一性原理来说，是一种信息交互的缺陷，比如LSTM它对于超过了20步长（timesteps）的信息就无法很好的记录所有的信息了，更别说长短期记忆模型还有很多遗忘门，专门用于忘记很久远的信息。他们在小于20步内的信息处理，非常优秀。（当然我非常喜欢LSTM模型，它的整个构架都让我爱不释手。它是整个AI发展史重要的一部分。）

问题在于，循环神经网络（RNN）有bottleneck problem。它是指在处理长序列时，模型难以捕捉长期依赖关系的问题。具体来说，RNN 在处理长序列时会面临梯度消失或梯度爆炸的问题，这导致了模型在训练过程中难以有效地学习长期依赖关系。

再具体来说，在传统的 RNN 结构中，信息只能在一个方向上流动，每个时间步的隐藏状态只能通过上一个时间步的隐藏状态来计算。因此，当序列长度增加时，梯度在反向传播过程中会经过多个时间步的累积，导致梯度逐渐消失或爆炸。这会导致模型在处理长序列时性能下降，无法捕捉到序列中的长期依赖关系。

由于RNN的结构特性，空间向量 z 往往会有一种信息堆积的问题，导致模型更多地关注序列的后半部分，而忽略了前半部分的信息。打个比方，当你说，这个电脑放不进包里，因为**它**太大了。这时候这个**它**字，模型无法像人一样理解，这是在说电脑大。

### 注意力机制 Attention

针对上述问题才出现了注意力机制模型。

注意力机制和循环神经网络的不同，在于它强调了对序列所有信息的关注，而不仅仅是最后一部分。其实这个想法最初是来自计算机视觉，也就是图像处理。我们人类在看图的时候，会注意图中的重要信息，比如物体边缘，特征细节等，我们看着图片的时候，如果用红外热度扫描就会看到人眼对各个地方的关注点是不同的，注意力机制也是基于这种思想，针对序列的句子，也是通过查看不同的部分，然后找到最重要的地方。

顺便说，Transformer 模型也可以用于图像处理，因为其实图像拉开也是一个序列，只要对各个像素块进行位置编码就是一个序列了，比如有名的ViT模型，就是使用Transformer，注意力机制，进行图像识别的优秀模型。

注意力机制包含以下几个关键组成部分：

- 查询（Query）：查询向量表示模型要关注的特定信息。在注意力机制中，查询用于计算每个输入位置的重要性分数。
- 键（Key）和数值（Value）：键和数值分别表示输入序列中的位置。键用于计算查询与输入序列中各个位置之间的关联度，而数值则提供了与每个位置相关的信息。
- 注意力分数（Attention Scores）：注意力分数表示查询与每个键之间的相关程度，通常使用一种度量方法（如点积、加性、缩放点积等）来计算。
- 注意力权重（Attention Weights）：通过对注意力分数进行归一化，可以得到注意力权重，表示模型在生成输出时给予每个输入位置的重要性。这些权重可以与数值相乘以加权求和，以得到输出的加权表示。
- 注意力输出（Attention Output）：通过将数值与注意力权重相乘并求和，可以得到最终的注意力输出，它是模型在生成输出时考虑到输入序列的加权表示。

接下来是关键部分，就是**如何计算注意力分数**：

在注意力机制中，注意力分数（Attention Scores）是通过查询（Query）和键（Key）之间的相关度来计算的。这通常通过一种度量方法来完成，最常见的是点积（Dot Product）、加性（Additive）和缩放点积（Scaled Dot Product）等方法。

**点积（Dot Product）**：点积是最简单和常见的注意力计算方法之一。对于查询向量Q和键向量K，点积注意力分数可以通过它们的点积来计算，然后再进行归一化处理。具体而言，点积注意力分数计算公式如下：

Attention(Q, K) = softmax(Q dot K^T)

其中，softmax函数用于将点积结果转换为注意力权重，使其符合概率分布的要求。

**加性（Additive）**：加性注意力计算方法通过将查询和键连接并通过一个全连接层进行映射，然后应用激活函数来计算注意力分数。具体而言，加性注意力分数计算公式如下：

Attention(Q, K) = softmax(W^T tanh(QW_q + KW_k))

其中，W_q 和 W_k 是用于映射查询和键的权重矩阵，W 是全连接层的权重矩阵，tanh 是双曲正切激活函数。

**缩放点积（Scaled Dot Product）**：缩放点积注意力在点积的基础上进行了缩放，以便更好地控制注意力分数的范围。具体而言，缩放点积注意力分数计算公式如下：

Attention(Q, K) = softmax(Q dot K^T / sqrt(d_k))
其中，d_k 是查询和键的维度，通过对注意力分数除以 sqrt(d_k)，可以确保在维度较高时，梯度不会变得太大，从而提高模型的稳定性。

以上就是几种常见的注意力计算方法，选择哪种方法通常取决于具体的任务和模型结构。不同的方法可能会对模型的性能产生不同的影响。

一个简单的注意力机制模块：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    """
    Attention模块用于计算输入序列y和隐藏状态h之间的注意力权重，并将权重应用于隐藏状态以生成加权和。
    """

    def __init__(self, y_dim: int, h_dim: int):
        """
        初始化Attention模块。
        
        参数:
            y_dim (int): 输入序列y的维度。
            h_dim (int): 隐藏状态h的维度。
        """
        super().__init__()
        self.y_dim = y_dim
        self.h_dim = h_dim

        # 定义可学习的权重矩阵W，形状为(y_dim, h_dim)
        self.W = nn.Parameter(torch.FloatTensor(
            self.y_dim, self.h_dim))

    def forward(self, y: torch.Tensor, h: torch.Tensor):
        """
        Attention模块的前向传播函数。
        
        参数:
            y (torch.Tensor): 输入序列y，形状为(batch_size, y_dim)。
            h (torch.Tensor): 隐藏状态h，形状为(batch_size, h_dim)。
        
        返回:
            torch.Tensor: 加权和，形状为(batch_size, h_dim)。
        """
        # 计算注意力分数score
        # (B, y) x (y, h) = (B, h) x (h.T) = (B, B)
        score = torch.matmul(torch.matmul(y, self.W), h.T)

        # 计算注意力权重z
        z = F.softmax(score, dim=0)

        # 应用注意力权重到隐藏状态，得到加权和
        return torch.matmul(z, h)

```
在上述代码中：

输入序列 y：

- 输入序列通常指的是模型需要处理的输入数据的序列。
- 在自然语言处理任务中，输入序列可以是一个句子或者一个文档，通常以词嵌入（word embeddings）的形式表示。
- 在时间序列预测任务中，输入序列可以是一系列时间步长的数据点。
- 在图像处理任务中，输入序列可以是图像的像素值序列。

隐藏状态 h：

- 隐藏状态是模型**内部的状态**，用于存储模型在**处理输入序列时学到的信息**。
- 在循环神经网络（RNN）中，隐藏状态是模型在处理每个时间步时的输出，同时也是下一个时间步的输入。
- 在注意力机制中，隐藏状态通常表示**编码器的输出，包含了输入序列的信息**。
- 在某些情况下，隐藏状态也可以表示解码器的状态，用于生成输出序列。
- 注意力机制能够根据输入序列 y 和隐藏状态 h 动态地计算注意力权重，以便模型在生成输出时更好地关注输入序列中的相关部分。

逐行分析计算部分的形状：

- 计算注意力分数 `score`：
   - 首先，对输入序列 y 与权重矩阵 W 进行矩阵乘法，得到一个中间结果，其形状为 (B, h_dim)。这是因为 y 的形状是 (B, y_dim)，而 W 的形状是 (y_dim, h_dim)，两者相乘得到的结果将是 (B, h_dim)。
   - 然后，将这个中间结果与隐藏状态张量 h 的转置 h^T 进行矩阵乘法，得到注意力分数 `score`，其形状为 (B, B)。

2. 计算注意力权重 `z`：
   - 对 `score` 进行 softmax 操作，沿着第一个维度进行 softmax，即对每个 batch 内的元素进行 softmax。
   - softmax 操作不会改变张量的形状，因此 `z` 的形状仍然是 (B, B) 。

3. 应用注意力权重到隐藏状态，得到加权和：
   - 将注意力权重 `z` 与隐藏状态张量 h 进行矩阵乘法，得到加权和。
   - 由于 `z` 的形状是 (B, B)，而 h 的形状是 (B, h_dim)，因此两者相乘得到的结果形状将是 (B, h_dim)。

因此，最终输出的形状是 (B, h_dim)，即与隐藏状态张量 h 的形状相同。这表示每个 batch 内的隐藏状态经过注意力加权后得到的加权和。

### 自注意力机制 Self-attention

如果说注意力机制是序列和序列的交互，那么自注意力机制就是序列自己和自己的交互。

在自注意力机制中，输入序列中的每个元素都会与序列中的其他元素进行交互，以计算其自身的注意力权重。这种交互和刚刚说的注意力机制一样，是通过将输入序列分别投影到三个不同的空间（查询、键和值空间）来实现的。

首先是计算查询（Query）、键（Key）和值（Value）：

- 首先，通过将输入序列投影到三个不同的空间（Query、Key和Value空间）来计算查询向量 Q 、键向量 K 值向量 V。这些投影是通过独立的权重矩阵完成的。
- 具体而言，对于给定的输入序列 X ，通过矩阵乘法和激活函数，分别得到查询向量 Q = X dot W_Q、键向量 K = X dot W_K 和值向量 V = X dot W_V ，其中 W_Q、 W_K 和 W_V 是可学习的权重矩阵。

然后是计算注意力分数：

- 计算查询向量 Q 和键向量 K 之间的相似度，通常采用点积或其他相似性度量。这将得到一个注意力分数矩阵，用于衡量每个查询与每个键的相似程度。
- 具体而言，注意力分数矩阵 S 的元素 S_i,j 表示查询 i 和键 j 之间的相似度。

最后计算注意力权重和加权和：

- 最后，通过对注意力分数矩阵应用 softmax 函数来得到注意力权重矩阵 A （Attention），以确保注意力权重的总和为1且非负。
- 然后，将注意力权重矩阵 A 与值向量 V 相乘，得到加权和向量 Z 。加权和向量 Z 包含了序列中每个位置的加权信息，其中每个元素都由序列中其他位置的值按注意力权重加权得到。

自注意力机制通过计算序列中每个元素之间的交互，并根据它们之间的关系分配注意力权重，能够捕捉到序列中的长距离依赖关系，并提取出更丰富的特征表示。这使得自注意力机制在自然语言处理等领域中被广泛应用，并成为了众多模型的核心组成部分。

如果讲一句话的各个 token 放在 graph 的数据结构中，就是一个无向有权重的图。这样就可以表示他们各个 token 和自己前后的 token 以及自己的相关度了。

```python
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, 
                query: torch.FloatTensor, 
                key: torch.FloatTensor,
                value: torch.FloatTensor, 
                mask: Optional[torch.ByteTensor] = None, 
                dropout: Optional[nn.Dropout] = None
                ) -> Tuple[torch.Tensor, Any]:
        """
        Args:
            `query`: shape (batch_size, n_heads, max_len, d_q)
            `key`: shape (batch_size, n_heads, max_len, d_k)
            `value`: shape (batch_size, n_heads, max_len, d_v)
            `mask`: shape (batch_size, 1, 1, max_len) 
            `dropout`: nn.Dropout
        Returns:
            `weighted value`: shape (batch_size, n_heads, max_len, d_v)
            `weight matrix`: shape (batch_size, n_heads, max_len, max_len)
        """
        
        d_k = query.size(-1)  # d_k = d_model / n_heads
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  
        if mask is not None:
            scores = scores.masked_fill(mask.eq(0), -1e9)
        p_attn = F.softmax(scores, dim=-1) 
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn
```

代码中：

- mask是指哪些位置需要被忽略。
- 在前向计算中，首先通过计算查询Q和键K之间的点积，然后进行缩放。这一步可以增强梯度的稳定性，并降低在计算过程中的数值范围。
- 如果提供了注意力掩码 mask，则将其应用于计算的注意力分数，以忽略掩码位置对注意力计算的影响。
- 接下来通过 softmax 函数对注意力分数进行归一化，得到注意力权重矩阵。
- 如果提供了 dropout 层，则在计算注意力权重时对其进行随机失活，以防止模型过拟合。
- 最后，将注意力权重矩阵与值张量相乘，得到加权后的值表示，同时返回注意力权重矩阵作为额外的输出。将注意力权重矩阵与值张量相乘的目的是实现加权求和，其中注意力权重决定了每个值向量对最终输出的贡献程度。这一步骤可以被视为对值张量进行加权平均，其中每个值的权重由注意力权重矩阵中对应的位置决定。值张量通常包含输入序列的信息，例如输入序列中每个位置的特征表示。在自注意力机制中，值张量通常是相同的张量作为查询和键张量，因此它包含了序列中每个位置的特征信息。在多头注意力机制中，可以使用不同的查询、键和值张量，从而使每个注意力头关注不同的信息。通过将注意力权重矩阵与值张量相乘，可以按照每个位置的注意力权重对值进行加权，以产生最终的注意力输出。这种加权求和的操作使得模型能够在处理序列数据时有效地关注不同位置的信息，并且能够根据不同的任务自动学习哪些信息更重要。

### 多头注意力机制 Multi-Head Self-Attention

多头注意力机制（Multi-Head Attention）是在注意力机制的基础上进行改进，旨在增强模型对不同关系的表达能力。它通过引入多个注意力头（Attention Heads），使得模型能够并行地学习多种不同的注意力表示，从而更好地捕捉序列中的复杂关系。

下面是多头注意力机制的基本原理：

- 在传统的注意力机制中，模型只使用单个注意力头来计算注意力权重，即得到一个注意力表示。而多头注意力机制引入了 h 个不同的注意力头，每个头都学习一种不同的注意力表示。
- 每个注意力头都有自己的权重矩阵 W_q^i、W_k^i 和 W_v^i，用于计算查询、键和值的表示。这些权重矩阵是通过模型学习得到的。
- 最终，每个注意力头得到的注意力表示会被**拼接起来**，并经过另一个学习的线性变换，得到最终的多头注意力表示。

多头注意力的计算流程：

- 对于输入序列 X ，首先，通过 h 个不同的注意力头计算 h 个注意力表示。
- 具体而言，对于每个注意力头 i，分别计算其查询 Q^i、键 K^i 和值 V^i的表示，然后使用这些表示计算注意力分数和注意力权重。
- 最后，将每个注意力头得到的注意力表示拼接起来，并通过另一个学习的线性变换（通常是一个全连接层）得到最终的多头注意力表示。

为什么会进化出多头注意力，它的优势是什么：

- 多头注意力机制能够捕捉到序列中更丰富的关系和信息。通过引入多个注意力头，**并行**地学习多种不同的注意力表示，模型能够同时关注序列中的多个方面，从而更好地表达序列的语义信息。
- 多头注意力机制还可以提高模型的泛化能力和鲁棒性，因为不同的注意力头可以学习不同的表示，从而使得模型能够处理不同的输入模式和任务。

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads: int, d_model: int, dropout: float = 0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // n_heads
        self.h = n_heads

        self.linears = nn.ModuleList(
            [copy.deepcopy(nn.Linear(d_model, d_model)) for _ in range(4)]
        )
        self.sdpa = ScaledDotProductAttention()
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query: torch.FloatTensor, 
                key: torch.FloatTensor, 
                value: torch.FloatTensor,
                mask: Optional[torch.ByteTensor] = None
                ) -> torch.FloatTensor:
        """
        Args:
            `query`: shape (batch_size, max_len, d_model)
            `key`: shape (batch_size, max_len, d_model)
            `value`: shape (batch_size, max_len, d_model)
            `mask`: shape (batch_size, max_len)

        Returns:
            shape (batch_size, max_len, d_model)
        """
        if mask is not None:
            # Same mask applied to all h heads. B*1*1*L
            mask = mask.unsqueeze(1).unsqueeze(1)
        
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        # x: B x H x L x D_v
        x, self.attn = self.sdpa(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        return self.linears[-1](x)
```

这段代码实现了多头注意力机制（Multi-Head Attention），用于在Transformer等模型中实现对序列数据的注意力处理。

具体解释如下：

- 在初始化方法 `__init__` 中，定义了多头注意力的参数：注意力头的数量 `n_heads`、输入维度 `d_model` 和 dropout 概率。同时，通过断言确保输入维度能够被注意力头的数量整除，并计算得到每个头的维度 `d_k`。
- 在初始化方法中，定义了多个线性变换层（`nn.Linear`），用于将输入序列投影到多头注意力所需的维度。这里使用了 `nn.ModuleList` 来存储多个线性层，并使用 `copy.deepcopy` 来复制线性层，以确保每个注意力头都有自己的参数。
- 初始化了一个缩放点积注意力层 `ScaledDotProductAttention`，用于计算注意力权重。
- `forward` 方法实现了多头注意力的前向计算。接受输入的查询（query）、键（key）、值（value）张量以及可选的掩码（mask）。
- 在前向计算中，首先对输入的查询、键和值分别应用线性变换，并将结果按注意力头的数量拆分。然后，将拆分后的张量传递给缩放点积注意力层进行注意力计算。
- 最后，将每个注意力头得到的输出张量拼接起来，并通过最后一个线性变换层进行投影，以得到最终的多头注意力表示。

### Q，K，V 的内容探索

当使用注意力机制时，Q（Query）、K（Key）、V（Value）通常是从输入数据中学习到的。这些向量的具体值会随着模型的训练而更新。在自注意力机制中，Q、K、V向量通常与输入序列的词嵌入向量相关联。

以一个简单的例子来说明：

假设我们有一个输入序列，其中包含4个词：[hello, how, are, you]。

1. 首先，我们将这些词转换为词嵌入向量。假设使用一个简单的词嵌入矩阵，它将每个词映射到一个3维的向量。

    ```
    hello -> [0.1, 0.2, 0.3]
    how -> [0.4, 0.5, 0.6]
    are -> [0.7, 0.8, 0.9]
    you -> [1.0, 1.1, 1.2]
    ```

2. 接下来，我们将这些词嵌入向量分别用于计算Q、K、V向量。在自注意力机制中，通常情况下，Q、K、V向量的维度相同，假设我们选择每个词嵌入向量的每个维度作为Q、K、V向量。

    ```
    Q(hello) = [0.1, 0.2, 0.3]
    K(hello) = [0.1, 0.2, 0.3]
    V(hello) = [0.1, 0.2, 0.3]

    Q(how) = [0.4, 0.5, 0.6]
    K(how) = [0.4, 0.5, 0.6]
    V(how) = [0.4, 0.5, 0.6]

    Q(are) = [0.7, 0.8, 0.9]
    K(are) = [0.7, 0.8, 0.9]
    V(are) = [0.7, 0.8, 0.9]

    Q(you) = [1.0, 1.1, 1.2]
    K(you) = [1.0, 1.1, 1.2]
    V(you) = [1.0, 1.1, 1.2]
    ```

在实际应用中，Q、K、V向量通常是通过矩阵乘法和偏置加法等操作从输入数据中学习得到的，因此它们的具体值会随着模型的训练而变化。以上是一个简单的例子，用于说明Q、K、V向量是如何与输入数据相关联的。但是在实际中初始化他们的权重是比较复杂的，需要神经网络模型在学习中得到。

在多头注意力机制中，首先，将输入序列通过一个线性变换层映射到更高维度的空间。这个映射通常是通过一个全连接层实现的，该层包括一个权重矩阵和一个偏置向量。假设输入序列的维度是d_model，映射后的维度是d_k。将映射后的向量分割成多个头（例如，如果有n_heads个头，就将其分割成n_heads个子向量）。这一步是为了引入多头注意力机制，以增加模型的表示能力。

对每个头分别计算Q、K、V向量。这一步通常是通过对映射后的向量进行切片操作来实现的。将计算得到的Q、K、V向量作为输入，应用注意力机制来计算加权和。这一步与普通的注意力机制相同，通过计算Q和K的点积来获取注意力分数，然后将分数与V向量相乘并加权求和。

最后，将多个头的结果合并起来，形成最终的输出。这一步通常是将多个头的结果连接起来，并通过一个线性变换层进行再次映射，以得到最终的输出。

这种方法通过引入线性变换层和多头注意力机制，能够提高模型的表示能力和学习能力，使得模型能够更好地捕捉输入序列的语义信息和相关性。

