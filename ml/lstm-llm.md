## 语言模型的工作原理

语言模型的目标是为文本序列中的单词分配概率，通过将每个词汇单词看作一个类别，使用前面的单词作为输入，计算每个词汇类别的概率，类似于**多类别分类问题**。

训练语言模型的主要思想是根据前面的单词预测序列中的每个单词。训练对由相等长度的输入和目标序列组成，目标序列是输入序列向右移一个单词。

为语言模型训练创建输入-目标序列时，需要考虑序列的长度。可以选择将训练对序列限制为**固定的最大长度**，以提高训练速度并避免过度拟合不常见的文本依赖关系。输出是输入的文本向后移动一个单词。

如下代码所示：要构建的LanguageModel对象中的`truncate_sequences`就是为了达到这个目的。

```python
import tensorflow as tf

def truncate_sequences(sequence, max_length):
    input_sequence = sequence[:max_length - 1]
    target_sequence = sequence[1:max_length]
    return (input_sequence, target_sequence)

# LSTM Language Model
class LanguageModel(object):
    # Model Initialization
    def __init__(self, vocab_size, max_length, num_lstm_units, num_lstm_layers):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.num_lstm_units = num_lstm_units
        self.num_lstm_layers = num_lstm_layers
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size)

    def get_input_target_sequence(self, sequence):
        seq_len = len(sequence)
        if seq_len >= self.max_length:
            input_sequence, target_sequence = truncate_sequences(
                sequence, self.max_length
            )
        else:
            # Next chapter
            input_sequence, target_sequence = pad_sequences(
                sequence, self.max_length
            )
        return input_sequence, target_sequence
```

总体而言，语言模型通过多类别分类的方式，根据上下文预测每个单词的概率。在训练中，使用输入-目标序列的方式进行，其中目标序列是输入序列的右移版本。同时，可以考虑限制序列长度以优化训练效果。

## 使用Padding的方法使输入等长

大多数神经网络处理固定长度的输入数据，因为它们具有前馈结构，使用多个固定大小的层来计算网络的输出。然而，由于文本数据涉及不同长度的文本序列（例如句子、段落等），语言模型需要能够处理不同长度的输入数据。因此，我们使用递归神经网络（在接下来的章节中详细讨论）作为语言模型的基础。

虽然递归神经网络结构允许语言模型接受不同长度的输入文本，但在训练批次`batch`中，仍然需要每个经过标记化的文本序列具有相同的长度。这是因为训练批次必须具有适当的张量`tensor`形状，即在这种情况下需要是一个二维矩阵。

为了模拟具有不同长度序列的适当二维矩阵形状，我们采用填充的方式。对于比最大序列长度短的每个序列，我们在其末尾添加一个*特殊的非词汇标记*，直到其长度等于最大序列长度。通常，特殊的填充标记被赋予*ID 0*，而每个词汇单词的ID是正整数。因为Tokenizer对象从来不用0作转换。

下面的代码中的`pad_sequences`辅助函数就实现了这个功能。

```python
import tensorflow as tf

def pad_sequences(sequence, max_length):
    padding_amount = max_length - len(sequence)
    padding = [0] * padding_amount
    input_sequence = sequence[:-1] + padding
    target_sequence = sequence[1:] + padding
    return (input_sequence, target_sequence)

# LSTM Language Model
class LanguageModel(object):
    # Model Initialization
    def __init__(self, vocab_size, max_length, num_lstm_units, num_lstm_layers):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.num_lstm_units = num_lstm_units
        self.num_lstm_layers = num_lstm_layers
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size)

    def get_input_target_sequence(self, sequence):
        seq_len = len(sequence)
        if seq_len >= self.max_length:
            input_sequence, target_sequence = truncate_sequences(
                sequence, self.max_length
            )
        else:
            input_sequence, target_sequence = pad_sequences(
                sequence, self.max_length
            )
        return input_sequence, target_sequence
```

## RNN&LSTM

 **RNN的结构和运行机制**

**核心点:**
- RNN 由单个单元组成，并具有输入、输出和循环连接。
- 在展开状态传递的图表中，每个时间步代表一个输入和输出。
- 箭头表示信息流动，包括输入、输出和**状态传递**。
- 循环连接在每个时间步之间传递状态，是 RNN 的核心。
- 状态存储了之前时间步的信息，用于处理序列依赖关系。

**关键概念:**
- 时间步（Time steps）：RNN 处理序列数据的序列长度。
- 状态（State）：存储之前时间步信息的变量，用于捕捉序列依赖关系。
- 展开图（Unrolled diagram）：用于表示 RNN 在每个时间步的操作。

但是RNN存在长距离依赖问题: 普通的RNN在处理文本序列时，会随着序列长度的增加，逐渐遗忘前面的信息，导致难以捕捉长距离依赖关系。

这时候LSTM（长短期记忆算法）的优势就在于: LSTM通过引入特殊的门控机制来解决RNN的长距离依赖问题。这些门控机制可以控制信息的流动，使得LSTM能够记住序列中较早出现的重要信息，并有效地处理长距离依赖关系。LSTM在自然语言处理（NLP）领域被广泛应用，例如机器翻译、文本生成、语音识别、情感分析等任务。

**1. 默认RNN单元的内部结构包括:**

- **两层全连接层:**
    - 第一层：使用tanh激活函数，用于计算当前时间步的细胞状态，基于前一个状态和当前输入。
    - 第二层：没有激活函数，用于计算当前时间步的记忆细胞输出。（其实这里的细胞cell也可以翻译成别的？但是感觉翻译成细胞就有种神奇的感觉，宛如是记忆细胞的意思）
- **隐藏单元:** 隐藏单元的数量对应全连接层中隐藏单元的数量，决定了模型的学习能力。

**2. LSTM单元的改进:**

- **添加门控机制:** LSTM在默认RNN单元的基础上增加了几个门控层，用于调节细胞状态的信息流动，解决长距离依赖问题。
- **门控层类型:**
    - 输入门（Input Gate）：控制哪些新信息进入细胞状态。
    - 遗忘门（Forget Gate）：控制哪些旧信息从细胞状态中遗忘。
    - 输出门（Output Gate）：控制哪些信息从细胞状态中输出作为最终输出。

**3. 门控机制的作用:**

- **选择性记忆：** 通过门控机制，LSTM可以选择性地记住或遗忘信息，使得它能够处理长距离依赖关系。
- **长距离依赖：** LSTM可以将重要信息在序列中长时间传递，并在适当的时候使用这些信息进行预测。

总之，默认RNN单元在处理长距离依赖方面存在问题。而LSTM通过引入门控机制来解决这个问题，并有效地处理长距离依赖关系。门控机制是LSTM的核心，使得它能够选择性地记忆和遗忘信息。

关注代码的`make_lstm_cell`初始化程序的方法，唯一需要关注的就是细胞的个数。这里用的是keras的内置方法。

```python
import tensorflow as tf

# LSTM Language Model
class LanguageModel(object):
    # Model Initialization
    def __init__(self, vocab_size, max_length, num_lstm_units, num_lstm_layers):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.num_lstm_units = num_lstm_units
        self.num_lstm_layers = num_lstm_layers
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size)

    def make_lstm_cell(self, dropout_keep_prob):
        cell = tf.keras.layers.LSTMCell(self.num_lstm_units)
        return cell
```

## Dropout

Dropout是一种正则化技术，通过随机丢弃网络中的部分节点来防止过拟合。在循环神经网络（RNN）中，Dropout可以应用于输入、输出和隐含层。

**应用方式:** 

- **输入Dropout:** 在训练过程中，随机丢弃部分输入数据，迫使模型学习更鲁棒的特征。
- **输出Dropout:** 在训练过程中，随机丢弃部分输出数据，防止模型过度依赖特定的输出模式。
- **隐含层Dropout:** 在训练过程中，随机丢弃部分隐含层节点，防止模型过度依赖特定的内部状态。

**作用:**

- **防止过拟合:** Dropout可以有效地防止RNN过拟合训练数据，提高模型的泛化能力。
- **提高模型鲁棒性:** Dropout可以使RNN模型更加鲁棒，对噪声和干扰数据具有更强的抵抗力。
- **缓解梯度消失和爆炸:** Dropout可以一定程度上缓解RNN训练过程中出现的梯度消失和爆炸问题。

**注意事项:**

- **Dropout率:** Dropout率控制着被丢弃节点的比例，需要根据具体的模型和数据集进行调整。
- **训练阶段:** Dropout只在训练阶段使用，在测试阶段需要关闭。
- **应用范围:** Dropout可以应用于各种类型的RNN，例如LSTM、GRU等。

**一些关于RNN中Dropout的额外信息:**

* 研究表明，在RNN中使用Dropout可以有效提高模型的性能，特别是在处理长序列数据时。
* 不同的Dropout策略对RNN模型的影响可能有所不同，需要进行实验找到最优策略。
* Dropout与其他正则化技术可以结合使用，以获得更好的效果。

Tensorflow2.0的封装方法：

```python
import tensorflow as tf

# 假设我们需要一个 LSTM 单元
units = 128  # 例如，LSTM单元的数量
dropout_keep_prob = 0.5  # 保留50%的输出

# 创建一个 LSTM 单元
lstm_cell = tf.keras.layers.LSTMCell(units)

# 将 LSTM 单元包装在一个 RNN 层中，并应用 dropout
dropout_cell = tf.keras.layers.RNN(
    lstm_cell,
    dropout=1 - dropout_keep_prob,  # 设置 dropout 概率
    return_sequences=True  # 如果需要返回完整的序列，而不仅仅是最终的状态
)

# 或者创建完整的 LSTM 层，并应用 dropout
lstm_layer = tf.keras.layers.LSTM(
    units,
    dropout=1 - dropout_keep_prob,  # 设置 dropout 概率
    return_sequences=True  # 如果需要返回完整的序列，而不仅仅是最终的状态
)
```

下面的代码中的`make_lstm_cell`方法实现了对输出进行dropout。

```python
import tensorflow as tf

# LSTM Language Model
class LanguageModel(object):
    # Model Initialization
    def __init__(self, vocab_size, max_length, num_lstm_units, num_lstm_layers):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.num_lstm_units = num_lstm_units
        self.num_lstm_layers = num_lstm_layers
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size)

    def make_lstm_cell(self, dropout_keep_prob):
        cell = tf.keras.layers.LSTMCell(self.num_lstm_units) 
        # apply dropout to output
        # 1.0 version
        # dropout_cell = tf.compat.v1.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=dropout_keep_prob)
        # dropout_keep_prob 是在每次训练迭代中保留输出的概率
        # 2.0 version
        dropout_keep_prob = 0.5  # 例如 0.5 表示保留50%的输出
        dropout_cell = tf.keras.layers.RNN(
            cell,
            dropout=1 - dropout_keep_prob,  # 设置 dropout 概率
            return_sequences=True  # 如果需要返回完整的序列，而不仅仅是最终的状态
        )
        return dropout_cell
```

## RNN的多层结构

和所有的深度学习人工神经网络一样，RNN也有多层结构。

代码中的 cell，是一个 Keras 层，允许将多个 RNN 单元堆叠在一起，从而形成一个多层的 RNN 结构。每一层的输出作为下一层的输入。这个层本质上是一个*容器*，可以包含多个 RNN 单元（如 SimpleRNNCell, LSTMCell, GRUCell 等）。

```python
import tensorflow as tf

# LSTM Language Model
class LanguageModel(object):
    # Model Initialization
    def __init__(self, vocab_size, max_length, num_lstm_units, num_lstm_layers):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.num_lstm_units = num_lstm_units
        self.num_lstm_layers = num_lstm_layers
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size)

    # Stack multiple layers for the LSTM
    def stacked_lstm_cells(self, is_training):
        dropout_keep_prob = 0.5 if is_training else 1.0
        cell_list = [self.make_lstm_cell(dropout_keep_prob) for i in range(self.num_lstm_layers)]
        cell = tf.keras.layers.StackedRNNCells(cell_list)
        return cell
```

## LSTM的输出

下面这段demo代码演示了如何使用TensorFlow创建一个基本的LSTM模型，并对序列数据进行处理。该模型可以用于解决各种序列问题，例如文本分类、时间序列预测等。通过理解这段代码，基本可以理解lstm的输出原理。

```python
import tensorflow as tf
# 设置一个序列长度的lens列表
lens = [4, 9, 10, 5, 10]
# 创建一个lstm单元，里面包含7个隐藏单元，这些隐藏单元可以记录前面时间步的信息
cell = tf.keras.layers.LSTMCell(units=7)
# input_sequences是一个输入占位符
# Shape: (time_steps, embed_dim)
# time_steps：表示序列的长度，这里为10。
# embed_dim：表示每个时间步的向量维度（特征），这里为20。
input_sequences = tf.keras.Input(
    shape=(10, 20),
    dtype=tf.float32
)
# 创建一个rnn层，使用前面创建的单元
# 设定return_sequences为true表示返回所有时间步的输出，反之只输出最后一个
# 设定input_length使得跳过处理padding部分的资源浪费
rnn=tf.keras.layers.RNN(
    cell,
    return_sequences=True,
    input_length=lens,
    dtype=tf.float32
    )
# 将输入序列 input_sequences 送入RNN层进行处理，得到输出 output。
# output 的形状为 (batch_size, time_steps, 7)，其中7表示每个时间步的输出向量维度。
output = rnn(input_sequences)
print(output)
```

那么这里模型的代码实现，关注`run_lstm`对lstm对输出的实现。

在代码中，二元序列（binary_sequences）的作用是用于计算输入序列的实际长度。具体来说，它将输入序列中的非零值转换为 1，填充值（通常为零）保持为 0。通过这种方式，可以忽略填充值，仅考虑实际的序列长度。

知道每个序列的实际长度可以避免对填充值的无效计算，从而提高计算效率。

序列模型（如 RNN、LSTM）在处理变长序列时，可以使用掩码 masking 来忽略填充值的影响。通过二元序列，可以生成相应的掩码，确保模型只关注实际的有效数据。

下面的demo演示了二元序列的计算：

```python
input_sequences = tf.constant([
    [1, 2, 3, 0, 0],  # 长度为 3
    [4, 5, 0, 0, 0],  # 长度为 2
    [6, 7, 8, 9, 0]   # 长度为 4
])

binary_sequences = tf.math.sign(input_sequences)
# binary_sequences = [[1, 1, 1, 0, 0], 
#                     [1, 1, 0, 0, 0], 
#                     [1, 1, 1, 1, 0]]

sequence_lengths = tf.math.reduce_sum(binary_sequences, axis=1)
# sequence_lengths = [3, 2, 4]
```

具体代码：

```python
import tensorflow as tf
import numpy as np
# LSTM Language Model
class LanguageModel(object):
    # Model Initialization
    def __init__(self, vocab_size, max_length, num_lstm_units, num_lstm_layers):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.num_lstm_units = num_lstm_units
        self.num_lstm_layers = num_lstm_layers
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size)

    # Create a cell for the LSTM
    def make_lstm_cell(self, dropout_keep_prob):
        cell = tf.keras.layers.LSTMCell(self.num_lstm_units, dropout=dropout_keep_prob)
        return cell

    # Stack multiple layers for the LSTM
    def stacked_lstm_cells(self, is_training):
        dropout_keep_prob = 0.5 if is_training else 1.0
        cell_list = [self.make_lstm_cell(dropout_keep_prob) for i in range(self.num_lstm_layers)]
        cell = tf.keras.layers.StackedRNNCells(cell_list)
        return cell

     # Convert input sequences to embeddings
    def get_input_embeddings(self, input_sequences):
        embedding_dim = int(self.vocab_size**0.25)
        embedding=tf.keras.layers.Embedding(
            self.vocab_size + 1, # 包含一个填充token
            embedding_dim, 
            embeddings_initializer='uniform', # 初始化方法均匀分布
            mask_zero=True,  # 将0作为填充标记
            input_length=self.max_length
        )
        input_embeddings = embedding(input_sequences)
        return input_embeddings
    
    # Run the LSTM on the input sequences
    def run_lstm(self, input_sequences, is_training):
        cell = self.stacked_lstm_cells(is_training)
        input_embeddings = self.get_input_embeddings(input_sequences)
        # 二元序列
        binary_sequences = tf.math.sign(input_sequences)
        sequence_lengths = tf.math.reduce_sum(binary_sequences, axis=1)
        rnn = tf.keras.layers.RNN(cell, 
                                  return_sequences=True,
                                  input_length=sequence_lengths,
                                  dtype=tf.float32)
        lstm_outputs = rnn(input_embeddings)
        return (lstm_outputs, binary_sequences)
```

## 损失计算

通过一个demo理解一下lstm的损失计算：

```python
import tensorflow as tf
# 定义一个占位符 lstm_outputs，用于接收来自LSTM层的输出数据。
# Shape: (batch_size, time_steps, cell_size)
# time_steps: 表示序列的长度，这里为10。
# cell_size: 表示LSTM单元的隐藏单元数量，这里为7。
lstm_outputs = tf.keras.Input(shape=(10, 7), tf.float32)
# 定义变量 vocab_size，表示词汇表的大小，即所有可能的词的数量。
vocab_size = 100
# print(lstm_outputs)
# 使用 tf.keras.layers.Dense 添加一个全连接层，将LSTM输出转换为最终的预测值。
# 全连接层的输出维度为 vocab_size，表示每个时间步预测每个词的概率。
logits = tf.keras.layers.Dense(units=vocab_size)(lstm_outputs)
# Target tokenized sequences：定义一个占位符 target_sequences，用于接收真实的目标序列数据。
# Shape: (time_steps, cell_size) 其中每个元素表示序列中每个位置的词对应的索引。
target_sequences = tf.keras.Input(shape=(None, 10), tf.int64)
# 使用 tf.nn.sparse_softmax_cross_entropy_with_logits 计算模型的损失函数。
# labels 表示真实的标签序列，logits 表示模型的预测值。
# 该函数会计算每个时间步的预测值与真实标签之间的交叉熵损失，并将所有时间步的损失求平均值。
loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
    labels=target_sequences,
    logits=logits)
```

引入一个padding mask防止对padding对部分进行无用的损失计算。如下demo代码。

```python
import tensorflow as tf
# loss: Softmax loss for LSTM
with tf.compat.v1.Session() as sess:
    print(repr(sess.run(loss)))

# Same shape as loss
pad_mask = tf.constant([
    [1., 1., 1., 1., 0.],
    [1., 1., 0., 0., 0.]
])

new_loss = loss * pad_mask
with tf.compat.v1.Session() as sess:
    print(repr(sess.run(new_loss)))
```

继续完成模型代码：关注`calculate_loss`方法，通过和二元运算符做乘法运算，排除了对padding处的loss计算。

```python
import tensorflow as tf

class LanguageModel(object):
    # Model Initialization
    def __init__(self, vocab_size, max_length, num_lstm_units, num_lstm_layers):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.num_lstm_units = num_lstm_units
        self.num_lstm_layers = num_lstm_layers
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size)


    def make_lstm_cell(self, dropout_keep_prob):
        cell = tf.keras.layers.LSTMCell(self.num_lstm_units, dropout=dropout_keep_prob)
        return cell

    # Stack multiple layers for the LSTM
    def stacked_lstm_cells(self, is_training):
        dropout_keep_prob = 0.5 if is_training else 1.0
        cell_list = [self.make_lstm_cell(dropout_keep_prob) for i in range(self.num_lstm_layers)]
        cell = tf.keras.layers.StackedRNNCells(cell_list)
        return cell_list

    # Convert input sequences to embeddings
    def get_input_embeddings(self, input_sequences):
        embedding_dim = int(self.vocab_size**0.25)
        embedding=tf.keras.layers.Embedding(
            self.vocab_size+1, embedding_dim, embeddings_initializer='uniform',
            mask_zero=True, input_length=self.max_length
        )
        input_embeddings = embedding(input_sequences)
        return input_embeddings

    def run_lstm(self, input_sequences, is_training):
        cell = self.stacked_lstm_cells(is_training)
        input_embeddings = self.get_input_embeddings(input_sequences)
        binary_sequences = tf.math.sign(input_sequences)
        sequence_lengths = tf.math.reduce_sum(binary_sequences, axis=1)
        rnn=tf.keras.layers.RNN(
            cell,
            return_sequences=True,
            input_length=sequence_lengths,
            dtype=tf.float32
        )
        lstm_outputs = rnn(input_embeddings)
        return lstm_outputs, binary_sequences
    
    def calculate_loss(self, lstm_outputs, binary_sequences, output_sequences):
        logits = tf.keras.layers.Dense(self.vocab_size)(lstm_outputs)
        batch_sequence_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels = output_sequences, 
            logits = logits)
        unpadded_loss = tf.cast(binary_sequences, tf.float32) * batch_sequence_loss
        overall_loss = tf.math.reduce_sum(unpadded_loss)
        return overall_loss
```

### 计算概率和单词预测

```python
import tensorflow as tf

# 假设我们有一个logits张量，形状为 (batch_size, 5, 100)
logits = tf.random.uniform(shape=(32, 5, 100))  # 示例张量，替代占位符
probabilities = tf.nn.softmax(logits, axis=-1)

print(probabilities)
```

```python
import tensorflow as tf

# 假设我们有一个probabilities张量，形状为 (batch_size, 5, 100)
probabilities = tf.random.uniform(shape=(32, 5, 100))  # 示例张量，替代占位符
word_preds = tf.argmax(probabilities, axis=-1)

print(word_preds)
```

虽然相比较注意力机制长短记忆模型已经过时了一点点但是我们依然需要了解和学习。

### 张量切片

和一般的python列表的切片不一样，tensor的切片在tf中使用gather和gather_nd方法。

`tf.gather` 和 `tf.gather_nd` 都是 TensorFlow 中用于从张量中选择元素的操作，但它们有一些关键的区别：

1. **维度选择：**
   - `tf.gather`: 主要用于在指定轴上选择特定的索引，可以选择单个元素或按指定轴选择一部分元素。
   - `tf.gather_nd`: 更为灵活，可以在多个维度上选择特定的索引，以实现更复杂的选择。

2. **索引的表示：**
   - `tf.gather`: 通过指定索引列表或张量来选择元素。可以选择单个索引、一维索引数组或多维索引数组。
   - `tf.gather_nd`: 通过指定一个多维索引的张量来选择元素。这个多维索引的张量的形状决定了选择的元素数量和维度。

下面是两个操作的简要使用示例：

使用 `tf.gather`:

```python
import tensorflow as tf

# 创建一个常量张量 t，包含元素 [1, 2, 3, 4, 5, 6]
t = tf.constant([1, 2, 3, 4, 5, 6])

# 使用 tf.gather，在轴 0 上选择索引为 [1, 3, 5] 的元素
selected_elements = tf.gather(t, [1, 3, 5])
```

使用 `tf.gather_nd`:

```python
import tensorflow as tf

# 创建一个常量张量 t，包含二维矩阵
t = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# 使用 tf.gather_nd，在轴 0 和轴 1 上选择指定索引的元素
indices = tf.constant([[0, 1], [2, 0]])
selected_elements = tf.gather_nd(t, indices)
```

如果你只需在一个轴上选择特定索引的元素，使用 `tf.gather` 更为简单。但如果你需要在多个维度上进行复杂的选择，那么 `tf.gather_nd` 更为适用。

### 预测下一个单词

`get_word_predictions`方法通过接收两个张量参数，`word_preds`和`binary_sequences`，以及一个表示批次大小的标量`batch_size`。

通过计算每个序列中最后一个位置的索引，构造了用于提取最后一个预测单词 ID 的索引。

使用 tf.gather_nd 操作从 word_preds 中提取最后一个预测单词 ID，并将结果返回。

```python
import tensorflow as tf

# LSTM Language Model
class LanguageModel(object):
    # Model Initialization
    def __init__(self, vocab_size, max_length, num_lstm_units, num_lstm_layers):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.num_lstm_units = num_lstm_units
        self.num_lstm_layers = num_lstm_layers
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size)

    # Predict next word ID
    def get_word_predictions(self, word_preds, binary_sequences, batch_size):
        row_indices = tf.range(batch_size)
        final_indexes = tf.math.reduce_sum(binary_sequences, axis=1) - 1
        gather_indices = tf.transpose([row_indices, final_indexes])
        final_id_predictions = tf.gather_nd(word_preds, gather_indices)
        return final_id_predictions
```
