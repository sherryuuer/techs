## Seq2seq

Seq2seq（Sequence to Sequence）是一种用于将一个序列转换为另一个序列的深度学习模型架构，常用于机器翻译、文本摘要等任务。它由编码器（encoder）和解码器（decoder）两部分组成。编码器将输入序列编码为一个固定长度的上下文向量，解码器根据该向量生成输出序列。

一切都是数字，而语言，声音，视频在计算机的世界，一切都是序列。

历史上，**贝叶斯统计**在序列处理中的应用包括隐马尔可夫模型（HMM）和动态贝叶斯网络（DBN）。

1. **隐马尔可夫模型（HMM）**：用于处理时间序列数据，特别是语音识别和生物信息学。HMM利用贝叶斯推断来确定隐藏状态和观察序列之间的关系。

2. **动态贝叶斯网络（DBN）**：扩展了HMM，通过贝叶斯网络表示时间序列中的随机变量及其依赖关系，广泛应用于时序预测和机器人定位等领域。

这些方法利用贝叶斯统计的原理进行概率推断，处理不确定性和时间依赖性。

**Encoder-decoder**是一种神经网络架构，主要用于处理序列到序列的任务，如机器翻译和文本摘要。

1. **编码器（Encoder）**：将输入序列转换为一个固定长度的上下文向量（或一组向量）。它通常由一系列RNN、LSTM或Transformer层组成。

2. **解码器（Decoder）**：从编码器生成的上下文向量中生成输出序列。解码器也通常由一系列RNN、LSTM或Transformer层组成。

编码器-解码器架构可以处理变长的输入和输出序列，并且广泛应用于自然语言处理领域。

## 训练任务和数据

对于 seq2seq 模型，我们使用包含**输入序列和输出序列的训练对**。例如，在机器翻译中，输入序列是一种语言的句子，而输出序列是另一种语言的句子的正确翻译。

在训练期间，我们执行两项任务：

- 输入任务：从输入序列中提取有用信息
- 输出任务：使用输入序列和*输出序列中前一个单词的信息*，计算每个输出时间步骤的单词概率

输入任务在NLP中很常见（例如文本分类），不需要对输入序列进行任何处理。输出任务相当于语言建模的任务，因此需要对输出序列进行处理。

我们将输出序列处理成两个独立的序列：一个是**真实值序列（ground truth sequence）**，另一个是**最终的预测序列（final token sequence）**。这种处理方式常见于模型训练和评估过程中，用于对比模型的预测结果与真实值，评估其性能和准确度。

1. **真实值序列（Ground Truth Sequence）**：
   - 对于seq2seq模型来说，真实值序列相当于语言模型中的输入序列。
   - 它表示用来计算输出序列中每个时间步上单词概率的序列前缀。
   - 也就是说，真实值序列用于指导模型在每一步生成正确的单词。

2. **最终预测序列（Final Token Sequence）**：
   - 对于seq2seq模型来说，最终预测序列相当于训练语言模型时的输出序列。
   - 它表示模型应该根据真实值序列中的前缀预测的“正确”单词。
   - 换句话说，最终预测序列是模型需要生成的正确输出，用于与真实值序列进行对比以评估模型的准确性。

总之，使用真实值序列来*计算预测的概率*，并生成最终的预测序列来*进行模型训练和评估*。

同时**SOS**和**EOS**是标记序列开始和结束的重要token。

一个训练数据的示例：

- Input Seq:["SOS", "he", "eats", "bread", "EOS"]
- Output Seq:["SOS", "il", "mange", "du", "pain", "EOS"]
- Ground Truth Seq:["SOS", "il", "mange", "du"]
- Final Token Seq:["mange", "du", "pain", "EOS"]

## Final State

LSTM的final state通常指的是最后一个时间步的隐藏状态（hidden state）和细胞状态（cell state）。这些状态总结了输入序列的信息，可以用于后续任务，如预测或分类。

他是encoder的输出。分别是隐藏状态和细胞状态。

隐藏状态（hidden state）和细胞状态（cell state）是LSTM网络中的两个重要状态。

- **隐藏状态（hidden state, h_t）**：它捕捉了直到当前时间步的*短期记忆*，反映了LSTM对当前输入的响应。隐藏状态用于输出当前时间步的结果。
- **细胞状态（cell state, c_t）**：它携带了LSTM的*长期记忆信息*，通过门控机制来保存或忘记信息，从而在整个序列中传递信息。

隐藏状态和细胞状态共同决定了LSTM的表现和记忆能力。

```python
import tensorflow as tf

# Input sequences (embedded)
# Shape: (batch_size（省略了）, max_seq_len, embed_dim)
input_embeddings = tf.keras.Input(shape=(None, 4))

cell = tf.keras.layers.LSTMCell(5)
rnn = tf.keras.layers.RNN(cell, return_state=True)
output, state_h, state_c = rnn(input_embeddings)

final_state = (state_h, state_c)

print(final_state)
```

下面是一个**Multi-layer的final state**：

```python
import tensorflow as tf

# Input sequences (embedded)
# Shape: (batch_size, max_seq_len, embed_dim)
input_embeddings = tf.keras.Input(shape=(None, 4))

cell1 = tf.keras.layers.LSTMCell(5)
cell2 = tf.keras.layers.LSTMCell(8)
multi_cell = tf.keras.layers.StackedRNNCells([cell1, cell2])
rnn = tf.keras.layers.RNN(multi_cell, return_state=True, dtype=tf.float32)
outputs = rnn(input_embeddings)

final_state_cell1 = outputs[1]
final_state_cell2 = outputs[2]
print(final_state_cell1)  # layer 1
print(final_state_cell2)  # layer 2
```

**双向长短期记忆网络（Bidirectional Long Short-Term Memory，BiLSTM）**是一种特殊的LSTM结构，它通过两个独立的LSTM层处理输入序列，一个从前向后处理（正向LSTM），另一个从后向前处理（反向LSTM）。这种结构使模型可以同时利用前后文信息，从而在许多自然语言处理任务中表现更好。

在BiLSTM中：
- **正向LSTM**：从序列的起点到终点处理数据。
- **反向LSTM**：从序列的终点到起点处理数据。
- **输出**：正向和反向LSTM的输出通常会连接起来（concatenate），然后输入到后续层。

这种结构可以捕捉到输入序列的双向依赖性，从而提升模型的性能。

下面是一个使用BiLSTM的代码示例：

```python
import tensorflow as tf

# Input sequences (embedded)
# Shape: (batch_size, max_seq_len, embed_dim)
input_embeddings = tf.keras.Input(shape=(None, 4))

bilstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(5, return_sequences=True, return_state=True))
outputs, forward_h, forward_c, backward_h, backward_c = bilstm(input_embeddings)

final_state_h = tf.concat([forward_h, backward_h], axis=-1)
final_state_c = tf.concat([forward_c, backward_c], axis=-1)

print(final_state_h)
print(final_state_c)
```

## Seq2SeqModel

```python
import tensorflow as tf
from tensorflow import feature_column as fc

# Seq2seq model
class Seq2SeqModel(object):
    def __init__(self, vocab_size, num_lstm_layers, num_lstm_units):
        self.vocab_size = vocab_size
        # Extended vocabulary includes start, stop token
        self.extended_vocab_size = vocab_size + 2
        self.num_lstm_layers = num_lstm_layers
        self.num_lstm_units = num_lstm_units
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(
            num_words=vocab_size)

    # Create a sequence training tuple from input/output sequences
    def make_training_tuple(self, input_sequence, output_sequence):
        truncate_front = output_sequence[1:]
        truncate_back = output_sequence[:-1]
        sos_token = [self.vocab_size]
        eos_token = [self.vocab_size + 1]
        input_sequence = sos_token + input_sequence + eos_token
        # 表示输出序列前缀，也就是不包括最后一个单词
        ground_truth = sos_token + truncate_back
        # 表示根据前缀预测的正确单词，不包括第一个但是有最后一个
        final_sequence = truncate_front + eos_token
        return (input_sequence, ground_truth, final_sequence)

    def make_lstm_cell(self, dropout_keep_prob, num_units):
        cell = tf.keras.layers.LSTMCell(num_units, dropout=dropout_keep_prob)
        return cell

    # Create multi-layer LSTM
    def stacked_lstm_cells(self, is_training, num_units):
        dropout_keep_prob = 0.5 if is_training else 1.0
        cell_list = [self.make_lstm_cell(dropout_keep_prob, num_units) for i in range(self.num_lstm_layers)]
        cell = tf.keras.layers.StackedRNNCells(cell_list)
        return cell

    # Get embeddings for input/output sequences
    # 这个函数的主要目的是通过嵌入层将输入序列转换为嵌入向量，以便在神经网络中进一步处理
    def get_embeddings(self, sequences, scope_name):
        with tf.compat.v1.variable_scope(scope_name, reuse=tf.compat.v1.AUTO_REUSE): # reuse 自动重用已有变量
            # 创建一个类别列，用于将序列转换为整数索引，范围是 [0, extended_vocab_size-1]
            cat_column = fc.categorical_column_with_identity('sequences', self.extended_vocab_size)
            # 根据词汇表大小计算嵌入维度，通常是词汇表大小的四次方根
            embed_size = int(self.extended_vocab_size**0.25)
            # 创建一个嵌入列，将类别列映射到低维向量空间
            embedding_column = fc.embedding_column(cat_column, embed_size)
            # 将输入序列包装成字典格式，以便于后续处理
            seq_dict = {'sequences': sequences}
            # 使用输入层函数生成嵌入。该函数会根据提供的嵌入列将输入序列映射到嵌入向量
            embeddings = fc.input_layer(seq_dict, [embedding_column])
            # 创建一个 tf.keras.Input 用于表示序列的长度，占位符允许在运行时输入实际的序列长度
            sequence_lengths = tf.keras.Input(shape=(), dtype=tf.int64, name=scope_name + "/sinput_layer/sequence_length")
            # 返回嵌入和序列长度（将序列长度转换为 int32 类型）
            return embeddings, tf.cast(sequence_lengths, tf.int32)

    # Create the encoder for the model
    def encoder(self, encoder_inputs, is_training):
        input_embeddings, input_seq_lens = self.get_embeddings(encoder_inputs, 'encoder_emb')
        cell = self.stacked_lstm_cells(is_training, self.num_lstm_units)

        rnn = tf.keras.layers.RNN(
            cell,
            return_sequences=True,
            return_state=True,
            go_backwards=True,
            dtype=tf.float32
        )
        Bi_rnn = tf.keras.layers.Bidirectional(
            rnn,
            merge_mode='concat'
        )
        input_embeddings = tf.reshape(input_embeddings, [-1, -1, 2])
        outputs = Bi_rnn(input_embeddings)

        states_fw = [outputs[i] for i in range(1, self.num_lstm_layers + 1)]
        states_bw = [outputs[i] for i in range(self.num_lstm_layers + 1, len(outputs))]

        for i in range(self.num_lstm_layers):
            bi_state_c, bi_state_h = self.ref_get_bi_state_parts(states_fw[i], states_bw[i])

        return outputs, states_fw, states_bw
```
