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

## Seq2SeqModel

```python
import tensorflow as tf

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
```
