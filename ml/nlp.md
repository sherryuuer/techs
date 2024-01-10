## Tensorflow 自然语言处理流程

---

### 自然语言处理的主要应用领域

机器翻译，文本处理，情感分析，文本分类。

### 在 tf 中的主要处理流程是什么

- 数据准备
- 数据处理，词向量化，词嵌入
- 建模（包括从 hub 提取好用的迁移模型）
- 编译模型
- 拟合模型
- 通过测试集预测和查看结果
- 更多的模型试验和最终模型比较
- 查看错误的预测，考虑如何通过标签修改提升模型性能
- 应用部署

### 数据前置处理很重要

_防止数据泄漏_

在学习中，第一次将数据打乱后，从打乱的数据中提取训练数据，很容易把测试数据也混淆进训练数据，造成数据泄漏，以至于训练后的结果异常的好。有时候不要在好的预测精度上高兴的太早，预测结果是否合理，要提前心里有数。

所以在数据分割上要做好功课，先分割，提前分割，尽早分割，然后将 test 集放在一边就不要再动了，直到需要测试的时候，再拿出来用。

活用 sklearn 的分割方法：

```python
from sklearn.model_selection import train_test_split
train_sentences, val_sentences, train_labels, val_labels = train_test_split(train_df_shuffled["text"].to_numpy(),
                                                                            train_df_shuffled["target"].to_numpy(),
                                                                            test_size=0.1,
                                                                            random_state=42)
```

_数据可视化_

即使是文本数据，也有必要进行可视化的处理，提前查看，文本的长度，内容，标签，是否正常。

例如：

```python
import random
random_index = random.randint(0, len(train_df)-5)
for row in train_df_shuffled[["text", "target"]][random_index:random_index+5].itertuples():
  _, text, target = row
```

查看全文本数据量，查看全标签是否不平衡。

是否有缺损数据。如果是 kaggle 的数据，大概率没有什么缺损可以直接拿来用。但是生产环境，就需要进行缺损数据的补全，有时候甚至需要机器学习算法来进行更好的补全。

### 词向量化和词嵌入

对于计算机来说，一切都是数字，所以 nlp 也不例外。

词转数字就是词向量化，很好理解，最简单的向量化，就像是 one-hot 的方法，给每个单词一个数字，然后交给一个向量，给他维度。

词嵌入可以理解为，不只是看一个单词，而是解析单词和单词之间的关系的矩阵，同时还有另一个矩阵在内部作用，context 矩阵，该矩阵装着上下文信息，通过每一个批次的处理，使得两个矩阵的数字不断调整，让关系更近的词汇更近，更远的词汇更远。通过学习，提高模型精度。

### 循环神经网络 RNN

这三个模型真的挺好玩的 LSTM，GRU，Bidirectional。

LSTM 是回忆过去的记忆，GRU 是控制是否遗忘记忆，Bidirectional 是预言者模式。

### 1 维 CNN 在处理时间序列和序列化数据中表现出色

虽然说 CNN 一说起来就是图像识别领域，但是在序列化数据中，使用 1 维的 CNN 模型表现也非常出色。

### 不要小看 sk 的模型

基线模型朴素贝叶斯只是用了 tf-idf 处理单词，表现就非常突出，在基础构架中，打败它的只有迁移学习模型 USE。
