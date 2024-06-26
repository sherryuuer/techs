## words = vectors：一切都是数字

词嵌入技术，让自然语言处理方面的问题，可以在深度学习大型网络中tensor flow。LLM的诞生归功于DL，更归功于词嵌入技术。

再说自然语言处理，NLP涵盖了处理人类口头或书面语言的各种任务，包括语言翻译、语音识别、文本分析和自动文本生成等。如语音助手（Siri、Alexa）和搜索引擎（Google、Bing），这些都依赖于NLP技术。尽管不是所有的NLP任务都需要机器学习，一些任务仍然可以通过非机器学习算法完成，比如搜索引擎的相关性匹配。然而，机器学习在NLP中的日益普及，并且预计这个趋势将在未来继续发展。

## 语料库（text corpus）和分词（Tokenization）

语料库是进行处理任务时候的文本集合，比如一整套论文合集，一整个wiki文档库。而整个语料库会对应一个词汇表vocabulary，所有的任务都是围绕词汇展开的。除此之外，还有基于每个字母的词汇表，但是基于单词的更常用。

将一段文本标记化，标记在英文中叫做token，通过将一句话tokenization也就是标记化或者分词化，使得一个句子可以用一个向量表示。关于标点符号，有时候他们也会发挥很重要的作用，因此有时候标点也会被标记化。

```python
import tensorflow as tf
tokenizer = tf.keras.preprocessing.text.Tokenizer()
text_corpus = ['bob ate apples, and pears', 'fred ate apples!']
tokenizer.fit_on_texts(text_corpus)
new_texts = ['bob ate pears', 'fred ate pears']
print(tokenizer.texts_to_sequences(new_texts))
print(tokenizer.word_index)

# [[3, 1, 5], [6, 1, 5]]
# {'bob': 3, 'and': 4, 'ate': 1, 'apples': 2, 'pears': 5, 'fred': 6}
```

在kerasAPI中就有对应的`tokenizer`分词器对象，拥有许多参数，可以将文本转化为token，可以规定最大的词汇数量`num_words`参数用于仅仅保留需要的最常用的词汇，可以规定OOV（out-of-vocabulary词汇表之外的单词）用什么表示。比如下面的例子。

```python
tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token='OOV')
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=2)
```

使用Tensorflow构架一个EmbeddingModel的初始化模型，规定词汇表大小。

设计一个method函数，将指定的texts文本变成句子向量。

```python
import tensorflow as tf

# Skip-gram embedding model
class EmbeddingModel(object):
    # Model Initialization
    def __init__(self, vocab_size, embedding_dim):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=self.vocab_size)

    # Convert a list of text strings into word sequences
    def tokenize_text_corpus(self, texts):
        self.tokenizer.fit_on_texts(texts)
        sequences = self.tokenizer.texts_to_sequences(texts)
        return sequences
```

## Embeddings

**词汇表达（Word representation）**：为了表达词汇之间的关系而不仅仅是将它们变成数字，需要进行嵌入向量操作。也就是向量的高维表达，在高维空间中，可以将向量想象为无数的长短不一的小尖头，他们之间的距离角度都可以用来表达词汇之间的关系。在Tensorflow的[嵌入生成器](https://projector.tensorflow.org/)中就可以看到壮观的宛如星云的词向量表示。

一般来说维度越高的向量，可以表达更丰富的信息，但是维度过高，则会导致计算量过大和过拟合风险变大。

一个词的向量可能非常长，那么**多高维度的向量**是最好的呢。前人经验来说是词汇表中总数量的除以10^3数量的向量，比如有10000个单词的词汇表，那么就表示为10维的向量。当然更多的尝试是最好的。

**目标词汇和上下文窗口（Target-context）**：嵌入向量的基础来自于单词目标和它的上下文窗口。机器学习中我们有目标label和训练数据，在这里，目标就是单个单词，训练材料就是它的上下文，上下文对称存在，所以一个窗口一般是奇数大小。其他的还有，根据上文给出下文等多种方式，你的输入法的候补词汇也是这个原理，它会给我们，一个单词后面可能出现的概率最大的单词们，供我们选择。

取得目标词汇和半窗口大小的函数实现：

```python
def get_target_and_size(sequence, target_index, window_size):
    target_word = sequence[target_index]
    half_window_size = window_size // 2
    return (target_word, half_window_size)
```

取得窗口切片的函数实现：

```python
def get_window_indices(sequence, target_index, half_window_size):
    left_incl = max(0, target_index - half_window_size)
    right_excl = min(len(sequence), target_index + half_window_size + 1)
    return (left_incl, right_excl)
```

然后就可以在model类中使用这两个函数取得目标词汇，和窗口切片索引。

```python
# Skip-gram embedding model
class EmbeddingModel(object):
    # Model Initialization
    def __init__(self, vocab_size, embedding_dim):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=self.vocab_size)

    # Convert a list of text strings into word sequences
    def get_target_and_context(self, sequence, target_index, window_size):
        target_word, half_window_size = get_target_and_size(
            sequence, target_index, window_size
        )
        left_incl, right_excl = get_window_indices(
            sequence, target_index, half_window_size)
        return target_word, left_incl, right_excl
```

## 两种模型Skip-gram/CBOW

Skip-gram和CBOW（Continuous Bag of Words）是用于自然语言处理中的词嵌入（Word Embedding）的两种主要模型。它们都是Word2Vec工具中的模型，旨在将词汇映射到具有连续值的向量空间。以下是对它们的简要解释以及它们之间的区别：

**Skip-gram模型**

Skip-gram模型的目标是从给定的中心词预测其周围的上下文词汇。具体来说，对于一个给定的句子，模型试图通过学习在该句子中每个词的上下文中生成该词的概率分布。这使得模型能够捕捉到词汇之间的语义关系。

它输入：一个中心词。输出：预测该中心词的上下文词。更适合于大型语料库，在处理罕见词汇时效果较好。

比如输入一句话：paul likes singing in french，那么该模型处理得到的训练组为：(singing,paul),(singing,likes),(singing,in),(singing,french)。

**CBOW模型**

CBOW模型与Skip-gram相反，它的目标是通过给定上下文中的词汇来预测中心词。在这个模型中，上下文词被看作是条件，而中心词则是目标。

同样对于上面的例子，这里的连续词袋模型输出，用于训练的数据就是：((paul,likes,in,french),singing)。

它在小型数据集上的训练速度较快。对于频繁出现的common的词汇效果较好。

在 skip-gram 和 CBOW 嵌入模型之间**进行选择**时，需要考虑一些事项。由于 skip-gram 模型会为每个上下文词创建一个训练对，因此它所需的实际数据比 CBOW 模型少得多。不过，这也意味着 CBOW 模型的训练速度更快。

此外，由于 skip-gram 模型为每个目标单词创建多个训练对实例，因此它可以比 CBOW 模型更好地表示罕见单词或短语。另一方面，CBOW 模型为更常见的单词提供了更准确的嵌入。

这里的例子选择Skip-gram模型的实现：

为之前的模型，增加一个函数`create_target_context_pairs`，输出一个pairs列表，是（target，context）组合的列表。也就是从中心词到周围词汇的映射对列表。

```python
# Skip-gram embedding model
class EmbeddingModel(object):  
    # Create (target, context) pairs for a given window size
    def create_target_context_pairs(self, texts, window_size):
        pairs = []
        sequences = self.tokenize_text_corpus(texts)      
        for sequence in sequences:
            for i in range(len(sequence)):
                target_word, left_incl, right_excl = self.get_target_and_context(
                    sequence, i, window_size)
                for j in range(left_incl, right_excl):
                    if j != i:
                        pairs.append((target_word, sequence[j]))
        return pairs
```

## 嵌入矩阵初始化

自然语言处理在深度学习模型中训练，和普通的深度学习一样，需要大型的神经网络，初始化神经网络也就是*初始化嵌入矩阵*，该矩阵就是词汇们的向量表示的初始化。通过深度学习，不断更新权重，最终可以得到一个可以用于推理的嵌入矩阵。使用random.uniform进行随机初始化是一个好的选择。

初始化实现：

```python
def get_initializer(embedding_dim, vocab_size):
    # embedding_dims是词表达的维度，vocab_size是词数量
    # bounds是定义的尺度范围
    initial_bounds = 0.5 / embedding_dim
    initializer = tf.random.uniform((vocab_size, embedding_dim), 
                                    minval=-1 * initial_bounds, 
                                    maxval=initial_bounds)
    return initializer
```

使用上面的初始化函数，构造前向传播forward函数：

- 构造初始化矩阵`self.embedding_matrix`
- 使用embedding_lookup方法，根据相应单词id的，找到在嵌入向量矩阵中的对应向量，并返回。

```python
import tensorflow as tf

# Skip-gram embedding model
class EmbeddingModel(object):
    # Model Initialization
    def __init__(self, vocab_size, embedding_dim):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=self.vocab_size)

    # Forward run of the embedding model to retrieve embeddings
    def forward(self, target_ids):
        initializer = get_initializer(
            self.embedding_dim, 
            self.vocab_size
        )
        # self.embedding_matrix = tf.compat.v1.get_variable('embedding_matrix', initializer=initializer)
        # Tensorflow 2.0
        # Initialize the variable
        self.embedding_matrix = tf.Variable(initializer)
        # embeddings = tf.compat.v1.nn.embedding_lookup(self.embedding_matrix, target_ids)
        embeddings = tf.nn.embedding_lookup(params=self.embedding_matrix, ids=target_ids)
        return embeddings
```

## 候选采样（负采样初始化）

我们知道通常在机器学习领域，你有正样本，也有负样本，但是文本处理方面从刚刚的流程中可以看出，只有正样本，所以这里出现了候选采样的概念，目的就是加入负样本，加速训练。

候选采样（Negative Sampling or Candidate Sampling）是一种用于训练嵌入模型（如Word2Vec）的技术，旨在加速训练过程。这种方法主要用于解决 softmax 函数在大型词汇上的计算效率低下的问题。

在传统的训练过程中，模型通常需要对整个词汇表进行 softmax 计算，以获得与目标词汇相关的概率分布。由于词汇表很大，这样的计算是非常昂贵的。候选采样通过选择一小部分负样本，只计算与这些负样本相关的概率，从而降低了计算成本。

具体步骤如下：

1. **正样本选择：**
   - 对于每个训练样本，有一个目标词汇，这是正样本。

2. **负样本采样：**
   - 从词汇表中随机选择一些词汇作为负样本。这些负样本不是实际上的上下文词汇，而是被认为不是目标词汇的词。

3. **目标函数设计：**
   - 对于每个样本，目标函数会尽量使正样本的得分高，同时使负样本的得分低。常见的目标函数是最大化正样本的概率，最小化负样本的概率。

4. **模型更新：**
   - 根据目标函数进行梯度下降等优化算法，更新模型参数，以提高正样本的预测概率，降低负样本的预测概率。

通过这种方式，候选采样允许模型在计算概率时只关注少量的负样本，从而大大提高了训练效率。虽然这会引入一些噪音，但实际上在训练中，对于大多数应用而言，这种噪音并不会显著影响最终的嵌入质量。

为了进行损失计算，需要一个取得权重和偏置的方法：两者都初始化为0。

```python
import tensorflow as tf

# Skip-gram embedding model
class EmbeddingModel(object):
    # Model Initialization
    def __init__(self, vocab_size, embedding_dim):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=self.vocab_size)

    # Get bias and weights for calculating loss
    def get_bias_weights(self):
        weights_initializer = tf.zeros([self.vocab_size, self.embedding_dim])
        bias_initializer = tf.zeros([self.vocab_size])
        weights = tf.Variable(weights_initializer)
        bias = tf.Variable(bias_initializer)
        return weights, bias
```

## 损失计算

这里会用NCE进行计算，但是比较两种主要方法的不同和原理：

NCE Loss（Noise Contrastive Estimation损失）和Sampled Softmax Loss都是用于训练大规模分类问题中的嵌入模型（如Word2Vec）时的损失函数，旨在解决softmax计算在大型词汇表上的计算效率低下的问题。

1. **NCE Loss（Noise Contrastive Estimation损失）：**

NCE Loss是由Tomas Mikolov等人提出的，其基本思想是通过与噪声分布进行对比估计模型参数。损失函数的目标是提高正样本（实际上的上下文词）的概率，降低负样本（噪声词）的概率。具体步骤如下：

- 对于每个训练样本，从噪声分布中采样一些负样本。
- 构建一个二分类任务，将正样本与负样本进行区分。
- 最小化二分类的交叉熵损失函数。

NCE Loss通过仅计算与正样本相关的概率来显著减少计算成本，避免了对整个词汇表进行softmax计算。

2. **Sampled Softmax Loss：**

Sampled Softmax Loss也是为了提高计算效率而设计的。与NCE Loss类似，它通过对正样本进行计算，以及对一小部分负样本进行采样，来估计softmax函数的损失。具体步骤如下：

- 对于每个训练样本，从噪声分布中采样一些负样本，同时保留正样本。
- 将样本划分为一个小型的子集，只对这个子集进行softmax计算。
- 最小化这个子集上的softmax交叉熵损失函数。

这种采样的方法在计算效率上也有所提高，避免了对整个词汇表进行softmax计算。

**区别：**

- NCE Loss更侧重于噪声对比估计，引入了二分类任务。
- Sampled Softmax Loss侧重于在softmax计算中进行子集采样，减少计算成本。

两者的目标都是为了在大规模分类问题中更高效地进行训练，尤其在处理大词汇表时，提高了计算速度。选择哪种损失函数通常取决于具体的应用和模型的特性。

tensorflow处理loss的函数包括以下参数：

- weights: 权重
- biases: 偏置
- labels: 标签，在这里是上下文id
- inputs: 输入，是中间词汇target的id嵌入向量
- num_sampled: 负采样分类的数量
- num_classes: 词汇表大小，分类类别数量

代码实现：

```python
import tensorflow as tf

# Skip-gram embedding model
class EmbeddingModel(object):
    # Model Initialization
    def __init__(self, vocab_size, embedding_dim):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=self.vocab_size)
    
    # Calculate NCE Loss based on the retrieved embedding and context
    def calculate_loss(self, embeddings, context_ids, num_negative_samples):
        weights, bias = self.get_bias_weights()
        nce_losses = tf.nn.nce_loss(weights,
                                    bias, 
                                    context_ids,
                                    embeddings,
                                    num_negative_samples, # 负采样的数量
                                    self.vocab_size)
        # 计算平均损失值
        overall_loss = tf.math.reduce_mean(nce_losses)
        return overall_loss
```

## 评估：余弦相似度

余弦相似度是一种衡量两个向量之间相似性的方法，尤其在高维空间中常被用于比较文本、图像或其他表示为向量的数据。余弦相似度衡量的是两个向量之间的夹角的余弦值，取值范围在[-1, 1]之间，其中**1表示完全相似，-1表示完全不同，0表示无关。**

在自然语言处理和嵌入模型中，余弦相似度通常用于衡量两个词向量之间的语义相似性。如果两个词的词向量在高维空间中更接近，它们的余弦相似度将更接近1，表示它们在语义上更相似。

在词嵌入模型中，可以使用余弦相似度来衡量词向量之间的语义关联性。通过计算余弦相似度，可以找到在嵌入空间中相似或相关的词汇。这对于诸如词语相似性搜索、信息检索和文本分类等任务非常有用。通过测量词向量之间的夹角，余弦相似度提供了一种有效的方式来捕捉词汇之间的语义关系。

计算其实就是嵌入向量和嵌入矩阵的dot运算。

```python
import tensorflow as tf

# Skip-gram embedding model
class EmbeddingModel(object):
    # Model Initialization
    def __init__(self, vocab_size, embedding_dim):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=self.vocab_size)
    
    # Compute cosine similarites between the word's embedding
    # and all other embeddings for each vocabulary word
    def compute_cos_sims(self, word, training_texts):
        self.tokenizer.fit_on_texts(training_texts)
        word_id = self.tokenizer.word_index[word]
        # 获得嵌入向量
        word_embedding = self.forward([word_id])
        # 归一化嵌入向量。归一化后，向量的 L2 范数为 1。
        normalized_embedding = tf.math.l2_normalize(word_embedding)
        # 归一化嵌入矩阵。
        normalized_matrix = tf.math.l2_normalize(self.embedding_matrix, axis=1)
        # 计算嵌入向量和嵌入矩阵的余弦相似度。转置以使得向量和矩阵的每一行进行计算。
        cos_sims = tf.linalg.matmul(normalized_embedding, normalized_matrix, transpose_b=True)
        return cos_sims
```

## 使用K邻近算法找到最相近的k个词汇

计算出了一个词和整个词矩阵的所有词的余弦相似度，然后使用K邻近算法，找到k个最高相似度的词，并返回。

K近邻算法是一种简单而直观的非参数方法，用于分类和回归任务。在寻找相似度最高的词时，KNN的应用同样直观：找到距离（相似度）最近的K个词。

在自然语言处理（NLP）中，单词通常表示为高维向量（如词嵌入）。KNN算法在高维空间中有效地工作，通过计算向量之间的距离（例如余弦相似度）找到最近的邻居。

KNN是一种惰性学习算法，不需要在训练阶段进行复杂的模型拟合。只需在推断阶段计算相似度并排序，这使得它非常适合在线计算和实时应用。

```python
import tensorflow as tf

# Skip-gram embedding model
class EmbeddingModel(object):
    # Model Initialization
    def __init__(self, vocab_size, embedding_dim):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=self.vocab_size)
    
    # Compute K-nearest neighbors for input word
    def k_nearest_neighbors(self, word, k, training_texts):
        cos_sims = self.compute_cos_sims(word, training_texts)
        # shape (1, self.vocab_size)
        squeezed_cos_sims = tf.squeeze(cos_sims)
        top_k_output = tf.math.top_k(squeezed_cos_sims, k)
        # Note that the output top_k_output is a tuple. The first element is the top K cosine similarities, while the second element is the actual word IDs corresponding to the top K nearest neighbors.
        return top_k_output
```
