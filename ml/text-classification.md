NLP的实质是对*序列*的处理和预测。不管是文本还是语音都是序列。

## 项目目的和流程：

分类识别推特的文本是否和灾难发生有关，也就是文本分类问题。

Text -> turn into numbers -> build a model -> train the model to find patterns -> use patterns (make predictions)

Talk is cheap, show the code only.

```python
# Check for GPU
!nvidia-smi -L
# Import series of helper functions for the notebook
from helper_functions import unzip_data, create_tensorboard_callback, plot_loss_curves, compare_historys
```

## 数据准备

下载和解压数据。

```python
# Download data (same as from Kaggle)
!wget "https://storage.googleapis.com/ztm_tf_course/nlp_getting_started.zip"

# Unzip data
unzip_data("nlp_getting_started.zip")
```

## 数据可视化

数据读取和可视化。

```python
# Turn .csv files into pandas DataFrame's
import pandas as pd
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
train_df.head()
```

The data includes: `id, keyword, location, text, target`

```python
# Shuffle training dataframe
train_df_shuffled = train_df.sample(frac=1, random_state=42) # shuffle with random_state=42 for reproducibility
train_df_shuffled.head()
```

frac means 100% of the datas.

```python
# How many examples of each class?
train_df.target.value_counts()
# How many samples total?
print(f"Total training samples: {len(train_df)}")
print(f"Total test samples: {len(test_df)}")
print(f"Total samples: {len(train_df) + len(test_df)}")
```

Visualize!

```python
# Let's visualize some random training examples
import random
random_index = random.randint(0, len(train_df)-5) # create random indexes not higher than the total number of samples
for row in train_df_shuffled[["text", "target"]][random_index:random_index+5].itertuples():
  _, text, target = row
  print(f"Target: {target}", "(real disaster)" if target > 0 else "(not real disaster)")
  print(f"Text:\n{text}\n")
  print("---\n")
```

### 数据分割

将 text 和 target 转化为 numpy array 格式并进行数据的分割，方便后续使用。

```python
from sklearn.model_selection import train_test_split

# Use train_test_split to split training data into training and validation sets
train_sentences, val_sentences, train_labels, val_labels = train_test_split(
    train_df_shuffled["text"].to_numpy(),
    train_df_shuffled["target"].to_numpy(),
    test_size=0.1, # dedicate 10% of samples to validation set
    random_state=42
) # random state for reproducibility

# Check the lengths
len(train_sentences), len(train_labels), len(val_sentences), len(val_labels)

# View the first 10 training sentences and their labels
train_sentences[:10], train_labels[:10]
```

## 文本分词

**Tokenization**，我还是很喜欢Daniel的解释，很简洁简单好懂。分词技术，以下几种：

- Word-level：这相当于给每个单词标记数字。
- Character-level：相当于给26个字母标记数字。
- Sub-word-level：这是所谓的分词，Token其实是组成了单词的部分。他们被叫做*token*。

**Embedding**，嵌入。是一个单词的向量表示。可以自己训练得到，也可以用pre-trained的嵌入层。

向量之间的相似度使用余弦相似度度量。

**嵌入矩阵和上下文矩阵**：

嵌入矩阵（Embedding Matrix）是将离散的词汇或类别映射到低维连续向量空间的矩阵。在自然语言处理中，通常使用词嵌入技术，将每个单词表示为一个固定长度的向量。这些向量在嵌入矩阵中进行存储，每行对应一个单词的嵌入。

上下文矩阵（Context Matrix）是用来表示某种上下文的矩阵。上下文可以是一个句子、一个文档，或者其他一些序列数据。这个矩阵捕捉了序列中的信息，并可用于下游任务，如情感分析、命名实体识别等。

这两个矩阵之间的关系在自然语言处理中非常密切。在很多任务中，嵌入矩阵用来将单词映射到向量空间，而上下文矩阵则用来表示单词所处的上下文信息。嵌入矩阵可以视为上下文矩阵的一部分，因为它包含了单词的语义信息，而上下文矩阵则更广泛地表示了整个序列的语义。

在深度学习模型中，通常会将嵌入矩阵作为模型的一部分，在训练过程中学习得到，以便模型能够更好地理解输入数据。上下文矩阵则可能由嵌入矩阵以及其他模型的隐藏状态等信息组成，用来表示更高级的语义和序列信息。

**TextVectorization**类的输出结果是经过向量化处理后的文本数据。这个类可以将输入的原始文本数据转换成数值化的表示，使其可以被深度学习模型所理解和处理。

输出结果通常是一个向量，其长度与词汇表的大小相同，每个位置对应一个词汇表中的单词或子词。每个位置上的值表示对应单词在文本中出现的次数、TF-IDF值或者其他形式的权重。

例如，如果有一个词汇表包含了单词 "cat"、"dog" 和 "bird"，那么一个包含 "cat" 和 "dog" 的句子可能被向量化为 [1, 1, 0]，其中第一个位置对应 "cat"，第二个位置对应 "dog"，第三个位置对应 "bird"。

一个普通的分词模型：
```python
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization

# Use the default TextVectorization variables
text_vectorizer = TextVectorization(max_tokens=None, # how many words in the vocabulary (all of the different words in your text)
                                    standardize="lower_and_strip_punctuation", # how to process text
                                    split="whitespace", # how to split tokens
                                    ngrams=None, # create groups of n-words?
                                    output_mode="int", # how to map tokens to numbers
                                    output_sequence_length=None) # how long should the output sequence of tokens be?
                                    # pad_to_max_tokens=True) # Not valid if using max_tokens=None
```

但是需要找到句子的平均长度：这里的结果是15，也就是句子的长度大约是15个单词。
```python
# find avarage number of tokens in training tweets
round(sum([len(i.split()) for i in train_sentences]) / len(train_sentences))
```

根据这个数据我们可以有自己的custom分词模型：
```python
# Setup text vectorization with custom variables
max_vocab_length = 10000 # max number of words to have in our vocabulary
max_length = 15 # max length our sequences will be (e.g. how many words from a Tweet does our model see?)

text_vectorizer = TextVectorization(max_tokens=max_vocab_length,
                                    output_mode="int",
                                    output_sequence_length=max_length)
```

将这个分词模型实例应用于训练数据：使用adapt方法，这相当于是拟合了一个*分词模型*。
```python
# Fit the text vectorizer to the training text
text_vectorizer.adapt(train_sentences)
```

如果将这个分词模型应用于一个具体的例子：
```python
# Create sample sentence and tokenize it
sample_sentence = "There's a flood in my street!"
text_vectorizer([sample_sentence])
```

那么它将会输出，该句子的嵌入向量。
```
<tf.Tensor: shape=(1, 15), dtype=int64, numpy=
array([[264,   3, 232,   4,  13, 698,   0,   0,   0,   0,   0,   0,   0,
          0,   0]])>
```
最后会出现0，是因为我们确定了最大长度，对于不足的部分，会用0进行填充。

最后可以检查我们的**分词器text_vectorizer**中的unique tokens的数量，以及最常见和最不常见的token。通过get_vocabulary()方法可以得到所有的unique tokens。他们按照出现的次数进行排列。

```python
# Get the unique words in the vocabulary
words_in_vocab = text_vectorizer.get_vocabulary()
top_5_words = words_in_vocab[:5] # most common tokens (notice the [UNK] token for "unknown" words)
bottom_5_words = words_in_vocab[-5:] # least common tokens
print(f"Number of words in vocab: {len(words_in_vocab)}")
print(f"Top 5 most common words: {top_5_words}") 
print(f"Bottom 5 least common words: {bottom_5_words}")

# Number of words in vocab: 10000
# Top 5 most common words: ['', '[UNK]', 'the', 'a', 'in']
# Bottom 5 least common words: ['pages', 'paeds', 'pads', 'padres', 'paddytomlinson1']
```

## 文本嵌入

和上面的分词不同，分词是**静态**表达，而嵌入则是**动态**的，可以进行学习和进化。这个嵌入层，将作为模型的一部分存在，所以在训练中，它的参数是可以被更新和进化的。

一个嵌入层如下：这里定义的128为输出的长度，意味着输出的嵌入矩阵的维度将是128维度。

```python
tf.random.set_seed(42)
from tensorflow.keras import layers

embedding = layers.Embedding(input_dim=max_vocab_length, # set input shape
                             output_dim=128, # set size of embedding vector
                             embeddings_initializer="uniform", # default, intialize randomly
                             input_length=max_length, # how long is each input
                             name="embedding_1") 

embedding
```

## Model 0: Naive Bayes (baseline/benchmark)

- TfidfVectorizer：用于将文本数据转换为TF-IDF（Term Frequency-Inverse Document Frequency）特征矩阵。TF-IDF是一种用于文本挖掘的特征值，反映了一个词在一个文档中的重要性。
- MultinomialNB：多项式朴素贝叶斯分类器，适用于特征值表示为多项式分布的情况，常用于文本分类。
- Pipeline：用于将多个处理步骤串联起来，使得工作流程更简洁清晰。

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Create tokenization and modelling pipeline
model_0 = Pipeline([
                    ("tfidf", TfidfVectorizer()), # convert words to numbers using tfidf
                    ("clf", MultinomialNB()) # model the text
])

# Fit the pipeline to the training data
model_0.fit(train_sentences, train_labels)
```

计算分数和进行预测。
```python
baseline_score = model_0.score(val_sentences, val_labels)
print(f"Our baseline model achieves an accuracy of: {baseline_score*100:.2f}%")
# Make predictions
baseline_preds = model_0.predict(val_sentences)
baseline_preds[:20]
```

## 创建一个评估函数

```python
# Function to evaluate: accuracy, precision, recall, f1-score
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def calculate_results(y_true, y_pred):
    """
    Calculates model accuracy, precision, recall and f1 score of a binary classification model.

    Args:
    -----
    y_true = true labels in the form of a 1D array
    y_pred = predicted labels in the form of a 1D array

    Returns a dictionary of accuracy, precision, recall, f1-score.
    """
    # Calculate model accuracy
    model_accuracy = accuracy_score(y_true, y_pred) * 100
    # Calculate model precision, recall and f1 score using "weighted" average
    model_precision, model_recall, model_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
    model_results = {"accuracy": model_accuracy,
                    "precision": model_precision,
                    "recall": model_recall,
                    "f1": model_f1}
    return model_results

# Get baseline results
baseline_results = calculate_results(y_true=val_labels,
                                     y_pred=baseline_preds)
baseline_results
```

## Model 1: Feed-forward neural network (dense model)

创建一个 tensorboard callback 函数：

```python
# Create tensorboard callback (need to create a new one for each model)
from helper_functions import create_tensorboard_callback

# Create directory to save TensorBoard logs
SAVE_DIR = "model_logs"
```

构建第一个模型。

```python
# Build model with the Functional API
from tensorflow.keras import layers
inputs = layers.Input(shape=(1,), dtype="string") # inputs are 1-dimensional strings
x = text_vectorizer(inputs) # turn the input text into numbers
x = embedding(x) # create an embedding of the numerized numbers
x = layers.GlobalAveragePooling1D()(x) # lower the dimensionality of the embedding (try running the model without this layer and see what happens)
outputs = layers.Dense(1, activation="sigmoid")(x) # create the output layer, want binary outputs so use sigmoid activation
model_1 = tf.keras.Model(inputs, outputs, name="model_1_dense") # construct the model

# Compile model
model_1.compile(loss="binary_crossentropy",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

# Get a summary of the model
model_1.summary()

# Fit the model
model_1_history = model_1.fit(train_sentences, # input sentences can be a list of strings due to text preprocessing layer built-in model
                              train_labels,
                              epochs=5,
                              validation_data=(val_sentences, val_labels),
                              callbacks=[create_tensorboard_callback(dir_name=SAVE_DIR, 
                                                                     experiment_name="simple_dense_model")])

# Check the results
model_1.evaluate(val_sentences, val_labels)
# check the embedding weights
embedding.weights
embed_weights = model_1.get_layer("embedding_1").get_weights()[0]
print(embed_weights.shape)

# Make predictions (these come back in the form of probabilities)
model_1_pred_probs = model_1.predict(val_sentences)
model_1_pred_probs[:10] # only print out the first 10 prediction probabilities

# Turn prediction probabilities into single-dimension tensor of floats
model_1_preds = tf.squeeze(tf.round(model_1_pred_probs)) # squeeze removes single dimensions
model_1_preds[:20]

# Calculate model_1 metrics
model_1_results = calculate_results(y_true=val_labels, 
                                    y_pred=model_1_preds)
model_1_results
```

创建一个函数用于各个模型和baseline的结果比较。
```python
# Create a helper function to compare our baseline results to new model results
def compare_baseline_to_new_results(baseline_results, new_model_results):
  for key, value in baseline_results.items():
    print(f"Baseline {key}: {value:.2f}, New {key}: {new_model_results[key]:.2f}, Difference: {new_model_results[key]-value:.2f}")

compare_baseline_to_new_results(baseline_results=baseline_results, 
                                new_model_results=model_1_results)
```

## Model 2: LSTM model

流程相似：Input (text) -> Tokenize -> Embedding -> Layers -> Output (label probability)

text_vectorizer可以复用，因为它不会被更新。但是Embedding层会被重新创建，因为需要更新。

```python
# Set random seed and create embedding layer (new embedding layer for each model)
tf.random.set_seed(42)
from tensorflow.keras import layers
model_2_embedding = layers.Embedding(input_dim=max_vocab_length,
                                     output_dim=128,
                                     embeddings_initializer="uniform",
                                     input_length=max_length,
                                     name="embedding_2")


# Create LSTM model
inputs = layers.Input(shape=(1,), dtype="string")
x = text_vectorizer(inputs)
x = model_2_embedding(x)
print(x.shape)  # (None, 15, 128)
# x = layers.LSTM(64, return_sequences=True)(x) # return vector for each word in the Tweet (you can stack RNN cells as long as return_sequences=True)
x = layers.LSTM(64)(x) # return vector for whole sequence
print(x.shape)  # (None, 64)
# x = layers.Dense(64, activation="relu")(x) # optional dense layer on top of output of LSTM cell
outputs = layers.Dense(1, activation="sigmoid")(x)
model_2 = tf.keras.Model(inputs, outputs, name="model_2_LSTM")

# Compile model
model_2.compile(loss="binary_crossentropy",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

# Fit model
model_2_history = model_2.fit(train_sentences,
                              train_labels,
                              epochs=5,
                              validation_data=(val_sentences, val_labels),
                              callbacks=[create_tensorboard_callback(SAVE_DIR, 
                                                                     "LSTM")])

# Make predictions on the validation dataset
model_2_pred_probs = model_2.predict(val_sentences)
model_2_pred_probs.shape, model_2_pred_probs[:10] # view the first 10

# Round out predictions and reduce to 1-dimensional array
model_2_preds = tf.squeeze(tf.round(model_2_pred_probs))
model_2_preds[:10]

# Calculate LSTM model results
model_2_results = calculate_results(y_true=val_labels,
                                    y_pred=model_2_preds)
model_2_results

# Compare model 2 to baseline
compare_baseline_to_new_results(baseline_results, model_2_results)
```
其实还是没有baseline的效果好。

## Model 3: GRU model

```python
# Set random seed and create embedding layer (new embedding layer for each model)
tf.random.set_seed(42)
from tensorflow.keras import layers
model_3_embedding = layers.Embedding(input_dim=max_vocab_length,
                                     output_dim=128,
                                     embeddings_initializer="uniform",
                                     input_length=max_length,
                                     name="embedding_3")

# Build an RNN using the GRU cell
inputs = layers.Input(shape=(1,), dtype="string")
x = text_vectorizer(inputs)
x = model_3_embedding(x)
# x = layers.GRU(64, return_sequences=True) # stacking recurrent cells requires return_sequences=True
x = layers.GRU(64)(x) 
# x = layers.Dense(64, activation="relu")(x) # optional dense layer after GRU cell
outputs = layers.Dense(1, activation="sigmoid")(x)
model_3 = tf.keras.Model(inputs, outputs, name="model_3_GRU")

# Compile GRU model
model_3.compile(loss="binary_crossentropy",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

# Get a summary of the GRU model
model_3.summary()

# Fit model
model_3_history = model_3.fit(train_sentences,
                              train_labels,
                              epochs=5,
                              validation_data=(val_sentences, val_labels),
                              callbacks=[create_tensorboard_callback(SAVE_DIR, "GRU")])

# Make predictions on the validation data
model_3_pred_probs = model_3.predict(val_sentences)
model_3_pred_probs.shape, model_3_pred_probs[:10]

# Convert prediction probabilities to prediction classes
model_3_preds = tf.squeeze(tf.round(model_3_pred_probs))
model_3_preds[:10]

# Calcuate model_3 results
model_3_results = calculate_results(y_true=val_labels, 
                                    y_pred=model_3_preds)
model_3_results

# Compare to baseline
compare_baseline_to_new_results(baseline_results, model_3_results)
```

## Model 4: Bidirectional-LSTM model

```python
# Set random seed and create embedding layer (new embedding layer for each model)
tf.random.set_seed(42)
from tensorflow.keras import layers
model_4_embedding = layers.Embedding(input_dim=max_vocab_length,
                                     output_dim=128,
                                     embeddings_initializer="uniform",
                                     input_length=max_length,
                                     name="embedding_4")

# Build a Bidirectional RNN in TensorFlow
inputs = layers.Input(shape=(1,), dtype="string")
x = text_vectorizer(inputs)
x = model_4_embedding(x)
# x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x) # stacking RNN layers requires return_sequences=True
x = layers.Bidirectional(layers.LSTM(64))(x) # bidirectional goes both ways so has double the parameters of a regular LSTM layer
outputs = layers.Dense(1, activation="sigmoid")(x)
model_4 = tf.keras.Model(inputs, outputs, name="model_4_Bidirectional")

# Compile
model_4.compile(loss="binary_crossentropy",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

# Get a summary of our bidirectional model
model_4.summary()

# Fit the model (takes longer because of the bidirectional layers)
model_4_history = model_4.fit(train_sentences,
                              train_labels,
                              epochs=5,
                              validation_data=(val_sentences, val_labels),
                              callbacks=[create_tensorboard_callback(SAVE_DIR, "bidirectional_RNN")])
# Make predictions with bidirectional RNN on the validation data
model_4_pred_probs = model_4.predict(val_sentences)
model_4_pred_probs[:10]

# Convert prediction probabilities to labels
model_4_preds = tf.squeeze(tf.round(model_4_pred_probs))
model_4_preds[:10]

# Calculate bidirectional RNN model results
model_4_results = calculate_results(val_labels, model_4_preds)
model_4_results

# Check to see how the bidirectional model performs against the baseline
compare_baseline_to_new_results(baseline_results, model_4_results)
```

## Model 5: 1D Convolutional Neural Network
## Model 6: TensorFlow Hub Pretrained Feature Extractor
## Model 7: Same as model 6 with 10% of training data
