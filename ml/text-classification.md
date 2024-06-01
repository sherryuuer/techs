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

## Downloading a text dataset

下载和解压数据。

```python
# Download data (same as from Kaggle)
!wget "https://storage.googleapis.com/ztm_tf_course/nlp_getting_started.zip"

# Unzip data
unzip_data("nlp_getting_started.zip")
```

## Visualizing text data

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

### Split data into training and validation data

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

## Converting text into numbers using tokenization

Tokenization，我还是很喜欢Daniel的解释，很简洁简单好懂。分词技术，以下几种：

- Word-level：这相当于给每个单词标记数字。
- Character-level：相当于给26个字母标记数字。
- Sub-word-level：这是所谓的分词，Token其实是组成了单词的部分。他们被叫做*token*。

Embedding，嵌入。

## Turning our tokenized text into an embedding
## Modelling a text dataset
### Starting with a baseline (TF-IDF)
### Building several deep learning text models
### Dense, LSTM, GRU, Conv1D, Transfer learning
## Comparing the performance of each our models
## Combining our models into an ensemble
## Saving and loading a trained model
## Find the most wrong predictions
