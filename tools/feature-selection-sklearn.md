## 数据处理框架Scikit-learn

---

在小数据和传统的机器学习中，scikit-learn是一个优秀的框架，最优秀的地方是它方便的和统一化的API，可以简单调用，进行属于的预处理。

这里总结一下该框架中常用的数据处理的代码和说明，并进行一个自己的复习。

### Standardization（标准化）

将数据缩放为均值为0，标准差为1的尺度。

```python
from sklearn.preprocessing import scale
# Standardizing each column of data, the data is numpy array
col_standardized = scale(data)
```

create a function to do the preprocessing

```python
def standardize_data(data):
  scaled_data = scale(data)
  return scaled_data
```

###  MinMaxScaler（最小最大正规化）

将数据缩放到一个data-range内。

```python
from sklearn.preprocessing import MinMaxScaler
default_scaler = MinMaxScaler() # the default range is [0,1]
transformed = default_scaler.fit_transform(data)

custom_scaler = MinMaxScaler(feature_range=(-2, 3))
transformed = custom_scaler.fit_transform(data)
```

fit和transform也可以分开进行处理。
```python
from sklearn.preprocessing import MinMaxScaler
default_scaler = MinMaxScaler() # the default range is [0,1]
transformed = default_scaler.fit_transform(new_data)

default_scaler = MinMaxScaler()  # new instance
default_scaler.fit(data)  # different data value fit
transformed = default_scaler.transform(new_data)
```

create a function to do preprocessing

```python
def ranged_data(data, value_range):
  min_max_scaler = MinMaxScaler(feature_range=value_range)
  scaled_data = min_max_scaler.fit_transform(data)
  return scaled_data
```

### RobustScaler（针对异常值的稳健型缩放）

"稳健型缩放" 是指在数据处理或统计分析中采用的一种技术，用于调整数据的尺度，以减少异常值的影响，并使数据更适合于建模或分析。

通常情况下，数据集中可能存在一些异常值或极端值，这些值可能会对统计分析或建模产生不良影响。稳健型缩放技术旨在减少这种影响，使得模型更加稳健，即对异常值更具鲁棒性。

```python
from sklearn.preprocessing import RobustScaler
robust_scaler = RobustScaler()
transformed = robust_scaler.fit_transform(data)
```

create a function to do preprocessing

```python
def robust_scaling(data):
  robust_scaler = RobustScaler()
  scaled_data = robust_scaler.fit_transform(data)
  return scaled_data
```

### L2 Normalization（归一化）

```python
from sklearn.preprocessing import Normalizer
normalizer = Normalizer()
transformed = normalizer.fit_transform(data)
```

create a function to do the preprocessing

```python
def normalize_data(data):
  normalizer = Normalizer()
  norm_data = normalizer.fit_transform(data)
  return norm_data
```

### 针对数据确实的插值

```python
from sklearn.impute import SimpleImputer
imp_mean = SimpleImputer()
transformed = imp_mean.fit_transform(data)
```

默认使用mean值进行插值，除此之外还有中值median，最频繁值，以及常数。可以通过策略参数进行定义。

```python
from sklearn.impute import SimpleImputer
imp_median = SimpleImputer(strategy='median')
transformed = imp_median.fit_transform(data)

imp_frequent = SimpleImputer(strategy='most_frequent')
transformed = imp_frequent.fit_transform(data)

from sklearn.impute import SimpleImputer
imp_constant = SimpleImputer(strategy='constant',
                             fill_value=-1)
transformed = imp_constant.fit_transform(data)
```

除此之外，数据填充不仅限于这四种方法。

还有更高级的填充方法，如k-Nearest Neighbors（根据kNN算法的相似度分数填充缺失值）和MICE（应用多个链式填充，假设缺失值在观测值之间是随机分布的）。

在大多数行业案例中，这些高级方法并不是必需的，因为数据要么已经完全清洗过，要么缺失值很少。然而，在处理开源数据集时，这些高级方法可能会有用，因为这些数据集往往更不完整。

### 降维：PCA主成分分析提取特征

```python
from sklearn.decomposition import PCA
pca_obj = PCA() # The value of n_component will be 4. As m is 5 and default is always m-1
pc = pca_obj.fit_transform(data).round(3)

pca_obj = PCA(n_components=3)
pc = pca_obj.fit_transform(data).round(3)

pca_obj = PCA(n_components=2)
pc = pca_obj.fit_transform(data).round(3)
```

create a function to do preprocessing

```python
def pca_data(data, n_components):
  pca_obj = PCA(n_components = n_components)
  component_data = pca_obj.fit_transform(data)
  return component_data
```

针对降维后的数据进行数据分割：

```python
def get_label_info(component_data, labels,
                   label, label_names):
  label_name = label_names[label]
  label_data = component_data[labels == label]
  return (label_name, label_data)

def separate_data(component_data, labels,
                  label_names):
  separated_data = []
  for label in range(len(label_names)):
    separated_data.append(get_label_info(component_data, labels, label, label_names))
  return separated_data

from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
bc = load_breast_cancer()
pca_obj = PCA(n_components=2)
component_data = pca_obj.fit_transform(bc.data)
labels = bc.target
label_names = bc.target_names
# Using the completed separate_data function
separated_data = separate_data(component_data,
                               labels, label_names)

# Plotting the data
import matplotlib.pyplot as plt
for label_name, label_data in separated_data:
  col1 = label_data[:, 0]  # 1st column (1st pr. comp.)
  col2 = label_data[:, 1]  # 2nd column (2nd pr. comp.)
  plt.scatter(col1, col2, label=label_name) # scatterplot
plt.legend()  # adds legend to plot
plt.title('Breast Cancer Dataset PCA Plot')
plt.show()
```
