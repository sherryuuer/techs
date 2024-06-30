## 机器学习算法之线性回归，特征选择方法，评估方法，非常用回归模型

---
### 线性回归重要性

现在还有人学习线性回归？我为什么还在线性回归我都Transformer都学过了。高级算法学完了就不需要学机器学习的入门算法了吗，什么是入门算法。这个世界的法则是什么，回归是否支配了我们的世界。

一天我在推特看到一张图，说在transformer的支配下，传统的机器学习算法没落了（权当胡说八道），然后在那张可视化图中，线性回归模型岿然不动，看来什么都逃不过这个世界回归的法则。

### 一些心得和思考

不再从基础概念开始而是直接写下我总结和补充的一些关于回归的心得：

- 多元线性回归中使用可视化的图标好处多多，比如箱线图帮我看到整体的分布和离异值。

- **特征缩放的好处：**
  - 它有助于基于梯度下降的算法更快地收敛。
  - 它有助于基于距离的算法在计算相似度时为每个特征赋予相同的权重。（基于距离的算法考虑数据集实例之间的距离或相似性来进行计算。）
  - 它有助于比较特征重要性。

- **正则化Regularization防过拟合：**
  - 岭回归Ridge是一种L2正则化，它的拉姆达参数是可以学习的。（过高会导致欠拟合，过低会导致对参数影响忽略不计。）
  - Lasso回归是一种L1正则化，所以它也可以用来进行特征选择，它会给我们一个稀疏解。
  - Elastic-Net回归则结合了上述两种回归模型。
    - l1_ratio是 ElasticNet 混合参数，其值介于 0 和 1 之间。
    - 当l1_ratio= 0 时，使用 L2 正则化。当 l1_ratio= 1，使用 L1 正则化。当 0 < l1_ratio< 1，使用 L1 和 L2 的组合。

- **可以进行回归的支持向量机：**
  - 支持向量机（Support Vector Machine，SVM）通常被认为是一种用于分类问题的算法，但实际上，它也可以用于解决回归问题。这个版本被称为支持向量回归（Support Vector Regression，SVR）。
  - SVR的基本思想与SVM类似，但目标是拟合数据点，使得数据点尽可能地接近或在边界上，并且在边界上存在尽可能少的数据点。在SVR中，我们不再寻找一个超平面（在分类问题中的分割边界），而是寻找一个函数，这个函数在输入空间中有较小的误差，即预测值与真实值之间的差异尽可能小，同时在预测值与真实值之间存在一个ε-tube，以此来控制模型的复杂度。
  - ε-tube用于限制预测值与真实值之间的误差范围。模型的目标是在训练数据上达到尽可能小的误差，同时保持模型的泛化能力，即在未见过的数据上也能表现良好。
  - 超平面在分类算法中是尽量使得数据点远离平面，在回归中则相反，是让数据点尽量距离这个平面近。分类中超平面表示决策边界，回归中，超平面表示拟合的线性函数，用于预测连续型目标变量的值。它是一个线性函数，可以被视为将输入特征映射到对应的目标值的函数。
  - SVR通过引入一个成本函数来实现这一点，该成本函数包括两个部分：一部分是误差的总和，另一部分是边界上的数据点。优化过程的目标是最小化这个成本函数。
  - SVR使用了核技巧（Kernel Trick），它可以将数据从输入空间映射到更高维的特征空间中，以便在新的特征空间中找到一个更好的超平面来拟合数据。这使得SVR在处理非线性问题时非常有效。
  - 尽管SVR是基于SVM的，但它被设计用于解决回归问题，可以有效地处理线性和非线性回归任务。

- **可以进行回归的最邻近算法Nearest Neighbour Regression：**
  - 一种基于邻居的非参数回归方法，用于解决回归问题。它的主要思想是利用训练数据集中的最近邻居的信息来预测新数据点的输出值。
  - 这个算法一开始也是在分类的算法中出现的，对于回归问题，可以将这 K 个邻居的输出值进行加权平均或简单平均，作为测试样本的预测输出值。加权平均通常根据距离进行加权，距离越近的邻居权重越大。
  - 但在处理大规模数据集或需要更高预测精度的情况下，可能需要考虑其他更复杂的回归模型。

- **Decision Tree Regression（决策树回归）：**
  - 基于树形结构的回归方法，它通过对输入特征空间进行递归的划分，将输入空间划分为一系列的矩形区域，并在每个区域内拟合一个简单的模型（通常是一个常数）。它的主要思想是通过构建一棵树来对输入特征进行分段，然后在每个叶节点上预测输出值。
  - 决策树可以用来解决回归问题的原因在于它的基本结构和算法机制允许它对连续型输出变量进行预测。虽然决策树通常与分类问题联系紧密，但实际上它也适用于回归问题。
  - 在决策树的叶节点上存储的不再是类别标签，而是该叶节点对应的连续型输出值。这意味着每个叶节点代表了一个特定的预测输出值。

- **对非数值型特征的处理one-hot-encoder方法要注意防止多重共线性。**

- **数值变量变换**：在统计分析和机器学习中，对数值变量进行变换是一种常见的数据预处理技术，旨在改善数据的分布特性、减少偏斜或者增强模型的性能。以下是常见的数值变量变换方法：
  - 对数变换（Log Transformation）：对数变换是常用的一种方法，特别适用于偏斜分布（长尾）的数据。通过取自然对数、以2为底的对数或者以10为底的对数等，可以使数据更加接近正态分布。
  - 平方根变换（Square Root Transformation）：平方根变换可以减少数据的右偏，并且对数据中的较小值影响较大，对大值的影响较小。
  - 反正弦变换（Arcsine Transformation）：适用于介于0和1之间的数据，如比率或百分比数据，可以将其进行反正弦变换来改善数据的分布。
  - Box-Cox 变换：Box-Cox 变换是一种广义的幂函数变换，可以自动确定最适合数据的变换指数，从而使数据更加接近正态分布。
  - Yeo-Johnson 变换：与 Box-Cox 变换类似，但可以处理负值。
  - 指数变换（Exponential Transformation）：指数变换可以增加数据的差异性，使其更加适合某些模型的假设。
  - 分位数变换（Quantile Transformation）：将数据转换为服从指定分位数的分布，如正态分布。
  - Rank 变换：将数据转换为其排名的百分比，以消除异常值的影响并减少偏斜。
  - 幂次变换（Power Transformation）：一般形式为 y = x^lambda，通过调整参数 lambda，可以对数据进行适当的调整。

- 幂次变换（Power Transformation）是一种常用的数据变换方法，用于处理数据的偏斜或不均匀方差等问题。在幂次变换中，通过引入一个幂次lambda，将数据 x 变换为 x^lambda 的形式。lambda 取值包括：当 lambda = 0 时，即进行对数变换，也称为对数转换（Log Transformation）。对数变换常用于处理右偏斜数据，使其更接近于正态分布。当 lambda = 1 时，不进行任何变换，即原始数据。lambda = -1 时，即进行倒数变换，也称为倒数转换。倒数变换常用于处理左偏斜数据。lambda = 0.5 或其他小于 1 的值：这些值可以用于减小数据的右偏斜，但比对数变换更轻。lambda = 2 或其他大于 1 的值：这些值可以用于增加数据的右偏斜，使其更加接近于正态分布。在实际应用中，可以使用诸如 Box-Cox 变换或 Yeo-Johnson 变换等方法来自动确定最优的 lambda 值。

### 特征选择方法

在统计学习中也学习过如何进行特征选择，简单说就是要选出所有特征中最重要的那几个，目的是防止过拟合，减少计算量，提高计算速度等。毕竟很多相关性很高的特征都用来预测也没什么用处。

**Scikit-learn中有自己内置的特征选择方法：**

使用 scikit-learn 中的 `SelectFromModel` 可以从给定的模型中选择重要的特征。这个模块可以与各种不同的模型一起使用，比如线性模型、树模型等，通过模型自身的特征重要性来选择最重要的特征。使用这个模块的时候需要选择一个模型，该模型能够给出特征的重要性排名。例如，你可以选择随机森林、梯度提升树或者线性模型等。下面是[官网示例](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFromModel.html)的代码：

```python
from sklearn.feature_selection import SelectFromModel
# 这里选了逻辑回归作为模型标准
from sklearn.linear_model import LogisticRegression
X = [[ 0.87, -1.34,  0.31 ],
     [-2.79, -0.02, -0.85 ],
     [-1.34, -0.48, -2.55 ],
     [ 1.92,  1.48,  0.65 ]]
y = [0, 1, 0, 1]
# 定义选择器的时候是可以自定义threshold的参数的，这里没有定义的情况内部会默认，大多数情况是mean
selector = SelectFromModel(estimator=LogisticRegression()).fit(X, y)
# 度量各个特征的相关性
selector.estimator_.coef_
# 打印阈值
selector.threshold_
# 返回一组布尔值，表达该特征是否支持
selector.get_support()
# 转换特征向量为选出的向量
selector.transform(X)
```

**Wrapper Methods：**

Wrapper Methods 是一种特征选择的方法，它通过在特征子集上训练模型并根据模型性能来评估特征的重要性。与过滤方法（如方差阈值、相关系数等）不同，Wrapper Methods 使用机器学习模型本身的性能来评估特征的贡献。

Wrapper Methods 的一般思想是不断尝试不同的特征子集，直到找到一个最优的子集，使得模型性能达到最佳。这个过程可以通过贪心搜索、递归特征消除等技术来实现。

常见的 Wrapper Methods 包括：

- 递归特征消除（Recursive Feature Elimination，RFE）：该方法通过不断训练模型并剔除最不重要的特征，直到达到指定的特征数量为止。在每一轮迭代中，根据模型性能选择要删除的特征，直到达到指定的特征数量或达到某个性能指标。
- 正向特征选择（Forward Feature Selection）：该方法从一个空特征集开始，每次迭代将最重要的特征添加到特征集中，直到达到指定的特征数量或达到某个性能指标。
- 后向特征选择（Backward Feature Selection）：与正向特征选择相反，该方法从所有特征开始，每次迭代将最不重要的特征从特征集中移除，直到达到指定的特征数量或达到某个性能指标。
- 基于模型的特征选择（Model-Based Feature Selection）：该方法使用特定的学习模型（如逻辑回归、支持向量机等）来评估特征的重要性，并选择对模型性能最有利的特征子集。

Wrapper Methods 的优点是可以考虑特征之间的交互作用，因为它们是基于模型性能来选择特征的。然而，由于需要在每一轮迭代中重新训练模型，因此 Wrapper Methods 的计算成本通常较高。

**特征选择的过滤方法：基于统计而不是基于模型**

过滤方法是一种特征选择的方法，它通过计算特征与目标变量之间的统计指标来选择最相关的特征，而不涉及到模型的训练。常见的过滤方法包括：

**方差阈值（Variance Thresholding）**：计算特征的方差，剔除方差低于某个阈值的特征。适用于移除方差过小的特征，这些特征可能对目标变量的预测没有太大帮助。

**单变量特征选择（Univariate Feature Selection）**：SelectKBest 和 SelectPercentile 都是Scikit-learn中用于单变量选择的类。
- 回归问题：使用诸如 F 检验或者皮尔逊相关系数等统计指标来评估特征与目标变量之间的相关性，然后选择与目标变量显著相关的特征。
- 分类问题：同样，使用统计指标（如卡方检验、t 检验等）来评估特征与目标变量之间的相关性，选择与目标变量显著相关的特征。

**互信息（Mutual Information）**：衡量特征与目标变量之间的非线性相关性，选择与目标变量信息量最大的特征。

**相关系数（Correlation Coefficient）**：计算特征与目标变量之间的线性相关系数，选择与目标变量具有高相关性的特征。

**基于树的特征选择（Tree-Based Feature Selection）**：利用决策树或者随机森林等树模型中特征的重要性评估来选择特征。这类方法通常用于回归和分类问题，因为树模型可以直接提供特征的重要性排序。

*代码示例1：*

```python
from sklearn.datasets import make_regression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression

# 生成数据集
X, y = make_regression(n_samples=100, n_features=100, n_informative=10)
# n_samples：样本数量为100，n_features：特征数量为100，n_informative：具有信息量的特征数量为10

# 定义特征选择器
fs = SelectKBest(score_func=f_regression, k=10)
# 使用 f_regression 作为评分函数，选择 10 个最佳特征的 SelectKBest 对象

# 应用特征选择
X_selected = fs.fit_transform(X, y)
# 使用特征选择器对数据进行特征选择，将原始数据 X 通过 SelectKBest 选择最佳的 10 个特征

# 打印所选择的特征的形状
print(X_selected.shape)
# 输出所选择的特征的形状
```

这段代码的作用是生成一个包含100个样本和100个特征的回归数据集，然后使用 `SelectKBest` 类来选择与目标变量最相关的前10个特征，并打印所选择的特征的形状。

f_regression 用于计算输入特征和输出列的相关性，将其转换为f分数，然后转化为p值。F 分数用于评估模型或组之间的整体差异，而 p 值用于确定这种差异是否显著。

*代码示例2：*

```python
from sklearn.datasets import make_classification
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

# 生成数据集
X, y = make_classification(n_samples=100, n_features=20, n_informative=2)
# n_samples：样本数量为100，n_features：特征数量为20，n_informative：具有信息量的特征数量为2

# 定义特征选择器
fs = SelectKBest(score_func=f_classif, k=2)
# 使用 f_classif 作为评分函数，选择 2 个最佳特征的 SelectKBest 对象

# 应用特征选择
X_selected = fs.fit_transform(X, y)
# 使用特征选择器对数据进行特征选择，将原始数据 X 通过 SelectKBest 选择最佳的 2 个特征

# 打印所选择的特征的形状
print(X_selected.shape)
# 输出所选择的特征的形状
```

这段代码的作用是生成一个包含100个样本和20个特征的分类数据集，其中只有2个特征具有信息量，然后使用 `SelectKBest` 类来选择与目标变量最相关的前2个特征，并打印所选择的特征的形状。

f_classif 是 f_regression 的分类问题版本。

*代码示例3：*

```python
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest, chi2

# 加载鸢尾花数据集
X, y = load_iris(return_X_y=True)
# 返回特征矩阵 X 和目标向量 y

print(X.shape)
# 打印原始特征矩阵的形状

# 选择两个最佳特征
X_new = SelectKBest(chi2, k=2).fit_transform(X, y)
# 使用卡方检验作为评分函数，选择最佳的两个特征，并将原始特征矩阵 X 进行特征选择

print(X_new.shape)
# 打印选择的两个最佳特征的新特征矩阵的形状
```

这段代码的作用是使用卡方检验（chi2）作为评分函数，选择鸢尾花数据集中最相关的两个特征，并打印选择的两个最佳特征的新特征矩阵的形状。

卡方检验常用于分析分类变量之间的关系，比如检验两个分类变量之间的独立性、检验分类变量对目标变量的影响等。

*代码示例4：*

```python
from sklearn.datasets import load_digits
from sklearn.feature_selection import SelectPercentile, chi2

# 加载数据集
X, y = load_digits(return_X_y=True)
# 返回特征矩阵 X 和目标向量 y

print(X.shape)
# 打印原始特征矩阵的形状

# 基于前 10% 的特征选择
X_new = SelectPercentile(chi2, percentile=10).fit_transform(X, y)
# 使用卡方检验作为评分函数，选择前 10% 的最佳特征，并将原始特征矩阵 X 进行特征选择

print(X_new.shape)
# 打印选择的特征的新特征矩阵的形状
```

这段代码的作用是使用卡方检验（chi2）作为评分函数，选择鸢尾花数据集中与目标变量最相关的前 10% 特征，并打印选择的特征的新特征矩阵的形状。

*代码示例5：*

```python
from sklearn.datasets import load_iris
import pandas as pd

# 加载鸢尾花数据集
iris = load_iris()

# 创建特征矩阵和目标向量
X = iris.data
y = iris.target

# 将特征矩阵转换为 DataFrame
df = pd.DataFrame(X, columns=["sepal_length", "sepal_width", "petal_length", "petal_width"])
# 将特征矩阵 X 转换为 DataFrame，并指定列名为 sepal_length、sepal_width、petal_length、petal_width

# 创建相关系数矩阵
corr_matrix = df.corr()
# 计算特征矩阵中各个特征之间的相关系数

print(corr_matrix)
# 打印相关系数矩阵
```

这段代码的作用是使用 Pandas 将鸢尾花数据集的特征矩阵转换为 DataFrame，并计算特征之间的相关系数，然后打印相关系数矩阵。

*嵌入式方法（Embedded Methods）：*

嵌入式方法（Embedded Methods）是一种特征选择的方法，它将特征选择过程嵌入到模型训练过程中。在嵌入式方法中，特征选择与模型训练同时进行，模型在训练过程中自动地选择最佳的特征子集，以提高模型的性能或泛化能力。

和其他方法不同，在嵌入式方法中，特征选择是作为模型训练的一部分，模型通过优化目标函数来选择特征，从而得到最佳的特征子集。

常见的嵌入式方法包括：

- L1 正则化（Lasso Regression）：通过在损失函数中加入 L1 正则项，促使模型系数稀疏化，从而实现特征选择。
- L2 正则化（Ridge Regression）：虽然 L2 正则化不会导致模型系数稀疏化，但它可以减小系数的大小，抑制特征的过拟合，从而达到特征选择的效果。
- 决策树算法：在决策树算法中，特征选择是通过计算信息增益或基尼不纯度等指标来完成的。在树的生长过程中，模型会自动选择最佳的特征进行分裂，从而实现特征选择。
- 基于模型的特征重要性评估：一些机器学习模型（如随机森林、梯度提升树等）可以直接提供特征重要性评估，模型可以通过这些评估来选择最佳的特征。

嵌入式方法的优点是它们能够充分利用模型的学习能力来进行特征选择，不需要额外的特征子集搜索过程，并且可以提高模型的性能和泛化能力。然而，嵌入式方法可能会受到选择的模型的限制，因此选择合适的模型对于嵌入式特征选择至关重要。

### 模型评估方法

**Explained Variance Score：**

解释方差得分衡量了模型对数据变化的解释程度，即模型所能解释的目标变量方差的比例。解释方差得分的取值范围为0到1，其中：

- 得分为1表示模型完美地解释了数据的变化，即模型的预测值与实际值完全一致。
- 得分为0表示模型未能解释目标变量的任何变化，即模型的预测值与实际值之间没有相关性。

相关代码：
```python
from sklearn.metrics import explained_variance_score
y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]
print(explained_variance_score(y_true, y_pred))
```

**Mean absolute error：**

平均绝对误差（Mean Absolute Error，MAE）是一种用于评估回归模型性能的指标，它衡量了模型预测值与实际值之间的平均绝对差异程度。

具体来说，对于每个样本，MAE 是预测值与实际值之间的绝对差的平均值。

MAE 的取值范围为 0 到正无穷，其值越小表示模型的预测能力越好。当 MAE 等于 0 时，表示模型的预测完全准确，每个样本的预测值与实际值完全一致。

MAE 的一个优点是它对异常值不敏感，因为它只关注了预测值与实际值之间的差异的绝对值。因此，MAE 在评估回归模型时是一个常用的指标之一。

相关代码：
```python
from sklearn.metrics import mean_absolute_error
y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]
print(mean_absolute_error(y_true, y_pred))
```

**Mean squared error：**

均方误差（Mean Squared Error，MSE）是一种用于评估回归模型性能的指标，它衡量了模型预测值与实际值之间的平均平方差异程度。

具体来说，对于每个样本，MSE 是预测值与实际值之间的平方差的平均值。

MSE 的取值范围为 0 到正无穷，其值越小表示模型的预测能力越好。当 MSE 等于 0 时，表示模型的预测完全准确，每个样本的预测值与实际值完全一致。

与 MAE 相比，MSE 更加关注预测值与实际值之间的差异的平方，因此它更加注重大误差的影响。然而，由于平方的存在，MSE 会对异常值更加敏感。

相关代码：

```python
from sklearn.metrics import mean_squared_error
y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]
print(mean_squared_error(y_true, y_pred))
```

**Median absolute error：**

中值绝对误差，这个用的确实不多，衡量数据集中数据分散程度。它衡量了数据点与数据集的中值之间的平均距离。

它对于异常值比较稳健，因为它用的是中值而不是均值。由于他对异常值抵抗能力强，所以在异常值敏感的情况下使用该指标，在这种情况下，我们通常关注数据分布的离散程度或分布形状，而不希望异常值对结果产生大的影响。

例如，在金融领域的风险管理中，我们希望能够识别出潜在的异常情况，而不希望异常值对风险评估产生过大的干扰。因此，使用 MAD 可以更准确地估计数据的分布特征，从而更好地应对异常值的影响。

相关代码：

```python
from sklearn.metrics import median_absolute_error
y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]
print(median_absolute_error(y_true, y_pred))
```

**R-Squared (R²)**

R^2（R squared）指标，也称为决定系数（Coefficient of Determination），是统计学中用于衡量 *回归模型拟合优度* 的一种常见指标。它表示模型解释因变量变异性的比例，即模型所解释的方差比例。

R^2 的取值范围在 0 到 1 之间，越接近 1 表示模型对观测数据的拟合程度越好，即模型能够解释较大比例的数据方差。而当 R^2 接近于 0 时，说明模型对数据的拟合程度较差，即模型无法解释观测数据的方差。

它的计算是 1 - （ 残差平方和 / 总平方和 ）残差平方和预示了预测和真实数据的差异。

相关代码：

```python
from sklearn.metrics import r2_score
y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]
print(r2_score(y_true, y_pred))
```

**要点**

- 均方根误差 (RMSE) 提供了比均方误差 (MSE) 更好的解释，并且在取平方根后进行了简化。
- 当特别不希望有大误差时，均方根误差 (RMSE) 应该更有用。
- MSE、RMSE 或 MAE 更适合用来比较不同回归模型之间的性能。
- R平方最适合向高管和其他非技术人员解释回归模型的结果。
- 如果我们不想惩罚较大的预测误差，我们可以使用 MAE。

### DummyRegressor

我们在跑模型的时候，总是会先跑一个baseline。比如我跑CNN的时候虽然是图像识别，但是一开始可能会跑一个全是线性模型堆叠的全联接层，作为基准。

DummyRegressor 是 Scikit-learn 中的一个回归模型，它是一种基本的回归模型，用于作为基准模型进行比较或评估其他更复杂的回归模型的性能。DummyRegressor 的作用是简单地使用一种规则或策略来预测目标变量的值，从而提供一个基准性能水平，以便评估其他回归模型的表现是否优于基准水平。

DummyRegressor 可以根据不同的策略生成预测值，常见的策略包括：

1. **均值策略**（mean strategy）：DummyRegressor 使用训练集中目标变量的均值来预测所有样本的目标变量值。这种策略适用于假设所有样本的目标变量值都接近于均值的情况。

2. **中位数策略**（median strategy）：DummyRegressor 使用训练集中目标变量的中位数来预测所有样本的目标变量值。这种策略适用于假设所有样本的目标变量值都接近于中位数的情况。

3. **常数策略**（constant strategy）：DummyRegressor 使用用户指定的常数值来预测所有样本的目标变量值。这种策略适用于特定场景下的预测需求，例如将所有样本预测为相同的值。

DummyRegressor 的主要作用是提供一个简单的基准模型，帮助我们评估其他更复杂的回归模型的表现是否优于基准水平。如果其他模型的性能不能明显优于 DummyRegressor，那么可能需要重新考虑模型的设计或数据的特征。DummyRegressor 在实践中经常用于回归问题的初步探索和模型选择阶段，以便快速评估模型的性能和稳健性。不过说实话这是我第一次看到这个模型，这模型确实很随便，如果真的有人用我真的想听听经验。

### Cross Validation

交叉验证的目的是为了提高模型的鲁棒性。使用train_test_split将数据集分为训练，验证，测试集。另外我们通常说的交叉验证是指 k-fold 交叉验证。

这种方法，将数据集分成训练集和测试集，多次重复地将数据集划分为不同的训练集和测试集，以评估模型的性能。

在交叉验证中，数据集通常被分为 k 个相似大小的子集，其中一个子集被保留作为验证模型的测试集，而其他 k−1 个子集被用作训练模型的训练集。然后，对于每个子集，使用训练集来训练模型，然后使用测试集来评估模型的性能。最终，将所有 k 次测试结果的评估指标（如准确率、精确度、召回率等）进行平均，作为模型的最终性能评估。

目的当然就是为了训练出更加稳健的模型，同时减少数据浪费（因为你把一部分数据用来验证，相当于减少了训练数据），同时降低过拟合风险。还有就是找到最好的超参数。

当然这里说的寻找最佳超参数，我们经常和网格搜索，随机搜索，贝叶斯优化等技术一起使用。比如网格搜索，就是将超参数组合定义为网格，然后进行交叉验证，找到最优组合。

下面这段代码是对四个样本进行样本数2的划分：可以看到得到的结果是两种情况，不是我们说的排列组合的C(4,2)，因为每次的测试样本不能重复。所以有两种分割方法。

```python
import numpy as np
from sklearn.model_selection import KFold

X = ["a", "b", "c", "d"]
kf = KFold(n_splits=2)
count = 0
for train, test in kf.split(X):
    print("Split  %i : Train %s Test %s " % (count, train, test))  
    count = count + 1

# Split  0 : Train [2 3] Test [0 1] 
# Split  1 : Train [0 1] Test [2 3] 
```

还有一种叫LOO的交叉验证分割数据方法，它在每一次划分中都留出单个样本作为测试集，其余样本作为训练集。LOO 交叉验证的优点是能够利用数据集中的每一个样本进行模型评估，因此对模型性能的评估更为准确。但是如果数据集太大，计算成本就很高了。

Leave-One-Out (LOO) 交叉验证方法适用于以下情况：

- 小样本数据集：当你拥有的样本数量比较少时，LOO 交叉验证是一个很好的选择。因为 LOO 每次只留下一个样本作为测试集，所以它可以在样本数量较少的情况下提供更准确的模型评估。
- 高度不平衡的数据集：当你的数据集中存在类别不平衡问题时，LOO 交叉验证可以提供更准确的评估，因为它可以确保每个类别都至少有一个样本作为测试集。
- 模型评估：LOO 交叉验证可以用于评估模型的性能，尤其是对于需要尽可能准确地评估模型泛化能力的情况下，如模型选择和调参。

下面的代码就是使用了这种方法，因为一共四个数据，所以每次都使用了一个数据作为验证，一共有四种方法。

```python
from sklearn.model_selection import LeaveOneOut
X = ["a", "b", "c", "d"]
loo = LeaveOneOut()
count = 0
for train, test in loo.split(X):
    print("Split  %i : Train %s Test %s " % (count, train, test))  
    count = count + 1

# Split  0 : Train [1 2 3] Test [0] 
# Split  1 : Train [0 2 3] Test [1] 
# Split  2 : Train [0 1 3] Test [2] 
# Split  3 : Train [0 1 2] Test [3] 
```

总的来说，从历史和前人惊艳来看，划分5-fold或者10-fold，在模型评估中会表现比较好。

### 接触很少的回归模型

1. **Least Angle Regression (LARS)**:
Least Angle Regression 是一种用于高维数据集的回归方法，它的主要特点是在每一步选择具有最小残差的特征，并沿着与目标变量最相关的方向移动。它与lasso回归（Lasso Regression）密切相关，并且在处理具有高度相关特征的数据集时表现良好。

```python
from sklearn.linear_model import Lars
# 创建LARS回归模型
lars_model = Lars()
```

2. **Polynomial Regression**:
多项式回归是一种基于多项式函数的回归方法，它允许通过使用多项式拟合数据来捕获输入特征和输出之间的非线性关系。通过在原始特征上添加高次项，多项式回归可以对数据进行更灵活的建模，以适应不同的数据模式。

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
# 创建多项式回归模型
degree = 3  # 多项式的阶数
poly_reg_model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
```

3. **Bayesian Regression**:
贝叶斯回归是一种基于贝叶斯统计理论的回归方法，它利用先验概率和观测数据的联合分布来估计参数。与传统的最小二乘回归相比，贝叶斯回归可以提供对参数不确定性的更好估计，并且能够在参数较少的情况下处理多个输入特征。

```python
from sklearn.linear_model import BayesianRidge
# 创建贝叶斯回归模型
bayesian_model = BayesianRidge()
```

4. **Robustness Regression**:
鲁棒性回归是一种对异常值或数据噪声具有鲁棒性的回归方法。它通过使用鲁棒性损失函数（如Huber损失函数）来减少异常值对拟合模型的影响，从而提高模型的稳健性和鲁棒性。

```python
from sklearn.linear_model import HuberRegressor
# 创建鲁棒性回归模型
huber_model = HuberRegressor()
```

5. **Isotonic Regression**:
保序回归（Isotonic Regression）是一种非参数的回归方法，它假设输出变量随输入变量的增加而单调递增或单调递减。保序回归通过将数据分段进行单调递增或递减的拟合，从而产生一个保持输入顺序的预测输出序列。

```python
from sklearn.isotonic import IsotonicRegression
# 创建保序回归模型
isotonic_model = IsotonicRegression()
```
世界上有很多算法，都是针对具体问题出现的，没有万能的算法，只有针对问题和情况最适用的算法。也许未来会出现真正的强人工智能，不，一定会的。
