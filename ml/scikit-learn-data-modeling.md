## 使用Scikit-learn进行建模学习：回归，分类，调参，聚类

---

内含Scikit-learn的主要数据模型包括分类，回归，超参调优，评估方法。这真的是一个很优雅的框架。在使用Tensorflow和Pytorch的人都知道，sklearn的东西是可以直接拿来辅助使用的。

他们更多是一种统计模型，因为日常我们不可能总是训练神经网络。更多的是在更小的数据集上进行统计分析，所以这个框架是日常建模的最佳选择。这么说起来其实大模型和现在的神经网络模型我们更适合的是直接

### 一，线性模型

```python
from sklearn import linear_model
def linear_reg(data, labels):
    reg = linear_model.LinearRegression()
    reg.fit(data, labels)
    return reg
```

优化算法是最小二乘法。目的是残差平方和的最小化。但是最小二乘法，依赖于特征之间互相独立，比如很多features都是描述货币的不同币种，那么就会产生很多噪声，因为他们之间不相互独立。

岭回归是一种正则化方法，通过引入一个正则项拉姆达，来降低噪声防过拟合。

普通的模型建模：

```python
from sklearn import linear_model
reg = linear_model.Ridge(alpha=0.1)
reg.fit(data, prices)
print('Coefficients: {}\n'.format(repr(reg.coef_)))
print('Intercept: {}\n'.format(reg.intercept_))
r2 = reg.score(pizza_data, pizza_prices)
print('R2: {}\n'.format(r2))
```

这里的coef是系数，intercept是截距，都是表达自变量对因变量的影响程度。最后的R2是一项评估，表达模型可以解释的变异程度，预测准确程度。

下面是使用了交叉验证的岭回归模型：

```python
from sklearn import linear_model
alphas = [0.1, 0.2, 0.3]
reg = linear_model.RidgeCV(alphas=alphas)
reg.fit(data, prices)
print('Coefficients: {}\n'.format(repr(reg.coef_)))
print('Intercept: {}\n'.format(reg.intercept_))
print('Chosen alpha: {}\n'.format(reg.alpha_))
```
最后的输出可以输出最好的alpha系数。

写成漂亮的函数：

```python
def cv_ridge_reg(data, labels, alphas):
    reg = linear_model.RidgeCV(alphas=alphas)
    reg.fit(data, labels)
    return reg
```

岭回归是一种L2正则化方法，那么Lasso回归就是L1正则化。L1正则又叫做稀疏正则化，面对特征值较多的情况使用稀疏正则化可以降低复杂度（权重系数为0），防止过拟合，提高模型泛化。但是不可否认一些副作用，有可能导致模型出现偏差，因为可能删掉重要的特征，导致模型精度下降。

另外MAE平均绝对误差的评价标准也叫L1Loss，MSE均方误差那么就是L2Loss了。（Pytorch学习时候想到。）

Lasso回归的代码：

```python
# predefined dataset
print('Data shape: {}\n'.format(data.shape))
print('Labels shape: {}\n'.format(labels.shape))
from sklearn import linear_model
reg = linear_model.Lasso(alpha=0.1)
reg.fit(data, labels)
print('Coefficients: {}\n'.format(repr(reg.coef_)))
print('Intercept: {}\n'.format(reg.intercept_))
print('R2: {}\n'.format(reg.score(data, labels)))
```
输出如下内容：（删除了空行）
```
Data shape: (150, 4)
Labels shape: (150,)
Coefficients: array([ 0.        , -0.        ,  0.40811896,  0.        ])
Intercept: -0.5337110569441175
R2: 0.8958211202747038
```
可以看到四个维度的特征，好几个被0掉了。然后写成好看的函数：
```python
def lasso_reg(data, labels, alpha):
    reg = linear_model.Lasso(alpha=alpha)
    reg.fit(data, labels)
    return reg
```

### 二，贝叶斯回归

首先还是要理解贝叶斯技术，还是让我想起以前一个讲解贝叶斯公式的[视频](https://www.bilibili.com/video/BV1a4411B7B4/?spm_id_from=333.337.search-card.all.click&vd_source=6f369a5f476f1c51d95e50188f9c4a81)。这个视频讲的很棒。对公式很清楚，但是公式背后讲的是什么。

```python
# predefined dataset from previous chapter
print('Data shape: {}\n'.format(data.shape))
print('Labels shape: {}\n'.format(labels.shape))

from sklearn import linear_model
reg = linear_model.BayesianRidge()
reg.fit(data, labels)
print('Coefficients: {}\n'.format(repr(reg.coef_)))
print('Intercept: {}\n'.format(reg.intercept_))
print('R2: {}\n'.format(reg.score(data, labels)))
print('Alpha: {}\n'.format(reg.alpha_))
print('Lambda: {}\n'.format(reg.lambda_))
```
最后得出的阿尔法是控制先验分布形状的参数，阿尔法越大分布越趋进于0，拉姆达是控制先验分布尺度的参数，拉姆达越大，先验分布越分散。他们都是控制模型复杂度的重要参数。然后上函数：
```python
def bayes_ridge(data, labels):
  reg = linear_model.BayesianRidge()
  reg.fit(data, labels)
  return reg
```
贝叶斯技术适用哪些问题？回归问题，空间统计，时间序列分析，还有机器学习，但是他的计算量很大，对于量化不确定性是一种强大的统计工具。

### 三，逻辑回归，分类！

名字容易混淆视听的一种模型，是一种分类模型，因为他输出的是logits是对结果的一种可能性预估，因为是连续空间的概率事件，所以也可以说是回归，但是本质是通过概率确定分类结果。

相关代码：
```python
# predefined dataset
print('Data shape: {}\n'.format(data.shape))
# Binary labels
print('Labels:\n{}\n'.format(repr(labels)))

from sklearn import linear_model
reg = linear_model.LogisticRegression()
reg.fit(data, labels)

# create data for prediction
new_data = np.array([
  [  0.3,  0.5, -1.2,  1.4],
  [ -1.3,  1.8, -0.6, -8.2]])
print('Prediction classes: {}\n'.format(
  repr(reg.predict(new_data))))
```
上面是一个二分类问题，同时逻辑回归还可以用于多分类问题。只要对模型加入参数multi_class，这个参数的默认设置是ovr（意思是One-vs-Rest，将该类和其他所有的类别分开，将多分类问题转化为二分类问题的方法），下面使用不同的策略，多分类策略。

下面的例子假定有三个分类。
```python
from sklearn import linear_model
reg = linear_model.LogisticRegression(
    solver = 'lbfgs',
    multi_class = 'multinomial',
    max_iter = 200
)
reg.fit(data, labels)

# create data for prediction
new_data = np.array([
  [  0.3,  0.5, -1.2,  1.4],
  [ -1.3,  1.8, -0.6, -8.2]])
print('Prediction classes: {}\n'.format(
  repr(reg.predict(new_data))))
```
max_iter是训练轮次，默认是100，但是如果模型不能收敛，就会收到“收敛警告”，所以可以设置的大一点这里就设置了200，一般小的数据集用100到500，中型数据量用500到1000，再大就1000往上了。

关于solver到选择，根据需要正则化和数据规模进行选择，可以参考[官网](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)。

比如代码：
```python
from sklearn import linear_model
reg = linear_model.LogisticRegression(solver='lbfgs', max_iter=1000)
```
最后总结成函数：
```python
def multiclass_lr(data, labels, max_iter):
    reg = linear_model.LogisticRegression(solver='lbfgs', max_iter=max_iter, multi_class='multinomial')
    reg.fit(data, labels)
    return reg
```

### 四，决定树分类算法，它来了！

世界到处都是0和1，一个问题无法就是发生和不发生，阴阳，有无，都是如此。决定树就是这样！布尔的美妙。世界也是离散分布的，看似连续的值，其实在无比精细的每一个点其实也就是0和1的问题，所以强大的决定树可以解决分类也可以解决回归问题！

```python
from sklearn import tree
clf_tree1 = tree.DecisionTreeClassifier()
reg_tree1 = tree.DecisionTreeRegressor()
clf_tree2 = tree.DecisionTreeClassifier(
  max_depth=8)  # max depth of 8
reg_tree2 = tree.DecisionTreeRegressor(
  max_depth=5)  # max depth of 5

# predefined dataset
print('Data shape: {}\n'.format(data.shape))
# Binary labels
print('Labels:\n{}\n'.format(repr(labels)))
clf_tree1.fit(data, labels)
```
在分类模型中，使用信息熵和基尼不纯度衡量每个节点的好坏，在回归模型中使用均方误差（MSE），平均绝对误差（MAE）等。

### 五，将数据分为训练集和测试集

最常用的sklearn方法。

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
```

小函数：

```python
def dataset_splitter(data, labels, test_size=0.25):
  split_dataset = train_test_split(data, labels, test_size=test_size)
  train_set = (split_dataset[0], split_dataset[2])
  test_set = (split_dataset[1], split_dataset[3])
  return (train_set, test_set)
```

注意这个方法内部会帮我们打乱数据。

### 六，交叉验证

使用sklearn可以很方便的实现交叉验证。当我们有很大的数据集，可以将数据分为training，validation，testing，三个数据集的时候，使用交叉验证方法来训练training，validation的不同组合，可以使得训练变的更加强大。

分类算法使用精确度accuracy作为最后的评价标准。
```python
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
clf = linear_model.LogisticRegression(max_iter=3000)
# Predefined data and labels
cv_score = cross_val_score(clf, data, labels, cv=3)  # k = 3
```

回归算法使用R^2作为最后的评价标准。
```python
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
reg = linear_model.LinearRegression()
# Predefined data and labels
cv_score = cross_val_score(reg, data, labels, cv=4)  # k = 4
```

注意到使用交叉验证的方法`cross_val_score`我们不需要手动进行fit了，因为他内部会帮我们fit！

内部使用的算法是`stratified K-Fold`算法，这个算法基本保证每个round中，0和1的标签数量比都相同。

解析下面的交叉验证代码：

```python
is_clf = True  # 假设这是一个分类算法的model
for depth in range(3, 8):
  # 进行交叉验证输出一个scores的列表
  scores = cv_decision_tree(
    is_clf, data, labels, depth, 5)  # k = 5
  mean = scores.mean()  # 算出平均精确度accuracy
  std_2 = 2 * scores.std()  # 两个标准区间，95%的置信区间
  print('95% C.I. for depth {}: {} +/- {:.2f}\n'.format(
    depth, mean, std_2))
```

最终就可以得到使用哪个参数可以得到更高的精确度，如果更大的超参数效果更好，那么可能需要继续扩大测试上线，使用8，9之类的。

下面是一个决策树使用交叉验证的函数：

```python
def cv_decision_tree(is_clf, data, labels, max_depth, cv):
  if is_clf:
    d_tree = tree.DecisionTreeClassifier(max_depth=max_depth)
  else:
    d_tree = tree.DecisionTreeRegressor(max_depth=max_depth)
  scores = cross_val_score(d_tree, data, labels, cv=cv)
  return scores  
```

### 七，模型衡量指标

回归一般是R方，均方误差，平均绝对误差，分类使用精确度或者混淆矩阵。

分类：

```python
clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_labels)
predictions = clf.predict(test_data)

from sklearn import metrics
acc = metrics.accuracy_score(test_labels, predictions)
print('Accuracy: {}\n'.format(acc))
```

回归：

```python
reg = tree.DecisionTreeRegressor()
reg.fit(train_data, train_labels)
predictions = reg.predict(test_data)

from sklearn import metrics
r2 = metrics.r2_score(test_labels, predictions)
print('R2: {}\n'.format(r2))
mse = metrics.mean_squared_error(test_labels, predictions)
print('MSE: {}\n'.format(mse))
mae = metrics.mean_absolute_error(test_labels, predictions)
print('MAE: {}\n'.format(mae))
```

### 八，GridSearch

如果数据集够小，可以详尽地进行超参数训练，那么可以使用GridSearch交叉验证的方法。

```python
reg = linear_model.BayesianRidge()
params = {
  'alpha_1':[0.1,0.2,0.3],
  'alpha_2':[0.1,0.2,0.3]
}
reg_cv = GridSearchCV(reg, params, cv=5)
# predefined train and test sets
reg_cv.fit(train_data, train_labels)
print(reg_cv.best_params_)
```

注意到它和一般的交叉验证的方法不同。上面的普通交叉验证是将model和data一起作为参数，得到scores，这里的GridSearch是将所有的参数组合，以及model作为参数，将cv实例化，然后进行fit。

由于将所有的超参数进行了组合，所以在大的训练集上会非常慢，比较适合数据集不太大的情况。

**注意**，CV更多的是一种评分方法，也可以帮我们找到最好的超参数，但是GridSearch更多的是为了帮我们详尽地找到最好的参数。

### 八，无监督的集合算法：余弦相似度cosine similarity

万物都是向量，将任何X向量化，那么衡量他们之间的相似程度，可以用他们在空间中的余弦夹角来计算。越接近1，关系越近。

```python
from sklearn.metrics.pairwise import cosine_similarity
data1 = np.array([
  [ 1.1,  0.3],
  [ 2.1,  0.6],
  [-1.1, -0.4],
  [ 0. , -3.2]])
data2 = np.array([
  [ 1.7,  0.4],
  [ 4.2, 1.25],
  [-8.1,  1.2]])
cos_sims = cosine_similarity(data1, data2)
print(cos_sims)
```
将输出一个array表达的是两个数据中点的余弦相似度的值矩阵。

下面是一个实现，找到每一行中对相关数据的index的方法。

```python
cos_sims = cosine_similarity(data)
# 填充对角线为0，排出了自己和自己的最大相似度
np.fill_diagonal(cos_sims, 0)
# 找到每一行的最大数的所在索引即可
similar_indexes = cos_sims.argmax(axis=1)
```

### 九，KNN算法找出最邻近的K个点

虽说KNN是一种有监督的算法，也就是空间中的点是有标签的，通过找到K个最邻近的点来投票，得出目标的分类。

但是也可以将它作为一种集合算法，找到距离目标点最近的K个点，返回这些点，和距离。

```python
data = np.array([
  [5.1, 3.5, 1.4, 0.2],
  [4.9, 3. , 1.4, 0.2],
  [4.7, 3.2, 1.3, 0.2],
  [4.6, 3.1, 1.5, 0.2],])

from sklearn.neighbors import NearestNeighbors
nbrs = NearestNeighbors(n_neighbors=2)
nbrs.fit(data)
new_obs = np.array([
  [5. , 3.5, 1.6, 0.3],
  [4.8, 3.2, 1.5, 0.1]])
dists, knbrs = nbrs.kneighbors(new_obs)

# nearest neighbors indexes
print(f'{repr(knbrs)}\n')
# nearest neighbor distances
print(f'{repr(dists)}\n')
```

### 十，K-means算法

对空间中的点进行集群，找到K个集合，在K-means++算法中，一开始初始化的质心是随机的，通过不断更新和每个质心的距离，以及更新每个集群的质心的迭代操作，最终达到稳定状态的结果就是集群的结果。在大批量的数据上因为效率很低，所以有mini-batch-clustering的方法可以在一个小批量上进行集群，以提高效率和即时处理的效果。

```python
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3)
# data is predefined
kmeans.fit(data)

# cluster assignments
print(f'{repr(kmeans.labels_)}\n')

# centroids
print(f'{kmeans.cluster_centers}\n')

new_obs = np.array([
  [5.1, 3.2, 1.7, 1.9],
  [6.9, 3.2, 5.3, 2.2]])
# predict clusters
print(f'{repr(kmeans.predict(new_obs))}\n')
```

`labels_`会给出每个点的标签，表示分到的第几组，如果是三个组可能就是`0，1，2`的数组。`cluster_centers`可以返回每个组的质心坐标。最后的`predict`返回要预测的每个点的所属的集合的标签。

如果执行`Mini-batch kmeans`：`batch_size`参数是每次选取的样本数量。

```python
from sklearn.cluster import MiniBatchKMeans
kmeans = MiniBatchKMeans(n_clusters=3, batch_size=10)
# predefined data
kmeans.fit(data)

# cluster assignments
print(f'{repr(kmeans.labels_)}\n')

# centroids
print(f'{repr(kmeans.cluster_centers_)}\n')

new_obs = np.array([
  [5.1, 3.2, 1.7, 1.9],
  [6.9, 3.2, 5.3, 2.2]])
# predict clusters
print(f'{repr(kmeans.predict(new_obs))}\n')
```

整合两种方法的函数：
```python
def kmeans_clustering(data, n_clusters, batch_size):
  if batch_size is None:
    kmeans = KMeans(n_clusters=n_clusters)
  else:
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=batch_size)
  kmeans.fit(data)
  return kmeans
```

### 十一，分层集合hierarchical clustering

注意到Kmeans方法中，数据点都是围绕着质心成为一个球形。这是它的既定假设，但是这个假设不能包括所有的集合情况，因此使用kmeans方法就可能出现错误的结果，这是不可避免的。

而分层集合的算法，可以处理不同特性的数据。

分层聚类算法可以通过两种不同的方法来构建聚类层次结构：自底向上（自下而上）和自顶向下（自上而下）。

方法一：分裂（divisive）自底向上（自下而上）：算法从单个数据点开始，并逐步将相似的数据点合并成更大的聚类，直到达到指定的停止条件为止。具体步骤如下：

- 初始化：将每个数据点视为一个单独的聚类。
- 合并：重复以下步骤直到达到停止条件：
   - 计算所有聚类之间的相似度或距离。
   - 合并最相似的两个聚类，形成一个更大的聚类。
- 停止条件：停止条件通常是达到指定的聚类数量或达到特定的相似度阈值。

方法二：聚集（agglomerative）自顶向下（自上而下）：算法从整个数据集开始，并逐步将其划分成更小的子集，直到每个子集都满足某种终止条件。具体步骤如下：

- 初始化：将整个数据集视为一个单独的聚类。
- 分裂：重复以下步骤直到达到停止条件：
   - 将当前聚类划分成更小的子集。
   - 对每个子集递归地应用相同的分裂过程。
- 停止条件：停止条件通常是达到指定的聚类数量、达到特定的深度（层数）、或达到某种相似度阈值。

区别：

- 合并与分裂：自底向上方法通过不断合并相似的聚类来构建聚类层次结构，而自顶向下方法则通过不断分裂聚类来构建层次结构。
- 起点与终点：自底向上方法从单个数据点开始，最终合并成一个大的聚类；自顶向下方法从整个数据集开始，最终分裂成多个小的子集。
- 停止条件：自底向上方法通常使用指定的聚类数量或相似度阈值作为停止条件；自顶向下方法通常使用聚类数量、深度或相似度阈值作为停止条件。
- 选择自底向上或自顶向下方法取决于数据的特性以及对聚类层次结构的需求。

其中聚合，自顶向下的方法更常用，下面是sklearn库代码，个人感觉是因为自顶向下使用的资源和步骤更小，如果是自底向上，那么每一个sample一开始都是一个类，每次合并都需要大量的计算，而自顶向下效率较高，并可以更快地到达需要的目标群组数量，当然我觉得如果目标群组数量本来就很多，甚至趋近于样本大小，那么自底向上也许就需要采纳了。灵活运用。

```python
from sklearn.cluster import AgglomerativeClustering
agg = AgglomerativeClustering(n_clusters=3)
# predefined data
agg.fit(data)
# cluster assignments
print(f'{repr(agg.labels_)}\n')
```

### 十二，Mean-Shift聚类

刚刚的两个聚类算法都是我们给定了要几个类的情况，那么总有我们不知道该分几个类的情况，mean-shift算法就是帮我们决定分几个类的。

Mean Shift 聚类是一种无参数的密度估计和聚类算法。它通过在数据空间中移动一组数据点来寻找数据的密度最大值（模式），从而发现聚类中心。它的工作原理类似于将数据点视为粒子，然后让它们沿着密度梯度方向移动，直到到达密度的最大值。

在工作原理上，首先它为每个点选择一个随机初始点作为中心。然后对于每个候选中心，计算它周围点的密度估计，常用的密度估计方法是通过核函数（如高斯核函数）对数据点之间的距离进行加权计算。对于每个候选聚类中心，将它沿着密度梯度方向移动到密度估计增加的方向，直到达到局部密度的最大值。重复上述移动步骤，直到所有候选聚类中心不再改变位置或达到收敛条件。将最终的候选聚类中心作为聚类中心，将数据点分配到最近的聚类中心。

和梯度下降算法有异曲同工之妙。

它的优点是不需要预先指定聚类数量，而是根据数据的密度分布自动发现聚类中心。不受数据点分布的稀疏性或密集性的影响，对于不规则形状的聚类可以表现良好。与一些传统的聚类算法相比，Mean Shift 聚类在高维数据上也能够表现较好。

但是它的计算复杂度通常较高，特别是在处理大规模数据集时。同时算法涉及到的参数，如核函数的带宽，可能需要进行调参来获得最佳的聚类效果。另外初始的候选聚类中心的选择可能会影响最终的聚类结果，因此需要进行一定的初始化策略。

Mean Shift 聚类算法在许多领域都有广泛的应用，特别是在计算机视觉、图像分割、物体跟踪等领域。

代码实现和之前的kmean很相似，感谢sklearn的封装。

```python
from sklearn.cluster import MeanShift
mean_shift = MeanShift()
# predefined data
mean_shift.fit(data)

# cluster assignments
print(f'{repr(mean_shift.labels_)}\n')

# centroids
print(f'{repr(mean_shift.cluster_centers_)}\n')

new_obs = np.array([
  [5.1, 3.2, 1.7, 1.9],
  [6.9, 3.2, 5.3, 2.2]])
# predict clusters
print(f'{repr(mean_shift.predict(new_obs))}\n')
```

### 十三，DBSCAN算法

这也是一个根据密度进行聚合分类的算法。

DBSCAN（Density-Based Spatial Clustering of Applications with Noise）是一种基于密度的聚类算法，它能够发现具有足够高密度的区域，并将这些区域视为聚类，并且能够在噪声点周围识别出不属于任何聚类的孤立点。不要被DB两个字迷惑成数据库。

几个关键的概念：

- 核心点Core Point：对于给定的半径epsilon和最小点数minpoints，一个数据点被称为核心点，如果在以该点为中心、半径为epsilon的区域内至少包含minpoints个数据点（包括该点本身）。
- 边界点Border Point：如果一个数据点在某个核心点的epsilon半径范围内，但它本身不是核心点，则称其为边界点。
- 噪声点Noise Point：如果一个数据点既不是核心点也不是边界点，则称其为噪声点。
- 聚类扩展：DBSCAN从一个核心点出发，利用核心点的可达性来逐步扩展聚类。对于每个核心点，它将其epsilon范围内的点添加到同一个聚类中。然后继续处理该聚类中的边界点，将它们也加入到相同的聚类中。这样逐步扩展，直到没有新的点可以添加到聚类中为止。
- 孤立点识别：识别噪声点，即那些不能被任何核心点的邻域所访问到的点。

sklearn的相关代码：

```python
from sklearn.cluster import DBSCAN
dbscan = DBSCAN(eps=1.2, min_samples=30)
# predefined data
dbscan.fit(data)
# cluster assignments
print(f'{repr(dbscan.labels_)}\n')
# core samples
print(f'{repr(dbscan.core_sample_indices_)}\n')
num_core_samples = len(dbscan.core_sample_indices_)
print(f'Num core samples: {num_core_samples}\n')
```

### 十四，聚类算法评估指标：ARI和AMI

这两个指标用法很相似，ARI是调整后的兰德指数（adjusted Rand index），AMI是调整后的互信息（adjusted mutual information）。

他们用来衡量两个聚类结果的相似度，标准在 -1 到 1 之间。0 代表具有随机性，-1 代表完全不相似，1 代表完美分类。并且它具有对称性，就是说作为参数的时候，真实标签和预测标签的顺序没所谓，得到的结果是一样的。

另外即使改变了标签的名称，也不会影响结果，比如[0, 0, 0, 1, 1, 1]和[1, 1, 1, 3, 3, 3]这两个就是完美分类，结果应该是1。

在sklearn中有相对应的metric可用。

```python
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import adjusted_mutual_info_score
true_labels = np.array([0, 0, 0, 1, 1, 1])
pred_labels = np.array([0, 0, 1, 1, 2, 2])
ari = adjusted_rand_score(true_labels, pred_labels)
ami = adjusted_mutual_info_score(true_labels, pred_labels)
```

### 十五，特征聚类：feature clustering

聚类算法还可以进行将维。使用特征聚类，可以将相似的特征归为一类达到降维的目的。属于数据处理的一部分了。

```python
from sklearn.cluster import FeatureAgglomeration
agg = FeatureAgglomeration(n_clusters=2)
new_data = agg.fit_transform(data)
```
