## 无监督学习

指从没有标签的数据中进行推断的学习。

无监督学习帮助我们识别数据集中的模式。

数据聚类是无监督学习技术之一，其中单个聚类中的元素具有相同的属性并被分配相同的标签。一些最著名的聚类算法包括：

- K-means算法
- 层次聚类
- DBSCAN聚类

除此之外还有关联规则挖掘，异常检测和降维。

最常听的主要是Kmeans，降维PCA。异常检测也是我经常听到的主题。

## K-Means算法

它是一种迭代算法，如何工作：

- 随机挑选K个质心。
- 通过计算相似度或者距离将每个实例添加到最近的质心范畴内。
- 重新计算每个类别的（叫做簇）质心，继续进行所有实例的归类。
- 重复步骤，直到各个实例的分类不再变化。

它是一种简单可以实现的算法，当然它也有很多缺点，就从第一步初始化开始就有很多漏洞，如何选到最好的质心关系到最后的结果。所有进化出一种K-Means++算法涉及初始化质心，使它们彼此远离。它比随机选择质心要好得多。

通常使用欧氏距离（Euclidean distance）作为数据点之间的距离度量。欧氏距离是指在几何空间中，两个点之间的直线距离。理论上可以使用余弦相似度来替代欧氏距离，但在实际应用中需要注意，余弦相似度适用于表示为向量的数据，因此在使用余弦相似度时，需要将数据点表示为向量形式。此外，余弦相似度不考虑向量的绝对大小，而是关注向量之间的方向，因此在某些情况下，余弦相似度可能更适合于描述数据的相似性。

欧氏距离是计算点之间的距离。余弦相似度关注向量的方向，它是（向量之间的dot运算/各个向量的欧几里得范数相乘）得到的。

### 余弦相似度

**余弦相似度**的推导首先是从空间向量的点乘公式得到的，这就是点乘公式：A · B = ||A|| ||B|| cos(θ)

**点乘公式**描述了两个向量有多大的投影在对方的方向上，如果两向量完全重合，则它们的点乘结果就等于各自模长的乘积，即A·B = ||A|| ||B||。

设有两个n维空间向量A和B: A = (a1， a2， ...， an)，B = (b1， b2， ...， bn)

我们先将它们分别展开到n个基向量的线性组合: 
```txt
A = a1e1 + a2e2 + ... + anen
B = b1e1 + b2e2 + ... + bnen
```
其中e1， e2， ...， en是n维空间的基向量。

根据向量的线性运算法则，我们有:
```txt
A·B = (a1e1+a2e2+...+anen)·(b1e1+b2e2+...+bnen) 
= a1b1(e1·e1) + a1b2(e1·e2) + ... + a1bn(e1·en)
+ a2b1(e2·e1) + a2b2(e2·e2) + ... + a2bn(e2·en)
...
+ anb1(en·e1) + anb2(en·e2) + ... + anbn(en·en)
```

根据基向量的正交性质: ei·ej = 0， (i≠j)， ei·ei = 1

上式可以化简为: A·B = a1b1 + a2b2 + ... + anbn = Σ(aibi)  (i=1，2，...，n) 这就是空间向量的点乘公式。

**基向量(Basis Vector)**是构成某一向量空间的一组线性无关的向量。在n维空间中，有n个单位基向量e1，e2，...，en，可以构成该空间的一个基底。
对于2维空间，基向量通常设为: e1 = (1，0)， e2 = (0，1)，对于3维空间，基向量通常设为: e1 = (1，0，0)， e2 = (0，1，0)， e3 = (0，0，1)。可以看出它就是一个矩阵中，主对角线都为1。

基向量有以下几个基本性质:

- 模长为1：例如在3维空间，||e1|| = ||e2|| = ||e3|| = 1
- 两两正交（也就是90度）：任意两个不同的基向量ei和ej的点乘为0，即ei·ej = 0 (i≠j) 这就是所谓的正交性质。
- 基向量可用于线性表示任意向量：任意n维向量A可表示为n个基向量的线性组合: A = a1e1 + a2e2 + ... + anen
- 正是由于基向量的这些特性， 我们可以利用它们构造向量空间， 并在向量空间中研究向量的性质和运算。比如点乘公式的推导，就需要利用基向量的正交性质。

所以基向量是构造和分析向量空间的基础，掌握它们的性质对于线性代数等数学理论至关重要！

## 层次聚类 Hierarchical Clustering

层次聚类涉及创建聚类的层次结构。以层次结构的形式表示数据对象对于数据汇总和可视化很有用。

例如，假设我们想要将组织中的人员分为主要组，例如主管、经理和员工。我们可以将上面的这些进一步划分为更小的子组。这就是层次聚类的基本思想。

层次聚类设计两种，自下而上的凝聚层次聚类，和自上而下的分裂层次聚类。

**凝聚层次聚类**使用自下而上的策略。在此策略中，每个实例最初都被视为一个集群。在该聚类算法的每次连续迭代中，一个聚类与其他最相似的聚类合并，直到仅形成一个大聚类。

凝聚层次聚类的工作原理：

- 计算实例之间的相似度并以矩阵的形式表示它们，也称为相似度或邻近矩阵。
- 将每个实例视为一个簇。
- 合并最近的两个簇并更新邻近矩阵。
- 更新矩阵涉及计算新合并的簇与每个其他簇之间的簇距离。这样，在下一次迭代中，将使用这个新的更新后的矩阵，并且我们再次使用更新后的矩阵执行合并操作。
- 不断重复步骤，直到满足条件或形成一个簇。该条件可能是执行合并操作后剩余的最小簇。

如何表达每个簇之间的关系，这里用的是**联动标准Linkage criteria**，和我们一般来说计算各种距离一样，这个关联标准也有几个种类。

- 单链接：两个簇之间的距离定义为每个簇中两点之间的最短距离。有点像SVM中的那种最短距离感。
- 完全链接：两个簇之间的距离定义为每个簇中两点之间的最长距离。
- 平均链接：两个簇之间的距离定义为一个簇中的每个点到另一个簇中的每个点之间的平均距离。
- 质心联动：找到簇1的质心和簇2的质心，然后计算合并前两者之间的距离。

至于使用哪种标准取决于你的具体问题，和聚类效果。

**分裂式层次聚类**与凝聚式层次聚类相反。它的工作原理是将所有点视为一个簇，然后在每次连续迭代中，最不相似的点从父簇中分离出来。最后，我们剩下K个簇，其中所需情况下的 K 可以是数据集中的点总数。

这似乎更适合我们解决问题的需求。

但是总的来说。层聚聚类算法的时间和空间复杂度很高，不适合大型数据集。根据需求使用。

## DBSCAN Clustering

这里的DB和数据库可没关系！它的意思是Density Based spatial clustering of Applications with Noise，翻译为**基于密度的噪声应用空间聚类**。

专业词汇一上来就开始脑袋转圈圈了，但是没关系不要被这次缩写吓到。

从翻译中就可以看到它时基于密度的聚类。K-means也是基于聚类的算法，但是K-means的密度簇时球型的。但是DBSCAN可以发现任意形状的簇，并可以有效处理噪声数据。

DBSCAN的核心思想是基于数据的密度关联，将高密度区域中的数据点分为同一个簇，而将低密度区域的数据点视为噪声。

算法的运行需要两个关键参数:

1. **Eps(ϵ)**:邻域半径，定义了点的ϵ-邻域。即一个点的ϵ-邻域是以该点为中心，半径为ϵ的球形区域。

2. **MinPts**:设置了点在某个ϵ-邻域内的最少点数，用于定义密集区域。

算法步骤:

1. **遍历数据集，标记每个点的性质**
    - 核心点(Core Point):如果ϵ-邻域内点的个数超过MinPts，则称该点为核心点
    - 边界点(Border Point):非核心点，但在核心点的ϵ-邻域内
    - 噪声点(Noise Point):不属于任何一个簇的剩余点

2. **从一个核心点开始，构建一个簇**
    - 将核心点标记为新簇的一部分
    - 递归地将所有可达的密度相连的点加入该簇

3. **继续下一个未处理的核心点，重复步骤2**

4. **最后将噪声点单独作为一个簇输出**

DBSCAN算法的优点:
- 可以发现任意形状、大小的簇，不受簇数量、簇形状、数据密度等影响
- 能有效处理噪声数据，将其区分为噪声簇
- 可以很好地应用于高维数据集

缺点:
- 对边界数据的聚类效果不太理想
- 参数Eps和MinPts对结果影响较大，需要合理设置：
    - 如果设置的Eps太小了，那么大多数的点都不会聚类，而是被标记为 -1，表示异常或者噪声。
    - 相反，如果Eps设置太大了，会导致接近的簇合并为一个簇，最终将整个数据集作为一个簇返回。
- 对于密度相差较大的数据簇，识别效果不佳

DBSCAN常用于发现空间聚类分布，如地理数据的聚类、基因数据分析、网络数据分析等。是无监督学习中比较实用和流行的一种聚类算法。

案例：Customer Segmentation在Github仓库project-drafts中的，同名文件夹可以找到。这个项目的可视化分析真的挺好看的。

在聚类中我们经常听到肘部法则，在这里解释意思：

肘部法则(Elbow Method)是用于确定最优聚类数量的一种 *启发式* 方法。它的原理如下:

1) 聚类的目标是将数据点分成若干个聚类，使得同一个聚类内部的数据点相似度高，不同聚类之间的数据点相似度低。

2) 聚类数量越多，簇内平方和(Within-Cluster Sum of Squares， WCSS)就越小。因为数据点会被分配到更多的聚类中，每个聚类内部的数据点就更加相似。

3) 但是，随着聚类数量的增加，每次增加一个新的聚类所能降低的WCSS值就会越来越小。这是因为剩余的数据点已经比较相近了，新增一个聚类的效果就不太明显。

4) 因此，当绘制聚类数量与WCSS值的关系线图时，这条线最初会是陡峭下降，但是当聚类数量增加到一定程度后，这条线的斜率就会变得很小。

5) 这条线的"肘部"(拐点处)，就是一个合适的聚类数量。在这个点之前，增加聚类数可以大幅降低WCSS;但在这之后，增加聚类数只会略微降低WCSS。

6) 肘部法则认为，这个拐点处的聚类数量就是最优聚类数，因为它实现了WCSS值的大幅降低，而不会过多增加聚类数导致模型过于复杂。

总的来说，肘部法则利用了聚类数量与WCSS值之间的这种"肘形"关系，通过视觉上识别拐点来确定最优聚类数。这种方法简单、可解释，但也有一定的主观性，需要结合其他指标共同评估聚类效果。

## 关联规则挖掘和它的相关算法

关联规则挖掘(Association Rule Mining)是数据挖掘的一个重要分支，主要用于发现数据集中有趣、频繁、意义重大的关联关系或相关模式。它通常应用于购物篮数据分析、网页挖掘、产品聚类等场景。

关联规则的形式为:A->B，表示如果事务数据集中包含项集A，则也很可能包含项集B。常用的两个指标来描述规则的质量:

- 支持度(Support)=P(A，B)，表示包含A和B的记录占总记录的比例
- 置信度(Confidence)=P(B|A)=P(A，B)/P(A)，表示已知A发生时，B发生的概率
- 置信度值越高,表明该规则的可信程度越高。通常会设置一个最小置信度阈值,低于该阈值的规则会被过滤掉。
置信度可以度量规则的确定性有多强,对发现有价值的关联模式很重要。但置信度较高的规则未必就是有趣的或重要的,因此还需要结合其他指标如支持度、lift等综合评估。

**Apriori算法**

Apriori算法是早期最有影响力的关联规则挖掘算法。它分两个步骤:

- 发现频繁项集:通过迭代从1-项集（只包含一个元素的项集）开始，利用支持度剪枝减少搜索空间，发现所有频繁项集。
- 生成关联规则:对每个频繁项集，只要其两个非空子集的置信度满足最小置信度要求，就构建一条规则。

**FP-Growth算法**

FP-Growth(Frequent Pattern Growth)算法通过构建FP-树来高效发现频繁项集，不需要像Apriori那样产生大量候选集。算法分两步:

- 构建FP-树:扫描两次数据集，发现频繁1-项集并构建FP-树。
- 在FP-树上挖掘频繁项集:以频繁模式基为起点，通过连接子节点构建条件FP-树，递归挖掘频繁项集。

总的来说，关联规则挖掘算法的关键是高效发现频繁项集，并生成满足约束的有趣规则。企业可以基于这些挖掘结果进行产品购买相关性分析、销售促销策略制定等决策。

这一部分现在我不是很了解，学习资料也没有详细再讲下去，而是作为聚类的一个部分被提起。因此这里只做一个了解吧。

## 降维技术：主成分分析

如同它的文字表述，它的目的是将高维度数据转换为低纬度数据。

换个表述方式就是将大量特征，变为较少的特征量。

因为维数过高会导致模型复杂和难以可视化。

降低维度的好处很多，有助于快速拟合模型，降低复杂度，可视化，消除多重共线性（一些特征彼此相关，提供了很多冗余的信息）问题。

降维技术中最重要的是主成分分析。简写为PCA。

**PCA的步骤拆解：**

*Standardization*

首先对每个特征进行正规化处理，将特征都转换为均值为0，方差为1的数据。这样做的好处是将所有特征进行统一。

计算公式为：x - 均值 / 标准差

具有较大范围值的特征，会比具有较小范围值的特征占据主导地位。

*CovarianceMatrix*

第二步是，根据上面的正规化后的特征，计算协方差矩阵。

协方差矩阵具有对称的性质，它的对角线上是方差，非对角线上是两两特征的协方差。

反应了变量之间的线性相关程度和方向。

*EigenDecomposition*

执行协方差矩阵的特征值分解。

它是将矩阵分解为一组特征值和特征向量的过程。

它在数学上，是将矩阵简化为对角线矩阵和另一个矩阵的乘积。

特征值反映了矩阵沿着哪些方向伸缩，特征值大小对应了伸缩的大小和程度。对应地，特征向量反映了，伸缩的方向。

特征值分解为我们研究矩阵的本质属性、发现隐含模式、降维和压缩提供了强有力的数学工具。

*SortByEigenValues*

主成分被构造为原始特征的线性组合。主成分的构造方式是将原始变量中的大部分信息（最大可能的方差）压缩到第一个成分中，然后挤压到其他成分中，依此类推。

假设总共构建了 n 个主成分其中 n 是数据集的维度总数。我们按照特征值的顺序对主成分进行排序，从最高到最低，也就是重要性进行了排序。

*ChoosePrincipalComponents*

决定保留的主成分数量 k 个，并选择最重要的 k 个主成分，并构建一个向量矩阵，称之为特征向量。

这些是我们最后得到的，在新的较低维度中的特征。

**在Scikit-learn中的实现**：

以鸢尾花数据集为例的PCA实现：

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
df = pd.read_csv(url, names=['sepal length','sepal width','petal length','petal width','target'])
print("The Iris Dataset before applying PCA is")
print(df.head().to_string())
x = df[['sepal length', 'sepal width', 'petal length', 'petal width']].values
y = df[['target']].values
x = StandardScaler().fit_transform(x)
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])
finalDf = pd.concat([principalDf, df[['target']]], axis = 1)
print("\n")
print("The Iris Dataset after applying PCA is")
print(finalDf.head().to_string())
```

原始数据如下：
```
The Iris Dataset before applying PCA is
   sepal length  sepal width  petal length  petal width       target
0           5.1          3.5           1.4          0.2  Iris-setosa
1           4.9          3.0           1.4          0.2  Iris-setosa
2           4.7          3.2           1.3          0.2  Iris-setosa
3           4.6          3.1           1.5          0.2  Iris-setosa
4           5.0          3.6           1.4          0.2  Iris-setosa
```

中间的进行正规化后的x输出前五行查看：
```
[[-0.90068117  1.03205722 -1.3412724  -1.31297673]
 [-1.14301691 -0.1249576  -1.3412724  -1.31297673]
 [-1.38535265  0.33784833 -1.39813811 -1.31297673]
 [-1.50652052  0.10644536 -1.2844067  -1.31297673]
 [-1.02184904  1.26346019 -1.3412724  -1.31297673]]
```

最终输出结果：
```
The Iris Dataset after applying PCA is
   principal component 1  principal component 2       target
0              -2.264542               0.505704  Iris-setosa
1              -2.086426              -0.655405  Iris-setosa
2              -2.367950              -0.318477  Iris-setosa
3              -2.304197              -0.575368  Iris-setosa
4              -2.388777               0.674767  Iris-setosa
```
