## Python进行统计分析

---
（项目学习收录在project-drafts里）

导入Python库。

```python
# Libraries
import pandas as pd
import scipy.stats as st
import math as m
import statsmodels.stats.api as sm
import numpy as np
```

### 推论统计：描述性统计

阐述一些概念：

统计中的**数据类型**：包括定量（quantitaive）连续型数据（continuous），离散型数据（discrete）。和定性（qualitative）类型名义（nominal）和序数（ordinal）。Nominal数据是分类标签数据，比如男女，而Ordinal是有固定数量等级的数据，他们看起来也是分类但是之间的关系不是平等的，比如小学，初中，高中。

**总体（population）和样本（sample）**。样本是总体的子集。研究总体是近乎不可能的，所以需要样本，样本应具有代表性。

通过`df.info()`读取样本信息。

**集中趋势分析central tendency**：包括均值mean，中位数median，众数mode。中位数更具有鲁棒性，因为对outlier异常值不会特别敏感。mode更多用于离散数据的统计。

Python的均值mean计算：(中位数和众数只需要使用`median()`和`mode()`)，通过查看均值和中位数的差，可以想象有没有很大的异常值。而众数则可以想象一下投票表决，以多胜少的情况。使用`df.col.hist()`方法可以查看分布，是表达众数的一种方法。

```python
# Mean of Runs Scored
df.RS.mean()

# Mean of Runs Scored by the Arizona Team (ARI)
df.loc[df.Team == "ARI"].RS.mean()

# Mean of Runs Scored by the Arizona team since 2005
df.loc[(df.Team == "ARI") & (df.Year >= 2005)].RS.mean()

# What is the average of Runs Allowed by the Chicago Tem (CHC)
# and before 2000 or after 2005
df.loc[(df.Team == "CHC") & ((df.Year < 2000) | (df.Year > 2005))].RA.mean()
```

**标准差和方差（Standard Deviation and Variance）**：方差开根是标准差，标准差的平方是方差。

根号内的计算：分子是每个元素和均值的差的平方。

分母是n但是在样本数据中使用（n-1）是为什么？因为是样本中的计算。总体的数量大于样本，计算的结果比样本要小很多，因此为了无偏估计，将样本计算的分母减去1来估计总体的方差标准差，这种矫正叫做贝塞尔校正。

使用Python对数据集的OOBP特征进行方差和标准差的计算：

```python
print(f"Mean: {df.OOBP.mean()}")
print(f"STD(sample): {df.OOBP.std()}") # 默认为样本数据计算，分母位n-1
print(f"STD(population): {df.OOBP.std(ddof=0)}") # 通过增加一个参数使计算对象变为总体，也就是分母为n
# 两种方差计算方法
print(f"Variance: {df.OOBP.std() * df.OOBP.std()}")
print(f"Variance: {df.OOBP.var()}")
```

在观察标准差，方差的时候，总是会同时输出mean均值，是为了了解数据的中心位置在哪里。

通过Python的`describe()`方法可以立刻查看各种指标：count，mean，std，min，max，各个百分位数。

**变异系数**：

"Coefficient of variation"（变异系数）将**标准差除以均值**，然后乘以100%，就可以得到以百分比表示的变异系数。

变异系数用来衡量数据的离散程度相对于其平均值的大小。它是标准差与平均值的比率，通常以百分比表示。

变异系数常用来判断不同数据集或者不同变量的变异程度，其主要作用包括：

1. **比较不同数据集的离散程度**：当数据具有不同的尺度或单位时，直接比较它们的标准差可能不合适，因为标准差受到尺度的影响。通过计算变异系数，可以消除尺度的影响，从而更好地比较不同数据集的离散程度。

2. **判断变量之间的变异性**：对于多个变量，变异系数可以帮助确定哪个变量具有更大的变异程度。较高的变异系数意味着变量的值更加分散或波动较大，而较低的变异系数则表示变量的值更加稳定。

3. **评估数据的相对稳定性**：在一些应用中，例如质量控制或风险评估，变异系数可用于评估数据的相对稳定性。较低的变异系数可能表示数据更加稳定，而较高的变异系数则可能意味着数据更不稳定或者波动较大。

总的来说，变异系数是一种有用的统计量，可用于比较不同数据集或变量的变异程度，并帮助评估数据的相对稳定性。

```python
# Coefficient of variation for Runs Scored variable
cv = df.RS.std() / df.RS.mean()
print(f"The CV for RS is {cv}")
#As a rule of thumb, data with variability has a cv > 1
```

Python自带的方法`cov()`也可以直接计算。

```python
# Covariance with continuous variables
df_co = df[["RS", "RA", "W"]]
df_co.cov()
```

**协方差**：

协方差（Covariance）是一种统计量，用于衡量两个随机变量之间的关系或者联合变化程度。它描述了这两个变量的变化趋势是否一致。如果两个变量的变化趋势一致（即一个变量的值增加时，另一个变量的值也增加），则协方差为正数。如果一个变量的增加与另一个变量的减少相关，则协方差为负数。如果两个变量之间没有明显的关系，则协方差接近于零。强调**方向**性的关系。

协方差的数值大小并不能直接用来衡量两个变量的相关性的强度，因为它的值受到变量单位的影响。为了消除这种影响，可以使用相关系数来衡量两个变量之间的线性关系的强度。

Cov(X,Y)= ∑ (Xi - Xˉ)(Yi − Yˉ) * 1/n（总体样本n，样本样本使用n-1）

**相关性**：

相关性（Correlation）是指两个或多个变量之间的关联程度或者关系的强度。在统计学和数据分析中，相关性描述了一个变量的变化如何与另一个变量的变化相关联。当一个变量的值发生变化时，如果另一个变量的值也随之发生变化，那么这两个变量就被认为是相关的。

相关性通常用相关系数来量化。相关系数是一个介于 -1 和 1 之间的数值，它表示两个变量之间的线性关系的强度和方向。具体来说：

- 当相关系数接近于 1 时，表示两个变量呈正相关，即一个变量的增加伴随着另一个变量的增加。
- 当相关系数接近于 -1 时，表示两个变量呈负相关，即一个变量的增加伴随着另一个变量的减少。
- 当相关系数接近于 0 时，表示两个变量之间没有线性关系。

需要注意的是，相关性并不意味着因果关系。即使两个变量之间存在相关性，也不能推断其中一个变量的变化直接导致了另一个变量的变化。因此，在分析数据时，需要谨慎区分相关性和因果关系。

相关性和方差与机器学习紧密相关的原因在于它们在不同方面都对机器学习模型的性能和特征选择起到了重要作用。

   - 相关性：
   - 在特征选择中，相关性可以帮助确定哪些特征与目标变量相关，从而有助于筛选出最具预测能力的特征。相关性分析可以帮助识别出对模型性能影响最大的特征，从而提高模型的效率和准确性。
   - 在特征工程中，可以利用特征之间的相关性来创造新的特征，或者通过降维技术去除高度相关的特征，从而减少模型的复杂性和冗余信息。

   - 方差：
   - 方差衡量了模型对训练数据的敏感性，即模型在不同训练数据上的表现是否稳定。在过拟合的情况下，模型的方差往往会很高，这意味着模型对训练数据中的噪声过于敏感，而对新数据的泛化能力较差。
   - 通过控制模型的方差，例如通过正则化技术（如L1或L2正则化）或集成方法（如随机森林或梯度提升树），可以提高模型的泛化能力，避免过拟合问题。

因此，理解和利用相关性和方差可以帮助优化机器学习模型的性能、提高模型的泛化能力，并对特征选择和特征工程进行更好的指导。

Python中计算相关性的方法，以及使用seaborn可视化相关性矩阵：

```python
# Correlation Matrix
df.corr()

# Visualization via Heatmap
import seaborn as sns
sns.heatmap(data = df.corr(),
            annot = True,
            center = 0,
            cmap = 'coolwarm',
            linewidths = 1,
            linecolor = 'black')
```

**协方差和相关性**：

协方差（Covariance）和相关性（Correlation）都是用来描述两个变量之间的关系的统计量，但它们在计量方法和解释方面有所不同。

1. **协方差（Covariance）**：
   - 协方差衡量了两个随机变量之间的总体线性关系的强度和方向。具体而言，它描述了这两个变量如何一起变化。如果两个变量的协方差为正值，表示它们之间存在正相关关系，即当一个变量增加时，另一个变量也往往增加；如果协方差为负值，则表示它们之间存在负相关关系，即一个变量增加时，另一个变量通常减少。
   - 由于协方差的数值大小受到变量单位的影响，所以协方差不能直接用来比较不同数据集之间的相关性。

2. **相关性（Correlation）**：
   - 相关性是协方差的标准化版本，它除以各自变量的标准差，从而消除了变量单位的影响，使得相关系数的取值范围始终在 -1 到 1 之间。
   - 相关系数（Correlation Coefficient）可以直接反映两个变量之间的关系的强度和方向。如果相关系数接近于 1，表示存在强正相关关系；接近于 -1，表示存在强负相关关系；接近于 0，表示两个变量之间几乎没有线性关系。
   - 与协方差相比，相关系数更直观、更易解释，并且不受变量单位的影响。

综上所述，协方差衡量了两个变量之间的线性关系，而相关系数不仅衡量了线性关系的强度和方向，还消除了单位的影响，因此更常用于比较不同数据集之间的相关性。

**正态分布**：

正态分布（Normal Distribution），也称为高斯分布（Gaussian Distribution），是统计学中最重要和最常见的分布之一。它的特点是呈钟形曲线，中心对称，左右对称，且由两个参数决定：均值（μ）和标准差（σ）。

正态分布的特点包括：
- 曲线在均值处取得最高点，并且对称于均值；
- 标准差决定了曲线的宽度，标准差越大，曲线越宽；
- 大约68% 的数据落在均值加减一个标准差范围内，大约95% 的数据落在均值加减两个标准差范围内，大约99.7% 的数据落在均值加减三个标准差范围内，这被称为正态分布的"68-95-99.7法则"或"三倍标准差原则"。

正态分布在自然界和社会科学中广泛出现，并且在统计学和机器学习中扮演着重要的角色，例如用于模拟随机变量、假设检验、参数估计等。

使用Python画出正态分布曲线：

```python
# Density plot for RAs
df.RA.plot.density()
```

通过手动计算，查看是否符合正态分布三个标准差百分比的法则：

```python
# Do the 68-95-99 check, use a for plot
for i in range(1,4):
  print(df.loc[(df.RA <= df.RA.mean() + i * df.RA.std()) &
        (df.RA >= df.RA.mean() - i * df.RA.std())].RA.count()/df.RA.count())

# output
# 0.674512987012987
# 0.9594155844155844
# 0.9983766233766234
```

通过假设检验（查看下面的内容）可以判断数据是否符合正态分布。

代码如下：

```python
# Create a function to read the p-value
def p_value_reader(p_value, alpha):
  if p_value < alpha:
    print("Reject the Null Hypothesis")
  else:
    print("Fail to reject the Null Hypothesis")

# Shapiro Wilks Test for Normality
# Null Hypothesis: The data looks normal
# Alternative Hypothesis: The data does not look normal
stat, p_value = st.shapiro(df.W)
print(f"The p-value is {p_value}")
p_value_reader(p_value, 0.05)
```

### 推论统计：置信区间

**样本均值标准误差**（Standard Error of the Sample Mean，SEM）是对样本均值估计的精确度的一种度量。它衡量了样本均值与总体均值之间的差异的估计精度。SEM的计算方法通常是将样本标准差除以样本容量的平方根，如下所示：

SEM = s / sqrt(n)

其中，
- SEM是样本均值标准误差，
- s是样本标准差，
- n是样本容量。

SEM的大小与样本标准差的大小以及样本容量的大小有关。当样本标准差较大或样本容量较小时，SEM通常较大，表示样本均值的估计不够精确；而当样本标准差较小或样本容量较大时，SEM通常较小，表示样本均值的估计更加精确。

SEM在统计学中通常用于计算置信区间或进行假设检验。

```python
# With the formula: SD / sqrt(n) -> Price
import math as m
print(df.Price.std() / m.sqrt(df.Price.count()))
# use scipy to do the same thing
import scipy.stats as st
print(st.sem(df.Price))
```
以上是两种计算方法。SciPy是一个开源的Python科学计算库，而`scipy.stats`是其中的一个模块，提供了大量的统计函数和概率分布，包括假设检验、描述性统计、概率分布拟合等功能。通过这个模块，你可以进行各种统计分析，如t检验、ANOVA、线性回归等。

**Z分数（Z-score）**是一个统计量，用于表示一个数据点距离均值的偏差程度，通常以标准差为单位。Z分数告诉你一个数据点在数据集中相对于均值的位置。

z = （x - 均值）/ 标准差

Z分数为正表示该数据点高于均值，为负表示低于均值，而Z分数为0表示数据点与均值相等。Z分数还可用于识别异常值：一般情况下，绝对值大于2或3的Z分数可以被认为是异常值。

Z分数在统计学中有多种应用，例如标准化数据、检测异常值、进行假设检验等。在标准正态分布中，Z分数的分布遵循标准正态分布（均值为0，标准差为1），这种性质使得Z分数在统计分析中非常有用。

**标准化**和Z分数实际上指的是同一件事情：将数据转换为具有特定均值和标准差的分布。标准化通常使用Z分数来实现，因此Z分数也称为标准化值。

具体来说，标准化是一种数据预处理技术，旨在使数据集的特征具有统一的尺度，以便更好地适用于某些模型或算法。在标准化过程中，通过减去均值然后除以标准差，将原始数据转换为均值为0、标准差为1的分布，这就是Z分数。

因此，通过标准化，原始数据中的每个数据点都会变成它在数据集中的位置的Z分数。这样做的**好处**是，使得不同特征之间的比较更为合理，同时还可以使某些机器学习算法更加稳定和有效地运行，尤其是那些对数据尺度敏感的算法，如支持向量机（SVM）和k近邻（KNN）等。

因此，可以将标准化视为将数据转换为Z分数的过程，而Z分数则是实际上被用来衡量标准化程度的指标。

```python
# Us using the formula for Delivery Time
df['col_standardized'] = (df['col'] - df['col'].mean()) / df['col'].std()
```

使用scikit-learn包同样可以简单实现标准化。

```python
# Using Sklearn
from sklearn import preprocessing
df['col_standardized2'] = preprocessing.scale(df['col'])
```
由于函数计算的精确性不同，这两种计算的结果会有略微的不同但是，这个差异无关紧要。

**置信度和置信区间**：

在统计学中，"置信度"（Confidence Level）是指对于某个统计量（比如均值、比例、方差等）的置信程度或置信水平。它表示对该统计量的估计的可信程度或者说确定程度。

常见的置信度水平通常以百分比形式表示，例如95%置信度、99%置信度等。置信度表示的是在进行一次抽样实验时，针对某个统计量的估计结果在多次重复实验中覆盖真实参数值的频率。举个例子，如果你得到一个95%置信度的区间，那么在重复的抽样实验中，有95%的置信水平这个区间会包含真实的参数值。

置信度和置信区间（Confidence intervals）紧密相关。置信区间是一个范围，通常是由一个下限和一个上限组成，表示了对某个参数的估计的不确定性。置信度就是对置信区间的可信度的度量。

在统计推断中，通过置信度和置信区间，我们可以对样本数据所代表的总体参数进行估计，并且了解到这个估计的准确程度。

在Python中，可以使用不同的库来计算置信区间和置信度，其中最常用的是`scipy.stats`和`statsmodels`。这两个库都提供了计算置信区间的函数。

下面是一个使用`scipy.stats`计算置信区间的示例：

```python
import numpy as np
from scipy import stats

# 假设我们有一个样本数据
data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# 计算样本均值和标准差
sample_mean = np.mean(data)
sample_std = np.std(data, ddof=1)  # 使用ddof=1进行无偏估计

# 设置置信水平
confidence_level = 0.95

# 计算置信区间
margin_of_error = stats.norm.ppf((1 + confidence_level) / 2) * (sample_std / np.sqrt(len(data)))
confidence_interval = (sample_mean - margin_of_error, sample_mean + margin_of_error)

print("置信区间:", confidence_interval)
```

这个示例首先计算了样本的均值和标准差，然后使用了正态分布的逆累积分布函数(`stats.norm.ppf`)来计算给定置信水平下的临界值。最后，根据临界值和样本标准差，计算了置信区间。

另一种方法是使用`statsmodels`库，它提供了更多的统计测试和模型拟合功能，其中包括了置信区间的计算。下面是一个使用`statsmodels`计算置信区间的示例：

```python
import numpy as np
import statsmodels.stats.api as sms

# 假设我们有一个样本数据
data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# 设置置信水平
confidence_level = 0.95

# 计算置信区间
confidence_interval = sms.DescrStatsW(data).tconfint_mean(alpha=1-confidence_level)

print("置信区间:", confidence_interval)
```

这个示例使用了`DescrStatsW`类来计算置信区间，`tconfint_mean`方法用于计算均值的置信区间。

**置信区间的计算公式**：

置信区间的计算取决于所使用的统计分布和估计方法。常见的情况是基于 正态分布 或 t分布 来计算置信区间。

1. **基于正态分布的置信区间（大样本）**：
   当样本容量大（通常大于30）且总体标准差已知时，可以使用正态分布来计算置信区间。置信区间的计算公式为：
   
   置信区间 = 均值 +- Z * （sigma / sqrt（n））

   - Z是正态分布的临界值，通常取决于所选择的置信水平，例如95%置信度时，Z约为1.96。
   - sigma是总体标准差，
   - n是样本容量。

2. **基于t分布的置信区间（小样本）**：
   当样本容量较小或总体标准差未知时，可以使用t分布来计算置信区间。置信区间的计算公式为：

   置信区间 = 均值 +- t * （s / sqrt（n））

   其中，
   - t是t分布的临界值，取决于所选择的置信水平和自由度（样本容量减1），
   - s是样本标准差，
   - n是样本容量。

在实际应用中，选择合适的置信水平和所用的分布是非常重要的。通常，95%置信度是最常见的选择，但也可以根据实际需求选择其他置信水平。

**置信区间的临界值怎么计算**：

```python
from scipy.stats import norm

# 计算95%置信度下的临界值
confidence_level = 0.95
critical_value = norm.ppf((1 + confidence_level) / 2)

print("95%置信度下的临界值:", critical_value)
```

**t分布下的置信区间**：

t分布（Student's t-distribution）是统计学中常用的一种概率分布，用于处理样本容量较小（通常小于30）或总体标准差未知的情况。它在形状上类似于标准正态分布，但是具有更宽的尾部。

t分布的形状由一个参数，自由度（degrees of freedom，通常记作 df）所决定。自由度表示了用于计算t分布的样本数量减去估计的参数个数。当样本容量增加时，t分布逐渐接近标准正态分布。

计算样本均值的置信区间时，通常使用t分布来考虑样本容量较小以及总体标准差未知的情况。置信区间的计算公式如下：

均值 +- t * （s / sqrt（n））

其中：
- t是t分布的临界值，可以在t分布表中查找，也可以使用统计软件计算。
- s是样本标准差。
- n开根是样本容量。

在计算置信区间时，t的选择通常是根据置信水平和样本自由度来确定。置信水平表示我们对估计结果的信心水平，常见的置信水平包括90%、95%和99%。自由度通常是样本容量减1。

一般来说，我们会计算出一个置信区间，该区间包含了真实总体参数（比如总体均值）的估计范围。例如，95%的置信水平意味着在重复抽样中，我们有95%的把握认为真实总体参数位于计算得到的置信区间内。

**小样本自由度n-1，以及样本均值对样本方差的约束**：

样本方差对于样本均值具有一定的约束，这主要是因为样本方差是基于样本均值计算的，因此它受到样本均值的影响。

具体来说，样本方差是样本中每个数据点与样本均值之差的平方和的平均值。这意味着样本方差受到样本均值的位置和大小的影响。如果样本均值发生变化，样本方差通常也会随之发生变化。

在统计学中，我们常常使用样本方差来估计总体方差，进而进行推断统计。然而，要注意的是，样本方差的估计受到样本均值的影响，特别是在小样本情况下。这意味着如果样本均值发生变化，样本方差的估计也可能会受到影响。

此外，样本方差还受到样本大小的影响。在小样本情况下，样本方差通常会高估总体方差，因为它除以 ( n-1 ) 而不是 ( n )，其中 ( n ) 是样本容量。这是因为 ( n-1 ) 自由度的使用考虑了样本数据中的不确定性，使得样本方差的估计更加准确。

因此，虽然样本方差是样本数据中变异性的一个重要度量，但在解释和使用时，需要注意样本均值对于样本方差的影响，特别是在小样本情况下。

**样本误差SE在置信区间计算中的作用**：

在计算置信区间时，样本误差（standard error，通常用SE表示）是为了估计样本均值的抽样分布的不确定性。这个不确定性源于我们只能观察到样本数据，而无法得知总体的全部信息。样本误差提供了一种衡量样本均值估计的精度的指标。

置信区间是用来估计总体参数（比如总体均值）的范围，使得我们可以用一定的置信水平（confidence level）来断定总体参数落在这个区间内的概率。样本误差在计算置信区间时起到了至关重要的作用，它是置信区间宽度的一部分，可以理解为置信区间的“半径”。

样本误差SE的大小直接影响了置信区间的宽度，即我们对总体参数的估计精度。较大的样本误差意味着我们的样本均值估计不够精确，因此置信区间会相对较宽；而较小的样本误差则表示我们的样本均值估计相对准确，置信区间会相对较窄。

因此，在计算置信区间时，样本误差用于考虑样本数据的随机变动，帮助我们评估总体参数的估计范围，提供了对估计结果的精度和可信程度的一种度量。

**bootstraping：有放回的抽样统计方法**：

自助法（bootstraping）是一种统计学上的重抽样技术，用于估计样本统计量的抽样分布或总体参数的抽样分布，而无需对真实总体分布做出假设。

自助法的基本思想是通过有放回地从原始样本中抽取多个样本，形成一个新的自助样本集合，然后利用这些自助样本来估计统计量的分布。由于自助样本是通过有放回抽样得到的，因此有些观测值可能在某些自助样本中出现多次，而有些观测值可能根本不出现。通过对多个自助样本的统计量进行计算，可以获得统计量的分布信息，从而进行推断。

自助法的优势在于它对原始数据的分布情况并不做出假设，因此适用于各种类型的数据，无论是正态分布还是非正态分布。它特别适用于样本量较小或难以获得大量样本的情况下，用于估计统计量的分布或进行假设检验。

自助法在统计学中有着广泛的应用，包括但不限于：估计参数的置信区间，估计参数的标准误差，评估回归模型的稳健性，评估分类模型的性能。

**置信区间的应用**：

置信区间在统计学中有着广泛的应用，主要用于以下几个方面：

1. **参数估计**：置信区间可用于估计总体参数的范围。例如，可以使用置信区间来估计总体平均值、总体比例、总体方差等。这些估计可以帮助我们了解总体特征，并对未来的观察或实验做出预测。

2. **假设检验**：在假设检验中，我们通常会将一个参数的真实值与某个假设值进行比较，以确定我们的样本数据提供了足够的证据来支持或反驳这个假设。置信区间提供了一个范围，我们可以检查假设值是否在这个范围内。如果假设值在置信区间内，则我们无法拒绝该假设；否则，我们可能会拒绝假设。

3. **决策制定**：在进行决策时，我们经常需要考虑不确定性。置信区间可以帮助我们评估决策的风险和不确定性。例如，如果某个产品的销售预测落在一个宽置信区间内，说明我们对销售量的估计有较大的不确定性，这可能会影响我们的库存管理和生产计划。

4. **研究结果的解释**：在科学研究中，置信区间可以用来解释研究结果的稳健性和可靠性。如果一个研究发现某个治疗方法的效果是显著的，并且效果的置信区间不包含零，则可以认为这个效果是真实存在的，并且具有一定的可靠性。

总而言之，置信区间是统计学中一种重要的工具，它提供了对参数估计的不确定性的度量，帮助我们更好地理解数据和做出合理的决策。

---

参考理解：

- [怎样全面理解95%置信区间](https://zhuanlan.zhihu.com/p/140194206)

关键点：

- 因为样本均数是服从正态分布的，依据95%法则，我们知道有95%的样本均数是在总体均数加减大概1.96个标准差范围内的，把这句话用概率的数学表达式写出来，稍作整理就得到了总体均数的95%置信区间。

- [关于正态分布的理解](https://zhuanlan.zhihu.com/p/128809461)

关键点：

- 点概率，由于某个区间上的一个点的概率其实无限趋近于零，所以对于连续随机变量，一般不研究一个随机点点概率，而是研究它在某个区间上的取值的概率。
- 概率密度函数，曲线下的面积（积分）就是概率，曲线越高，代表这个区间的概率越密集。
- 均值和标准差决定了正态分布。均值决定分布的中心位置，标准差决定分布的扁平程度。
- 当一个数据样本符合正态分布的时候，第一时间想到要求均值和标准差。
- 将一个符合正态分布的数据样本进行标准化，或者z变换，就会变成标准正态分布。和变换之前的不同就是，均值会变成0，但是曲线的形状不会发生变化。
- 查表求概率：z分布表中间是概率小于某值的概率密度。这个某值的个位和小数点后第一位查看列，小数点后第二位查看行，交叉的值就是这个概率密度。比如要找坐落于a到b之间的概率密度，就用b的z分布值减去a的z分布值就得到结果了。具体看文章中的例子，写的很清楚。
- 68-95-99.7的概率分别坐落于一个，两个，三个标准差范围内。

- [关于抽样和抽样分布](https://zhuanlan.zhihu.com/p/136889276)

关键点：

- 多种多样的抽样方法：多阶段抽样，用各种方法进行抽样。简单随机抽样。系统抽样，设置抽样间隔进行抽样。分层抽样。整群抽样。为了反应整体的性质，最好进行各种抽样方法相结合。
- 抽样分布中，样本均数是一个随机变量。这是因为每抽取一组样本都会有一个样本均数。
- 中心极限定理：在任意总体中随机抽取一个样本量为n的样本，如果样本容量较大（通常大于30即可），那么通过这个样本计算的样本均数近似服从正态分布。

使用llm写出置信区间计算函数：

```python
def calculate_confidence_intervals(df, confidence=0.95):
    intervals = {}

    for column in df.select_dtypes(include=[np.number]).columns:
        data = df[column].dropna()
        mean = np.mean(data)
        std_dev = np.std(data, ddof=1)  # use ddof=1 to calculate sample std deviation
        if len(data) > 30:
            interval = st.norm.interval(confidence, loc=mean, scale=std_dev/np.sqrt(len(data)))
        else:
            interval = st.t.interval(confidence, len(data)-1, loc=mean, scale=std_dev/np.sqrt(len(data)))
        intervals[column] = interval

    return intervals

# usage
print(calculate_confidence_intervals(df, confidence = 0.9))
```


### 推论统计：假设检验

在维基百科有更为全面的解释，同时其中的统计学框架可以更好的看到整个知识体系的全貌，一定要[点一下](https://zh.wikipedia.org/zh-cn/%E5%81%87%E8%AA%AA%E6%AA%A2%E5%AE%9A)。

**假设检验（Hypothesis Testing）**是统计学中的一种方法，用于评估统计推断中的假设。根据样本数据，分析总体数据的一些事实。

在假设检验中，我们通常考虑两种假设：零假设（Null Hypothesis）和备择假设（Alternative Hypothesis）。零假设通常是我们想要进行测试的假设，而备择假设则是对零假设的补充或者反面假设。

在假设检验中，零假设（Null Hypothesis）通常是我们想要对其进行检验的假设。零假设通常表达的是一种默认或者基本的假设，它表明在没有观察到显著差异或效应时的情况下，我们所假设的状态。*我们只能证伪零假设，而不能证真零假设*。

在很多情况下，零假设通常表达的是“无效果”、“无差异”或“无关联“，即*变量之间本身没有关系*。但在某些情况下，零假设也可能表达其他假设，具体取决于研究的背景和问题的设置。

备择假设（Alternative Hypothesis）则是对零假设的补充或者反面假设，通常表达了我们希望得到支持的结论或者观察到的效应。备择假设通常是我们在零假设*被拒绝时所支持的假设。如果我们对零假设证伪失败，代表我们没有足够的证据证明备择假设，但是不代表备择假设一定是错的。*

因此，在假设检验中，我们的目标通常是通过样本数据来判断是否拒绝零假设，从而支持备择假设。如果样本数据提供足够的证据表明零假设不成立，我们就有理由拒绝零假设，接受备择假设。

假设检验的基本步骤如下：

1. 提出假设：明确零假设和备择假设。
2. 选择显著水平：通常用符号α表示，代表拒绝零假设的概率阈值。
3. 收集数据：收集与假设相关的数据样本。
4. 计算统计量：根据样本数据计算一个统计量，该统计量用于评估零假设的真实性。
5. 做出决策：通过比较计算得到的统计量与显著水平的临界值，决定是否拒绝零假设。

如果统计量落在拒绝域（即超过了显著水平的临界值），则我们拒绝零假设，否则我们接受零假设。拒绝零假设意味着我们认为备择假设更有可能成立。

假设检验在科学研究、质量控制、医学实验等领域都有广泛的应用，它帮助我们根据样本数据做出对总体的推断，并提供了一种量化的方法来评估假设的成立程度。

**P-value**：

P值（P-value）是统计学中用于评估假设检验结果的一个指标。它表示在原假设为真的情况下，观察到的统计量或更极端情况发生的概率。换句话说，P值告诉我们观察到的数据或更极端情况发生的可能性大小。

在假设检验中，我们通常会提出一个原假设（null hypothesis），然后收集样本数据来对该假设进行检验。P值是根据观察到的样本数据计算出来的，它表示了如果原假设为真，观察到的样本数据或更极端情况的出现概率。

- 通常，如果P值很小（小于事先设定的显著性水平，通常为0.05），则我们有足够的证据拒绝原假设，接受备择假设（alternative hypothesis）。换句话说，我们认为观察到的数据在原假设为真的情况下发生的概率很低，因此我们认为原假设可能不成立。

- 反之，如果P值较大（大于显著性水平），则我们没有足够的证据拒绝原假设。这并不意味着原假设一定是正确的，而只是表示我们没有足够的证据去否定它。在这种情况下，我们通常会说我们 "无法拒绝" 原假设。

- 低P值表示低偶然性。高P值表示不太可能是偶然的。

需要注意的是，P值并不直接告诉我们原假设是真还是假，它只是提供了关于观察数据与原假设之间的一种概率信息。因此，在解释P值时，我们应该谨慎地考虑它与显著性水平的关系，并结合实际背景和领域知识做出合适的判断。

用Python设置一个P值函数：alpha是显著性水平。

```python
# Create a function to read the p-value
def p_value_reader(p_value, alpha):
  if p_value < alpha:
    print("Reject the Null Hypothesis")
  else:
    print("Fail to reject the Null Hypothesis")
```

**显著性水平**：

显著性水平（Significance Level）是在假设检验中设定的一个阈值，用于判断样本数据是否提供了足够的证据来拒绝原假设。

在进行假设检验时，我们通常提出一个原假设（Null Hypothesis，H0）和一个备择假设（Alternative Hypothesis，H1）。显著性水平表示我们愿意接受原假设被错误拒绝的风险程度。

常见的显著性水平包括：
- 0.05（5%的显著性水平）
- 0.01（1%的显著性水平）

这意味着如果我们将显著性水平设定为0.05，那么在进行假设检验时，我们愿意接受在原假设为真的情况下，我们错误地拒绝原假设的风险不超过5%。换句话说，如果得到的p值小于0.05，我们将拒绝原假设。

显著性水平的选择通常取决于研究的领域、研究问题的重要性以及对研究结果的可信度要求。较低的显著性水平意味着更高的证据要求，因此需要更强的数据支持才能拒绝原假设。

**一类错误和二类错误**：

在统计学和假设检验中，一类错误和二类错误是两种可能发生的错误类型，通常与假设检验的结论相关联。

1. **一类错误（Type I Error）**：错判-false positive
   - 一类错误指的是在原假设为真的情况下，错误地拒绝了原假设的概率。换句话说，一类错误是错误地认为存在效应或差异，而实际上并不存在。
   - 通常用α表示一类错误的概率，称为显著性水平。显著性水平通常设定为0.05或0.01，表示在5%或1%的情况下会犯错。

2. **二类错误（Type II Error）**：漏判-false negative
   - 二类错误指的是在原假设为假的情况下，错误地接受了原假设的概率。换句话说，二类错误是未能检测到实际上存在的效应或差异。
   - 通常用β表示二类错误的概率。与一类错误不同，二类错误的概率通常不明确指定，而是取决于样本大小、效应大小和显著性水平等因素。

举例来说，假设一医学研究人员要测试一种新药是否能够治疗某种疾病。研究人员设置了以下两个假设：
- 原假设（H0）：新药对治疗疾病没有效果。
- 备择假设（H1）：新药对治疗疾病有效果。

在进行假设检验时，研究人员可能会做出以下两种错误之一：

- **一类错误**：研究人员错误地拒绝了原假设，即错误地得出新药对治疗疾病有效果的结论。这意味着他们可能会错误地推广新药的疗效，尽管实际上新药可能并不比安慰剂更有效。

- **二类错误**：研究人员未能拒绝原假设，即未能得出新药对治疗疾病有效果的结论。这意味着他们可能会忽略新药实际上具有的治疗效果，而继续使用不太有效的传统治疗方法。

因此，在假设检验中，研究人员需要在一类错误和二类错误之间进行权衡，并根据研究的具体情况和实际需求来选择适当的显著性水平和样本大小，以最小化两种类型错误的概率。

---
**关于Publish Bias**：

在统计学中，"publication bias"（出版偏倚）指的是在科学研究中发表结果的倾向性或偏向性，这可能导致研究结果的不准确或误导性。

**发表偏倚**可能出现在以下情况下：

1. **正向结果偏倚**：研究人员和期刊编辑倾向于发表结果显著或正面的研究，而忽略结果无关或消极的研究。这可能是因为正面结果更具吸引力，更有可能被引用和报道。

2. **选择性报道**：研究人员倾向于选择性地报告他们所发现的显著结果，而忽略那些未能得到显著性的结果。这可能导致读者获得不完整或误导性的信息。

3. **出版倾向性**：期刊编辑和出版者可能更愿意接受具有显著结果的研究，而不愿意接受结果无关或无显著性的研究。这可能导致未发表的研究结果被忽略或被排除在外。

4. **语言偏倚**：在一些国际期刊上，发表结果的语言可能会导致偏倚。例如，英语期刊上发表的研究结果可能更有可能受到全球关注，而其他语言期刊上发表的研究结果可能会被较少注意。

**出版偏倚**对科学研究和学术进展可能产生严重影响：

- 可能导致对真实情况的误解。
- 可能导致对研究结果的过度解读。
- 可能导致科学界的研究方向偏向性。

为了减少出版偏倚的影响，一些措施已经被采取：

- 注册前分析计划（Registered Pre-Analysis Plans）：在开始研究之前，研究人员可以注册研究设计和分析计划，以减少选择性报道的可能性。
- 发表前审查（Preprint Review）：研究人员可以选择在发表之前在预印本服务器上公开他们的研究结果，从而避免出版倾向性。
- 数据共享政策（Data Sharing Policies）：一些期刊和出版商实施了数据共享政策，鼓励研究人员在发表之后共享他们的数据，以验证和重现研究结果。

通过这些措施，可以帮助减少出版偏倚，提高科学研究的透明度和可靠性。

---

**在已知总体方差的情况下进行假设检验（z检验）的步骤**：

- 抽取n个样本，计算出均值。
- 根据方差，和计算的均值，计算出z分数。
- 根据z分数在z-table中知道Probability。
- 比较Probability和显著性水平（一般是0.05）的比较判断是否接受或者拒绝原假设。

Z检验（Z-test）是一种统计方法，用于检验*一个样本的均值是否与总体的均值有显著差异*。它适用于大样本（通常指样本量大于30）和已知总体标准差的情况。Z检验基于正态分布的性质，利用样本均值与总体均值之间的差异，结合样本的标准差和样本大小，计算出一个Z值，然后与标准正态分布的Z分布相比较，以确定差异是否显著。

Z检验通常用于以下场景：
1. 总体标准差已知，样本大小大于30的情况下，用于检验样本均值与总体均值的差异；
2. 比较两个样本的均值差异，这时候也可以用Z检验，前提是两个样本都满足正态分布且总体标准差已知。

Z检验的步骤包括：
1. 提出假设：原假设（H0）通常是样本均值等于总体均值，备择假设（H1）是样本均值不等于总体均值。
2. 选择显著性水平（α），通常设定为0.05或0.01。
3. 计算Z值：根据样本数据计算出Z值。
4. 判断拒绝域：根据显著性水平和自由度确定Z分布的临界值。
5. 做出决策：比较计算得到的Z值与临界值，如果Z值落在拒绝域内，则拒绝原假设，否则接受原假设。

Z检验的结果告诉我们样本的均值与总体均值之间是否有显著差异，以及这种差异的可能性大小。

Python代码执行z检验：

```python
# Info
mean_pop = 54
sd_pop = 2
confidence = 0.95
# 置信水平相对应的显著性水平
alpha = 1 - confidence
mean_sample = df_main['Cars Produced'].mean()
print(f"The sample mean is {mean_sample}")
sample_size = df_main['Cars Produced'].count()
print(f"The sample size is {sample_size}")

# Z Test formula (sample mean - pop mean) / (pop sd ( sqrt(sample size)))
z_score = (mean_sample - mean_pop) / (sd_pop / np.sqrt(sample_size))
print(f"The Z-score is {z_score}")

# Calculate the p_value from the z-score (two tails)
tails = 2
p_value = st.norm.sf(abs(z_score)) * tails
print(f"The p-value is {p_value}")

# Interpret the p_value
if p_value < alpha:
  print("Reject the Null Hypothesis")
else:
  print("Fail to reject the p_value")
```

或者将上面的代码整合为一个函数：

```python
# Build a function to compute the z-test
def ztest(mean_pop, mean_sample, sample_size, sd_pop, alpha, tails):
  # Z Test formula (sample mean - pop mean) / (pop sd ( sqrt(sample size)))
  z_score = (mean_sample - mean_pop) / (sd_pop / np.sqrt(sample_size))
  print(f"The Z-score is {z_score}")

  # Calculate the p_value from the z-score (two tails)
  p_value = st.norm.sf(abs(z_score)) * tails
  print(f"The p-value is {p_value}")
  p_value_reader(p_value, alpha)

# Apply the function
ztest(mean_pop, mean_sample, sample_size, sd_pop, alpha, tails)
```

---
**根据z分数计算p值的过程：**

计算 p 值的一种常见方法是使用标准正态分布的累积分布函数 (CDF)。在 Z 分数已知的情况下，我们可以使用标准正态分布的累积分布函数来计算 p 值。

通常，p 值是由标准正态分布的累积分布函数计算得出的，具体步骤如下：

1. 计算 Z 分数的绝对值：`abs(z_score)`
2. 使用累积分布函数 (`cdf`) 来计算累积概率。由于标准正态分布是对称的，所以可以使用 `1 - cdf(z_score)` 或 `cdf(-z_score)` 来计算右侧的概率，这两者是等价的。
3. 如果是双侧检验，需要将得到的概率乘以 2。

代码中使用了 `st.norm.sf(abs(z_score))`，其中 `st.norm.sf()` 是 SciPy 中的函数，用于计算标准正态分布的累积分布函数的补码 (1 - CDF)，而 `abs(z_score)` 则是确保计算绝对值的 Z 分数。

所以，p 值计算的代码段为：

```python
p_value = st.norm.sf(abs(z_score)) * tails
```

其中 `tails` 表示双侧检验中的尾数，因为双侧检验需要考虑两个尾部。

---
**在不知道总体方差的情况下进行z检验**：

和之前的情况稍有不同而已，这次使用样本方差，并且使用t分布。

```python
# Information
target_mean = 2.2
mean_sample = df_main['Defects Found'].mean()
print(f"The sample mean is {mean_sample}")
sample_size = df_main['Defects Found'].count()
print(f"The sample size is {sample_size}")
confidence = 0.95
alpha = 1 - confidence
sample_sd = df_main['Defects Found'].std()
print(f"The SD is {sample_sd}")

# Calculate the t-score
t_score = (mean_sample - target_mean) / (sample_sd / np.sqrt(sample_size))
print(f"The T-score is {t_score}")

#Calculate the p_value
tails = 2
# 自由度设置为-1
p_value = st.t.sf(abs(t_score), df = (sample_size - 1)) * tails
print(f"The p-value is {p_value}")

#Interpret the p_value
p_value_reader(p_value, alpha)
```

使用 SciPy 库中的 `ttest_1samp` 函数进行一个样本 t 检验：

```python
# How to do the 2-tailed test with unknown pop variance
t_score, p_value = st.ttest_1samp(a = df_main['Defects Found'],
                                  popmean = target_mean,
                                  alternative = 'two-sided')
print(f"T-score: {t_score}")
print(f"p-value: {p_value}")
p_value_reader(p_value, alpha)
```

`alternative='two-sided'` 参数指定了双侧检验，这意味着它将检查样本均值是否与总体均值不同，而不是只关注样本均值是否比总体均值大或小。

---

**配对T检验**

配对 t 检验（paired t-test）是一种用于*比较两组相关样本之间平均值差异是否显著*的统计检验方法。它通常用于分析同一组个体在两种不同条件下的观察结果，比如在不同时间点或者不同处理条件下的观察值。

在进行配对 t 检验时，首先要对每个个体或观察单位进行两次测量，然后比较这两组测量值的平均数。配对 t 检验的假设是这两组测量值的平均数没有显著差异，也就是说，它们来自同一总体。如果在统计上发现这两组测量值的平均数存在显著差异，那么我们就可以拒绝原假设，认为这两组测量值来自不同的总体，或者在两种条件下有显著不同的平均值。

配对 t 检验的优点在于可以减少个体间的变异性对结果的影响，因为它比较的是同一组个体在不同条件下的变化，而不是不同个体之间的差异。这使得配对 t 检验在实验设计中非常有用，特别是在涉及时间序列数据或者重复测量的情况下。比如pre-post用药前后对比数据等。

使用Python进行配对t检验：

```python
# Data
differences = df_paired['Month 2'] - df_paired['Month 1']
mean_difference = differences.mean()
sd_difference = differences.std()
sample_size = differences.count()
print(f"The mean difference is {mean_difference}")

# Info of the test
dof = sample_size - 1
tails = 2
confidence = 0.95
alpha = 1 - confidence

# Computing the t-score: (x1_avg - x2_avg) / (SD_diff / SQRT(sample size))
t_score = mean_difference / (sd_difference / np.sqrt(sample_size))
print(f"The T-score is {t_score}")

# Compute the p_value
p_value = st.t.sf(abs(t_score), df = dof) * tails
print(f"The p-value is {p_value}")
p_value_reader(p_value, alpha)
```

同样，更简便的，使用`scipy.stats`中的 ttest_rel 函数进行配对 t 检验。这个函数的第一个参数是第一个月份的数据，第二个参数是第二个月份的数据。alternative='two-sided' 参数表示进行双侧检验（即检验是否存在显著差异的可能性在两个方向上），这是默认设置。函数会返回计算得到的 t 分数和对应的 p 值。

```python
# Perform a paired t-test with 2 tails
t_score, p_value = st.ttest_rel(df_paired['Month 1'],
                                df_paired['Month 2'],
                                alternative='two-sided')

print(f"T-score: {t_score}")
print(f"p-value: {p_value}")
p_value_reader(p_value, alpha)
```
---
**两个样本T检验**

两个样本t检验（Independent Samples t-test）和韦尔奇（Welch）测试都是用于比较两个样本均值之间是否存在显著差异的统计方法。它们都适用于当两个样本之间是独立的情况，即一个样本的观测值与另一个样本的观测值没有关联。

1，两个样本t检验（Independent Samples t-test）：

这种方法假设两个样本的方差相等，并且两个样本都来自于服从正态分布的总体。如果这些假设成立，两个样本t检验可以通过比较两个样本的均值和它们的方差来判断它们之间是否存在显著差异。

**步骤：**
1. 提出假设：原假设（H0）通常是两个样本的均值相等，备择假设（H1）是两个样本的均值不相等。
2. 选择显著性水平（α）。
3. 计算t值：根据两个样本的均值、方差和样本大小，计算出一个t值。
4. 确定自由度：根据两个样本的大小计算自由度。
5. 判断拒绝域：根据显著性水平和自由度确定t分布的临界值。
6. 做出决策：比较计算得到的t值与临界值，如果t值落在拒绝域内，则拒绝原假设，否则接受原假设。

2，韦尔奇（Welch）测试：

韦尔奇测试是一种用于两个样本均值差异的统计检验方法，它不假设两个样本的方差相等，因此更加灵活。它的假设是两个样本都来自于正态分布的总体。

**步骤：**
1. 提出假设：与两个样本t检验相同。
2. 选择显著性水平（α）。
3. 计算t值：韦尔奇测试计算的t值与两个样本的均值、标准差和样本大小有关。
4. 确定自由度：韦尔奇测试使用自由度的近似公式。
5. 判断拒绝域：与两个样本t检验相同。
6. 做出决策：与两个样本t检验相同。

韦尔奇测试的优点是它对于样本大小不同或方差不相等的情况也能给出可靠的结果，但是计算过程相对复杂一些。

3，Levene测试：如何判断方差是否相等，就用到了这个测试，在这个测试后，在选择用两个样本t检验还是韦尔奇检验。

Levene测试是一种用于检验两个或多个总体方差是否相等的统计检验方法。它是用来验证方差齐性（homogeneity of variance）的，即各组的总体方差是否相同。Levene测试通常用于方差分析（ANOVA）等统计方法的前提检验，因为这些方法对方差齐性有一定的要求。

Levene测试的原假设是各组的总体方差相等（方差齐性），备择假设是至少有一组的总体方差不相等。Levene测试的统计量基于各组观测值与该组均值之间的差异来计算。如果Levene测试的结果显示p值较大，通常大于选择的显著性水平（如0.05），则我们接受原假设，即各组的总体方差相等；反之，如果p值较小，则我们拒绝原假设，认为各组的总体方差不等。

Levene测试是一种鲁棒性较好的方法，即在数据不满足正态性假设或样本量不同等情况下也能给出相对可靠的结果。这使得它在实际应用中得到了广泛的应用，特别是在ANOVA等分析中。

**使用Python进行两个样本t检验：**

这个函数通过levene测试先判断方差是否相等，然后使用相对应的t检验进行计算。

```python
# Build a function that performs 2 sample Test
# based on the outcome of Levene's test

def test_2sample(sample1, sample2, alpha, alternative='two-sided'):
    #levene's test
    stat, p_value = st.levene(sample1, sample2)
    #interpret the test
    if p_value < alpha:
        equal_var = False
        print("Reject the Null Hypothesis. Variances are unequal. Perform Welch's Test")
    else:
        equal_var = True
        print("Fail to reject the Null Hypothesis. Variances are equal. Perform 2-sample T-test")
    # 2 sample test
    t_statist, p_value = st.ttest_ind(
        sample1,
        sample2,
        equal_var = equal_var,
        alternative = alternative)
    print(f"The p-value is {p_value}")
    p_value_reader(p_value, alpha)

test_2sample(sample1, sample2, 0.05, 'two-sided')
```
---
**单尾测试**

到上面为止关注的都是双尾测试（Two-tailed test）：

在双尾测试中，假设检验关注的是统计指标在两个方向上的差异。原假设（H0）和备择假设（H1）涉及到统计分布的两个尾部。这意味着在统计分布的两个尾部都定义了拒绝原假设的临界值。双尾测试通常用于我们对效应的方向没有先验假设的情况下。

例如，假设我们要检验一种新的教学方法是否会改善学生的成绩。我们可能关心的是学生的成绩是否在任一方向上发生了变化，无论是提高还是降低。在这种情况下，我们会使用双尾测试。

单尾测试（One-tailed test）和双尾测试（Two-tailed test）都是统计学中常用的假设检验方法，它们用于检验某个统计指标是否在给定假设下具有显著性差异。

在单尾测试中，假设检验关注的是一个方向上的差异。也就是说，原假设（H0）和备择假设（H1）只针对分布中的一个尾部。这意味着在统计分布的一个尾部定义了拒绝原假设的临界值。单尾测试通常用于预先有方向性假设的情况，或者当我们只关心一个方向上的效应。

例如，假设我们要检验一种新药是否比已有的药物更有效，我们可能只关心新药的效果是否更好，而不关心是否更差。在这种情况下，我们会使用单尾测试。

单为测试的代码其实和上面的双尾是一样的，只是将tails变成1。使用上面构造的function进行。最后的参数1就是tails尾数。

```python
ztest(mean_pop, mean_sample, sample_size, sd_pop, alpha, 1)
```

在未知总体方差的情况下使用scipy的包进行计算：

```python
# How to do the 1-tailed test with unknown pop variance
t_score, p_value = st.ttest_1samp(a = df_main['Defects Found'],
                                  popmean = target_mean,
                                  alternative = 'greater')
print(f"T-score: {t_score}")
print(f"p-value: {p_value}")
p_value_reader(p_value, alpha)
```

配对T检验：

```python
# Perform a paired t-test with 1 tail
t_score, p_value = st.ttest_rel(df_paired['Month 1'],
                                df_paired['Month 2'],
                                alternative='greater')

print(f"T-score: {t_score}")
print(f"p-value: {p_value}")
p_value_reader(p_value, alpha)
```

使用提前准备好的函数进行两个样本T检验。

```python
test_2sample(sample2, sample1, 0.05, 'less')
```

**卡方检验**：

卡方检验（Chi-Square Test）是一种统计学中常用的假设检验方法，用于确定两个或多个分类变量之间是否存在关联性。它的名称来自于它所使用的统计量，即卡方统计量。

卡方检验通常应用于分析两个分类变量之间的关系，例如，性别与吸烟习惯之间是否存在关联。但它也可以扩展到更多的分类变量，进行更复杂的关联性分析。

卡方检验的基本思想是比较观察到的频数与期望的频数之间的差异，如果这种差异超出了随机误差的范围，则可以得出结论，两个变量之间存在显著的关联性。

卡方检验的步骤包括：
1. 制定原假设（H0）和备择假设（H1）。
2. 构建一个列联表，将观察到的频数与期望的频数进行比较。
3. 计算卡方统计量，衡量观察到的频数与期望的频数之间的差异。
4. 根据卡方统计量和自由度，计算出一个P值。
5. 判断P值是否小于预先设定的显著性水平（通常为0.05或0.01），以决定是否拒绝原假设。

如果P值小于显著性水平，则拒绝原假设，认为两个变量之间存在显著的关联性；如果P值大于显著性水平，则不能拒绝原假设，认为两个变量之间不存在显著的关联性。

卡方检验在各种领域中广泛应用，包括医学、社会科学、生物统计学等。

```python
# Actual frequency
contingency_table = pd.crosstab(index = df_chisquare['Factory'],
                                columns = df_chisquare['Category'],
                                values= df_chisquare['Count'],
                                aggfunc = np.sum)

# Chi Square Test
stat, p_value, dof, expected_freq = st.chi2_contingency(observed = contingency_table)

# Print and interpret the p_value
print(f"The p-value is {p_value}")
p_value_reader(p_value, 0.05)
```

### 回归分析: 线性回归

从统计学的角度（因为还有ml角度）来解释线性回归，它是一种用于建立变量之间关系的统计模型，通常用于以下几个方面：

1. **描述性分析**：线性回归可用于描述自变量与因变量之间的线性关系。通过线性回归模型，可以评估自变量对因变量的影响程度（impact），并确定它们之间的相关性。

2. **预测**：线性回归可以用于预测因变量的值。通过已知的自变量值，可以利用线性回归模型来估计因变量的值。这对于预测未来事件或趋势具有重要意义。这就是机器学习视角了。

3. **假设检验**：线性回归模型的拟合可以用于检验各个自变量对因变量的影响是否显著。通过对回归系数进行假设检验，可以确定自变量是否对因变量有显著影响。

4. **模型评估**：统计学角度还包括对线性回归模型的评估。这涉及检查模型的拟合优度、残差的分布是否符合假设、模型是否满足统计假设等方面。

总的来说，线性回归通过建立自变量和因变量之间的线性关系，提供了一种解释和预测数据的方法。在统计学中，线性回归是一种常用的工具，可以帮助研究人员从数据中提取信息，做出推断，并进行科学分析。

相关的Python库：

```python
# Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
```

一般使用散点图表达回归模型：

```python
# Scatter plot
plt.scatter(X, y)
#Customizing the graph
plt.xlabel("Carat")
plt.ylabel("Price")
plt.title("The relationship of Carats and Price")
plt.show()
```

向数据中添加一个常数列（新增一个列，比如这个列全是1），通常被称为截距项或偏置项。这是因为在线性回归模型中，除了自变量之外，通常还会有一个常数项，用于拟合数据中可能存在的偏移量或基准值。

```python
# Adding a constant
X = sm.add_constant(X)
X.head(2)
```

执行线性回归模型分析，`sm.OLS(endog = y, exog = X)`：这一行创建了一个最小二乘线性回归模型对象。`endog`参数是因变量（即被解释变量）`y`，`exog`参数是自变量（即解释变量）`X`。OLS代表“最小二乘”，是一种用于拟合线性回归模型的常见方法。

通过打印模型摘要，会看到关于回归系数、标准误、t统计量、p值等统计指标的信息，这些指标有助于评估模型的拟合程度以及自变量对因变量的影响是否显著。通常，摘要还包括了拟合优度指标（如R-squared）、残差分析和其他诊断信息，帮助你理解模型的适用性和准确性。

```python
# Equation : y = a + b*X + e
model1 = sm.OLS(endog = y, exog = X).fit()
print(model1.summary())
```

可视化模型。这段代码主要用于绘制数据点和线性回归拟合曲线的图形。

`b, a = np.polyfit(X, y, 1)`：这一行代码利用 NumPy 库的 `polyfit` 函数拟合了一条直线（即一次多项式）到数据上。它返回了拟合直线的斜率 `b` 和截距 `a`。这里的 `1` 表示拟合的多项式阶数为 1，即线性拟合。

`plt.plot(X, b * X + a)`：这一行代码绘制了通过线性拟合得到的直线。其中 `b * X + a` 表示利用斜率 `b` 和截距 `a` 构建的线性方程，即回归方程。通过将自变量 `X` 代入回归方程，得到了相应的预测值，从而得到了拟合曲线。

这段代码的作用是在同一张图上绘制数据点的散点图以及利用线性回归模型拟合的直线。这样的可视化可以帮助我们直观地理解数据的分布情况，并且可以评估线性回归模型对数据的拟合效果。

```python
# Plotting the regression curve
X = df.carat

# Plotting the Curve and dots
plt.plot(X, y, 'o')
b, a = np.polyfit(X, y, 1)
plt.plot(X, b * X + a)

#Customizing the graph
plt.xlabel("Carat")
plt.ylabel("Price")
plt.title("The relationship of Carats and Price")

plt.show()
```
注释：以上的数据列表使用的是钻石克拉对钻石价格的回归分析。

**二元自变量的二元回归**：

如果自变量是二元的（即只有两种可能的取值，通常为0和1），这种情况下的回归分析称为二元回归或者双重（双值）回归。这种情况下的回归模型通常被称为二元逻辑回归（Binary Logistic Regression）。

二元逻辑回归是一种用于解决分类问题的统计模型，其中因变量是二元的（通常是二分类问题，如“是”或“否”、“成功”或“失败”等），而自变量可以是连续的、分类的或二元的。这种回归分析的目的是确定自变量与因变量之间的关系，并用于预测因变量的类别。

以下代码示例：(其实这个可视化的图挺怪的)

```python
# Create a dummy variable with carat
X_binary = np.where(df.carat > 0.6, 1, 0)
X_binary # output will be 0 or 1

# Add a constant
X_binary_const = sm.add_constant(X_binary)

# Build the second regression model
model2 = sm.OLS(y, X_binary_const).fit()
print(model2.summary())

# Plotting the Curve and dots
plt.plot(X_binary, y, 'o')
b, a = np.polyfit(X_binary, y, 1)
plt.plot(X_binary, b * X_binary + a)

#Customizing the graph
plt.xlabel("Carat - dummy variable")
plt.ylabel("Price")
plt.title("The relationship of Carats and Price")

plt.show()
```
