## 使用Scikit-learn进行建模学习

---
内含Scikit-learn的主要数据模型包括分类，回归，超参调优，评估方法。这真的是一个很优雅的框架。在使用Tensorflow和Pytorch的人都知道，sklearn的东西是可以直接拿来辅助使用的。

他们更多是一种统计模型，因为日常我们不可能总是训练神经网络。更多的是在更小的数据集上进行统计分析，所以这个框架是日常建模的最佳选择。这么说起来其实大模型和现在的神经网络模型我们更适合的是直接

### 线性模型

```python
from sklearn import linear_model
def linear_reg(data, labels):
    reg = linear_model.LinearRegression()
    reg.fit(data, labels)
    return reg
```

优化算法是最小二乘法。目的是残差平方和的最小化。
