## 评估方法：混淆矩阵（Confusion Matrix）

---
### 概念解释

混淆矩阵（Confusion Matrix）是用于评估分类模型性能的表格，总之看起来他是一个非常有用的评估方法。我感觉到他最有用的时候是可视化热图的时候，可以明显看出分类器在哪些类别分类上出了问题。

混淆矩阵的四个基本术语如下：

1. **真正例，真阳性（True Positive，TP）：** 模型正确地预测为正类别的样本数。（正类我理解为目标样本）
2. **真负例，真阴性（True Negative，TN）：** 模型正确地预测为负类别的样本数。
3. **假正例，假阳性（False Positive，FP）：** 模型错误地将负类别样本预测为正类别的样本数（误报）。
4. **假负例，假阴性（False Negative，FN）：** 模型错误地将正类别样本预测为负类别的样本数（漏报）。

混淆矩阵的一般形式如下：

```
|              | 实际类别  | 实际类别  |
|--------------|-----------|-----------|
| **预测类别** | 正类别 (P) | 负类别 (N) |
| 正类别 (P)   |   TP      |   FP      |  **精确率**
| 负类别 (N)   |   FN      |   TN      |
|             | **召回率** |**假正例率**|
```

混淆矩阵的主要应用包括：

1. **准确率（Accuracy）：** 衡量模型整体的分类性能，计算方法为 (TP + TN) / (TP + TN + FP + FN)。

也就是所有样本中预测正确的样本比例。
  
2. **精确率（Precision）：** 衡量模型在**预测为正类别**的样本中有多少是真正例，计算方法为 TP / (TP + FP)。

主要是分母经常记不清，这里是精确率，所以关注的是**预测为正类**标签的样本中正确的比例，所以分母的两个都是P。关注错报！
  
3. **召回率（Recall）：** 衡量模型在实际为正类别的样本中有多少被正确地预测为正类别，计算方法为 TP / (TP + FN)。

分母这里是，**实际为正类**的样本中，正确正类的比例。关注实际。所以关注的漏报！

别名：真正例率，灵敏度。TPR（与之相反有FPR假正例率）
  
4. **F1分数：** 综合考虑精确率和召回率，计算方法为 2 * Precision * Recall / (Precision + Recall)。

是精确率和召回率的综合评估。

### ROC，AUC，PR

关注上面召回率中提到的TPR真阳性率和，FPR假阳性率指标，这两个指标，可以得到ROC曲线。X轴是假阳性率，Y轴是真阳性率。

把TPR放在X轴上，把FPR放在Y轴上，就可以得到一条ROC曲线。这条曲线阈值0到1，当全都预测正确的时候，也就是FPR是0TPR是1的时候，曲线是Y轴和Y=1平行于X轴的，这两条线的形状。这个时候曲线下面的面积AUC就是1，当相反，也就是全预测错了的时候，AUC是0，曲线是贴着下面X轴和X=1的垂直线的形状。

为什么ROC是一个很好的指标，因为有时候我们不一定设置阈值为0.5，而这条曲线可以评估**各种阈值情况下的模型优劣**。

PR曲线分别在 y 轴和 x 轴上取 Precision 和 Recall，分别针对 0 和 1 之间的不同阈值。同样它也是计算曲线下的面积。该值越大，模型的性能越好。这很好理解，面积越大意味着精确率和召回率都很高。

### Python代码

混淆矩阵的计算。

```python
def confusion_matrix_calculation(y_true, y_pred):
    """
    This function calculates the following values (TP, FP, TN, FN).
    It takes in y_true (which are actual class labels) and y_pred(which 
    are predicted class labels).
    """
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_pred)): 
        if y_true[i]==y_pred[i]==1:          
           TP += 1
        if y_pred[i]==1 and y_true[i]!=y_pred[i]:
           FP += 1
        if y_true[i]==y_pred[i]==0:
           TN += 1
        if y_pred[i]==0 and y_true[i]!=y_pred[i]:
           FN += 1

    return(TP, FP, TN, FN)

y_true = [1, 0, 1, 0, 0, 1]
y_pred = [0, 0, 1, 1, 0, 1]
TP, FP, TN, FN = confusion_matrix_calculation(y_true, y_pred)
print("The True Class Labels are {}.".format(y_true))
print("The Predicted Class Labels are {}.".format(y_pred))
print("The number of True Positives are {}.".format(TP))
print("The number of False Positives are {}.".format(FP))
print("The number of False Negatives are {}.".format(FN))
print("The number of True Negatives are {}.".format(TN))
```

准确率分数：

```python
from sklearn.metrics import accuracy_score
y_pred = [0, 2, 1, 3]
y_true = [0, 1, 2, 3]
print(accuracy_score(y_true, y_pred))
```

分类结果报告：

```python
from sklearn.metrics import classification_report
y_true = [0, 1, 2, 2, 0]
y_pred = [0, 0, 2, 1, 0]
target_names = ['class 0', 'class 1', 'class 2']
print(classification_report(y_true, y_pred, target_names=target_names))

# output
#              precision    recall  f1-score   support

#     class 0       0.67      1.00      0.80         2
#     class 1       0.00      0.00      0.00         1
#     class 2       1.00      0.50      0.67         2

# avg / total       0.67      0.60      0.59         5
```
