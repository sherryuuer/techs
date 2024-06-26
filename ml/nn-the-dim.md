## 神经网络：从维度的角度重新解构

---
### 一，高维线性分类器

很多时候我们去学习神经网络的时候，一般会从感知机开始。我不知道你们，但我是这样，一般一开始会有一个开篇告诉你，灵感来自于模仿人的神经元，然后有输入输出层，更多的隐藏层就会成为一个深度的神经网络。

但是最近的学习更多的让我偏向于从**维度，矩阵**的角度去看。从线性分类器的观点去理解一个很大的神经网络，也是非常有意思的视角，而且可以在脑海中展开一个不一样的高维世界，但是实际上是我也不知道高维世界是什么样子，但是姑且我认为很多人脑海中有自己对高维世界的定义吧。我的脑海中高维度的印象来自三体中的描述，蓝色空间宇宙飞船中，四维空间的人，可以瞬间在三维空间穿梭一样，人也可以将一张纸对折，相当于将二维世界进行折叠的高维操作。总之维度打击是我心中最高级的打击。不管是科幻，还是人的认知。

那么在线性分类器的世界，也同样是从低维度到高维度的扩展而已。而神经网络就是一个最显然的高维线性分类器的升级版本。

一条线上的居民，只需要用一个点就能进行分类，但是这条线上的点可能不是均匀地在线的两边分别站好，让他们只用一个点就能分开数据。于是一维世界的科学家们，发明了二维的分类器，但是由于他们自己也没见过二维长什么样子，所以只能假设有更高维度的分类器来帮他们分类。他们的那些数据点，在他们的一维世界里杂乱无章，就像是二维平面里的点在一条线上的投射（这种投射，正是我们的PCA降维法）！没错，正因为如此，所以他们就可以用二维世界中的直线来分开他们的这些点了，尽管他们不知道二维世界长什么样子。

二维平面的居民，也需要做分类任务，他们的点在一个平面中散乱不均，如何分类，你想到了吗，对，支持向量机！这是一个绝美对算法。二维平面的居民也无法分开平面上的点，于是他们想像更高的三维空间，尽管他们不知道三维空间长什么样子，但是他们使用比自己维度更高一个维度的视角，使用一个三维世界中的平面将自己世界中的杂乱的点分开，这时候你脑海中那个支持向量机的图就可以出场了，没错那就是高维空间投射下的线性分类器。

三维空间中的我们，更需要做分类任务，这时候我们的点已经散布在广阔的三维空间中了，所以我们需要更高维度的分类器，这个分类器的样子就是矩阵，也就是神经网络中那张大网，那个用大写的W表示的weight网络，神经网络只是一个更加深不可测维度上的分类器而已。在比我们更高次元的生物的脑海中，可能就像我们看一条直线那么简单。

人类最重要的能力就是想象力，创造力。打开思维，提升维度，是最终极的解决问题的能力。

### 二，激活函数激活了什么

刚刚在说到维度居民使用更高一个维度的分类器来分类最的世界的数据的时候可能会出现疑惑，一维数据上的点有时候也不能被一条直线分开吧，二维数据上的点有时候也不能被一个平面分开吧！是的！但是如果他们是弯曲的呢～如何弯曲，激活函数，激活函数激活的就是分类器的S属性。

一条直线怎么能作为直线维度上居民的更高维度的分类器呢，只有让直线弯曲起来，跃动起来，才能展现出他的高维属性。一个平面分类器怎么能作为平面居民的更高维度的分类器呢，只有让平面跃动起来，卷曲起来，才能展现出一个平面的三维属性。因为一个平面只有凹凸不平，我么才能看出它是一个三维物体。激活函数的重要性，就是激活分类器的更高维度的属性！

### 三，神经网络所做的就是降维打击的事情

**感知机和多层感知机的实质**

一个感知机也就是一个神经元，也就是一个分类器，经过激活函数就是一个稍微高维度的线性分类器。多层感知机呢？我们知道很深的神经网络，只有到了最后才开始做真正的分类任务，那么前面的那些层所做的事情是什么：特征提取，这个方面在计算机视觉CNN算法中有更具象化的体现。卷积核直接就是学术上所称的特征提取器。何为特征，不管是卷积还是池化都是将特征合并，降解维度。

**输入值和输出值**

输入值我们成为特征集，输出值则是分类标签。他们的实质又是什么，在我看来，输出的标签只是更高维度的一些向量的具象化表达，神经网络这个大型高维分类器，通过不断的**特征提取**，将重要的特征提取成更加具象化的新的矩阵，感受到了吗，这是一个不断的降维打击。最终将特征大型矩阵，打击成了一个零维度的特征标签。整个神经网络为什么厉害，因为它是一个N向箔。

使用多层感知机，对输入的内容进行维度降级，就是整个网络所在做的事情，最后一层，将所有的特征打击为一维然后得出最后的标签，那个点就是最终的目的。是一场宏大的宇宙战争呢。

### 四，损失函数到底损失了什么

先抛开回归问题，由于在无限细小的维度上，回归问题其实是一种分类问题，因为在无限小的粒度中，可以将回归问题的预测结果进行归类，变成一个很大的多分类问题，姑且放置不谈。

脱离官方的表达，损失函数说白了就是预测值和真实值之间相差有多大。

那么什么是预测值，什么是真实值？他们实际上就是向量表达，这时候自然语言处理中的那个wordbedding的三维向量群图就可以在脑海中出场了，所有的元素，物体，人格，在更加高维的视角上，都可以用向量表达的话，那么他们之间就有了夹角和距离，我对余弦相似度一见钟情，那种在空间中的差距表达能力，很让人着迷，每个人都有自己在高维空间中的投射，每个向量都有自己的方向，方向是如此重要，方向相同，余弦相似度趋近于1，反之趋近-1。通过这种差距比较两个概念之间的差距。

损失函数虽然有很多种计算方法，针对不同的问题也是选用不同的计算方法，最让我着迷的是**交叉熵**。它是信息论中的概念，用于比较两个概率分布之间的差异性。一开始它假定的就是概率分布，概率分布必然带有不确定性，同时在熵的概念下，信息参杂噪音，这才是这个世界的真实样貌。更高的维度，意味着更多的信息，意味着更多的噪音，所以就需要更更高维度的分类器来对抗这种噪音，降低这种熵损失。

所以损失函数的意义，就是衡量这些带着不确定性，带着噪音的信息之间的，在高维空间中的差异的指标。

### 五，优化器和反向传播

下坡。对，这是对一个学习者进行解释的时候，最好的解释方案。使用具象化的比喻表达总是能让人一下就理解其中的精妙之处，但是也失去了很多更高的理解角度。首先这里谈到了坡度，是为了说一下反响传播。在普通的机器学习算法中，比如简单的线性分类器，只需要画出损失函数曲线，我们就懂了需要求导，让梯度不断下降来进行优化，在反向传播中我们有时候则会晕头转向，但其实，它只是利用链式法则，将求导过程变成了求偏导数的过程，实质上是一样的，由于反向传播包含的推导较多，所以被单独拿出来讲解的地方很多，但是实际上，它是神经网络进行优化的一个部分。那么讲回优化器。

梯度下降的Python表达：

```python
for t in range(steps):
    dw = gradient(loss, data, w)
    w = w - learning_rate * dw
```

优化器（Optimizer）是一种用于调整神经网络模型参数的算法。在神经网络训练中，目标是最小化一个损失函数，而优化器的作用就是根据损失函数的梯度信息来更新模型的参数，使得损失函数达到最优或接近最优值。

优化器通常与学习率（Learning Rate）搭配使用。学习率决定了每次参数更新的步长大小，过大的学习率可能导致参数更新过大，错过最优值；而过小的学习率可能导致收敛速度过慢。因此，选择合适的学习率对于训练的成功至关重要。学习率可以是固定的，也可以随着训练过程逐渐减小。

梯度下降算法是一种常用的优化算法之一，它的基本思想是沿着损失函数梯度的反方向更新参数，从而逐渐降低损失函数值。梯度下降算法有多种变体，包括批量梯度下降（Batch Gradient Descent）、随机梯度下降（Stochastic Gradient Descent）和小批量梯度下降（Mini-batch Gradient Descent）等。

下面是优化器、学习率和梯度下降的结合计算过程：

- 首先，通过前向传播计算神经网络的输出，并将输出与真实标签进行比较，从而得到损失函数的值。
- 计算梯度：使用反向传播算法计算损失函数对于网络中各个参数的梯度。
- 选择优化器：根据选择的优化器的不同，可以使用不同的算法来更新参数。优化器可能会考虑之前的梯度信息、动量等因素。
- 调整学习率：如果使用了动态学习率策略，可以根据当前的训练轮数或其他规则来调整学习率。
- 更新参数：根据优化器给出的更新规则和学习率，更新神经网络模型的参数。
- 重复迭代：重复以上步骤，直到达到停止训练的条件，如达到最大迭代次数、损失函数收敛等。

通过这个过程，神经网络模型的参数会不断地调整，使得损失函数逐渐降低，从而实现模型的优化和训练。

通过再次整理上述的内容和过程，在维度的视角下重新理解这个过程。如果说真实标签是一个向量坐标，那么每一层神经网络就是一个高维度的**航标**，一开始的航标是随机的，歪七八扭，优化器的作用就是对这些向量进行不断的调整。让所有的航标，在时空跳跃点转移的过程中，不断的指向正确的方向。**The space jump**，Can you jump to the right point of the universe?学习率的动态变化是必要的，这也就是为什么**Adam**先生如此优秀，因为他是动态调整的，当你的指针已经非常接近你要去的那颗星星的时候，你必然会微调你的指针，那些被调节的夹角，就是学习率。

### 六，Pytorch对上述的整体构架

我很喜欢我看的一句话：

> Pytorch is a library that helps you perform mathematical operations between matrices in their basic form because, as you will realize, deep learning is just simple linear algebra.

是的其实整个运算过程，就是进行简单的线性代数的过程，我觉得线性代数四个字低估了这个学科的力量，《鸢尾花书》的姜老师有一本书叫做《矩阵力量》，线性代数正是这样一种强大的存在，他应该叫做高维数学才对。

Pytorch的基础数据结构也不是向量，而是张量tensor。向量只是一种特殊的张量而已。张量才是高维空间中的指向标。

### 七，一些编码

一个神经元or一个线性分类器：

```python
import torch

def neuron(input):
    weights = torch.Tensor([0.5, 0.5, 0.5])
    bias = torch.Tensor([0.5,])
    return torch.add(torch.matmul(input, weights), bias)
```

函数式高维构架：

```python
def fnn(input):
    x = nn.Linear(10, 128)(input)
    x = nn.ReLU()(x)
    x = nn.Linear(128, 64)(x)
    x = nn.ReLU()(x)
    x = nn.Linear(64, 2)(x)
    return x
```

序列式：

```python
model = nn.Sequential(nn.Linear(10, 128),
                        nn.ReLU(),
                        nn.Linear(128, 64),
                        nn.ReLU(),
                        nn.Linear(64, 2)
                    )
return model(input)
```
当然两者经常结合起来使用，尤其是卷积网络这种复杂的构架。

### 八，一些启发

我们看事物的维度越多就越容易发现问题和答案。

按照三体世界的理论，宇宙最初是有十个维度的，只是因为文明之间不断进行降维打击，导致宇宙的维度不断降低，以后甚至会坍塌为一个点，回归宇宙最初的状态，而文明中智慧体的智能则是一个不断升级维度的过程，只有更高维度的智能才能得到终极的提升。这也是所有大师所说的要提升维度的意思。真正的成长是维度级别的成长。
