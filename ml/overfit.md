## 防止过拟合

---

### 什么是过拟合

就是在训练后发现训练集精度远远超过了测试集的精度。就好像学习，你只会做课本上的题，考试的时候题目改了一下你就不会了，相当于学习的时候课本读到思维僵化了好吧。

为什么会产生或拟合：

1-数据集太小：数据不足，学习到的东西太少了，一旦给一个大测试集，模型不会了。

2-模型过于复杂：想象一个学习的人他想的太多了，书呆子，不知道变通，人太过聪明反而不好要学习泛华的能力，有时候想的简单一点也许更好。具体来说深度网络的层数太多了，神经元太多了等。

3-特征选择不当：学习材料错了的感觉，肯定在考试test中得不到好结果。

4-训练时间过长：不要学习太久，一样的道理，训练太久了就会陷入死脑筋。

### 防止过拟合的方法

1， 正则化技术

包括L1和L2正则化技术。区分方法就是L1是权重求**和**，L2是权重求**平方和**。

L1，因为有负数的存在，求和后有些特征权重会归零，并且计算机内部计算效率不高。相比较来说L2则不会出现负数的情况，所有的特征都会被考虑，只是大小不同，不会丢失信息。

但是有时候也需要L1，因为它是一种很好的降维技术。降维后就可以弥补计算效率不高的问题，可能在一百个特征中，只有十个有用，这时候使用L1就能达到这种效果。当然，如果觉得所有的特征都重要就用L2。

2， 交叉验证


3， 增加数据量


4， 降低模型复杂度：比如神经网络，就减少神经元数量，减少层数。在CNN中经常会用到**DropOut**技术，就是在训练中，随机丢弃一些神经元，以防止在某些神经元上发生过拟合现象。


5， 提前停止**EarlyStopping**：针对训练时间过长的问题，发现结果的精度在不停震荡，并且已经无法再提升精度了，就利用内部的算法，进行停止操作。比如设置在五个Epoch内，精度不再提升，则触发callback停止训练。
