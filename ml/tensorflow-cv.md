## Tensorflow 计算机视觉

---

### 计算机是怎么看图分类的

其实感觉和人类也没有很大不同，虽然人类在识别图像的时候，更多的是靠注意力。
此处是不是应该艾特注意力模型。NLP 以后再说，也许就是说，计算机读书需要更多的注意力，看图说话就很随意哈哈。

同样是靠着 RGB 三原色，在 CNN 中的池化处理中，计算机也是提取所谓的最大特征。
人类也是一样的。你如何通过双眼识别物体，也是通过特征辨别。
想象光线很暗的时候，你看不清前面的东西，但是也可以通过一些固有的物体特征去识别物体。
看不清全貌没关系，只要一些特征在，你就能猜。
其实计算机也是猜测，毕竟它的输出结果是一对的 prob，也就是可能性。
我们在猜的时候，如果将话语加上“哦，这个东西有百分之 89.56 的可能性是一只企鹅”，那我感觉我和计算机也差不多了就。

### 过拟合

无处不在的过拟合。如果训练精度在训练集表现良好，但是在测试集很辣鸡，那就是可能过拟合了。

如何在图像识别上防止过拟合：

方法 1，增加最大池化层。最大池化可以这么解释，比如给你一个图像矩阵，里面都是像素点的大小数字，那么每次只取一个最大值。这样训练的图像大小就会越来越小。这个最大的值指的是图像中最突出的特征。

方法 2，数据增强。Tensorflow 有自己的进行数据增强的工具。所谓的数据增强，简单的说就是让图像以各种方式变得扭曲，从而提高模型的翻化能力，也就是鲁棒性（我还专门查找了这个词什么意思，嗯来自英语的音译）。因为在现实中，我们无法总是看到完美的图像，比如你识别一个人类也是，他朝你走来的样子不可能每一帧都是一个完美图像吧，旋转跳跃我闭着眼。所以为了提高计算机的识别能力，同样的我们将数据集中的图片变得扭曲。

### 继续提升模型该做些什么

1. 增加模型的层数：脑力提升

   **有时候需要简化模型，想得简单点也许能得到更好的答案**

   过于复杂的模型，会对训练数据学习的太过了，降低了泛化能力，所以需要降低复杂度，学习最关键的特征，然后停下向细节方面的学习。

2. 增加过滤器的数量：多看几遍图片
3. 训练更长的时间 epochs：看的时间长一点
4. 找到更好的学习率：找到最好的学习速度
5. 获取更多数据：多看点图
6. 使用迁移学习：用别人的方法和模型看图

### 控制变量法

在寻找因果的方法的时候，第一次出现在自己的 toolkits 中的关键方法。在寻找模型中也很有用。

跟踪自己的模型，不断试验，每次都改变其中一个变量，记录效果，以此找到最关键的部分。

比如控制模型保持不变使用 clone model 方法克隆之前的模型（学习过的模型参数不会被克隆），然后使用增强数据进行训练，看数据的影响有多大。
