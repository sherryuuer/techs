## Pytorch初见就有点爱的框架

---

### 第一次接触Pytorch

给人的感觉一句话形容，就是像python那样让人亲切以及灵活，像一个朋友，一上来就爱了。

一开始我学的是Tensorflow，因为是Google的库所以一开始信赖感很强，很多模块都是内建的，只要拿来用就好，但是经过Pytorch两节课的洗礼，深深的感受到了python到力量哈哈。把神经网络层层拨开，写成你看得懂的类和函数，然后自己去灵活编辑的那种清晰的感觉。挺美好的。

Pytorch在前沿深度学习论文中的使用率已经超越了Tensorflow的使用率，成为最流行的深度学习框架，内置的反向传播机制，帮助使用者更方便得学习和训练。

### 简单的线性回归模型模板

```python
import torch
import torch.nn as nn

class LinearRegressionModel(nn.Model):
    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.linear(x)
        return out

input_dim = 1
ouput_dim = 1

model = LinearRegressionModel(input_dim, output_dim)

print(model)
```

`forward`函数是一个前向传播方法。一个简单的线性回归模型框架，通过对其中很多部分的扩展，可以变成很多很大框架的模型。

指定参数和损失函数，也是使用torch内部的类：

```python
epochs = 1000
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
# loss function
loss_fn = nn.MSELoss()
```

在之后的模型训练中会使用到以上参数，`epochs`是训练的批次数量，`learning_rate`学习率，是每次训练的过程中进行梯度下降的速度设置，损失函数是进行反向传播的时候，进行梯度优化的时候的计算标准。

具体的训练也是一个标准化模板：

```python
for epoch in range(epochs):
    epoch += 1
    # turn data into tensor
    inputs = torch.from_numpy(x_train)
    labels = torch.from_numpy(y_train)

    # set grad to zero
    optimizer.zero_grad()

    # forward calculate
    outputs = model(inputs)

    # calculate the loss
    loss = loss_fn(outputs, labels)

    # backward
    loss.backward()

    # update the weights
    optimizer.step()

    if epoch % 50 == 0:
        print(f"Epoch: {epoch}, Loss: {loss.item()}")
```

这是一个自动梯度下降的机制，方便进行简单的训练代码编写。流程来说就是：

梯度归零 -> 前向传播 -> 计算损失 -> 损失反向传播 -> 更新权重

整个流程非常符合直觉。

在工业界tensorflow是一个很好的选择，在学术界，Python使用者选择PyTorch，相当有利于学习和理解。

### 其他模块

强大的torchvision模块内置数据集以及经典的神经网络模型可以直接使用，比如VGG，ResNet等。可以说是非常懒惰了，比如：

```python
import torchvision.models as models
resnet18 = models.resnet18()
vgg16 = models.vgg16()
```

`torchvision`除了上面的`models`模块，还有`transforms`,`datasets`等模块也非常常用。


