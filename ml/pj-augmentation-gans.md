## 项目代码：简单的数据增强基于GANs实现

---

这是一个简单的线性数据集，基于GANs技术实现的例子。

### 1 - Import the libraries

```python
import time
import torch
from torch import nn
import matplotlib.pyplot as plt
from matplotlib import rcParams
from IPython import display
%matplotlib inline
```

### 2 - Create the data

生成一个标准正态分布1000x2的数据集。并自定义张量，用于数据集的生成。然后使用TensorDataset封装数据集，并生成数据集加载器。

```python
# create the data
X = torch.normal(0.0, 1, (1000, 2))
A = torch.tensor([[1, 2], [-0.1, 0.5]])
b = torch.tensor([1, 2])
data = torch.matmul(X, A) + b

# scatter the data
plt.scatter(
    data[:100, 0].detach().numpy(),
    data[:100, 1].detach().numpy()
)

# generate the dataloader
batch_size = 8
dataset = torch.utils.data.TensorDataset(data)
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size,
    shuffle=True
)
```

### 3 - Define the models

生成网络是一个简单的2-2，判别网络是2-5-3-1的结构。这里定义的是一个基本的生成网络模型。通过这种方式定义生成器和判别器，我们可以建立一个训练过程，使得生成器不断生成更加逼真的假数据，而判别器不断提高鉴别真假数据的能力，从而达到最终的目标：生成逼真的数据样本。

```python
Gen = nn.Sequential(
    nn.Linear(in_features=2, out_features=2)
)

Disc = nn.Sequential(
    nn.Linear(2, 5),
    nn.Tanh(),
    nn.Linear(5, 3),
    nn.Tanh(),
    nn.Linear(3, 1)
)
```

### 4 - Define the function for discriminator updates

```python
def update_D(X, Z, nnet_D, nnet_G, loss, trainer_D):
    """
    更新判别器模型的函数。

    Args:
    - X: 真实数据样本
    - Z: 用于生成假数据样本的随机噪声或潜在空间向量
    - nnet_D: 判别器模型
    - nnet_G: 生成器模型
    - loss: 用于计算损失的损失函数
    - trainer_D: 判别器模型的优化器

    Returns:
    - loss_D: 判别器模型的损失
    """

    # 获取批大小
    batch_size = X.shape[0]

    # 创建标签张量，用于真实数据和生成的假数据的标签
    ones = torch.ones((batch_size,), device=X.device)
    zeros = torch.zeros((batch_size,), device=X.device)

    # 将判别器的梯度置零
    trainer_D.zero_grad()

    # 计算判别器在真实数据上的输出
    real_Y = nnet_D(X)

    # 生成假数据样本
    synth_X = nnet_G(Z)

    # 将生成器的输出张量分离，避免梯度回传到生成器
    synth_Y = nnet_D(synth_X.detach())

    # 计算判别器的损失
    loss_D = (loss(real_Y, ones.reshape(real_Y.shape)) +
              loss(synth_Y, zeros.reshape(synth_Y.shape))) / 2

    # 反向传播并更新判别器模型参数
    loss_D.backward()
    trainer_D.step()

    # 返回判别器的损失
    return loss_D
```

这段代码定义了一个函数 `update_D`，用于更新判别器模型的参数。在生成对抗网络（GAN）的训练中，判别器的目标是能够区分真实数据和生成的假数据，因此需要通过不断优化判别器模型来提高其区分能力。函数中的步骤如下：

1. 获取批大小。
2. 创建真实数据和生成的假数据的标签张量，用于计算损失。
3. 将判别器的梯度置零。
4. 计算判别器在真实数据上的输出。
5. 生成假数据样本。
6. 将生成器的输出张量分离，避免梯度回传到生成器。
7. 计算判别器的损失，损失由真实数据和生成的假数据的交叉熵损失的平均值构成。
8. 反向传播并更新判别器模型参数。
9. 返回判别器的损失。

纯代码，用于笔记本粘贴：

```python
def update_D(X, Z, nnet_D, nnet_G, loss, trainer_D):
    batch_size = X.shape[0]
    ones = torch.ones((batch_size,), device=X.device)
    zeros = torch.zeros((batch_size,), device=X.device)
    trainer_D.zero_grad()
    real_Y = nnet_D(X)
    synth_X = nnet_G(Z)
    synth_Y = nnet_D(synth_X.detach())
    loss_D = (loss(real_Y, ones.reshape(real_Y.shape)) +
              loss(synth_Y, zeros.reshape(synth_Y.shape))) / 2
    loss_D.backward()
    trainer_D.step()
    return loss_D
```

### 5 - Perform generator updates

```python
def update_G(Z, nnet_D, nnet_G, loss, trainer_G):
    """
    更新生成器模型的函数。

    Args:
    - Z: 用于生成假数据样本的随机噪声或潜在空间向量
    - nnet_D: 判别器模型
    - nnet_G: 生成器模型
    - loss: 用于计算损失的损失函数
    - trainer_G: 生成器模型的优化器

    Returns:
    - loss_G: 生成器模型的损失
    """

    # 获取批大小
    batch_size = Z.shape[0]

    # 创建标签张量，用于生成的假数据的标签
    ones = torch.ones((batch_size,), device=Z.device)

    # 将生成器的梯度置零
    trainer_G.zero_grad()

    # 生成假数据样本
    synth_X = nnet_G(Z)

    # 计算判别器在生成的假数据上的输出
    synth_Y = nnet_D(synth_X)

    # 计算生成器的损失
    loss_G = loss(synth_Y, ones.reshape(synth_Y.shape))

    # 反向传播并更新生成器模型参数
    loss_G.backward()
    trainer_G.step()

    # 返回生成器的损失
    return loss_G
```

这段代码定义了一个函数 `update_G`，用于更新生成器模型的参数。在生成对抗网络（GAN）的训练中，生成器的目标是生成尽可能逼真的假数据，以骗过判别器。函数中的步骤如下：

1. 获取批大小。
2. 创建生成的假数据的标签张量，全部为1，用于计算生成器的损失。
3. 将生成器的梯度置零。
4. 生成假数据样本。
5. 计算判别器在生成的假数据上的输出。
6. 计算生成器的损失，通常是生成的假数据与标签为1的交叉熵损失。
7. 反向传播并更新生成器模型参数。
8. 返回生成器的损失。

纯代码，用于笔记本粘贴：

```python
def update_G(Z, nnet_D, nnet_G, loss, trainer_G):
    batch_size = Z.shape[0]
    ones = torch.ones((batch_size,), device=Z.device)
    trainer_G.zero_grad()
    synth_X = nnet_G(Z)
    synth_Y = nnet_D(synth_X)
    loss_G = loss(synth_Y, ones.reshape(synth_Y.shape))
    loss_G.backward()
    trainer_G.step()
    return loss_G
```

### 6 - Initialize the parameters

```python
def init_params(Discriminator, Generator, lr_D, lr_G):
    """
    初始化判别器和生成器模型的参数以及优化器，并创建损失图像的画布。

    Args:
    - Discriminator: 判别器模型
    - Generator: 生成器模型
    - lr_D: 判别器模型的学习率
    - lr_G: 生成器模型的学习率

    Returns:
    - loss: 用于计算损失的损失函数
    - trainer_D: 判别器模型的优化器
    - trainer_G: 生成器模型的优化器
    - fig: 用于绘制损失图像的画布对象
    - axes: 画布上的子图对象
    - loss_D: 用于保存判别器模型的损失的列表
    - loss_G: 用于保存生成器模型的损失的列表
    """

    # 定义损失函数
    loss = nn.BCEWithLogitsLoss(reduction='sum')

    # 初始化判别器和生成器模型的参数
    for w in Discriminator.parameters():
        nn.init.normal_(w, 0, 0.02)
    for w in Generator.parameters():
        nn.init.normal_(w, 0, 0.02)

    # 定义判别器和生成器模型的优化器
    trainer_D = torch.optim.Adam(Discriminator.parameters(), lr=lr_D)
    trainer_G = torch.optim.Adam(Generator.parameters(), lr=lr_G)

    # 创建用于绘制损失图像的画布对象和子图对象
    fig, axes = plt.subplots(2, 1, figsize=(5, 9))

    # 初始化保存判别器和生成器模型损失的空列表
    loss_D = []
    loss_G = []

    # 返回初始化的参数和对象
    return loss, trainer_D, trainer_G, fig, axes, loss_D, loss_G
```

这段代码定义了一个函数 `init_params`，用于初始化判别器和生成器模型的参数、优化器，并创建损失图像的画布。在训练生成对抗网络（GAN）时，判别器和生成器的初始化至关重要，它们的初始参数需要足够随机以避免模型陷入局部最优点。函数中的步骤如下：

1. 定义用于计算损失的损失函数，这里采用二元交叉熵损失函数 `BCEWithLogitsLoss`。
2. 初始化判别器和生成器模型的参数，采用均值为0，标准差为0.02的正态分布进行初始化。
3. 定义判别器和生成器模型的优化器，这里采用Adam优化器，并传入对应的学习率参数。
4. 创建用于绘制损失图像的画布对象和子图对象，其中子图对象包含两个子图，用于分别绘制判别器和生成器的损失。
5. 初始化保存判别器和生成器模型损失的空列表。
6. 返回初始化的参数和对象，用于后续的训练过程。

纯代码，用于笔记本粘贴：

```python
def init_params(Discriminator, Generator, lr_D, lr_G):
    loss = nn.BCEWithLogitsLoss(reduction='sum')
    for w in Discriminator.parameters():
        # tensor, mean, std
        nn.init.normal_(w, 0, 0.02)
    for w in Generator.parameters():
        nn.init.normal_(w, 0, 0.02)
    trainer_D = torch.optim.Adam(Discriminator.parameters(), lr=lr_D)
    trainer_G = torch.optim.Adam(Generator.parameters(), lr=lr_G)
    fig, axes = plt.subplots(2, 1, figsize=(5, 9))
    loss_D = []
    loss_G = []

    return loss, trainer_D, trainer_G, fig, axes, loss_D, loss_G
```

### 7 - Compute the losses

```python
def compute_losses(X, net_D, net_G, loss, trainer_D, trainer_G, batch_size, latent_dim, data_iter):
    """
    计算判别器和生成器模型的损失。

    Args:
    - X: 真实数据样本
    - net_D: 判别器模型
    - net_G: 生成器模型
    - loss: 用于计算损失的损失函数
    - trainer_D: 判别器模型的优化器
    - trainer_G: 生成器模型的优化器
    - batch_size: 批大小
    - latent_dim: 潜在空间向量的维度
    - data_iter: 数据迭代器，用于迭代训练数据

    Returns:
    - metrics: 包含判别器和生成器模型的损失以及数据样本数量的列表
    """

    # 初始化损失指标为0
    metrics = [0.0]*3 
    
    # 遍历数据迭代器
    for (X,) in data_iter:
        # 获取批大小
        batch_size = X.shape[0]

        # 生成潜在空间向量
        Z = torch.normal(0, 1, size=(batch_size, latent_dim))

        # 更新判别器和生成器模型，并计算损失
        metric = [update_D(X, Z, net_D, net_G, loss, trainer_D),  # 判别器模型是损失累计器
                  update_G(Z, net_D, net_G, loss, trainer_G),  # 生成器模型的损失累计器
                  batch_size]  # 数据样本的数量

        # 更新指标值，每次进行累计
        metrics = [sum(i) for i in zip(metric, metrics)]

    # 返回包含判别器和生成器模型的损失以及数据样本数量的列表
    return metrics
```

这段代码定义了一个函数 `compute_losses`，用于计算判别器和生成器模型的损失。在训练生成对抗网络（GAN）时，需要周期性地计算损失，并根据损失来更新模型的参数。函数中的步骤如下：

1. 初始化损失指标为0。
2. 遍历数据迭代器，迭代训练数据。
3. 获取每个批次的数据样本，并根据批大小生成相应数量的潜在空间向量。
4. 调用 `update_D` 和 `update_G` 函数更新判别器和生成器模型，并计算相应的损失。
5. 将每个批次的损失和数据样本数量累加到指标中。
6. 返回包含判别器和生成器模型的损失以及数据样本数量的列表。

在每次迭代中，`metric` 是每个批次的损失和样本数量的累加值，而 `metrics` 是总的损失和样本数量的累加值。通过迭代训练数据集，我们会不断更新 `metrics` 中的值，并将每个批次的损失值和样本数量累加到 `metrics` 中，最终得到的 `metrics` 中的值就是整个训练过程中的总损失值和总样本数量。

这一步将前面的更新和这里的loss计算结合在了一起。

纯代码，用于笔记本粘贴：

```python
def compute_losses(X, net_D, net_G, loss, trainer_D, trainer_G, batch_size, latent_dim, data_iter):
    metrics = [0.0]*3 
    for (X,) in data_iter:
        batch_size = X.shape[0]
        Z = torch.normal(0, 1, size=(batch_size, latent_dim))
        metric = [update_D(X, Z, net_D, net_G, loss, trainer_D),
                    update_G(Z, net_D, net_G, loss, trainer_G),
                    batch_size]
        metrics = [sum(i) for i in zip(metric, metrics)]

    return metrics
```

### 8 - Display generated distributions

```python
def display_gen_dist(net_G, axes, latent_dim, data):
    """
    显示生成的数据分布。

    Args:
    - net_G: 生成器模型
    - axes: 画布上的子图对象
    - latent_dim: 潜在空间向量的维度
    - data: 真实数据样本

    Returns:
    - None
    """

    # 生成潜在空间向量
    Z = torch.normal(0, 1, size=(100, latent_dim))

    # 使用生成器生成假数据样本，并将其转换为NumPy数组
    synth_X = net_G(Z).detach().numpy()

    # 清除第二个子图对象的内容
    axes[1].cla()

    # 绘制真实数据样本的散点图
    axes[1].scatter(data[:, 0], data[:, 1])

    # 绘制生成的假数据样本的散点图
    axes[1].scatter(synth_X[:, 0], synth_X[:, 1])

    # 添加图例，标识真实数据和生成的假数据
    axes[1].legend(['real', 'generated'])
```

这段代码定义了一个函数 `display_gen_dist`，用于显示生成的数据分布。函数中的步骤如下：

1. 生成 `100` 个潜在空间向量 `Z`，每个向量的维度为 `latent_dim`。
2. 使用生成器模型 `net_G` 生成假数据样本，并通过 `detach()` 方法分离出生成的张量以避免梯度回传。
3. 将生成的假数据样本转换为 NumPy 数组，以便后续绘图操作。
4. 清除第二个子图对象的内容。
5. 绘制真实数据样本和生成的假数据样本的散点图。
6. 添加图例，标识真实数据和生成的假数据。
7. 返回 `None`。

纯代码，用于笔记本粘贴：

```python
def display_gen_dist(net_G, axes, latent_dim, data):
    Z = torch.normal(0, 1, size=(100, latent_dim))
    synth_X = net_G(Z).detach().numpy()
    axes[1].cla()
    axes[1].scatter(data[:, 0], data[:, 1])
    axes[1].scatter(synth_X[:, 0], synth_X[:, 1])
    axes[1].legend(['real', 'generated'])
```

### 9 - Display the losses

```python
def display_losses(metrics, loss_D, loss_G, axes, fig, epoch):
    """
    显示损失值并更新损失列表。

    Args:
    - metrics: 包含判别器和生成器模型的损失以及数据样本数量的列表
    - loss_D: 判别器模型的损失列表
    - loss_G: 生成器模型的损失列表
    - axes: 画布上的子图对象
    - fig: 画布对象
    - epoch: 当前训练的轮数

    Returns:
    - loss_D: 更新后的判别器模型的损失列表
    - loss_G: 更新后的生成器模型的损失列表
    """

    # 计算并保存判别器和生成器模型的损失
    D = metrics[0] / metrics[2]
    loss_D.append(D.detach())
    G = metrics[1] / metrics[2]
    loss_G.append(G.detach())

    # 在第一个子图对象上绘制损失曲线
    axes[0].plot(range(epoch+1), loss_D, c="blue")
    axes[0].plot(range(epoch+1), loss_G, c="green")

    # 添加图例，标识判别器和生成器模型的损失
    axes[0].legend(['Discriminator loss', 'Generator loss'])

    # 显示画布
    display.display(fig)

    # 清空输出，并等待下一次更新
    display.clear_output(wait=True)

    # 返回更新后的损失列表
    return loss_D, loss_G
```

这段代码定义了一个函数 `display_losses`，用于显示损失值并更新损失列表。函数中的步骤如下：

1. 根据 `metrics` 列表中的值计算并保存判别器和生成器模型的损失，这里将累计的损失值除以数据样本数量得到平均损失。
2. 在第一个子图对象上绘制判别器和生成器模型的损失曲线。
3. 添加图例，标识判别器和生成器模型的损失。
4. 显示画布，即在 notebook 中显示损失曲线图。
5. 清空 notebook 输出，并等待下一次更新，以便实时显示损失曲线的变化。
6. 返回更新后的判别器和生成器模型的损失列表。

```python
def display_losses(metrics, loss_D, loss_G, axes, fig, epoch):
    D = metrics[0]/metrics[2]
    loss_D.append(D.detach())
    G = metrics[1]/metrics[2]
    loss_G.append(G.detach())
    axes[0].plot(range(epoch+1), loss_D, c="blue")
    axes[0].plot(range(epoch+1), loss_G, c="green")
    axes[0].legend(['Discriminator loss', 'Generator loss'])
    display.display(fig)
    display.clear_output(wait=True)

    return loss_D, loss_G
```

### 10 - Create the training function

```python
def train(net_D, net_G, data_iter, num_epochs, lr_D, lr_G, latent_dim, data):
    """
    训练生成对抗网络。

    Args:
    - net_D: 判别器模型
    - net_G: 生成器模型
    - data_iter: 数据迭代器，用于迭代训练数据
    - num_epochs: 训练的轮数
    - lr_D: 判别器模型的学习率
    - lr_G: 生成器模型的学习率
    - latent_dim: 潜在空间向量的维度
    - data: 真实数据样本

    Returns:
    - None
    """

    # 记录训练开始时间
    tik = time.perf_counter()

    # 初始化参数和变量
    loss, trainer_D, trainer_G, fig, axes, loss_D, loss_G = init_params(net_D, net_G, lr_D, lr_G)

    # 遍历每个训练轮次
    for epoch in range(num_epochs):
        # 计算当前轮次的损失值
        metrics = compute_losses(X, net_D, net_G, loss, trainer_D, trainer_G, batch_size, latent_dim, data_iter)
        
        # 可视化生成的样本分布
        display_gen_dist(net_G, axes, latent_dim, data)
        
        # 显示损失曲线
        loss_D, loss_G = display_losses(metrics, loss_D, loss_G, axes, fig, epoch)
    
    # 记录训练结束时间
    tok = time.perf_counter()

    # 显示训练统计信息
    print(f'loss_D {loss_D[-1]}, loss_G {loss_G[-1]}, {(metrics[2]*num_epochs) / (tok-tik):.1f} examples/sec')
```

这段代码定义了一个函数 `train`，用于训练生成对抗网络（GAN）。函数中的步骤如下：

1. 记录训练开始时间。
2. 初始化判别器和生成器模型的参数以及其他相关变量。
3. 遍历每个训练轮次。
4. 在每个训练轮次中，计算当前轮次的损失值。
5. 可视化生成的假数据分布。
6. 显示损失曲线。
7. 记录训练结束时间。
8. 显示训练统计信息，包括最后一个损失值以及训练速度（样本/秒）。

这个函数包含了整个训练过程中的关键步骤，它会遍历指定的训练轮次，每轮训练会计算损失、更新模型参数、可视化生成的假数据分布以及显示损失曲线。

纯代码，用于笔记本粘贴：

```python
def train(net_D, net_G, data_iter, num_epochs, lr_D, lr_G, latent_dim, data):
    # Start timer
    tik = time.perf_counter()
    # Init variables
    loss, trainer_D, trainer_G, fig, axes, loss_D, loss_G = init_params(net_D, net_G, lr_D, lr_G)
    for epoch in range(num_epochs):
        # Train one epoch
        metrics = compute_losses(X, net_D, net_G, loss, trainer_D, trainer_G, batch_size, latent_dim, data_iter)
        # Visualize generated examples
        display_gen_dist(net_G, axes, latent_dim, data)
        # Show the losses
        loss_D, loss_G = display_losses(metrics, loss_D, loss_G, axes, fig, epoch)
    # End timer
    tok = time.perf_counter()
    # Display stats
    print(f'loss_D {loss_D[-1]}, loss_G {loss_G[-1]}, {(metrics[2]*num_epochs) / (tok-tik):.1f} examples/sec')
```

### 11 - Train the model

```python
lr_D, lr_G, latent_dim, num_epochs = 0.05, 0.005, 2, 30

# 调用 train 函数进行生成对抗网络的训练
train(Disc, Gen, dataloader, num_epochs, lr_D, lr_G, latent_dim, data[:100].detach().numpy())
```

这段代码设置了训练生成对抗网络（GAN）所需的超参数，并调用了 `train` 函数进行训练。具体解释如下：

1. `lr_D`: 判别器模型的学习率，设置为 `0.05`。
2. `lr_G`: 生成器模型的学习率，设置为 `0.005`。
3. `latent_dim`: 潜在空间向量的维度，设置为 `2`。
4. `num_epochs`: 训练的轮数，设置为 `30`。

设置学习率是深度学习中的一个重要超参数，它直接影响着模型的训练效果和速度。对于生成对抗网络（GAN）的训练，通常需要仔细调整学习率以获得良好的结果。

在给定的代码中，设置了两个学习率 `lr_D` 和 `lr_G`，分别对应着判别器模型和生成器模型的学习率。这种不同的学习率设置是因为在训练过程中，判别器模型和生成器模型可能会有不同的训练需求。

一般来说，判别器模型需要更高的学习率，这样可以更快地学习区分真实样本和生成样本之间的差异，从而提高判别器的准确性。而生成器模型则需要较低的学习率，因为生成器需要更加稳定的训练过程，避免模式崩溃或生成不稳定的样本。

具体到这个例子中，判别器的学习率 `lr_D` 设置为 `0.05`，而生成器的学习率 `lr_G` 设置为 `0.005`。这种设置可能是经过了一定的试验和调优，以获得较好的训练效果。当然，最佳的学习率设置可能会因数据集、模型结构、训练任务等因素而有所不同，需要通过实验来确定最佳的超参数设置。

然后，调用 `train` 函数进行生成对抗网络的训练，传入了判别器模型 `Disc`、生成器模型 `Gen`、数据加载器 `dataloader`、训练轮数 `num_epochs`、判别器和生成器模型的学习率以及潜在空间向量的维度和部分真实数据样本作为参考数据。

纯代码：

```python
lr_D, lr_G, latent_dim, num_epochs = 0.05, 0.005, 2, 30
train(Disc, Gen, dataloader, num_epochs, lr_D, lr_G, latent_dim, data[:100].detach().numpy())
```

笔记本连接：[project folder](https://github.com/sherryuuer/projects-drafts/blob/main/gans-linear-dataset/dataset-augmentation-GANs.ipynb)

最终的生成器的损失不断下降，判别器的损失开始上升，因为它渐渐识别出不生成器的结果。损失开始接近。
