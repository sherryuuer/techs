## 生成对抗网络 GAN

---

生成对抗网络（Generative Adversarial Networks，GANs）是一种用于生成模型的框架，由两个神经网络组成：生成器（Generator）和鉴别器（Discriminator）。GAN 的核心思想是让这两个网络进行对抗训练，生成器试图生成与真实样本相似的样本，而鉴别器则试图区分真实样本和生成样本。通过不断的对抗训练，生成器可以逐渐生成更加逼真的样本，从而实现生成模型的训练。

### 1 - 对抗性攻击 Adversarial attacks

因为深度学习模型对基于输入微小修改的攻击非常敏感。假设你有一个经过训练的分类器，在测试时能够正确识别图像中的对象，并给出正确的标签。有可能构造出一个对人类视觉来说无法区分的图像，即对抗样本。

这些对抗样本可以通过向图像添加噪声来构造。然而，这样的图像会被错误分类。为了解决这个问题，一种常见的方法是将对抗样本注入到训练集中。这被称为对抗训练。这样做可以提高神经网络的鲁棒性。这种类型的示例可以通过添加噪声、应用数据增强技术，或者在梯度相反方向扰动图像（以最大化损失而不是最小化损失）来生成。

```python
import torch

## FGSM 攻击代码
def fgsm_attack(image, epsilon, data_grad):
    """
    FGSM（Fast Gradient Sign Method）攻击代码。

    Args:
    - image: 输入图像张量
    - epsilon: 扰动的大小，控制攻击强度
    - data_grad: 输入图像的梯度

    Returns:
    - perturbed_image: 扰动后的图像张量
    """

    # 收集数据梯度的逐元素符号
    sign_data_grad = data_grad.sign()

    # 创建受扰动影响的图像，调整输入图像的每个像素
    perturbed_image = image + epsilon * sign_data_grad

    # 添加剪裁以保持 [0,1] 范围内，高于或者低于的都被设置为0，1
    perturbed_image = torch.clamp(perturbed_image, 0, 1)

    # 返回受扰动影响的图像
    return perturbed_image
```

这段代码实现了 FGSM（快速梯度符号法）攻击，用于生成对抗样本。具体解释如下：

- `image`: 输入图像张量。
- `epsilon`: 扰动的大小，用于控制攻击的强度。较大的 epsilon 值会导致更大程度的图像扰动。
- `data_grad`: 输入图像的梯度，用于确定应该对图像进行怎样的扰动。

攻击的主要步骤如下：

1. 收集输入图像梯度的逐元素符号，即保留梯度的方向信息。
2. 使用符号梯度调整输入图像的每个像素，以产生扰动后的图像。
3. 添加剪裁操作，将扰动后的图像张量限制在 [0,1] 范围内。
4. 返回受扰动影响的图像。

这种扰动图像的作用是在视觉上对人类不易察觉的情况下，使深度学习模型产生误判。这种图像称为对抗样本。

对抗样本的产生可以在原始图像的基础上添加微小的扰动，使得人眼难以察觉这些扰动，但对深度学习模型的影响很大。通过对原始图像进行微小的变换，可以导致模型在识别和分类任务上产生错误的结果。对抗样本的产生和利用可以帮助研究人员更好地理解深度学习模型的行为，并且对于提高模型的鲁棒性和安全性具有重要意义。

### 2 - 从对抗性攻击到生成式学习 Generative learning

那么如果我们不关心拥有一个鲁棒的分类器，而是想要，一个制作样本的过程，那么就可以让网络创造出在视觉上吸引人的不同例子。这就是生成性项起作用的地方了，在生成对抗学习中，专注于生成数据集的代表性示例，而不是使网络对扰动更加鲁棒！

利用另一个网络来完成这项工作，一个能够生成与训练样本相似、逼真的例子的网络。他就是生成器 Genarator。

生成器（Generator）：

- 生成器是一个神经网络，其输入通常是一个随机噪声向量（通常服从某种先验分布，如均匀分布或正态分布），输出是一个样本，例如一张图像。
- 生成器的目标是生成与真实样本相似的样本，以欺骗鉴别器。
- 生成器网络通常包含多个层，包括全连接层、卷积层、反卷积层等，用于将输入噪声向量映射到输出样本空间。

生成器的输入和输出（input&output）：

- 输入是从小范围实数中采取的随机噪声，就是类似于在自动编码器中说的，潜在或者连续空间 Z。
- 输入体现了**随机性**的内容，但是很简单。**由于随机输入的可能性无限多，所以输出也是无限可能的。**
- 输出虽然无限可能，但是在生成器网络中，我们要生成的往往是一个特定的输出。比如猫猫狗狗的照片。

鉴别器（Discriminator）：

- 鉴别器也是一个神经网络，其输入是一个样本（可以是真实样本也可以是生成器生成的样本），输出是一个标量值，表示输入样本为真实样本的概率，0代表为虚假图片，1代表是真实图片的分布。
- 鉴别器的目标是对生成器生成的样本与真实样本进行区分。但它区分的不是正类，而是真实样本的分布。
- 在一开始训练的时候，所有的真实图片标签为1，生成图片的标签为0。
- **零和博弈**：理想的生成器，会让概率变成0.5，也就是造出无法被区分的图像。在零和博弈中，双方玩家都知道对方的均衡策略，并且没有玩家可以通过仅改变自己的策略来获得任何收益。

生成对抗网络（GAN）的训练过程：

- 在训练过程中，生成器和鉴别器交替进行训练，每次迭代都先更新鉴别器，然后再更新生成器。他们一定是一个在训练一个是不训练的。
- 训练鉴别器的过程中必须冻结鉴别器的权重，如此提高训练的稳定性。如果在训练生成器的过程中，更新鉴别器的权重，会导致鉴别器重新学习新的特征，导致生成器的训练不稳定。
- 生成器通过生成，和真实图片**相似的分布**来欺骗鉴别器，鉴别器则试图确定**分布的来源**。
- 在鉴别器的更新过程中，通过最小化真实样本的损失和最大化生成样本的损失来训练鉴别器，使其能够更好地区分真实样本（概率接近1）和生成样本（概率接近0）。
- 在生成器的更新过程中，通过最小化生成样本的损失来训练生成器，使其生成的样本更加逼真，从而欺骗鉴别器。

### 3 - 关于学习过程和熵最大化的相似性思考 thinking

在计算重构图像和原图像的时候使用了交叉熵。我一直没有很好的思考这个损失函数，直到在GANs这里一切都是这么具象化。因为在GANs训练的二元博弈中，就是为了让生成器和鉴别器打成平手，这个时候分布近乎达到了熵最大化！

生成对抗网络（GAN）的过程和最大化熵的过程在某种程度上很相似。在生成对抗网络中的动态变化，使得生成器和鉴别器不断地互相提升和对抗，最终达到一个动态平衡。

在最大化熵的过程中，我们也是在追求一种状态的不确定性或混乱程度的最大化。通过增加系统的不确定性，我们可以使系统处于更加多样化和混乱的状态，这也可以看作是一种对抗性的过程，只是在这种情况下，我们是在与系统的初始状态或其它限制性条件进行对抗。但是反过来想，初始条件，就是一边很清晰，一边都是噪声。

这个宇宙是不确定的分布概率，是幂律还是正态分布，是稳定还是不稳定，我觉得是一个动态变化的过程，一开始是幂律的系统，整个系统很开放，但是渐渐的就趋于正态分布，宇宙趋于稳态，多样化达到了最大！但是熵也是最大的，最终会充斥着多样化，再次回归另一种稳态。

顺便python的交叉熵代码：

```python
import numpy as np

def cross_entropy(true_labels, predicted_probs):
    """
    计算交叉熵
    
    参数：
    true_labels: 真实标签，一个形状为 (n,) 的numpy数组
    predicted_probs: 预测概率，一个形状为 (n, num_classes) 的numpy数组，每行对应一个样本的预测概率分布
    
    返回值：
    交叉熵，一个标量值
    """
    # 确保预测概率不为0
    predicted_probs = np.clip(predicted_probs, 1e-15, 1 - 1e-15)
    # 计算交叉熵
    cross_entropy = -np.sum(true_labels * np.log(predicted_probs))
    return cross_entropy
```

再次被这个公式震撼。

### 4 - 使用CIFAR10数据集训练网络

我使用cifar数据集进行了两个比较简单的网络的训练，网络基本是由线形layer组成的，可能性能不是很好，但是可以看到整个训练的过程，并且我将loss结果储存在两个列表中，用于最后的可视化。

关注下面的训练部分的代码，可以注意到：对判别器和生成器的训练其实就是反向传播真值，和假值，分别和1之间的差距，然后将它们相加。

```python
def train_discriminator(discriminator, optimizer, real_data, fake_data, loss):
    N = real_data.size(0)
    # Reset gradients
    optimizer.zero_grad()

    # Train on Real Data
    prediction_real = discriminator(real_data)
    # Calculate error and backpropagate
    error_real = loss(prediction_real, Variable(torch.ones(N, 1)))
    error_real.backward()

    # Train on Fake Data
    prediction_fake = discriminator(fake_data)
    # Calculate error and backpropagate
    error_fake = loss(prediction_fake, Variable(torch.zeros(N, 1)))
    error_fake.backward()

    # Update weights with gradients
    optimizer.step()

    # Return error and predictions for real and fake inputs
    return error_real + error_fake, prediction_real, prediction_fake

def train_generator(discriminator, optimizer, fake_data, loss):
    # Reset gradients
    N = fake_data.size(0)  
    
    # Sample noise and generate fake data
    optimizer.zero_grad()  
    
    # Calculate error and backpropagate
    prediction = discriminator(fake_data)  
    error = loss(prediction, Variable(torch.ones(N, 1)))
    
    # Update weights with gradients
    error.backward()  
    optimizer.step()  
    
    # Return error
    return error
```

关注下面的主训练代码的一部分：

- 损失函数使用二元交叉熵函数。
- 优化器使用最常用的Adam。
- 在训练判别器的时候，每一轮都由生成器生成fake图像，然后调用函数对判别器进行训练，反向传播。
- 在训练生成器的时候，每一轮也是由生成器生成图像，然后调用函数对生成器进行训练，反向传播。这时候判别器什么也不做。

```python
def train():

    # Models, optimizers and losses
    discriminator = DiscriminatorNet()
    generator = GeneratorNet()
    loss_d = nn.BCELoss()
    loss_g = nn.BCELoss()
    d_optimizer = optim.Adam(discriminator.parameters(), 0.0002)
    g_optimizer = optim.Adam(generator.parameters(), 0.0002)

    data_loader = load_data()
    
    num_epochs = 1
    num_batches = len(data_loader)
    
    num_test_samples = 48
    test_noise = noise(num_test_samples)

    d_errors, g_errors = [], []
    d_pred_real_means, d_pred_fake_means = [], []
    for epoch in range(num_epochs):
        for n_batch, data in enumerate(data_loader):

            (real_batch, labels) = data
            N = real_batch.size(0)

            # 1. Train Discriminator
            real_data = real_batch.view(real_batch.size(0), -1)

            # Generate fake data and detach so gradients are not calculated for generator)
            latent_space_data = noise(N)
            fake_data = generator(latent_space_data).detach()

            d_error, d_pred_real, d_pred_fake = train_discriminator(discriminator, d_optimizer, real_data,
                                                                          fake_data,
                                                                          loss_d)
            # 2. Train Generator

            # Generate fake data TO train Generator
            latent_space_data = noise(N)
            fake_data = generator(latent_space_data)
            # Train G
            g_error = train_generator(discriminator, g_optimizer, fake_data, loss_g)  # Log batch error
```

完整的[notebook](https://github.com/sherryuuer/projects-drafts/tree/main/gans-cifar10)和训练结果可以在我的GH中看到。
