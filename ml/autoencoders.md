## 生成式AI：模型原理，自动编码器和变分自编码器

---
### 1 - 理解判别式模型和生成式模型（Discriminative vs generative models）

判别式（Discriminative）模型和生成式（generative）模型。

Discriminative 模型和 Generative 模型是机器学习中两种不同的建模方法，它们的核心区别在于所关注的任务和所建模的概率分布。

1 - **Discriminative 模型**：

- 基本原理：Discriminative 模型关注的是对观察数据和标签之间的条件分布进行建模，即给定输入数据，预测其对应的标签。因此，它们直接学习并建模了类别之间的边界或决策边界。
- 工作原理：Discriminative 模型通常通过学习条件概率分布 P(Y|X) 来实现，给定输入 X 来预测输出 Y，其中 X 表示输入特征，Y 表示标签或输出变量。这些模型主要关注于给定输入 X 来预测输出 Y，而对输入数据的分布并不关心。常见的 Discriminative 模型包括逻辑回归、支持向量机、决策树、神经网络等。
- 它更适合用于分类，标注等任务。

2 - **Generative 模型**：

- 基本原理：Generative 模型关注的是对观察数据的联合概率分布进行建模，即对观察数据的生成过程进行建模。因此，它们试图理解数据背后的潜在分布和生成机制。
- 工作原理：Generative 模型通常通过学习联合概率分布 P(Xn) 来实现，有些情况中，联合概率分布是X和Y的分布，X和Y分别是数据和标签，但是这里要说的生成式模型是不包括标签的。在生成式模型中，学习的确实是数据的联合概率分布，但这并不意味着这个联合概率分布一定与标签有关。
- 生成式模型学习的是观察到的数据点 x 的联合概率分布 p(x)，其中 x 是输入的数据点。这个概率分布描述了数据点在观察空间中的分布情况，而不包含标签信息。通过学习这个概率分布，生成式模型能够生成新的数据样本，填充缺失值，进行推理和预测等任务，而无需外部标签的指导。在生成式模型中，数据本身“竞争”概率密度，即模型学习如何平衡数据点之间的概率分布，使得观察到的数据点具有较高的概率密度。这意味着模型需要尽可能准确地学习数据的分布，以便能够有效地生成新的数据样本，并与观察到的数据点相符合。
- 常见的 Generative 模型包括朴素贝叶斯、高斯混合模型、潜在语义分析、生成对抗网络等。
- 更适合用于生成新的数据样本，填充缺失值，特征提取等任务。
- PS：联合概率分布是什么：它是指同时描述多个随机变量的概率分布。在一个联合概率分布中，每个可能的事件都与每个变量的值的组合相关联，并且分配了一个概率值。这种概率分布描述了多个变量之间的关系，以及它们如何共同决定观察到的数据。假设有两个随机变量 X 和 Y，它们的联合概率分布可以表示为 P(X, Y)。这个分布可以描述在给定 X 和 Y 的情况下，观察到某个事件的概率。例如，如果 X 表示一个骰子的投掷结果，Y 表示另一个骰子的投掷结果，那么联合概率分布可以告诉我们同时观察到两个骰子的特定组合的概率，如 (X=1, Y=3)、(X=2, Y=5) 等。联合概率分布包含了每个变量的边际概率分布以及它们之间的相互作用。通过联合概率分布，我们可以进行诸如边际化、条件化等操作，来计算关于单个变量或者一组变量的各种概率。

3 - **Conditional Generative 模型**：

- 条件生成模型是另一类模型，试图学习数据的概率分布 p(x) 在标签 y 的条件下的分布。这通常表示为 p(x|y)。在这种情况下，我们再次让数据“竞争”概率密度，但这次是针对每个可能的标签。
- “竞争”的概念。概率密度函数 p 是一个归一化函数，其在整个值域上的积分值等于 1。这意味着在给定一组数据点的情况下，它们的概率密度函数应该总和为 1。在条件生成模型中，我们学习的是数据点在给定标签的条件下的概率密度函数 p(x|y)。因此，对于每个可能的标签，我们都有一个对应的概率密度函数。
- 条件生成模型的目标是通过学习数据在给定标签的条件下的分布来实现各种任务。例如，给定一张图片和一个标签，我们可以使用条件生成模型来生成与该标签相符的新图片；或者在给定一些文本描述和一个标签时，生成与描述和标签相符的新文本。这种模型在图像生成、文本生成等任务中具有广泛的应用。

4 - **p(x|y) = p(y|x) * p(x) / p(y)**：

前面提到的模型们，在一定程度上是相互关联的，考虑贝叶斯定理：p(x|y) = p(y|x) * p(x) / p(y)

这个公式告诉我们，我们可以将每种类型的模型建立为其他类型的模型的组合。

这个公式所表示的贝叶斯定理是基于条件概率和边缘概率之间的关系。在这个公式中，p(x|y) 是在给定标签 y 的情况下观察到数据 x 的条件概率。而 p(y|x) 是在给定数据 x 的情况下观察到标签 y 的条件概率。p(x) 和 p(y) 分别是数据 x 和标签 y 的边缘概率。

这个公式告诉我们，我们可以通过对条件概率和边缘概率进行组合，从而构建出各种类型的模型。例如，我们可以从判别式模型（p(y|x)）出发，结合数据的边缘分布（p(x)）来构建生成式模型（p(x|y)）。或者从生成式模型（p(x|y)）出发，结合标签的边缘分布（p(y)）来构建判别式模型（p(y|x)）。

贝叶斯定理为我们提供了一个框架，说明了各种模型之间的联系和互相转换的可能性。这种联系使得我们能够更加灵活地建立和理解各种类型的模型。真是强大又神奇。

### 2 - 理解潜变量模型（Latent variable models）

潜变量模型（Latent Variable Models）是一类统计模型，用于描述观察到的数据与未观察到的潜在变量之间的关系。在潜变量模型中，假设观察到的数据是由潜在变量和随机噪声共同决定的，而潜在变量通常不能直接观察到，需要通过观察到的数据进行推断和估计。

潜变量模型通常用于以下几个方面：数据降维和特征提取，模式识别和分类，填充缺失值，探索数据生成过程。

常见的潜变量模型包括潜在类别模型、潜在因子模型、混合模型等。这些模型在各种领域都有广泛的应用，例如在机器学习、统计学、社会科学、生物医学等领域。潜变量模型为我们提供了一种强大的工具，可以帮助我们理解和利用数据背后的潜在结构和规律。

这里为什么要提起这个概念，因为在机器学习领域，这个模型描述的是**潜在变量的概率分布**，存在于数据点集合的**连续的低维空间**。

有的时候我感觉数学公式让直觉理解变的困难，但是在这个情形下，数学公式贝叶斯，让这里变的清晰明了。

这里的潜变量分布，其实是一种先验分布。整理一下看看：

- 数据集X遵循一种分布p(x)，它是到潜变量分布p(z)的map对应。
- 先验分布 piror distribution p(z) 是对潜变量分布进行建模。
- 似然估计（可能性） likehood p(x|z) 定义了潜变量如何映射到数据点。
- 联合分布 joint distribution p(x,z) = p(x|z)p(z) 是似然估计和先验分布的乘积，本质上它描述了模型。
- 边际分布 marginal distribution p(x)是原属数据分布，它描述了生成数据点的可能性，是模型的最终目标。
- 后验分布 posterior distribution p(z|x)描述了特定数据可以产生的潜在变量。

关注上面步骤中两个步骤其实就是生成和推理的过程。

- 生成 Generation：就是由潜在数据点，计算实际数据点的过程就是p(x|z)似然估计的计算。
- 推理 Inference：就是寻找潜在变量的过程，就是p(z|x)后验估计的计算。

```
      Generation p(x|z)
 -----------------------------
↑                               ↓
p(z)                          p(x)
↑                               ↓
 -----------------------------
      Inference p(z|x)
```
如上图推理和生成互为逆过程。为了实现生成，可以从p(z)进行对z对采样，然后从p(x|z)实现对x的采样。为了实现推理，可以从p(x)实现对x的采样，然后从p(z|x)实现对z的采样。

### 3 - 自动编码器（Autoencoder）

自动编码器的两个主要部分是编码器（encoder）和解码器（decoder）。编码器接收输入数据，并将其编码成低维空间中的潜在变量（latent variables），通常用一个向量表示，称为“潜在向量”（latent vector）。解码器则接收这个潜在向量，并尝试将其解码回原始的输入数据。

在编码器和解码器之间的潜在空间中是一个连续的低维空间，这意味着潜在向量表示了输入数据的重要特征，但并不一定对应着具体的类别或标签。

通过学习数据的潜在表示，自动编码器可以有效地压缩数据，减少数据的维度，或者去除数据中的噪声。此外，潜在向量还可以用于处理新的数据，如数据增强、异常检测等。

自动编码器通常使用重构损失（reconstruction loss）（是一种损失函数，重构损失衡量了重构数据与原始数据之间的差异，通常采用原始数据与重构数据之间的距离来表示。）来训练，其最简单的形式就是重构数据与原始数据之间的平方欧氏距离。编码器和解码器可以采用不同的网络结构，例如全连接网络或卷积神经网络（CNN），具体取决于输入数据的特点和应用场景。

下面的代码尝试使用 PyTorch 复现一个自动编码器的结构（如下）。给定输入和输出的尺寸`(-1,3,32,32)`，任务是搭建一个相应的自动编码器模型。

```
Conv2d-1	[-1, 12, 16, 16]	588
ReLU-2	[-1, 12, 16, 16]	0
Conv2d-3	[-1, 24, 8, 8]	4,632
ReLU-4	[-1, 24, 8, 8]	0
Conv2d-5	[-1, 48, 4, 4]	18,480
ReLU-6	[-1, 48, 4, 4]	0
Conv2d-7	[-1, 96, 2, 2]	73,824
ReLU-8	[-1, 96, 2, 2]	0
ConvTranspose2d-9	      [-1, 48, 4, 4]	73,776
ReLU-10	[-1, 48, 4, 4]	0
ConvTranspose2d-11	[-1, 24, 8, 8]	18,456
ReLU-12	[-1, 24, 8, 8]	0
ConvTranspose2d-13	[-1, 12, 16, 16]	4,620
ReLU-14	[-1, 12, 16, 16]	0
ConvTranspose2d-15	[-1, 3, 32, 32]	579
Sigmoid-16	[-1, 3, 32, 32]	0
```

```python
import torch
import torch.nn as nn

seed = 172
torch.manual_seed(seed)


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 12, 4, stride=2, padding=1),  # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.Conv2d(12, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
            nn.ReLU(),
            nn.Conv2d(24, 48, 4, stride=2, padding=1),  # [batch, 48, 4, 4]
            nn.ReLU(),
            nn.Conv2d(48, 96, 4, stride=2, padding=1),  # [batch, 96, 2, 2]
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(96, 48, 4, stride=2, padding=1),  # [batch, 48, 4, 4]
            nn.ReLU(),
            nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
            nn.ReLU(),
            nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1),  # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(12, 3, 4, stride=2, padding=1),  # [batch, 3, 32, 32]
            nn.Sigmoid()
        )

    def forward(self, x):
        encodered = self.encoder(x)
        decodered = self.decoder(encodered)
        return decodered
```

代码中的`__init__`部分分为编码和解码部分，编码使用的是一般的卷积操作，将图像不断减半，解码部分使用的是反卷积操作，将图像不断放大。PyTorch内部会帮我们计算，但是在写代码的时候，我是通过反向计算来确定怎么写stride和padding的。将input和output反过来想，针对输出图片，如何输出输入图片的情形来编写就可以了。这里因为encoder的部分直接是正向操作，所以直接照搬就OK了。

### 4 - 变分自动编码器（Variational Autoencoder）

变分自编码器（Variational Autoencoder，VAE）结合了自动编码器（Autoencoder）和概率图模型的思想，用于学习数据的潜在表示并生成新的样本。VAE 是由 Kingma 和 Welling 在 2013 年提出的。

在传统的自动编码器中，编码器将输入数据映射到潜在空间的固定点，然后解码器将潜在表示映射回重构的输入数据。与之不同的是，VAE 在编码器的输出中引入了潜在变量的概率分布，这使得编码器输出的不再是一个固定的点，而是一个**概率分布**。这个概率分布表示了输入数据在潜在空间中的分布情况，也就是**编码器对输入数据的不确定性的建模**。

具体来说，VAE 的工作流程如下：

- 编码器将输入数据 x 映射到潜在空间中的均值和方差，这些参数构成了潜在变量的高斯分布，也就是编码器的输出p(z|x)。分布之需要用均值和方差这两个参数来确定即可，所以这里就使用了参数来定义了 z 的分布。（对，因为一个分布就是由均值和方差确定的）

- 损失函数（Loss Function）：VAE 的训练目标是最大化数据的边际对数似然，即最大化给定数据 x 的情况下潜在变量 z 的后验概率p(z|x)。这等价于最小化重构损失和潜在变量的 KL 散度（KL Divergence），KL 散度用于度量编码器输出的潜在变量分布 **q(z|x)** 与先验分布 p(z) 之间的差异。

- 重参数化技巧（Reparameterization Trick）：为了能够对潜在变量进行有效的反向传播，VAE 使用重参数化技巧将随机性从潜在变量的采样过程中移除。具体来说，通过从标准正态分布 N(0, 1) 中采样随机噪声，然后利用编码器输出的均值和方差来计算潜在变量 z。 

- 解码器（Decoder）：解码器接收潜在变量 z 作为输入，并尝试从潜在空间中重构输入数据。解码器的输出表示为条件概率分布p(x|z)，它表示了给定潜在变量 z 的情况下，重构输入数据的可能性。这一部分和普通的自动编码器没什么不同。


**概率图模型**：刚刚谈到了这个概念，拿出来补充一下。

概率图模型（Probabilistic Graphical Models，PGMs）是一种用图论方法来表示随机变量之间的概率关系的模型。它是一种统计模型，用于描述随机变量之间的概率依赖关系，并且可以用于推断、估计和预测等任务。

概率图模型通常基于概率论的基本原理，将随机变量表示为图中的节点，将变量之间的依赖关系表示为图中的边。根据变量之间的关系，概率图模型分为两种主要类型：

1. **贝叶斯网络（Bayesian Networks）**：也称为有向图模型，它使用有向无环图（DAG）来表示变量之间的条件依赖关系。在贝叶斯网络中，节点表示随机变量，边表示变量之间的依赖关系，而边的方向表示了因果关系。贝叶斯网络通常用于表示因果关系明确的场景，例如疾病诊断、推荐系统等。

2. **马尔科夫网络（Markov Networks）**：也称为无向图模型，它使用无向图来表示变量之间的相关性，其中节点表示随机变量，边表示变量之间的相关性。与贝叶斯网络不同，马尔科夫网络中的边没有方向，表示的是变量之间的相关性而不是因果关系。马尔科夫网络通常用于表示变量之间的相关性较为复杂的场景，例如图像分割、社交网络分析等。

概率图模型具有直观的图形化表示，能够有效地表示和推断复杂的概率关系，因此在机器学习、人工智能和统计学等领域得到了广泛的应用。它们被用于模式识别、决策分析、数据挖掘、自然语言处理等任务，并且在大数据分析和概率推断等领域发挥着重要作用。

### 5 - 损失函数解构 ELBO

ELBO 是 Evidence Lower Bound（证据下界）的缩写。在概率模型的变分推断（Variational Inference）中，ELBO 是一种用来近似求解后验分布的方法。在变分推断中，我们通常会遇到一个难以处理的后验分布，ELBO 的出现就是为了解决这个问题。因为训练是为了变分后验分布，无限接近真正的后验分布。数学公式很复杂，暂且不在这里弄的很复杂，但是用代码表达就会比较好懂。

```python
def elbo(reconstructed, input, mu, logvar):
    """
        Args:
            `reconstructed`: The reconstructed input of size [B, C, W, H],
            `input`: The orignal input of size [B, C, W, H],
            `mu`: The mean of the Gaussian of size [N], where N is the latent dimension
            `logvar`: The log of the variance of the Gaussian of size [N], where N is the latent dimension

        Returns:
            a scalar
    """
    # 参数中，reconstructed是重构图像，input是原始输入图像，mu和logvar是潜在空间的均值和方差
    # 定义了一个二进制交叉熵损失函数
    bce_loss = nn.BCELoss(reduction='sum')
    # 计算重建损失，两个参数分别是模型重建的图像和原始输入图像
    BCE = bce_loss(reconstructed, input)
    # 计算KL散度
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD
```

这段代码实现了一个计算 ELBO（Evidence Lower Bound）的函数。KL 散度是用来度量两个概率分布之间的差异的指标。具体来说是衡量后验分布q(z∣x)和先验分布p(z)之间的差异的指标。可以描述为，通过观察x后，对z潜在变量的不确定性减少的程度。同时也可以说，是衡量了我们用后验分布近似先验分布的过程中损失了多少信息，如果完全相同，那么KL就是0，反之KL就是正数，表示了两个分布之间的差异，或者也可以说是距离。上面的代码中的KLD的计算就是它的算法。

在这里，我们希望模型学习一个潜在空间的分布，使其接近于一个给定的标准正态分布。最后返回了 ELBO 的值，即重建损失和 KL 散度的和。ELBO 是用来近似求解后验分布的一个下界，在变分推断中，我们试图最大化 ELBO 来使得变分分布接近真实后验分布。

ELBO损失函数同时包含交叉熵和KL散度两部分，这样可以**同时优化模型的重构能力和潜在表示的连续性**，从而更好地训练变分自编码器。

### 6 - 重参数化技巧 Reparameterization Trick

它是一种用于训练基于梯度的概率生成模型。是变分自编码器的常用技巧。竟然可以将随机采样操作，转化为可微分的操作，从而可以进行反向传播。

我们知道，在变分自编码器中，为了生成和原图像相似的图像，需要在潜在空间的分布中进行采样。但是直接进行采样是不可导的，没法训练。而重参数化技巧通过以下步骤实现了可导。

首先从一个固定的分布（例如标准正态分布）中采样一个固定的随机向量（通常称为噪声或者随机参数）。然后将这个随机向量通过一个确定的变换，例如乘以一个标准差并加上一个均值，来生成我们想要的潜在向量。这样，采样过程就可以被表示为一个可微分的操作，因为我们可以对均值和标准差求导。这使得我们可以使用反向传播算法来优化模型参数，而不需要考虑采样过程的不可导性。

在变分自编码器中，重参数化技巧通常应用于编码器网络，用来生成潜在变量的均值和方差，使得我们可以通过均值和方差来生成潜在向量，并且可以通过梯度下降来优化编码器网络。

以下的代码实现了该过程。

```python
import torch

seed = 172
torch.manual_seed(seed)

def reparameterize(mu, log_var):
    """
        Args:
            `mu`: mean from the encoder's latent space
            `log_var`: log variance from the encoder's latent space

        Returns:
            the reparameterized latent vector z
    """
    var = torch.exp(log_var)  # standard deviation
    eps = torch.randn_like(var)  # `randn_like` as we need the same size
    sample = mu + (eps * var)  # sampling as if coming from the input space
    return sample
```

### 7 - 变分自编码器的示例代码

使用了最基本的线性模型，可以修改成其他。尚且在研究，实际训练项目后准备再次学习一遍。

```python
import torch
from torch import nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.features = 16
        # encoder
        self.encoder1 = nn.Linear(in_features=3072, out_features=128)
        self.encoder2 = nn.Linear(in_features=128, out_features=self.features * 2)

        # decoder
        self.decoder1 = nn.Linear(in_features=self.features, out_features=128)
        self.decoder2 = nn.Linear(in_features=128, out_features=3072)

    def forward(self, x):
        # encoding
        x = F.relu(self.encoder1(x))
        x = self.encoder2(x).view(-1, 2, self.features)  # 输出形状为(batch_size, 2, self.features)

        # get mu and log_var
        mu = x[:, 0, :]  # the first feature value is mean
        log_var = x[:, 1, :]  # the second feature value is variance

        # get the latent vector through reparameterization
        z = self.reparameterize(mu, log_var)

        # decoding
        x = F.relu(self.decoder1(z))
        reconstruction = torch.sigmoid(self.decoder2(x))
        return reconstruction, mu, log_var
    
    def reparameterize(self, mu, log_var):
        # mu is the mean from the latent space, log_var is the log variance from the latent space
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        sample = mu + (eps * std)
        return sample
```
**关于编码和解码器的输入输出维度**

编码器和解码器的输入输出维度上，编码器最后乘以2，解码器输入不需要乘以2，是因为，编码器和解码器的结构是对称的，但并不是输入和输出维度一定要相同的。重点在于保持编码器和解码器之间的匹配。

在编码器部分，最后一层 `self.encoder2` 的输出维度是 `self.features * 2`。这里的 `* 2` 是因为变分自编码器中常用的技巧是将潜在空间的均值和方差分别预测出来。因此，输出的维度被设计为 `self.features * 2`，其中前一半部分代表均值，后一半部分代表方差。

在解码器部分，`self.decoder1` 的输入维度是 `self.features`。这是因为在解码器中，我们只需要用到潜在向量的表示，而不需要用到其方差信息。所以，解码器的输入维度只需要是潜在向量的维度 `self.features` 即可，而不需要考虑方差信息。因此，解码器部分的输入不需要乘以2，因为解码器只需要潜在空间的均值信息，而不需要方差信息。

**关于view**

`view` 方法的作用是将原始张量重塑为一个具有指定形状的新张量。括号中是一个元组，表示新张量的形状。新张量的元素数量必须与原始张量相同，否则会抛出错误。

在上述代码中，`x` 是经过编码器部分处理后的张量，`self.encoder2(x)` 输出的张量形状为 `(batch_size, features * 2)`，即每个样本有 `features * 2` 个特征。然后，通过 `view(-1, 2, self.features)` 方法，将这个张量重新调整为一个形状为 `(batch_size, 2, self.features)` 的张量。这里的 `-1` 表示该维度的大小会根据其他维度和原始张量的总元素数量自动推断出来，而 `2` 和 `self.features` 分别表示第二个维度和第三个维度的大小。

这样，通过 `view` 方法的调整，我们将原始的一维特征向量重新组织为一个二维的形状，这在某些情况下可以更方便地进行后续的处理。

损失函数如下。这里计算了之前所说的KL散度。

```python
def final_loss(bce_loss, mu, logvar):
    '''
    BCE: reconstruction loss
    KL-divergence: 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    params:
    bce_loss: reconstruction loss
    mu: the mean from latent vector
    logvar: log variance from latent vector
    '''
    BCE = bce_loss
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD
```

训练代码如下。

```python
import torch.optim as optim
import torch
from torch import nn

def train(model, training_data):

    optimizer = optim.Adam(model.parameters(), lr= 0.001)
    criterion = nn.BCELoss(reduction='sum')

    running_loss = 0.0

    for epoch in range(1):  # loop over dataset n times

        for i, data in enumerate(training_data, 0):
            inputs, _ = data
            inputs = inputs.view(inputs.size(0), -1)  # flatten all the other dimensions except batch size

            optimizer.zero_grad()
            reconstruction, mu, logvar = model(inputs)
            bce_loss = criterion(reconstruction, inputs)
            loss = final_loss(bce_loss, mu, logvar)

            running_loss += loss.item()
            loss.backward()
            optimizer.step()

            # print statistics
            if i % 2000 == 1999:  # print every 2000 mini batchs
                print('[{epoch + 1}, {i + 1}] loss : {running_loss / 2000 :.3f}')
            running_loss = 0.0

    PATH = './cifar_net.pth'
    torch.save(model.state_dict(), PATH)

    print('Finished Training')
```

最后是数据准备和训练。

```python
import os
import sys
import torchvision
from torchvision import transforms
cwd = os.getcwd()
# add CIFAR10 data in the environment
sys.path.append(cwd + '/../cifar10') 
from Cifar10Dataloader import CIFAR10
batch_size = 32
def load_data():
    
    # convert the images to tensor and normalized them
    transform = transforms.Compose([
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    trainset = CIFAR10(root='../cifar10',  transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=False, num_workers=1)
    return trainloader
```

最后是两行训练代码。

```python
model = VAE()
train(model, load_data())
```
