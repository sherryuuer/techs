## AWS中机器学习的相关服务和网络构架

---
时间戳2024年2月。

### Sagemaker的训练和部署模型

以下是一个例子。

```
[client app]
[endpoint]

[s3]
model artifacts
training data

[sagemaker]
model development/ hosting
model training

[ECR]
inference code image
training code image

```

最顶端是客户应用client app。通过endpoint和Sagemaker进行交互。

S3中存放训练数据training data，sagemaker使用ECR中的推理和训练代码，对S3中的训练数据进行训练，得到的model模型，保存在S3中。

当客户端通过端点发来请求，sagemaker会通过推理代码调用模型，进行推理，然后将结果返回给客户。

ECR中有内置的模型供使用。sagemaker则可以运行各种框架，比如Pytorch，Tensorflow等，灵活地进行训练和host。同时sagemaker可以选择各种计算资源，包括GPU，和Spark集群GPU等高级计算资源。

代码，模型，image，框架可以使用AWS的也可以是自己准备的。

**两种部署方法**

1， 持续性的endpoint，为个人提供推理服务。

2， Sagemaker batch transform 为整个数据集进行推理。

3， 其他优质选择：

    - Inference pipelines 可以进行更加复杂的处理。
    - Sagemaker Neo进行边缘部署。
    - Elastic Inference可以加速深度学习模型。
    - 自由伸缩scaling可以进行endpoint的自由增减。
    - ShadowTesting可以进行新模型的测试比较。

### 数据类型RecordIO和Protobuf

多次出现这种数据类型。

RecordIO和Protobuf都是用于序列化数据的格式，通常在机器学习中使用。

1. **RecordIO**：RecordIO是一种二进制数据格式，用于高效地存储和传输大规模数据集。它通常用于存储图像、音频、文本等类型的数据，并且能够有效地处理大规模数据，提高数据处理的效率。RecordIO格式通常与深度学习框架如MXNet等结合使用。

2. **Protobuf**（Protocol Buffers）：Protobuf是一种轻量级、高效的序列化数据格式，由Google开发。它使用简单的接口描述语言来定义数据结构，然后通过编译器生成相应的代码，用于在不同的平台和编程语言中进行数据的序列化和反序列化操作。Protobuf在机器学习中常用于定义模型结构、数据格式等，因为它具有高效的序列化和反序列化速度，能够有效地处理大规模数据。

这两种格式在机器学习中比较常见的原因包括：
- **效率**：它们都是二进制格式，相比于文本格式（如JSON、XML等），更加紧凑和高效，能够减少数据传输和存储的开销。
- **跨平台和语言支持**：它们都提供了跨平台和跨语言的支持，能够在不同的系统和编程语言中进行数据的序列化和反序列化，使得数据交换更加灵活和便捷。
- **数据结构定义**：它们都支持定义复杂的数据结构，能够满足机器学习中多样化的数据表示需求，包括模型参数、训练数据等。

总的来说，RecordIO和Protobuf都是为了解决大规模数据处理和跨平台数据交换而设计的，因此在机器学习中得到了广泛的应用。

### Sagemaker的build-in models

- 最好参考最新的[官方内置模型文档](https://docs.aws.amazon.com/sagemaker/latest/dg/algorithms-choose.html)内有全部的模型和说明。
- Linear Learner：线性模型，分类或着回归任务。
- XGBoost：在XGBoost中，eta参数（也称为学习率）是控制每次迭代中模型参数更新的幅度的一个重要超参数。较小的学习率可以防止模型过度拟合训练数据。另外max_depth如果过深，也会有过拟合风险。
- Seq2seq：序列到序列处理。机器翻译，文本总结等，用RNN，CNN模型。
- DeepAR：预测时间序列数据。
- BlazingText：有监督文本分类，Word2vec文本嵌入。
- Object2Vec：高维对象投射为低维向量，和word2vec对功能一样，将对象处理为嵌入。
- ObjectDetection：图像内的物体检测。使用MXNet，Tensorflow模型。这两种模型除了物体检测，还可以进行图像分类。
- Semantic Segmentation：像素级别图像内物体检测，可以检测出图像边界。只能用GPU进行训练。
- Random Cut Forest：AWS引以为豪的算法。异常检测，用于各种服务和场景。
- Neural Topic Model：神经网络和主题建模的结合。
- LDA：没有神经网络的主题建模。无监督的聚类。
- KNN：有监督的分类和回归算法。
- K-means：无监督的聚类算法。
- PCA：利用协方差矩阵进行特征分解，降维算法。
- Factorization Machine：因子分解机，用于处理推荐系统、回归、分类等任务。Factorization Machine 的提出是为了在高维稀疏数据上进行建模，同时考虑到特征之间的交互。比如推介系统，用户-物品的特征，采用矩阵内积的因式分解，然后得到用户对于物品的喜爱程度的排序。
- IP Insights：分析地址是否可疑。


### Sagemaker的生态服务

- Amazon SageMaker Autopilot/AutoML：似乎是我以前就用过的一个功能，只需要选择数据和目标对象，就可以进行自动的，模型选择（甚至可以将模型集成），数据预处理，调参实验，预测推理等功能。也可以进行人工指导。根据文档，从23年11月开始，这个功能集成到了Canvas功能去了。
- Amazon SageMaker Canvas：使用场景是商业分析。它和studio同属于sagemaker-domin下面的tab。有完整的modeling过程的操作仪表盘：标签选择，建模，分析，预测等过程被完整提供。
- Amazon SageMaker Data Wrangler：studio中。从GUI界面简化流程，数据选择，清理，探索，可视化，大规模处理，最终可以生成一个notebook的代码。
- Amazon SageMaker Ground Truth：打标签。
- Amazon SageMaker JumpStart：从Sagemaker studio的界面进入的，pretrained自动解决方案，内置很多热门模型仓库。进行微调，部署，评估等工作。
- Amazon SageMaker Pipelines
- Amazon SageMaker Studio：可视化的IDE机器学习集成开发环境。方便和团队进行协作开发，还可以方便地切换坐落于AWS上的硬件。
- Amazon SageMaker Experiments：查找，比较，整合，组织，在SageMaker中进行的各种ML实验。
- Amazon SageMaker Debugger：集成于Studio的一个插件，可以回顾过去的梯度和张量，以发现问题，可以将数据集成到一个仪表盘，适用于很多框架。这让我想到了Tensorboard，感觉功能是很相似的。
- Amazon SageMaker Studio Lab  
- Automatic Model Tuning：边调整参数边学习的智能调优功能。
- Apache Spark with SageMaker：将spark在大数据上的pre-processing能力和Sagemaker的机器学习能力相结合。使用Sagemaker-spark-library和SagemakerEstimator中的内置模型（K-means,PCA,XGBoost）可以更快速和方便的进行模型训练，并且可以使用Sagemaker中的很多其他功能，比如参数调优。
- Model Monitoring：Sagemaker的模型监控，监控数据偏差，异常值等，以及新的特征，集成SageMaker Clarify可以监控是否有数据偏差，并且和CloudWatch集成，进行SNS通知。
- Deployment Guardrails and Shadow tests：实时推理端点的蓝绿测试，自动rollback功能。一次性部署，小部分部署和监控，以及线性部署。以及可以进行在生产环境运行测试代码的影子测试。
- Pre-trained Bias Metrics in Clarify：内置的关于偏见评估的预训练的指标。
- Amazon SageMaker Training Compiler：对模型进行编译和优化，以在GPU上更快地训练。
- Amazon SageMaker Feature Store：存储，处理，共享模型特征，可以来自streaming，或者batch。
- Amazon SageMaker ML Lineage Tracking：跟踪，存储，复制，控制工作流（从数据准备到模型发布）的信息。形成图结构。

### 关键概念

- 正规化L1（降维），L2（特征权重化）
- 数据打乱shuffle
- 批大小batch-size
- 学习率
- 过拟合和如何防止
- Dropout
- Earlystopping
- 激活函数，优化方法，损失函数


### ML相关的AWS服务

**立刻上手无需代码**

- Amazon Comprehend：自然语言处理，文本分析，文本分类，信息提取，事件监测，
- Amazon Lex：是Alexa为base的自然语言聊天机器人引擎，通过将得到的文本传递给其他的服务，进行下一步处理。他没有处理语音的能力，如果要处理语音（Transcribe）和返回语音（Polly）需要其他的服务支持。有个Amazon Lex Automated Chatbot Designer可以帮助进行设计系统。比如如何嵌入意图slot文本等。
- Amazon Polly：文本到语音服务。lexicons，SSML可以定制语音说法。
- Amazon Rekognition：计算机视觉服务，物体检测，面部检测，图片和视频流（Kinesis）都可以。
- Amazon Textract：OCR，对文件和手写文件进行识别和分析，包括表单和数据等。
- Amazon Transcribe：语音到文本服务。可以自动监测语言，甚至不用设置。可以定制单词表。应用领域包括电话内容分析，医学分析，实时字幕等。
- Amazon Translate：深度学习算法进行机器翻译的服务。

**其他**

- Contact Lens：电话服务中心，提取电话内容进行分析，分类，主题检测。
- Amazon Kendra：企业内部AI的IT支持。Alexa的妹妹。使用内部系统文件进行系统搜索。
- Amazon Augmented AI（A2I）：Amazon的人工审查机器学习预测服务。构建更好的workflow。集成于Sagemaker，Textact，Rekognition等服务。
- Amazon Bedrock：生成式人工智能构建，托管服务。
- AWS Trainium：AWS专门为超过 1000 亿个参数模型的深度学习训练打造的第二代机器学习 (ML) 加速器。**AWS Neuron**SDK，由编译器、运行时和分析工具组成，可以使用这些工具在 AWS Trainium 支持的 Amazon EC2 Trn1 实例上运行高性能训练。
- AWS Inferentia：是 AWS 设计的一款机器学习推理加速器，可在云中提供高性能和低成本的机器学习推理。
- Amazon Titan：Amazon Bedrock 独有的 Amazon Titan 系列模型。Amazon Titan 基础模型（FM）通过完全托管的 API 为客户提供广泛的高性能图像、多模式和文本模型选择。主要是生成式人工智能的AWS的自己训练的模型。

**时间序列预测**

- Amazon Forecast：进行时间序列数据预测。很想DeepAR功能，在引擎下使用这个模型。其他还有一些很贵的模型**DeepAR+**，**CNN-QR（QR是分位数回归）**，**Prophet**是一个非线性时间序列模型，不会很贵但是效果不错。其他还有比较便宜的**NPTS**可以处理稀疏数据，如果数据较少但是想预测季节性数据，另外少于一百个数据等时候，**ARIMA**自动回归移动平均算法，**ETS**指数平滑算法等。

**教育娱乐**

- Amazon DeepRacer：强化学习赛车比赛。 
- Amazon DeepLens：机器学习研究用的摄像机。
- AWS Composer：AI驱动的键盘，教育使用。
- AWS Panorama：计算机视觉边缘，将计算机视觉集成在你的IP的照相机上。

**监控检测服务**

- Amazon Lookout：设备故障检测。
- Amaozn Monitron：端到端的系统监控服务。
- Amazon Fraud Detector：欺诈检测。评估新账户的风险。

**SDK**

- AWS Neuron：SDK，为了在AWS的推理芯片上优化你的模型。和Sagemaker集成。

**代码服务**

- Amazon CodeGuru：自动代码审查。（Java，Python）
- CodeWhisperer：代码伴驾，自动代码提示，自动完成。
- Amazon SageMaker Data Wrangler：也算是一个自动生成notebook代码的服务。

**部署：可扩展，加速，可靠，安全**

- SageMaker + Docker：部署主要依靠ECR中的Docker文件，也可以使用自己的镜像通过`sagemaker.estimator`使用扩张机能。
- SageMaker Neo + IoT Greengrass：Neo将推理代码进行编译。Greengrass则将代码送到edge设备中，构建，部署，管理设备软件。比如自动汽车。
- SageMaker 安全：在存储和传输中使用各种传统的加密。IAM权限管理。VPC联合使用的时候注意，SageMaker默认联网，所以需要设置和S3的endpoint，PrivateLink，以及Nat开通outbound。Logging，Monitor可以使用CloudWatch，CloudTrail服务。
- SageMaker 资源管理：GPU有助于加速训练（P3，g4dn），推理使用C5等。训练为了省钱可以用Spot instance，但是会被打断，使用checkpoints将中途训练保存在S3中是有必要的。
- Elastic Inference：深度学习框架推理加速器可以和CPU一起使用，但是现在被废止了。
- 灵活部署：推理节点自动伸缩功能（Automatic Scaling）和CloudWatch一起工作，决定节点的增减，可以设定最大最小capacity。可以多instance多AZ部署推理节点。另外还有用多少花多少的 Serverless Inference 服务。
- SageMaker Inference Recommender：如果你不知道用哪种instance type，它可以通过load test推介使用哪种instance type。（Inference & Endpoint （custom load test） Recommendations）
- SageMaker Inference Pipelines：可以将多个Inference Containers串联起来形成一个pipeline。

