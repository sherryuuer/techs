## AWS中机器学习的相关服务和网络构架

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
- Linear Learner：线性模型，分类或着回归任务。逻辑回归类似于线性回归，不同之处在于逻辑回归生成二元输出。您无法使用逻辑回归预测数值。
- XGBoost：在XGBoost中，eta参数（也称为学习率）是控制每次迭代中模型参数更新的幅度的一个重要超参数。较小的学习率可以防止模型过度拟合训练数据。另外max_depth如果过深，也会有过拟合风险。
- Seq2seq：序列到序列处理。机器翻译，文本总结等，用RNN，CNN模型。
- DeepAR：预测时间序列数据。
- BlazingText：有监督文本分类，Word2vec文本嵌入。
- Object2Vec：高维对象投射为低维向量，和word2vec对功能一样，将对象处理为嵌入。
- ObjectDetection：图像内的物体检测。使用MXNet，Tensorflow模型。这两种模型除了物体检测，还可以进行图像分类。
- Semantic Segmentation：像素级别图像内物体检测，可以检测出图像边界。只能用GPU进行训练。
- Random Cut Forest：AWS引以为豪的算法。异常检测，用于各种服务和场景。
- Neural Topic Model：神经网络和主题建模的结合。创建主题，不是摘要。
- LDA：没有神经网络的主题建模。无监督的聚类。是一种分类算法。
- KNN：有监督的分类和回归算法。
- K-means：无监督的聚类算法。
- PCA：利用协方差矩阵进行特征分解，降维算法。
- Factorization Machine：因子分解机，有监督模型。用于处理推荐系统、回归、分类等任务。Factorization Machine 的提出是为了在高维稀疏数据上进行建模，同时考虑到特征之间的交互。比如推介系统，用户-物品的特征，采用矩阵内积的因式分解，然后得到用户对于物品的喜爱程度的排序。
- IP Insights：分析地址是否可疑。


### Sagemaker的生态服务

- Amazon SageMaker Autopilot/AutoML：似乎是我以前就用过的一个功能，只需要选择数据和目标对象，就可以进行自动的，模型选择（甚至可以将模型集成），数据预处理，调参实验，预测推理等功能。也可以进行人工指导。根据文档，从23年11月开始，这个功能集成到了Canvas功能去了。
- SageMaker Autopilot 进行的 AUC 优化创建了高质量的 ML 模型，即使是*不均衡*的类数据。
- Amazon SageMaker Canvas：使用场景是商业分析。它和studio同属于sagemaker-domin下面的tab。有完整的modeling过程的操作仪表盘：标签选择，建模，分析，预测等过程被完整提供。
- Amazon SageMaker Data Wrangler：studio中。从GUI界面简化流程，数据选择，清理，探索，可视化，大规模处理，最终可以生成一个notebook的代码。
- Amazon SageMaker Ground Truth：打标签。
- Amazon SageMaker JumpStart：从Sagemaker studio的界面进入的，pretrained自动解决方案，内置很多热门模型仓库。进行微调，部署，评估等工作。通过创建一个SageMaker Domain（是一个独立的运行环境，包含用户，权限，VPC配置等，这些实体共享一个开发环境）进行。
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
- *数据缺失*：深度学习更适合分类数据的插补。数值型数据使用KNN可以更好地满足数据插补需求。虽然简单地删除缺失数据的行或使用平均值要容易得多，但它们不会导致最好的结果。
- 降低成本：使用对数scale进行参数训练（比如学习率0.1，0.01这样），减少并发数。


### ML相关的AWS服务

**立刻上手无需代码**

- *Amazon Comprehend*：自然语言处理，文本分析，文本分类，信息提取，事件监测（SAP考试）
- *Amazon Comprehend Medical*：使用NLP分析医学text，使用的API叫做DetectPHI API（Protected Health Information）
  - 可以使用S3进行存储，可以用kinesis进行实时分析
  - 将文本笔记，变成结构化的，清晰的病例，症状，笔记，突出关键词等，看hands-on视频的时候蛮震撼的
- *Amazon Lex*：是Alexa为base的自然语言聊天机器人引擎，通过将得到的文本传递给其他的服务，进行下一步处理。他没有处理语音的能力，如果要处理语音（Transcribe）和返回语音（Polly）需要其他的服务支持。有个Amazon Lex Automated Chatbot Designer可以帮助进行设计系统。比如如何嵌入意图slot文本等。（SAP考试）
  - call center的案例可以用*AWS Connect*是一个虚拟的contact center，结合Lex，可以处理客户订单，背后结合Lambda，就可以将信息写入CRM（客户关系管理系统）
- *Amazon Polly*：文本到语音服务。lexicons（自定义词典），SSML（Speech Synthesis Markup Language）是一种用于控制文本转语音（TTS）引擎如何合成和朗读文本的标记语言）可以定制语音说法。（SAP考试）
- *Amazon Rekognition*：计算机视觉服务，物体检测，文字检测，面部检测，图片和视频流（Kinesis）都可以，打标签打框框。（SAP考试）
  - 通过设置一个*最小自信度阈值*来检测一些不好的内容：content moderation内容审核
  - 可以使用A2I进行打标签flag
- *Amazon Textract*：OCR，对文件和手写文件进行识别和分析，包括表单和数据等。（SAP考试）
- *Amazon Transcribe*：语音到文本服务。可以自动监测语言，甚至不用设置。可以定制单词表。应用领域包括电话内容分析，医学分析，实时字幕等。（SAP考试）
  - 深度学习，自动语音识别技术
  - 可以自动移除PII个人信息
  - 多语种语音识别
- *Amazon Translate*：深度学习算法进行机器翻译的服务。（SAP考试）

**其他**

- Contact Lens：电话服务中心，提取电话内容进行分析，分类，主题检测。
- *Amazon Kendra*：企业内部AI的IT支持。Alexa的妹妹。使用内部*系统文件进行系统搜索*。Knowledge Index powered by ML（SAP考试）
- *Amazon Augmented AI（A2I）*：Amazon的人工审查机器学习预测服务。构建更好的workflow。集成于Sagemaker，Textact，Rekognition等服务。（SAP考试）
- Amazon Bedrock：生成式人工智能构建，托管服务。
- AWS Trainium：AWS专门为超过 1000 亿个参数模型的深度学习训练打造的第二代机器学习 (ML) 加速器。**AWS Neuron**SDK，由编译器、运行时和分析工具组成，可以使用这些工具在 AWS Trainium 支持的 Amazon EC2 Trn1 实例上运行高性能训练。
- AWS Inferentia：是 AWS 设计的一款机器学习推理加速器，可在云中提供高性能和低成本的机器学习推理。
- Amazon Titan：Amazon Bedrock 独有的 Amazon Titan 系列模型。Amazon Titan 基础模型（FM）通过完全托管的 API 为客户提供广泛的高性能图像、多模式和文本模型选择。主要是生成式人工智能的AWS的自己训练的模型。
- AWS Panorama：是机器学习 (ML) 设备及软件开发工具包 (SDK) 的集合，可在本地互联网协议 (IP) 摄像头中集成 CV 功能。
- *Amazon Personalize*：（SAP考试）全托管实时个性化推介 ，和Amazon网站用的同一个推介系统（红豆泥？），可以直接集成你现在的系统

**Kinesis Analysis中的两种算法**

- Random Cut forest：异常检测，只允许使用最近的数据（recent history）。
- Hotspots：数据密集区域检测。

**时间序列预测**

- *Amazon Forecast*：进行时间序列数据预测。很像DeepAR功能，在引擎下使用这个模型。其他还有一些很贵的模型**DeepAR+**，**CNN-QR（QR是分位数回归）**，**Prophet**是一个非线性时间序列模型，不会很贵但是效果不错。其他还有比较便宜的**NPTS**可以处理稀疏数据，如果数据较少但是想预测季节性数据，另外少于一百个数据等时候，**ARIMA**自动回归移动平均算法，**ETS**指数平滑算法等。（SAP考试）

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
- CodeWhisperer：代码伴驾，自动代码提示，自动完成。安全扫描。历史代码追踪。偏见避免。使用TLS安全传输协议。（但是Amazon其实可以挖掘你的代码）
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
- Production Variants：SageMaker 支持将多个模型（称为生产变体）部署到单个 SageMaker 终端节点中。您可以配置生产变体，以使一小部分实时流量路由到您要验证的模型。您可以对 SageMaker 服务进行简单的调用，以收集有关模型有效性的统计数据以及更改权重。

### 考试中遇到的我觉得自己需要再查一下的东西

**OpenSearch**

AWS OpenSearch Service 是亚马逊网络服务（AWS）提供的一项托管的搜索和分析服务。它是基于开源的 Elasticsearch 和 Kibana 构建的，用于实时搜索、分析和可视化大规模数据的解决方案。

以下是 AWS OpenSearch Service 的主要特性：

1. **实时搜索和分析：** OpenSearch Service 提供了快速、实时的搜索和分析功能，能够处理大规模数据的实时查询和聚合。

2. **弹性伸缩：** OpenSearch Service 可以根据需求自动扩展和收缩，以适应数据量和查询负载的变化。

3. **高可用性：** OpenSearch Service 提供了高可用性和可靠性，支持多个可用区部署，以确保服务的稳定性和容错能力。

4. **安全性：** OpenSearch Service 提供了多种安全功能，包括身份验证、访问控制、加密传输等，以保护数据的安全性和隐私性。

5. **可视化和监控：** OpenSearch Service 集成了 Kibana 工具，提供了丰富的可视化和监控功能，帮助用户实时监控数据的状态和性能指标。

6. **托管服务：** OpenSearch Service 是一项托管的服务，AWS 负责管理基础设施和服务的运行，用户只需关注数据和查询即可，无需担心基础设施的管理和维护。

总的来说，AWS OpenSearch Service 提供了一个灵活、可扩展和高可用的搜索和分析平台，适用于各种场景，包括日志分析、实时监控、数据可视化等。

---

**残差图**

残差图（Residual Plot）是一种用于评估统计模型拟合质量的可视化工具。在回归分析中，残差是观测值与模型预测值之间的差异，残差图显示了每个观测值的残差与相应的预测值之间的关系。

在残差图中，通常将横轴表示模型的预测值，纵轴表示残差。每个点代表一个观测值，其位置表示该观测值对应的预测值和实际观测值之间的偏差。

残差图的主要用途包括：

1. **评估模型拟合质量：** 残差图可以帮助识别模型是否存在偏差或方差的问题。如果模型拟合良好，残差图中的点应该均匀分布在零线附近，而不应该出现任何明显的模式或趋势。

2. **检测异方差性：** 异方差是指残差的方差随着预测值的变化而变化。在残差图中，如果残差的方差随着预测值的增加而增加或减少，则可能存在异方差性。

3. **识别异常值和离群点：** 残差图可以帮助识别异常值和离群点，即与其他观测值相比残差较大的点。

4. **检验线性假设：** 对于线性回归模型，残差图应该显示出随机分布的点，如果出现明显的非线性模式，则可能违反了线性假设。

通过观察残差图，分析人员可以更好地了解模型的拟合情况，并采取相应的措施来改进模型或调整分析方法。

----

**SagaMaker生命周期配置（life cycle configuration）**

Amazon SageMaker 生命周期配置（Lifecycle Configuration）是一种 AWS SageMaker 服务提供的功能，它允许您在训练实例启动或终止时自动执行特定的操作。这些操作可以包括预安装软件包、配置环境变量、下载数据等。

Lifecycle Configuration 允许您在训练实例启动时自动执行一系列步骤，这些步骤可以定制化以满足您的需求，提高工作效率并减少手动干预的需要。您可以通过 SageMaker 控制台、SDK 或 AWS CLI 来创建和管理 Lifecycle Configuration。

以下是创建 SageMaker Lifecycle Configuration 的一般步骤：

1. **创建 Lifecycle Configuration 脚本：** 首先，您需要编写一个包含要在训练实例启动或终止时执行的操作的脚本。这个脚本可以是 Shell 脚本、Python 脚本或其他脚本语言。

2. **上传 Lifecycle Configuration 脚本：** 将编写好的脚本上传到 Amazon S3 存储桶中，以便后续在 SageMaker 中引用。

3. **创建 Lifecycle Configuration：** 在 SageMaker 控制台、SDK 或 AWS CLI 中创建 Lifecycle Configuration，并指定上传的脚本的 S3 路径。

4. **将 Lifecycle Configuration 关联到 SageMaker 作业：** 在创建 SageMaker 训练作业时，将之前创建的 Lifecycle Configuration 关联到作业中。

5. **启动训练作业：** 启动训练作业后，SageMaker 将在训练实例启动或终止时自动执行 Lifecycle Configuration 中指定的操作。

通过使用 Lifecycle Configuration，您可以自动化许多与训练实例相关的任务，使得整个机器学习工作流程更加高效和可靠。

---

**`GetRecord` API 和 `BatchGetRecord` API**

是 Amazon SageMaker Feature Store 提供的两种不同的 API，用于检索特征记录，它们之间的主要区别在于处理的数据量和返回结果的方式。

1. **单个记录 vs 批量记录：**
   - `GetRecord` API 用于检索单个特征记录。您需要指定要检索的特征组名称和单个记录的标识符。
   - `BatchGetRecord` API 则用于批量检索多个特征记录。您可以同时指定多个特征组名称和记录标识符，以一次性检索多个记录。

2. **返回结果的方式：**
   - `GetRecord` API 返回一个特定记录的特征值。如果您只需要检索单个记录的特征值，则可以使用此 API。
   - `BatchGetRecord` API 返回多个记录的特征值。您可以一次性检索多个记录，并获取它们的特征值。这在需要批量处理大量记录时非常有用。

3. **效率：**
   - 对于单个记录的检索，`GetRecord` API 是更有效的选择，因为它专门设计用于获取单个记录。
   - 对于批量检索多个记录，`BatchGetRecord` API 则更加高效，因为它允许您在一次 API 调用中检索多个记录，减少了网络开销和调用次数。

因此，选择使用 `GetRecord` API 还是 `BatchGetRecord` API 取决于您的具体需求。如果您只需要检索单个记录的特征值，则使用 `GetRecord` API。如果您需要同时检索多个记录的特征值，则使用 `BatchGetRecord` API。

---

**网络隔离训练**

网络隔离训练是一种将训练任务隔离在独立的网络环境中执行的方法。这样做的目的是确保训练任务对外部网络的影响最小化，并提高训练任务的安全性和稳定性。

网络隔离训练通常涉及以下几个方面：

1. **虚拟私有云（VPC）：** 在云计算环境中，您可以使用 VPC 将训练任务隔离在专用的网络环境中。这样可以确保训练任务的流量不会通过公共网络，并减少与其他网络资源的干扰。

2. **访问控制：** 通过访问控制列表（ACL）或安全组等方式，限制训练任务对外部网络资源的访问权限。这可以减少训练任务与外部网络资源之间的通信，提高训练任务的安全性和稳定性。

3. **子网隔离：** 将训练任务部署在专用的子网中，以确保训练任务与其他网络资源之间的隔离。这可以减少训练任务与其他网络资源之间的干扰，并提高训练任务的稳定性。

4. **流量控制：** 使用流量控制机制（如流量限制、流量调节等）控制训练任务对外部网络资源的访问。这可以避免训练任务对外部网络资源的过度使用，提高网络资源的可用性和稳定性。

通过采取这些网络隔离措施，您可以确保训练任务在独立的网络环境中执行，并最大程度地减少对外部网络资源的干扰，从而提高训练任务的安全性和稳定性。

---

**snappy压缩**

Snappy 压缩是一种快速数据压缩算法，旨在在保持较高压缩率的同时提供非常快的压缩和解压缩速度。它由 Google 开发，最初用于 Google 的内部系统，后来被开源并广泛应用于各种领域，特别是大数据处理和网络传输方面。

Snappy 压缩的特点包括：

1. **快速：** Snappy 压缩和解压缩速度非常快，通常比许多其他压缩算法（如Gzip）要快得多。这使得它特别适合需要快速处理大量数据的场景。

2. **低延迟：** Snappy 压缩和解压缩的延迟非常低，这意味着它可以提供快速的响应时间，适用于需要实时性能的应用程序。

3. **无损耗：** Snappy 压缩尽可能地减少对压缩率的牺牲，同时保持了相对较高的压缩率。它主要关注在速度和压缩率之间找到一个平衡点，而不是追求最大的压缩率。

Snappy 压缩通常用于各种应用场景，包括网络传输、大规模数据分析、分布式系统等，特别是在需要快速处理大量数据的环境中。然而，值得注意的是，Snappy 压缩的压缩率可能不如一些其他压缩算法（如Gzip）那么高，因此在一些对压缩率要求较高的场景中，可能需要考虑其他压缩算法。
