## 概览

云的基础是虚拟化：服务器，存储，网络。服务器是远程计算机的逻辑分区。存储是物理硬盘的逻辑划分。网络则是虚拟私有云。

谷歌是唯一一个拥有全球私有基础设施的公司；他们的谷歌云基础设施没有任何一部分通过公共互联网。换句话说，谷歌的云基础设施完全建立在他们自己的私有网络上，而不是依赖于公共互联网的网络基础设施。

谷歌以其为其服务开发革命性技术而闻名，一旦他们有了更好的版本，旧版本就会向全世界开源。

TensorFlow就是这些技术之一。 TensorFlow 是 Google 的 AI 引擎，也是 Google 搜索的核心。现在最新的是kubeFlow！

**安全的全球专用网络**：Google Cloud 是一个大型专用网络；没有任何东西通过公共互联网传播。使用 Cloud CDN 时，来自源的流量会穿过此专有光纤网络，并通过最靠近客户的位置进行传送。大多数研究公司选择 GCP，因为没有网络连接到互联网。

**Kubernetes 引擎的创建者**：Kubernetes 是由 Google 设计和创建的容器编排工具，用于管理其基础设施。部署了数百万个容器来托管 Google 服务，而 Kubernetes 是这些容器的核心。 Google 开源了这项技术，并提供了 Google 管理的 Kubernetes 引擎作为服务。

收费方式按秒计费，新用户前三个月有300美刀信用额度。

**人工智能和机器学习服务**：毫无疑问，谷歌是数据之王。数据越多，模型在人工智能世界中产生的结果就越准确。谷歌的机器学习服务是最好的。 Google 提供了多种 ML 服务来训练模型并处理模型训练管道。Kubeflow现在是 ML 操作的趋势。同样，KubeFlow 基于 Google 内部方法将 TensorFlow 模型部署到 Kubernetes，称为 TensorFlow Extended。

超好的官方资源[codelabs](https://codelabs.developers.google.com/)

### ALL of them are APIs

所有的资源服务都是一种API。从API&Service中就可以查看所有的API。有些服务在创建项目的时候就已经激活，有些则需要在使用的时候手动激活。

API Explorer可以轻松从网页尝试调试API（就算没激活都可以试用，厉害）。一般来说如果 API 请求失败，要么是没设置账户，要么是该服务还没被你激活。

Cloud Operations：Monitoring，Dashboard，Metrics Explorer，Alerting，通过group可以整合资源管理，通过setting可以增加pj和account统一管理。其他还有logging，error report，trace服务。

### 计费 Billing

**计费账户**:可以为每个部门单位创建计费账户，或者别的什么单位。可以在项目Project上（三个点）设置它的计费账户。或者可以在新建项目的时候选择收费账户。这样就可以查看每个project的总费用。

操作需要的权限：（Billing Administrator role）或者 （Project Creator and Billing Account User）

Command:`gcloud alpha/beta billing accounts list`

注意：一开始的300美元积分，仅限第一个积分账户。最好把测试用的PJ都关联到第一个计费账户来使用。

**预算和警报**：预算和警报，是以计费账户为单位的。通过阈值在到达指定金额时，发送提醒邮件给我们，好让我们采取行动。

**BillingExport**：计费输出将费用数据输出到BigQuery。需要激活Bigquery Data transfer API。

如果你是发票结算账户，付款需要联系GCP销售团队，可以电汇或支票结算，每月将收到发票。

价格计算服务：**Pricing Calculator**：keyword：updated prices, latest prices, cost estimation

### 和谷歌云的交互方式

- 通过控制台交互：Google Cloud Console
- 通过命令行交互：Cloud SDK and Cloud Shell
- 通过IOS和Android系统：Cloud Mobile App
- 通过定制化应用（Custom Applications）交互：REST-based API

- Cloud Shell 本质上是在 Google 的基础设施上运行的一个轻量级虚拟机（通常在 Google Kubernetes Engine 上运行的一个 Pod），这使得它能够迅速启动和关闭，同时提供一个完整的 Linux 环境。

## Security & Management

### Resource Manager

整个GCP层级如下：组织（如果你的账户域名是组织类型比如google workspace账号的话）- 文件夹folder - 项目projects - 资源resource（gmail账号是无组织的）

### Data Catalog

- 将BQ，GCS，Pub/Sub，Dataproc Metastore等数据集成

### IAM

basic role，predefined role，custom role

IAM对GCS管理包括均一管理也就是只使用IAM，和细管理，也就是同时使用IAM和ACL的管理。

IAM包括四种账户：Google Account, Service Account, Google Groups, Cloud Identity Domain。

IAM Policy：用json方式写的权限控制方式，成员，角色，资源，以及按需条件。

### VPCSC

- 基于IP的访问控制。

### 数据加密多层防御

- 认证：Cloud Identity - Google workspace - MFA - password
- 认可：IAM，ACL，columus level control（BQ）- VPCSC - network访问控制
- table加密：认可UDF（用户定义函数）加密
- storage加密：Cloud KMS
- CLoud DLP：Data Loss Prevention 数据损失防止服务

### Cloud Operations suite

### Monitoring

- 监控dashboard，监控，通知，alarts。监察日志。
- 结合Pub/Sub的filter可以只输出想要的内容
- Custom Metric（网络连接，diskIO之类的元数据）收集，需要安装特别的agent，比如OpenCensus开源软件

### Logging

- Cloud Operations：Monitoring，Dashboard，Metrics Explorer，Alerting，通过group可以整合资源管理，通过setting可以增加pj和account统一管理。其他还有logging，error report，trace服务。这里讲Logging。
- 通过Cloud Logging API和Log Router存储和整合。
- Log viewer 功能可以query log
- Logs dashboard
- Logs-based metrics
- Log Storage：50GB/项目免费配额，审计日志存储在单独的存储桶中，最长保留期为 400 天。对于其余日志，保留期限为 30 天。可以为此创建警报，以便一旦超过 50 GB 就用排除项目删除不必要的日志。
- 可以输出到GCS作为担保性的archive，*如果需要是随时可以查看的形式，可以输出到BQ*

- 日志流处理构架：
  Logging -> Pub/Sub -> Dataflow -> Bigquery

- **Security Command Center**：日志异常检测
  - Security Health Analytics
  - Event Threat Detection：使用了机器学习以及基于规则的异常检测

- Cloud Audit Logs：记录特定的user，在什么服务，什么时间，做了什么的日志服务
  - 管理活动监察日志
  - 数据操作监察日志
  - 系统活动监察日志：system event log，资源的构成变化
  - Policy拒否监察日志：违反了安全policy的拒否日志

### Trace

- 追踪端点延迟，检测瓶颈问题
- 对象服务：
  - Compute Engine
  - GKE
  - APP Engine
  - Cloud Run
- codelabs：https://codelabs.developers.google.com/codelabs/cloud-function-logs-traces#6

### Profiler

- Cloud Profiler 是一种统计性、低开销的分析器，可以持续从生产应用程序收集 CPU 使用情况和内存分配信息。
- 根据不同的编程语言支持不同的分析内容，似乎Go的最多
- codelabs：https://codelabs.developers.google.com/codelabs/cloud-profiler#0， 这中间有很多模拟程序，很有趣

### Debugger

实时调试云应用程序，无需重新启动或停止

### Error Reporting

自动收集和分析错误和异常


## Compute

### Comparsion

- 从操作量来说（从多到少）：Compute Engine - Kubernetes Engine - (Cloud Run / Cloud Functions / App Engine)
- 从workload包罗范围来（从多到少）：Compute Engine - (Kubernetes Engine / Cloud Run) - Cloud Function - App Engine
- IaS：
  - Compute Engine：完全控制OS系统，无需重写代码，自定义虚拟机
  - GKE：不依赖特定OS系统，速度和可操作性强，容器化生产环境，分布式系统
- PaS：
  - App Engine：专注写代码，开发速度快，最小化操作，*适合*网站，应用，游戏后段，IoT应用
  - Cloud Run：可扩展，用多少花多少，支持API端点开发，代码迁移（portable）方便*适合*网站构架和微服务
  - Cloud Functions：事件驱动工作流，可扩展，用多少花多少，*适合*统计分析，图像标签处理，发送消息


### App Engine

- 网络应用，手机后端，特定语言Python，HTTP only，专注快速开发，谷歌托管Docker。
- 是一个托管代码的平台，Platform as service。不涉及底层系统。

### Cloud Functions

- 事件触发，语言：Javascript，Go，Python，Java，不关心操作系统，时间执行540秒之内。因为它是事件触发所以对于workload（各种计算任务）更有用。
- --trigger-event google.storage.object.finalize：Finalize（敲定）event trigger when a write to Cloud Storage is complete

### GKE

- 容器化服务，需要定义集群，可移植性，容器为部署单元。
- 容器，所以不涉及底层OS系统。


### Compute Engine

- 可以自己*控制OS操作系统*的服务。
- 创建时候的设置内容：
  - Preemptive VMs：抢占式虚拟机，24小时存活，适合处理批量作用，很像AWS的Spot，在创建regular Engine的时候选择它即可。便宜80%。
  - Predefined VMs：Standard, memory-optimized, compute-optimized
  - Instance:ssh for Linux, RDP protocol for windows.
  - 要素：Image, Snapshot, Metadata(hostname, instance id, startup&shutdown scripts, custom metadata, service accounts info)
  - 新建一个GCE可以从public image,custom image, snapshot, 或者任何可以启动的disk来创建。可以设置SA账号，并设置该服务器可以access的其他API。还可以设置防火墙firewall（http或者https访问）。
  - Startup script：这个相当于AWS的user data，是在启动的时候执行的命令。
- N2 is a balanced machine type, which is recommended for medium-large databases.
- 管理Instance集群，创建template，进行更新操作，使用PROACTIVE模式可以进行一次一个instance的rolling更新。

### Auto Scaling

- Predictive autoscaling：针对instance group进行预测性的扩展，一般在组创建三天后生效，因为需要预测的base data。
- Cool down period：从启动到可用的时间。
- Minimum&Maximum number of instances：和AWS一样，是扩展的范围。
- Auto healing：健康检查功能，检测如果发现异常，会重建实例的功能，如果不开健康检查功能，那么只有在实例不跑了的时候才会重建。
- Rolling updata/restart/replace：滚动更新，有助于重新启动或替换组中的实例。可能需要重新启动或类似的维护补丁。策略：maximum surge（最大的更新速率或数量）/maximum unavailable（重启更新时候的最大不可用数量）。

### Google Kubernetes Engine/Cluster

- 术语：
  - *Pod* 是一个集合，里面可以有多个容器，共享一个PodIP，一个Pod就是一个application的copy。
  - *Node* 是一个虚拟机实例，一个Node可以有一个或者多个Pod，docker和kubernetes预装在这些虚拟机里。
  - *Services* 是endpoint for user，PodIP在重启的时候IP会变，但是service可以提供不变的服务端点。
  - *Deployment* 翻译为部署，每个pod都是容器构成的，一个容器其实就是一段代码，部署就是这些代码的复制，它的作用就是确保pod维持需要数量的运行。
  - *Deamonset*：Node节点级别的控制，确保软件的copy在每个节点运行。
  - *Secrets*：运行容器的时候使用的敏感数据，作为环境变量存储。runtime时候使用。现在只能用kubecli CLI设置
  - *Configmaps*：非敏感数据的环境变量。在runtime时候使用。现在只能用kubecli CLI设置

- 工作方式：
  - 通过deployment.yaml文件定义pod的config以及要配置的数量。kube会据此创建实例，并在数量不够的时候进行替换。
  - 如果我们需要用Service暴露实例，每次更新实例，服务的endpoint都会映射到新的podIP。

- 创建GKE的三个部分是：Cluster basics, Node pools, Cluster（automation, networking, security（可以设置SA）, metadata, features）
- GKE mode: GKE has to modes to choose from: autopilot mode and standard mode. Autopilot is fully-provisioned and managed. *Autopilot* clusters are regional managed at the pod level. You are charged according to the resources pods use as you deploy and replicate them based on the pod spec. *Standard* mode provides you flexibility to define and manage the cluster structure yourself. *Standard* clusters can be zonal.
- Nodes run containers. Nodes are VMs (in GKE they're Compute Engine instances).

- 对于Workload可以进行设置和deploy。设置内容：application name, namespace, labels, cluster。这里namespace很重要，他是区分不同code的方式，比如一个node可以有很多版本的code，这里就是通过namespace区分的。

- Service是attach pod也就是组织pod是，以及整合他们的IP为一个endpoint给用户，并且可以进行load balancer。通过对应用的*expose*进行设置。我认为这是一个设置对外端口的步骤，所以用暴露这个单词。
  - IP组织方式：Load Balancer IP（对外开放的接口）--> Cluster IP --> NodeIP

### Cloud Run

- 事件驱动。web服务restAPI后端。轻量级数据转换。使用webhooks的业务工作流程。
- image可以来自Container Registry。
- 部署image后会有一个端口URL，通过requests请求就可以执行在image中部署的代码了。

- 托管的GKE服务，用法和GKE基本相似除了：集群由谷歌管理，工作负载容器化，docker image，便宜，之需要关心应用的高可用性即可。
- 正因为它只是容器，所以不涉及对OS层级的管理。而是交给谷歌管理。
- 根据requests进行scaling
- 可以有长达一小时的timeout请求

### APP Engine

- HTTP/HTTPS应用程序开发平台。web开发和移动后端开发。
- 每个项目只能有一个APP Engine。
- 版本控制
- `split traffic`可以进行canary test。
- 可以缓存内置静态资源。
- 可以开启本地开发服务器，一个命令即可。
- 数据库的首选是NoSQL数据库Firestore。

codelabs：https://codelabs.developers.google.com/codelabs/cloud-app-engine-python3#0

## Storage

### Comparsion

- 关系型 SQL
  - Cloud SQL：适合网络构架，比如商务网站，CMS（内容管理系统）
  - Cloud Spanner：关系型数据仓库，高可用可扩展性
- 非关系型 NoSQL
  - Firestore：document数据库，层级管理，手机，网络，用户profiles，游戏状态数据
  - Cloud Bigtable：以极低的延迟存储大量单键数据。比如IoT数据，动态实时查询，时间序列，图数据
- 对象存储 Object
  - GCS：二进制或对象数据，图像视频，备份，静态服务器
- 数据仓库
  - BigQuery：企业级数据仓库，分析和仪表盘

### Google Cloud Storage

- 根据使用频率可以分几个classes：
  - Multi-regional - Part of Standard now
  - Regional - Part of Standard now
  - Nearline：30天/访问频率
  - Coldline：90天/访问频率
  - Archive：一年/访问频率

- 可托管静态网页，和S3一样。
- 可署名URL
  - 使用Service Account的私钥来为生成的 URL 生成加密签名
  - `gcloud auth activate-service-account --key-file [path/to/key_file.json]`
  - 需要的包：`pip3 install pyopenssl`
  - 创建url：`gsutil signurl -d 10m -u gs://[bucket_name]/demo.txt`
- 创建bucket的时候不指定location，就会默认美国区使用。

### Bigtable

- NoSQL
- 每行中的单个值都被索引，该值称为行键。
- Cloud Bigtable 非常适合以极低的延迟存储大量单键数据。比如*IoT*数据，动态实时查询，时间序列，图数据。
- 它支持低延迟的高读写吞吐量，是MapReduce操作的理想数据源。

codelabs：https://codelabs.developers.google.com/codelabs/cloud-bigtable-intro-java#0

### BigQuery

- 经济高效的现代数据云仓库。关键词：分析，历史数据，SQL语法
- BigQuery Data Transfer Service支持从 Google SaaS 应用（Google Ads、Cloud Storage）、Amazon S3 和其他数据仓库（Teradata、Redshift）将数据传输到 BigQuery。
- 可以使用 Cloud Dataflow pipeline、Cloud Dataproc jobs或直接使用 BigQuery 流提取 API 将流数据（例如日志或 IoT 设备数据）写入 BigQuery。
- `bq query --use_legacy_sql=false --dry_run 'SELECT * FROM bigquery-public-data.stackoverflow.posts_answers LIMIT 1000'`其中的`dry_run`可以提示计算成本。
- Partitioning and Clustering（提高查询效率，降低成本）：分区是将数据分割成较小的独立单元，以提高性能和可扩展性，而聚类是将相关数据放在一起以提高查询性能和减少磁盘 I/O 操作。分区通常是水平的，而聚类则是垂直的。水平分区是按行分割数据，减少单个存储单元上的数据量，而垂直分区是按列分割数据，将相关的数据物理上放置在一起。
- Load data to BQ 的方式：
  - Data transfer是一种最简单的方式。
  - bq command and cron it
  - BigQuery connectors for big data products such as Spark or Hadoop
  - Cloud Composer, a Google Cloud managed version of Apache Airflow
  - Real-time use cases you can stream data into BigQuery using the Streaming API
  - Dataflow and [Apache Beam](https://www.cnblogs.com/zlslch/p/7609417.html)：A possible use-case for this is to trigger a Cloud Function when an event happens. The Cloud Function could contain the logic to start an Apache Beam pipeline using a Dataflow runner that would execute transformations required and then save your data into BigQuery when it is done.

### Cloud SQL

- 对标传统关系型数据库MySQL，PostgreSQL，SQLServer，提供这些数据库的托管服务
- 使用 CloudSQL 作为 Drupal 或 WordPress 等平台的 CMS（内容管理服务）后端
- Cloud SQL 非常适合轻松入门或提升和迁移现有 SQL 数据库。
- 不支持user-defined functions。
- 但对于现代云数据库，Cloud SQL 存在一些局限性。诸如水平扩展、区域方面的全球可用性等限制。 GCP 的 Cloud Spanner 服务解决了这些限制，并为解决方案提供了无需停机即可水平扩展的能力。总体而言，CloudSQL 的常见用例是将 SQL 数据库从本地提升并转移到云端。

### Cloud Spanner

- Modern cloud-based RDBMS-现代的基于云的关系型数据库。
- 计算和存储分离，跨区复制的高可用性。（可以一个region多个zone，或者多个region）
- 跨区域复制以提供高可用性。目前最多可对 4 个区域进行复制。
- 自动分片sharding（水平分割）
- 高可用性、全球范围内的强一致性、RDBMS和水平扩展。无需管理高可用性副本和只读副本。

### Cloud Firestore/Firebase

- 自动扩展、高性能和易于应用程序开发而构建的 NoSQL 数据库。
- Firestore更像是document数据库，collection group适合应用开发后台数据库
- ACID原子属性交易事物，高读写操作。
- 关键词：key-value pair、NoSQL database、之类的关键字时App Engine or app，SQL like query language
- 独特术语：
  - Kind - > Table
  - Entity - > row
  - Property - > Column
  - Key - > Primary key
- Firebase 存储在幕后使用storage bucket。所以，它是带有移动SDK（移动软件开发工具包）的云存储。

## Networking

### VPC

- 谷歌的VPC是全球资源。子网是区域资源。VPC就像是谷歌里的一个大城市，子网就像是街区，里面的instance就像是大楼。
- VPC设置的IP分配有自动模式和自定义模型。
- mode：auto & custom：auto是预定义了一个region一个subnet，预定义了IP range。新的region增加，会自动增加新的subnet。
- 可以implement Cloud VPN tunnels 来和本地通信。

### Load Balancer

- 三种类型：HTTPS Load Balancer（layer7），TCP Load Balancer（layer4），UDP Load Balancer.
- 根据流量来源，来决定是内部internal还是外部external的HTTPSLB
  - 外部包括：https，SSL，TCP
  - 内部包括：TCP/UDP，http(s)，network pass-through（网络直通cool）
  - http(s)是layer7，其他的都是layer4 of OSI model
- TCP-LB提供单区域或多区域的LB
  - 要求SSL offload的情况使用SSL Proxy
  - 不要求SSL但是需要全球traffic或者IPv6的情况使用TCP Proxy

![gcp_load_balancer](gcp_lb.png)

### Cloud DNS

- 低延迟，高可用性
- 域名系统
- public&private

## Data Pipeline

### Dataproc

- Spark分布式计算
- 托管的Hadoop和Spark服务，专门用于在Google Cloud Platform (GCP) 上进行批处理、查询和流式处理数据。
- 它让用户能够快速简便地运行**分布式计算工作负载**，同时避免复杂的集群管理。
- 使用Ephemeral Clusters（临时集群），只在需要的时候运作，可伸缩性，灵活性，成本效应
- 支持Apache Hadoop生态系统中的多种开源工具，如Spark、Hive、Pig 和 MapReduce。
- 支持通过 Initialization Actions 和 Preemptible VMs 来自定义集群的配置和优化成本。集群任务可以按需启动和停止，这使得资源使用更加高效。
- 利用 Spark MLlib 或集成 TensorFlow 进行分布式机器学习训练和预测。
- 适合需要处理大量数据的应用，如**日志处理、数据湖管理、实时数据流处理、复杂数据转换和大数据分析**等。
- 创建datalake的很好选择（各种形式的大数据存储和分析）
- DataprocHub：JupyterHub

### Composer

- 托管的 Apache Airflow 服务，专注于工作流编排和任务调度。
- 支持自动化跨多个系统和服务的任务，如数据传输、API调用、数据处理任务等。
- 通过重试策略、任务依赖和优先级管理来确保工作流的可靠执行。
- 适合需要编排和自动化多个步骤或系统之间任务的应用，如定期数据导入、跨服务的ETL流程、数据管道管理、复杂的数据转换和加载过程等。

### 两个服务的区别和结合

**目标和用途:**

- Dataproc 专注于大规模数据处理和分析，适合需要运行分布式计算作业的场景。
- Composer 主要用于工作流编排和任务自动化，适合需要管理复杂任务链和调度多个系统之间的任务的场景。

**核心技术:**

- Dataproc 基于 Apache Hadoop 和 Spark 技术，处理大规模数据集。
- Composer 基于 Apache Airflow，用于编排和管理工作流。

**典型使用场景:**

- Dataproc 常用于处理需要大量计算资源的批处理作业、实时数据处理和大规模数据分析。
- Composer 通常用于管理和自动化跨系统的数据流和任务调度。

**与其他 GCP 服务的集成:**

- Dataproc 常与 Cloud Storage、BigQuery 等数据存储和处理服务结合使用。
- Composer 通常用于协调跨 GCP 服务的工作流，包括 Dataflow、BigQuery、Cloud Functions 等。

**一个使用流程的案例：**

- 数据收集和预处理:使用 Cloud Composer 编排任务，从多个数据源（如云存储、数据库、API等）收集日志数据。调用 Dataproc 集群来预处理和清洗这些数据，可能需要进行一些转换和格式化。
- 数据分析和存储:通过 Dataproc 运行复杂的分析作业，如Spark或Hive查询，将处理后的数据存储在 BigQuery 或 Cloud Storage 中。使用 Cloud Composer 调度这些分析作业，以便它们按计划或按需运行。
- 报告和通知:使用 Cloud Composer 创建和调度将分析结果导出并发送到报告系统或通知团队的任务。自动化报告生成和发送的工作流管理。

## Event trigger

### Cloud Functions

- 最高memory：4GB
- 最长执行time：9minites
- 支持语言：Python，Java，Go，Node.js
- usecase：文件处理，视频处理，小型微服务移动后端
- codelabs：https://codelabs.developers.google.com/codelabs/cloud-starting-cloudfunctions#0

### Cloud PubSub

- 可以将 PubSub 视为 Apache Kafka 的替代方案，后者是开源替代方案。
- Publisher > Topic > Message & Storage > Subscription > Subscriber > ACK(应答后会删除message)
- 如果长时间收不到应答就会多次发送message，要注意
- delivery方式：push / pull / BQ Subscription(can load table to BQ)/ GCS Subscription
  - *pull*功能的意思是，client进行pull的请求，subscriber从topic进行拉取，并将message和ackid返回给client，所以pull的不是client，而是subscriber，作为client的我们，还是需要请求，并被推送，还需要返回ack消息
  - *push*是另一种机制，client要创建Endpoint服务，subscriber会不断的从Topic进行拉取，然后不断的对你的Endpoint进行Http Post，并期望得到2xx响应的机制！
  - 这在比如实时应用，视频网站发布上很有用。比如和CloudRun联动，使用其Endpiont进行视频的实时推送。
- ACK功能要手动开启
- 关键词：Capture Streaming data、Pubsub、Decoupled（解藕），Asynchronous application architecture（异步应用构架）
- CloudShell：装载了开发工具的虚拟机(list account: `gcloud auth list`;list pjs: `gcloud config list project`)
- Pull message by cloud shell: `gcloud pubsub subscriptions pull --auto-ack MySub_name`
- Enable bucket object upload topic: `gsutil notification create -t MyTopic -f json gs://bucket_name`
  - [Link for gsutil tools of GCS](https://cloud.google.com/storage/docs/gsutil/commands/notification)
  - [Topic filter](https://cloud.google.com/pubsub/docs/subscription-message-filter?authuser=1)

### Dataflow

是一个完全托管的流数据分析服务,可用于实时处理无限数据流。它基于Apache Beam编程模型,能够在多种执行环境中运行相同的数据处理代码,包括批处理和流式处理。

主要特点:

1. 无服务器 - 完全托管服务,无需预置集群,自动扩展
2. 统一模型 - 批处理和流式处理使用相同的编程模型
3. 多语言 - 支持Java,Python,Go等多种语言
4. 开箱即用 - 提供多种源和接收器连接器与Google产品集成
5. 可扩展 - 每秒可处理数百万个记录
6. 容错 - 具备自动重试和重新恢复能力
7. 监控 - 提供丰富的监控和日志记录能力

Dataflow广泛应用于ETL、实时数据处理、数据集成、数据分析等场景。用户可以在托管集群环境或自己的私有集群上运行Dataflow管道。它与BigQuery、Cloud Storage、Cloud Pub/Sub等GCP产品紧密集成。总的来说,Dataflow提供了一种简单、统一且富有弹性的方式来处理大规模数据。

Dataflow vs. Cloud Composer:

- Dataflow 主要用于数据处理和转换，尤其是需要高效处理流数据和批数据的场景。
- Cloud Composer 是一个工作流编排工具，用于管理和调度复杂的工作流，适合在跨系统或跨服务的任务之间进行协调和自动化。

## developer

- CI/CD：Deployment Manager
  - yaml文件整合资源
  - 一个resource代表一个API resource
  - Manifest文件是当前部署的只读主文件。它包含用户定义的资源和配置，以及由部署管理器创建的附加资源，以支持有用资源的创建。
- Cloud Marketplace：Google Cloud Platform的一键部署解决方案。您可以从云市场部署任何流行的软件、CMS、工具或 API。 比如WordPress 是一个非常流行的内容管理系统。

## 关于Network的一些解释
---
SSL offload（SSL 卸载）是一种网络安全技术，旨在减轻服务器负载和提高性能。在 SSL offload 中，SSL（安全套接层）和 TLS（传输层安全）的加密和解密操作从服务器转移到专门的硬件设备或者专用的 SSL 加速器中进行处理。

通常情况下，当客户端与服务器之间建立安全连接时，会使用 SSL/TLS 协议对通信进行加密。加密和解密操作对服务器的 CPU 和内存等资源造成了较大的负担，特别是在高负载情况下。为了减轻服务器的负载并提高性能，可以使用 SSL offload 技术。

SSL offload 的工作原理如下：

1. 客户端发起连接请求时，请求通过负载均衡器或者专用的硬件设备（如 SSL 加速器）。
2. 负载均衡器或者 SSL 加速器接收到连接请求后，会负责 SSL 握手过程中的密钥交换、加密和解密操作。
3. 一旦安全连接建立完成，负载均衡器或者 SSL 加速器将未加密的请求转发到后端的服务器上。
4. 后端的服务器只需要处理未加密的请求，不需要负责 SSL 握手和加解密操作，从而减轻了服务器的负载。

使用 SSL offload 技术的主要优势包括：

- **降低服务器负载**：通过将 SSL 加解密操作从服务器转移到专用的硬件设备或者 SSL 加速器中，可以减轻服务器的负载，提高服务器的处理性能和吞吐量。
- **提高性能**：减轻了服务器的负载后，可以更有效地利用服务器资源，提高应用程序的性能和响应速度。
- **简化管理**：SSL offload 技术可以集中管理 SSL/TLS 证书和密钥，简化了 SSL/TLS 配置和管理的复杂性。

总的来说，SSL offload 技术是一种有效的网络安全技术，可以帮助提高服务器的性能和可用性，并简化 SSL/TLS 配置和管理。

---
SSL Proxy（SSL 代理）是一种网络设备或服务，用于在客户端和服务器之间拦截、检查和修改 SSL/TLS 加密通信流量。SSL Proxy 在传输层上建立连接，同时充当客户端和服务器之间的中间人，使得它能够对加密通信进行解密和重新加密，以便检查、修改或过滤通信内容。

SSL Proxy 的主要功能包括：

1. **解密和重新加密**：SSL Proxy 拦截经过其的 SSL/TLS 加密流量，将其解密以便检查通信内容，然后重新加密并转发给目标服务器或客户端。

2. **内容过滤**：SSL Proxy 可以检查加密通信的内容，过滤掉不安全或不符合策略的内容，如恶意软件、广告、敏感信息等。

3. **访问控制**：SSL Proxy 可以基于特定的访问策略对加密通信进行访问控制，包括阻止或允许特定的域名、URL、IP 地址等。

4. **流量监控和分析**：SSL Proxy 可以监控和分析 SSL/TLS 加密流量，以便了解网络流量的模式、趋势和性能，并提供报告和分析。

SSL Proxy 在网络安全和内容过滤方面具有重要作用，尤其是在企业网络中用于监控和控制员工访问互联网的行为，以及在安全网关和防火墙中用于检测和防范网络威胁。然而，使用 SSL Proxy 也可能引发隐私和安全方面的顾虑，因为它涉及到对加密通信进行解密和重新加密的操作，可能会暴露用户的敏感信息。因此，在部署 SSL Proxy 时需要谨慎考虑隐私和安全问题，并遵循适当的法律法规和隐私政策。

---
TCP Proxy（传输控制协议代理）是一种网络设备或服务，用于在传输层拦截、检查和转发 TCP 数据流量。TCP Proxy 充当客户端和服务器之间的中间人，允许它在建立 TCP 连接时拦截数据，检查数据内容，并根据特定的规则对数据进行处理或者转发给目标服务器。

TCP Proxy 的主要功能包括：

1. **数据转发**：TCP Proxy 可以将来自客户端的 TCP 请求转发给目标服务器，并将服务器的响应转发给客户端，使得客户端和服务器之间的通信能够顺利进行。

2. **流量监控和分析**：TCP Proxy 可以监控和分析 TCP 数据流量，包括连接建立、数据传输、连接关闭等过程，以便了解网络流量的模式、趋势和性能，并提供报告和分析。

3. **访问控制**：TCP Proxy 可以根据特定的访问策略对 TCP 数据流量进行访问控制，包括阻止或允许特定的 IP 地址、端口号、协议等。

4. **负载均衡和高可用性**：TCP Proxy 可以用于实现负载均衡和故障转移，将请求转发给多个目标服务器，并在服务器发生故障时自动切换到备用服务器。

5. **安全防护**：TCP Proxy 可以用于实现网络安全防护，包括防火墙、入侵检测和防护等功能，以保护网络免受网络攻击和恶意活动的威胁。

TCP Proxy 在网络通信和安全方面具有重要作用，尤其是在企业网络中用于监控和控制对外网络访问的行为，以及在负载均衡和高可用性方面用于优化网络性能。它可以帮助组织提高网络安全性、可用性和性能，并提供更好的用户体验。

---
SSL Proxy 和 TCP Proxy 在功能和应用方面有一些重要的区别：

1. **协议支持**：
   - SSL Proxy 主要用于处理 SSL/TLS 加密的通信流量，它可以拦截、解密和重新加密 SSL/TLS 加密的数据流量。因此，SSL Proxy 通常用于检查和修改 HTTPS 流量、SMTPS 流量等使用 SSL/TLS 加密的通信。
   - TCP Proxy 则用于处理基于传输控制协议（TCP）的通信流量，它可以拦截和转发 TCP 数据流量，但通常不涉及对数据的解密和重新加密操作。TCP Proxy 主要用于 TCP 协议的代理、负载均衡、访问控制等。

2. **加密处理**：
   - SSL Proxy 对 SSL/TLS 加密的通信进行解密和重新加密操作，以便检查、修改或过滤通信内容。它需要能够识别 SSL/TLS 握手并拥有相应的证书和私钥。
   - TCP Proxy 不涉及对加密通信进行解密和重新加密操作，它只是简单地转发 TCP 数据流量，因此不需要处理 SSL/TLS 加密。

3. **应用场景**：
   - SSL Proxy 通常用于处理需要解密 SSL/TLS 加密的通信流量的场景，如防火墙、代理服务器、安全网关等。它可以用于检查和修改 HTTPS 流量、拦截恶意软件、过滤敏感信息等。
   - TCP Proxy 则用于处理 TCP 协议的通信流量的场景，如负载均衡、访问控制、入侵检测等。它通常用于转发 TCP 数据流量、实现高可用性、优化网络性能等。

总的来说，SSL Proxy 和 TCP Proxy 在处理的通信协议、加密处理方式和应用场景上有所不同。SSL Proxy 主要用于处理 SSL/TLS 加密的通信流量，并进行解密和重新加密操作，而 TCP Proxy 则用于处理基于 TCP 协议的通信流量，但不涉及对加密通信的解密操作。

---
"Preserve Client IP" 是一种网络技术，用于确保在经过代理或负载均衡器时，服务端能够获取到客户端的真实 IP 地址。在常规的网络通信中，客户端的 IP 地址可能会被代理或负载均衡器替换为其自身的 IP 地址，导致服务端无法获取到客户端的真实 IP 地址，而是获取到了代理或负载均衡器的 IP 地址。

为了解决这个问题，可以使用 "Preserve Client IP" 技术来确保服务端能够获取到客户端的真实 IP 地址。这通常涉及到在代理或负载均衡器上进行配置，以便将客户端的真实 IP 地址传递给服务端。具体实现方式包括：

1. **传递 HTTP 标头**：代理或负载均衡器可以将客户端的真实 IP 地址作为 HTTP 请求的一个标头字段（如 X-Forwarded-For）传递给服务端。服务端可以从这个标头字段中获取客户端的真实 IP 地址。

2. **设置自定义标头**：有些代理或负载均衡器支持设置自定义的标头字段来传递客户端的真实 IP 地址。服务端需要根据代理或负载均衡器的配置来获取这个自定义的标头字段。

3. **使用代理协议**：有些代理或负载均衡器支持一些特定的代理协议，这些协议可以在通信中包含客户端的真实 IP 地址。服务端需要根据代理协议来解析客户端的真实 IP 地址。

通过使用 "Preserve Client IP" 技术，服务端就可以获得客户端的真实 IP 地址，从而进行相关的访问控制、日志记录、统计分析等操作，而不是获取到代理或负载均衡器的 IP 地址。这对于需要了解客户端的真实来源和行为的应用场景非常重要。
## SDK

三种方法安装，docker，非docker，服务台的CloudShell。

```bash
# docker install
docker pull gcr.io/google.com/cloudsdktool/cloud-sdk:latest
docker run --rm gcr.io/google.com/cloudsdktool/cloud-sdk:latest gcloud version
# docker config
docker run -ti --name gcloud-config gcr.io/google.com/cloudsdktool/cloud-sdk gcloud auth login
docker run --rm --volumes-from gcloud-config gcr.io/google.com/cloudsdktool/cloud-sdk gcloud config list
# not docker config
gcloud init
```
SDK中所有的组建：使用`gcloud components list`可以列出来。知道了组件甚至可以猜出命令行。组件如下：

- gcloud: The main google cloud component.
- gcloud alpha: Set of commands used for early testing of new features.
- gcloud beta: Beta release of new commands.
- bq: Known as BigQuery component
- gsutil: Used for Cloud storage operations.
- core: Shared libraries for all the other components.
- kubectl: Kubectl is used to control the Kubernetes cluster.

更新命令：`gcloud components update`

安装新的组件的推介方法：`sudo apt-get install google-cloud-sdk-minikube`

命令构成：

`gcloud + release level (optional:alpha/beta) + component + entity + operation + positional args + flags`

For example: `gcloud + compute + instances + create + example-instance-1 + --zone=us-central1-a`

登陆：`gcloud auth login`

配置：使用命令设置默认项目`gcloud config set project <project ID>`

当您登录 gcloud CLI 时，命令`gcloud config list`显示了当前配置，但是，我们可以有多个配置，`gcloud config configurations list`并将列出所有可用的配置。

要创建新配置，请使用`gcloud config configurations create <name>`命令。

撤销当前设置的PJ`gcloud config unset project`

激活默认的配置`gcloud config configurations activate default`
### 附录gcloud

gcloud命令：

- 使用默认设置启动一个GCE：`gcloud beta compute --project=[PROJECT_NAME] instances create instance-2 --zone=us-central1-a`
```bash
$ gcloud compute instances create myinstance
Created [...].
NAME: myinstance
ZONE: us-central1-f
MACHINE_TYPE: n1-standard-1
PREEMPTIBLE:
INTERNAL_IP: 10.128.X.X
EXTERNAL_IP: X.X.X.X  # 这里的外部IP可以用于外部的网络访问
STATUS: RUNNING
```
- 列出所有GCE：`gcloud beta compute instances list`
- 删除GCE：`gcloud beta compute instances delete instance_name --zone zone_name`
- 设置防火墙80端口
```bash
$ gcloud compute firewall-rules create allow-80 --allow tcp:80
Created [...].
NAME: allow-80
NETWORK: default
DIRECTION: INGRESS
PRIORITY: 1000
ALLOW: tcp:80
DENY:
DISABLED: False
```
- 进行ssh连接：`gcloud compute ssh --zone us-central1-a [username]@[instance_name/host_name]`，另外所有的ssh key都在GCE的metadata页面中。在这个页面可以设置key的pub文件。
```bash
$ gcloud compute ssh myinstance
Waiting for SSH key to propagate.
Warning: Permanently added 'compute.12345' (ECDSA) to the list of known hosts.
...

yourusername@myinstance:~#
```
- 使用自定义的启动脚本创建GCE
```bash
$ gcloud compute instances create nginx \
         --metadata-from-file startup-script=startup.sh
```
- 创建一个服务器集群用于负载均衡的过程：创建一个模板，然后创建目标池，这可以用于之后的负载均衡，然后在池中创建两个目标GCE，最后列出所有服务器，最后创建负载均衡
```bash
$ gcloud compute instance-templates create nginx-template \
         --metadata-from-file startup-script=startup.sh
$ gcloud compute target-pools create nginx-pool
$ gcloud compute instance-groups managed create nginx-group \
         --base-instance-name nginx \
         --size 2 \
         --template nginx-template \
         --target-pool nginx-pool
$ gcloud compute instances list
$ gcloud compute forwarding-rules create nginx-lb \
         --ports 80 \
         --target-pool nginx-pool
$ gcloud compute forwarding-rules list
NAME: nginx-lb
REGION: us-central1
IP_ADDRESS: X.X.X.X  # 这个地址可以访问lb地址了
IP_PROTOCOL: TCP
TARGET: us-central1/targetPools/nginx-pool
```
- 所有的清理命令
```bash
$ gcloud compute forwarding-rules delete nginx-lb
$ gcloud compute instance-groups managed delete nginx-group
$ gcloud compute target-pools delete nginx-pool
$ gcloud compute instance-templates delete nginx-template
$ gcloud compute instances delete nginx
$ gcloud compute instances delete myinstance
$ gcloud compute firewall-rules delete allow-80
```
- 以上都是来自官方的codelabs，真的很好:https://codelabs.developers.google.com/codelabs/cloud-compute-engine?hl=zh-cn#0

如果是windowsGCE需要用RDP客户端，这对于Linux就是ssh。

- Cloud SQL 命令行合集：（也可以使用UI）

```bash
# create sql instance
gcloud sql instances create [instance-name]
# create database in the instance
gcloud sql databases create [database-name] --instance [instance-name]
# connect to CloudSQL / need activate CloudSQL Admin API
gcloud sql connect [project name] --user=root --quiet
# or user mysql client
mysql -h [IP-of-instance] -u [user] -p
# delete the instance
gcloud sql instances delete [instance-name]
```

- Cloud Spanner

```bash
gcloud spanner instances list
gcloud spanner databases list --instance [INSTANCE-ID]
gcloud spanner instances delete [Instance-ID]
```

- BQ

```sql
CREATE OR REPLACE TABLE `stackoverflow.questions_2018_clustered`
PARTITION BY
  DATE(creation_date)
CLUSTER BY
  tags AS
SELECT
  id, title, accepted_answer_id, creation_date, answer_count , comment_count , favorite_count, view_count, tags
FROM
  `bigquery-public-data.stackoverflow.posts_questions`
WHERE
  creation_date BETWEEN '2018-01-01' AND '2019-01-01';
```

- GCS
- 文件操作

```bash
# create
gsutil mb gs://<bucketname>
# list
gsutil ls
# upload
gsutil mv ./demo.txt gs://[bucketname]
gsutil cp ./demo.txt gs://[bucketname]
# multithread upload
gsutil -m mv . gs://cloudstoragelab/
# upload bigfile by chunks
gsutil -o GSUTIL:parallel_composite_upload_component_size=10M mv [bigfile] gs://[bucketname]
```

- 文件生命周期设置
```bash
# get
gsutil lifecycle get gs://cloudstoragelab/
# set rules by json file
gsutil lifecycle set rules.json gs://[bucket_name]
```

- GKE
- 删除cluster：`gcloud container clusters delete [cluster-name] --zone [zone]`

- 命令行合集：

```bash
# enable api
gcloud services enable container.googleapis.com
# create cluster
gcloud container clusters create fancy-cluster --num-nodes 3
# create container
# enable cloud build api
gcloud services enable cloudbuild.googleapis.com
# create image
gcloud builds submit --tag gcr.io/${GOOGLE_CLOUD_PROJECT}/monolith:1.0.0 .
# deploy container to gke
kubectl create deployment monolith --image=gcr.io/${GOOGLE_CLOUD_PROJECT}/monolith:1.0.0
# delete pod
kubectl delete pod/<POD_NAME>
# check the status of pod service and deployment
kubectl get all
# espose the service
kubectl expose deployment monolith --type=LoadBalancer --port 80 --target-port 8080
# get the service info of cluster ip, external ip and ports
kubectl get service
# scale the deployment to 3
kubectl scale deployment monolith --replicas=3
# rebuild application image with a new version!
gcloud builds submit --tag gcr.io/${GOOGLE_CLOUD_PROJECT}/monolith:2.0.0 .
# deploy the new version without downtime
kubectl set image deployment/monolith monolith=gcr.io/${GOOGLE_CLOUD_PROJECT}/monolith:2.0.0
```
codeslabs:https://codelabs.developers.google.com/codelabs/cloud-deploy-website-on-gke#0

- Auto Scaling
- gcloud：
```bash
gcloud compute instance-groups managed \
set-autoscaling instance-group-2 \
--max-num-replicas 3 \
--min-num-replicas 1 \
--target-cpu-utilization 0.60 \
--cool-down-period 120 \
--zone=us-central1-a
```
