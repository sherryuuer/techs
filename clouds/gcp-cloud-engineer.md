## GCP云工程师学习笔记

### 概览

云的基础是虚拟化：服务器，存储，网络。服务器是远程计算机的逻辑分区。存储是物理硬盘的逻辑划分。网络则是虚拟私有云。

谷歌是唯一一个拥有全球私有基础设施的公司；他们的谷歌云基础设施没有任何一部分通过公共互联网。换句话说，谷歌的云基础设施完全建立在他们自己的私有网络上，而不是依赖于公共互联网的网络基础设施。

谷歌以其为其服务开发革命性技术而闻名，一旦他们有了更好的版本，旧版本就会向全世界开源。

TensorFlow就是这些技术之一。 TensorFlow 是 Google 的 AI 引擎，也是 Google 搜索的核心。现在最新的是kubeFlow！

**安全的全球专用网络**：Google Cloud 是一个大型专用网络；没有任何东西通过公共互联网传播。使用 Cloud CDN 时，来自源的流量会穿过此专有光纤网络，并通过最靠近客户的位置进行传送。大多数研究公司选择 GCP，因为没有网络连接到互联网。

**Kubernetes 引擎的创建者**：Kubernetes 是由 Google 设计和创建的容器编排工具，用于管理其基础设施。部署了数百万个容器来托管 Google 服务，而 Kubernetes 是这些容器的核心。 Google 开源了这项技术，并提供了 Google 管理的 Kubernetes 引擎作为服务。

收费方式按秒计费，新用户前三个月有300美刀信用额度。

**人工智能和机器学习服务**：毫无疑问，谷歌是数据之王。数据越多，模型在人工智能世界中产生的结果就越准确。谷歌的机器学习服务是最好的。 Google 提供了多种 ML 服务来训练模型并处理模型训练管道。Kubeflow现在是 ML 操作的趋势。同样，KubeFlow 基于 Google 内部方法将 TensorFlow 模型部署到 Kubernetes，称为 TensorFlow Extended。


### 项目PJ和权限IAM

整个GCP层级如下：组织（如果你的账户域名是组织类型的话）- 文件夹folder - 项目projects - 资源resource

IAM包括四种账户：Google Account, Service Account, Google Groups, Cloud Identity Domain。

IAM Policy：用json方式写的权限控制方式，成员，角色，资源，以及按需条件。

所有的资源服务都是一种API。从API&Service中就可以查看所有的API。有些服务在创建项目的时候就已经激活，有些则需要在使用的时候手动激活。

API Explorer可以轻松从网页尝试调试API（就算没激活都可以试用，厉害）。一般来说如果 API 请求失败，要么是没设置账户，要么是该服务还没被你激活。

Cloud Operations：Monitoring，Dashboard，Metrics Explorer，Alerting，通过group可以整合资源管理，通过setting可以增加pj和account统一管理。其他还有logging，error report，trace服务。

### 计费 Billing

**计费账户**:可以为每个部门单位创建计费账户，或者别的什么单位。可以在项目Project上（三个点）设置它的计费账户。或者可以在新建项目的时候选择收费账户。

操作需要的权限：（Billing Administrator role）或者 （Project Creator and Billing Account User）

Command:`gcloud alpha/beta billing accounts list`

注意：一开始的300美元积分，仅限第一个积分账户。最好把测试用的PJ都关联到第一个计费账户来使用。

**预算和警报**：预算和警报，是以计费账户为单位的。通过阈值在到达指定金额时，发送提醒邮件给我们，好让我们采取行动。

**BillingExport**：计费输出将费用数据输出到BigQuery。需要激活Bigquery Data transfer API。

如果你是发票结算账户，付款需要联系GCP销售团队，可以电汇或支票结算，每月将收到发票。

### SDK

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

### Compute

从操作量来说（从多到少）：Compute Engine - Kubernetes Engine - (Cloud Run / Cloud Functions / App Engine)

从workload包罗范围来（从多到少）：Compute Engine - (Kubernetes Engine / Cloud Run) - Cloud Function - App Engine

**App Engine**：网络应用，手机后端，特定语言Python，HTTP only，专注快速开发，谷歌托管Docker。

**Cloud Functions**：事件触发，语言：Javascript，Go，Python，Java，不关心操作系统，时间执行540秒之内。因为它是事件触发所以对于workload（各种计算任务）更有用。

**GKE**：容器化服务，需要定义集群，可移植性，容器为部署单元。

**Cloud Run**：托管的GKE服务，用法和GKE基本相似除了：集群由谷歌管理，工作负载容器化，docker image，便宜，之需要关心应用的高可用性即可。

**Compute Engine**：能力越大责任越大。相当于EC2，没有容器化工作负载。

Preemptive VMs：抢占式虚拟机，24小时存活，适合处理批量作用，很像AWS的Spot，在创建regular Engine的时候选择它即可。便宜80%。

Predefined VMs：Standard, memory-optimized, compute-optimized

Instance:ssh for Linux, RDP protocol for windows.

Image, Snapshot, Metadata(hostname, instance id, startup&shutdown scripts, custom metadata, service accounts info)

### Storage



### Networking

