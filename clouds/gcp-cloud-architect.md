## Management

### Resource Manager

- **GCP构架层**：根为组织，下面是folder层，下面是Project层，pj里面是resources
- PJ的ID和Number是不可变的，全球唯一，Name可以随时变更
- **组织Policy**是和AWS的SCP类似的东西，下面的层级会继承它的Policy
- 创建**组织**不能用个人账户，需要*Google Workspace*或者*Cloud Identity*
- 坚守**最小权限原则**，但是也要尽量削减*管理Overhead*（在管理活动中额外消耗的资源或成本，这些成本并不直接产生效益，但为了确保系统或组织的正常运作而必须承担）
- 环境分离：test，staging，prod，最好是组织层级分离
- 使用*会计组织*管理请求书，其他的组织不拥有阅览权

### CDM（Deployment Manager）

- IaC工具组，Yaml格式，CloudFormation也是这个格式
- `gcloud deployment-manager deployments create/update/delete service --config file.yaml`
- 命令行：服务，组件，动作，对象，选项

### Cloud Foundation Toolkit

- 也是IaC，比CDM更好，对应*所有的service*（CDM只能一部分服务），更简单，可用*Terraform*，这么看来CDM很没用啊
- 里面的代码可以用外部的Github进行版本管理，而CDM只能在GUI上创建和保管

### IAM

- 谁，能做什么，针对什么对象
- **Role**是一种很重要的概念，代表一组资源和权限的组合，像是一顶帽子，GCP有predefined role和custom role
- *bindings*：将role和member进行绑定的概念和CEL语法
- RBAC和ABAC，前者基于role后者基于属性，属性有时候是很难定义的，AWS主要通过tag，GCP有自己的属性设置方式
- *Common Expression Language*（CEL）是一种用于定义条件逻辑的表达式语言，主要用于 Google Cloud 中的权限控制和策略管理。它允许用户编写简单的布尔表达式来定义访问控制的条件，例如访问时间、用户角色等，但是AWS的IAM是用json定义的，两者不同
- 基本（Primitive role）的Owner/Editor/Viewer已经被废除，缺乏灵活性和安全性
- IAM的变更日志，可以通logging服务，*输出*到GCS或者BigQuery进行保存和分析
- *权限继承*，组织-folder-PJ，高一层的权限会被低层继承
- **最佳实践原则**
  - 使用单一的Identity Provider
  - 使用特权管理（Cloud Workspace/Cloud Identity）
  - 使用SA进行服务权限管理
  - 经常更新ID认证最新流程
  - 设置SSO单点登录和MFA认证
  - 使用*最小权限原则*进行权限分散的设置
  - 访问监察，比如使用logging，以及AWS的CloudTrail都是该功能的实现
  - 自动化Policy管理，我想应该是用IaC等
  - 设置对resource的访问限制

### Service Account

- 是*应用或者服务作为主体*，进行认证和使用的，IAM是用户为主体
- 用密码或者认证key进行API执行
- 种类：用户自己创建/内置的default的SA
- AWS没有这个概念，而是用IAM Role控制服务主体的权限

### Identity Platform

- 适合开发者和应用程序构建者，用于为面向终端用户的应用实现身份验证和管理，它支持多种身份验证方式，广泛适用于 Web 和移动应用的场景
- AaaS（Authentication as a Service）
- 步骤：选择IdP，追加用户，创建login和认证界面

### Cloud Identity

- 类似于AWS的Identity Center，统合管理
- 实现User管理，Device管理，SSO登录管理，以及各种安全功能（威胁检测，Endpoint管理，多要素认证等）

### Operation Suite

- 系统监视的工具和对应，记录log进行监察等
- **Cloud Logging**：收集几乎所有的日志，实时分析功能，定制metric
  - 构架：Logging - Pub/Sub - Dataflow - BQ，这一连串流程依赖Logging API编码
  - Log Bucket：日志保存用的桶，分为required（监察，存留400天不可更改设置）和default（存留40天，可以更改设置）
  - *Cloud Audit Log*：如AWS的Cloud Trail
    * 管理活动日志，通过SDK，API，Console进行资源操作的日志
    * 数据访问日志
    * 系统事件日志，非用户活动导致的事件日志
    * Policy Denied监察日志，因IAM权限不足而被拒绝访问的情报

- **Cloud Monitoring**：可视化，infra和应用的活动监控，还可以*监控其他云平台比如AWS的活动*，主要功能如下：
  - 收集metric：
    * 主要可以收集的是CPU利用率，Network进出流量，和DiskIO
    * 高级metric比如memory利用率，需要安装*Monitoring Agent*，或者用*Monitoring API*则什么都可以发送
  - 仪表盘，也就是可视化
  - Alart，根据事先设置，发出警报
  - Uptime Check，定期检查你的Web服务的可用性（百分之XX的指标）
  - SLO（Service Level Objectives）指标报告书的生成
    * SLA（agreement）则是关于服务水平和客户制定的，必须达到的协议

- **Cloud Trace**：对GCP上的应用性能的瓶颈，和延迟原因，进行特定的，分散型追踪系统
  - 分析，收集，可视化，request的详细情报
  - 适合*微服务*构架，因为很多request发送/接收的数据

- **Cloud Service Health**：健康仪表盘
  - 可视化和通知，云服务的稼动状况，incident，未来maintainance的历史和计划等

### Secret Manager

- 秘密info管理
- 有version管理功能，方便rollback
- 和Cloud Audit Log统合，方便监察
- Functions可以帮助自动无效化和更新密码
- 如果没有location限制，在创建的时候会自动有效化复制Policy
- 防止代码中秘密情报的硬编码，使用SecretManager并对API的使用，执行最小权限原则的权限赋予
- 删除前要先无效化

### Cloud KMS

- Key Management Service
- 加密键的创建，使用，管理（使用version管理，防止找不到老的键）
- 统合IAM，进行访问控制管理
- 统合Audit Logs，进行监察管理
- 键可以设置为global或者region，但是global的泄露影响较大，最好还是慎重选择region
- HSM可用
- 第三方Key管理服务可用
- FIPS 140-2:美国和加拿大政府，加密，安全评价标准

### Security Command Center（SCC）

- 有点像*AWS Config和各种异常检测服务的合体*
- tier有Standard和Premium
- Standard功能：
  - Security Health Analytics健康分析，IAM，Network，数据管理构成config的问题检测
  - Web Security Scanner：检测web应用的脆弱点（高级功能中这个是可以自动的，标准功能中需要客户手动操作）
  - Anomaly Detection：比如VM加密货币挖掘滥用检测
- Premium功能：
  - Event Thread Detection：Logging数据加机器学习手法进行病毒检测
  - Container Thread Detection：对容器不正常举动的检测
  - VM Thread Detection：对VM的不正常举动的检测

### Cloud Data Loss Prevention（DLP）

- 机密info的检测，分类，保护，AWS同类服务Macie
- 可以对机密情报进行masking，置换操作等
- 统合Bigquery，GCS，Datastore服务，通过API可以和Custom的数据流统合
- 根据敏感数据出现的频率，分析风险水平

### Migration & Transfer

- 各种服务和数据的移行服务，包括服务器，数据，数据库，SQL变换工具等，自己用过的Storage Transfer，BQ transfer，Database Migration Service（转换SQL的那个）

### Disaster Recovery

- 把握基本的*RPO*和*RTO*的概念，恢复点目标和恢复时间目标，两者越短，cost越高
- 几个tier：
  - backup&restore：最便宜，恢复时间长
  - cold standby：发生灾害，移行数据到prod
  - warm standby：复制一个最小构成环境，除了数据库之外基本都具备
  - hot standby/multi-side/active-active：完全稼动的相同的prod环境
- 各个服务都有自己的DR策略
- **SRE（Site Reliability Engineering）**，谷歌2003开发的软件工程框架
  * 减少手动运维，使用自动化工具管理
  * 服务质量定量评估指标SLO，Error Budget
  * Toil是人工手动反复操作的task，应该尽量减少这部分内容，而转向有价值的输出
  * 防止问题反复的自动化构架
  * 从过去的错误和故障中进行学习和反思
  * 提高开发速度，提高运维可信赖度，这两者应该是平衡的

### Tag & Lable

- lable：方便资源管理整合的属性情报，metadata，用于resource管理，氪金管理
- tag的使用场景：网络防火墙的rule使用于VM的tag群组 / CI/CD工具的version管理tag / 组织的层级用不同的tag进行Policy设置
- tag主要用于*权限管理*，是在*组织层级*设置的，本身就是*一种resource*
- 注意，GCP的Label对照AWS和Azure的Tag，GCP的tag则是不同的东西
- Afiniti Lable：用于关系管理的label，比如一台VM之于node group，就是这种亲和关系

## Compute

### GCE

- 以秒为单位计费
- Preemptable VM & Spot VM：后者比前者更便宜，但是前者可以使用标准VM，后者则不固定
- burst：突然的CPU高负荷，不会有追加加算费用
- 持续存储disk：分为standard（HDD）/balanced/SSD/extreme性能区分，ZONE/Region存储位置区分
- Local SSD是本地存储，关闭Server，数据消失
- IP分类：静态固定IP/动态ephemeralIP/内部私有IP，固定IP会被氪金
- Disk快照：增量备份和全量备份，有GCS料金，Meta数据和Tag不会被备份，可以在稼动中备份
- Cstom Image：可以在稼动中创建，可用于环境迁移，是全量备份
- Machine Image：Disk + Image + Policy的全量备份
- 知识补充：差量备份，是和最初的备份相比的增量，所以这种备份每次都会变多，但是它restore比增量的快，因为只需要初始的部分，和最近一次备份的部分
- 云服务器解决方案的分类：
  * Cloud Engine：公有云解决方案
  * Cloud VMware Engine：私有云，和OnPre的VM统合
  * Bare Metal Solution：私有云，使用Google提供的物理机箱，高安全，低延迟需求
- 云迁移：Migrate for Compute Engine
- MIG（Managed Instance Group）相当于动态扩展组ASG
  * InstanceTemplate -> 使用模板创建MIG -> LoadBalancing设置和状态检查 -> AutoScaling设置
  * 相对的也有UIG（Unmanaged Instance Group），可以用不同种类的VM构成，但是完全需要用户手动管理和扩展
  * 分为stateless（batch处理的构架，可以随时停止和缩小的）和statefull（连接数据库的构架）
  * Server数量必须至少有一台，不能是min0max0

### GKE

- Kubernetes的所有功能，本身是一个容器编排服务，自动修复自动升级
- 支持DockerImage的部署：通过Deployment（API）的Yaml文件指定image文件（放置于GCR），然后通过kubectl命令行部署
- **K8S的最大单位是Cluster**：
  - User控制是通过kubectl
  - Control Plane(Master Node) -> Nodes(Worker Node) -> Other Services
  - Nodes是K8S的服务器主体，有Health Check功能
  - K8S的服务器定义可以通过manifest文件来定义，使用manifest还支持rollback
  - Pod是部署的最小单位
- 两种模式，一种是标准模式，需要用户手动控制，另一种是Autopilot模式，可以自动化管理
- 两种命令行体系：gcloud用于最大scope的Cluster的管理，kubectl用于内部的Pods等的细化管理
- API众多：Deployment，ReplicaSet，StatefullSet
- 支持Rolling Deployment蓝绿发布，自动LB切换（使用Deployment + ReplicaSet）
- **GKE网络构成要素**：
  - ClusterIP：可以连接到集群内部的IP
  - NodePort：Node和外部疏通用的Port
  - LoadBalancer：TCP负载均衡，Layer4，通过network来的流量
  - ingress：HTTP/S负载均衡，Layer7，通过URL和hostname来的流量
  - ExternalName：服务的DNS命名解决通过CNAME解决
- **冗长编排方式**：
  - Zone Cluster：Master和Worker都在同一个Zone，抗风险能力小
  - Multi-Zone Cluster：Master在单独一个Zone，Worker在多个Zone分布
  - Regional Cluster：Master和Worker都在多Zone中分布
- **自动伸缩功能Auto Scaling**：
  - Worker Node层级：背后使用GCE，设置CA*Cluster AutoScaler*有效化，就可以伸缩Node Pool，但是不能同时有效化GCE的自动伸缩，两者会冲突
  - Pod层级：设置HPA*Horizontal Pod AutoScaler*有效化
  - *Pod虽然可以伸缩，但是如果Cluseter容量不足，也伸不开，两者是包含的关系*
- **Helm**：是 Kubernetes 的一个包管理工具，类似于 Linux 上的 apt 或 yum，但专门用于 Kubernetes 环境。它允许你以可重用的模板化方式定义、安装和管理 Kubernetes 应用程序。Helm 提供了一种简化部署和管理 Kubernetes 集群中复杂应用的方法，特别是在多服务和微服务架构中。


### GCK & AR

- GCR是Container Register，是存放image的，有自动Scan病毒的功能
- AR是Artifact Register：Image，SourceCode，二进制文件，构成文件，文档等的存放
  - 安全和管理功能比较强化，RBAC，加密，Image历史记录追踪可

### PaaS分类

- APP分为HTTP和Event驱动两种
- HTTP中分为需要设置K8S硬件系统的Cloud Run for Anthos和不需要设置的Cloud Run（现在内置Function了）
- 不在不需要设置K8S（底层硬件系统）的分类中分为不受编程语言限制的Cloud Run和受到限制的Cloud Function，以及退休的App Engine

### App Engine

- PaaS，全托管快速发布应用，已废止使用了似乎
- 支持蓝绿发布，和各种CI/CD工具联动可

### Cloud Function

- Faas，Function as a Service
- 可以只驱动代码执行，是事件驱动型的微服务构架组件，高并发，自动伸缩
- 安全：端点是HTTPS类型，使用IAM管理访问权限
- TimeOut：HTTP函数的时间是60分钟，事件驱动函数的时间是9分钟

### Cloud Run

- 基于Docker Image的事件驱动服务
- 统合：HTTPS，GCS，PubSub，CloudBuild（组合完成CI/CD管道流程）
- Cloud Run for Anthos：可以使用GKE为基盘的Run，使用GKE的功能，更加灵活

### CI/CD工具组

- Cloud Build支持并行build，创建Docker Image，并无缝连接地Push到GCR中，与firebase统合，iamge自动加密
- 上游Git环境发生Push事件，couldbuild.yaml定义构成文件，deploy代码到各种GCP环境
- Test策略：Canary（要考虑下位互换性），A/B测试，Shadow测试
- Deploy策略：AllAtOnce（In-Place），Rolling，Blue/Green

### Anthos

- 混合云和多云应用管理服务
- 方便将原有的应用迁移到Container环境，使用微服务构架
- Anthos本身的*底层服务集群构架*是用K8S服务集群
- Anthos Service Mesh：*上层服务*，可视化，监视，管理服务，可以设置Alart和SLO（Service Level Objective服务水平目标）
- DevOps/CI/CD统合服务发布
- 开源项目，可与第三方统合（但是我觉得现在大家很少这样做）
- *Istio*：微服务架构中管理服务通信的工具，开源的服务网格（Service Mesh）平台，用于帮助管理、保护、连接和监控微服务架构中的网络流量。它通过抽象出应用程序的网络层，简化了微服务的管理，特别是在大规模分布式系统中
  * 数据平面（Data Plane）：由Envoy 代理组成。Envoy 是一个轻量级的代理，它在每个服务的旁路中运行，*处理微服务之间的网络流量*，执行流量管理、安全策略、监控和日志记录。
  * 控制平面（Control Plane）：主要组件为Pilot、Mixer 和 Citadel，负责配置代理、执行策略和管理服务身份验证。

## Storage

## Network

## Database

## Data Analytics

## AI & ML
