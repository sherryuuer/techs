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

- 系统监视的工具和对应，记录log进行监察等，Log explora功能
- **Cloud Logging**：收集几乎所有的日志，实时分析功能，定制metric
  - 构架（log router）：*Logging - Pub/Sub - Dataflow - BQ*，这一连串流程依赖Logging API编码
  - Log Bucket：日志保存用的桶，分为required（监察，存留400天不可更改设置）和default（存留40天，可以更改设置）
  - *Cloud Audit Log*（监察日志）：如AWS的Cloud Trail
    * 管理活动日志，通过SDK，API，Console进行资源操作的日志
    * 数据访问日志
    * 系统事件日志，非用户活动导致的事件日志
    * Policy Denied监察日志，因IAM权限不足而被拒绝访问的情报

- **Cloud Monitoring**：可视化，infra和应用的活动监控，还可以*监控其他云平台比如AWS的活动*，主要功能如下：
  - 收集metric：
    * 主要可以收集的是CPU利用率，Network进出流量，和DiskIO
    * 高级metric比如memory利用率，需要安装*Monitoring Agent*，或者用*Monitoring API*则什么都可以发送
  - 仪表盘，也就是可视化
  - Alart，根据事先设置，发出警报，通知到webhook，slack等
  - Uptime Check，定期检查你的Web服务的可用性（百分之XX的指标）
  - SLO（Service Level Objectives）指标报告书的生成
    * SLA（agreement）则是关于服务水平和客户制定的，必须达到的协议

- **Cloud Trace**：对GCP上的应用性能的瓶颈，和延迟原因，进行特定的，*分散型*追踪系统
  - 分析，收集，可视化，request的详细情报
  - 适合*微服务*构架，因为很多request发送/接收的数据
  - 追踪**性能瓶颈问题**

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
- **处理信用卡机密情报的时候，使用token化技术，将机密情报转换为非机密情报进行存储，是一种泛用的方法，不知道的话那就不知道了**

### Migration & Transfer

- 各种服务和数据的移行服务，包括服务器，数据，数据库，SQL变换工具等，自己用过的Storage Transfer，BQ transfer，Database Migration Service（转换SQL的那个）
- 移行的几种方式复习：
  - *Lift&Shift*：将现有环境，原封不动地移行到云端，这种方式的workload是最小的，很快
    * *Modernize*：这种方式的移行后，使用container等，使得服务更加适合云环境的生长方式
  - *Improve&Move*：为了适应云的环境会对config进行修改，workload稍微大一点
  - *Rip&Replace*：完全使用新的config方式进行移行，我所做的第二个GCP项目就是这种类型的移行，完全使用了新的batch和服务

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

- **灾害恢复，需要考虑的事项：（影响最小，速度最快）**
  * 数据恢复：数据备份和恢复方法，多区域配置方案，大数据的话需要考虑传输速率
  * 服务冗长构成：多区域构成
  * 网络构成：VPC/DNS构成是否合适和有弹性
  * 安全：
    - *standby环境*应当和prod环境有同样的安全构成和Compliance构成
    - *权限*应当用*IaC*来构成
    - *网络设置*上，防火墙和VPC构成都是必须要素
    - *监察日志和秘密情报的保护*
  * 用户训练和测试：是必备的


### Tag & Lable

- lable：方便*资源管理*整合的属性情报，metadata，用于resource管理，*氪金管理*
- tag的使用场景：网络防火墙的rule使用于VM的tag群组 / CI/CD工具的version管理tag / 组织的层级用不同的tag进行Policy设置
- tag主要用于*权限管理*，是在*组织层级*设置的，本身就是*一种resource*
  - docker发布的时候用的那个tag，注意如果你用latest作为tag的话那么每次都会被覆盖丢失
  - 在*CI/CD*等版本管理中的那个tag如果使用commit的hash数值，则比较好，因为那个东西不会重复
- **注意，GCP的Label对照AWS和Azure的Tag，GCP的tag则是不同的东西**
- **Afiniti Lable*：用于关系管理的label，比如一台VM之于node group，就是这种亲和关系

### 测试和发布方法

- 金丝雀测试：小环境和prod环境同等配置只是范围较小，影响范围基本没有，可以快速rollback，同时可以测试在prod环境中早期容易发现的性能问题等
- AB测试：一部分装载了新功能，目的是效果比较
- 影子测试：完全是和prod环境同样的环境进行测试，用LB连接，这种测试cost较高要注意
- 一次性再生性发布：一下子发布，中间会有downtime，对于服务水平的维持有影响
- rolling发布：会比较慢，环境是新旧混合的，但是没有downtime，要处理好客户的session
- 蓝绿发布/红黑发布：两个环境会一起跑，费用较高，*基本没有rollback*，比较安全，维持SLA的最好选择
- 将一个基盘，分割成许多*microservice*进行发布和运维，也可以降低rollback的概率，因为各个组件是疏结合的，相互之间的影响很小

## Compute

### GCE

- 以秒为单位计费
- Preemptable VM & Spot VM：后者比前者更便宜，但是前者可以使用标准VM，后者则不固定
- burst：突然的CPU高负荷，不会有追加加算费用
- 持续存储disk：分为standard（HDD）/balanced/SSD/extreme性能区分，ZONE/Region存储位置区分
- Local SSD是本地存储，关闭Server，数据消失
- IP分类：静态固定IP/动态ephemeralIP/内部私有IP，固定IP会被氪金
- Disk快照：增量备份和全量备份，有GCS料金，Meta数据和Tag不会被备份，可以在稼动中备份
- Custom Image：可以在稼动中创建，可用于环境迁移，是全量备份
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
  - Node的集合是Cluster，Pod的集合是Service，多Pod进行负荷分散（Ingress）
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


### GCR & AR

- GCR是Container Register，是存放image的，有自动Scan病毒的功能
- AR是Artifact Register：Image，SourceCode，二进制文件，构成文件，文档等的存放
  - 安全和管理功能比较强化，RBAC，加密，Image历史记录追踪可

### PaaS分类

- APP分为HTTP和Event驱动两种
- HTTP中分为需要设置K8S硬件系统的Cloud Run for Anthos和不需要设置的Cloud Run（现在内置Function了）
- 在不需要设置K8S（底层硬件系统）的分类中分为不受编程语言限制的Cloud Run和受到限制的Cloud Function，以及退休的App Engine

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
- *Anthos*适合混合云的移行

### CI/CD工具组

- Cloud Build支持并行build，创建Docker Image，并无缝连接地Push到GCR中，与firebase统合，iamge自动加密
- 上游Git环境发生Push事件，couldbuild.yaml定义构成文件，deploy代码到各种GCP环境
- Test策略：Canary（要考虑下位互换性），A/B测试，Shadow测试
- Deploy策略：AllAtOnce（In-Place），Rolling，Blue/Green

### Anthos

- 混合云和多云应用管理服务
- 方便将原有的应用迁移到Container环境，使用*微服务构架*
- Anthos本身的*底层服务集群构架*是用K8S服务集群
- Anthos Service Mesh：*上层服务*，可视化，监视，管理服务，可以设置Alart和SLO（Service Level Objective服务水平目标）
- DevOps/CI/CD统合服务发布
- 开源项目，可与第三方统合（但是我觉得现在大家很少这样做）
- *Istio*：微服务架构中管理服务通信的工具，开源的服务网格（Service Mesh）平台，用于帮助管理、保护、连接和监控微服务架构中的网络流量。它通过抽象出应用程序的网络层，简化了微服务的管理，特别是在大规模分布式系统中
  * 数据平面（Data Plane）：由Envoy 代理组成。Envoy 是一个轻量级的代理，它在每个服务的旁路中运行，*处理微服务之间的网络流量*，执行流量管理、安全策略、监控和日志记录。
  * 控制平面（Control Plane）：主要组件为Pilot、Mixer 和 Citadel，负责配置代理、执行策略和管理服务身份验证。

## Storage

### GCS

- object存储对象，object最大size是5TB
- 根据使用频率可以分几个classes：
  - Multi-regional - Part of Standard now
    - 注意这个多区域也是在一个大陆上的，要在多个大陆复制，要用多个Multi-region桶
  - Dual-region：可以自己控制在哪两个region的复制，比Multi-region更灵活一些
  - Regional - Part of Standard now
  - Nearline：30天/访问频率
  - Coldline：90天/访问频率
  - Archive：一年/访问频率
- LifeCycle管理可，在class之间的移动和删除可
- 数据保护：存储加密，传输SSL/TLS加密，Version管理（文件本身是Key，版本是ID，删除的时候是Deleted Marker，可取回）
- Bucket Lock：为了满足Compliance要求的一定期间对桶内object，不可删除，不可修改的设置，又叫Retension（保留）Policy
  - Object Holds：是针对单个object的锁定
- 签名URL（署名付きURL）是临时权限的URL，两种，针对resource的和针对scope的：前者针对特定资源，后者针对特定操作权限
  - 创建的时候需要使用特定权限的Service Account，下载SA的秘密key，Call创建签名URL的函数进行创建
- 均一管理（Uniform）和细化管理（Fine-Grained）
  - 均一管理针对bucket全体，不需要ACL管理object，后者则使用ACL管理object层级的权限，比较麻烦
- 有Metadata管理功能
- gsutil操作，它是Python应用哎，针对大object和streaming数据的上传和下载有更多的选项可以使用
  - 大object通过并行处理和分割处理会提高传输效率
- *Transfer Appliance*：物理数据传输，发送你黑盒子发送数据通过物理运输等方式发送到谷歌，放置于GCS中
  - 适用于TB级别的传输，放置网络问题导致的数据破损
  - 但是因为物理发送原因，要花费10天以上，一般来说10TB之类的，网络没问题的话，用网络传输效率更高
- *allUsers*这个用户权限选项可以用于静态网页全公开！


### Firestore

- NoSQL文件document数据库/Firebase是realtime数据库，是一种BaaS（Backend as a Service）
- 高IOPS低延迟
- 几个版本选择：
  - Firestore Basic：文件系统，软件开发后端，GKE
  - Firestore Enterprise：基盘应用比如SAP等
  - Firestore High Scale：比如金融交易，HPC等

## Network

### VPC

- VPC是global的，subnet是可以跨zone的，但是注意，AWS中VPC时跨几个AZ，subnet是每个AZ一个
- VPC Peering功能有，可以跨Region进行Peering，通过防火墙进行通信限制，服务本身不花钱，但是通信会花钱
  - 和AWS一样：IP cider不能有重叠，也不支持推移传递连接，要连接哪个就需要直接连接
  - 最大连接上限100个
- Shared VPC：是一个host的Project的VPC分享私有网络给多个Project，成为主从关系，共享一个私有网络的IP和资源
- VPCSC：公开API的安全境界设置，放置不安全的数据外流
  - Context Access Manager：可以设置账号，IP等的限制，等于说在IAM的设置外多了一层防护
- *Private Google Access*：只拥有私有IP，也可以和Google的服务/API进行连接的功能，安全

### Cloud VPN/Dedicated Connection

- VPN使用Site2site的VPN，使用IPSec加密方法
- 除了基本的VPN，还提供一种HA（High Availability）VPN
  - 高可用VPN支持DualStack（IP4+IP6），仅仅支持动态路由BGP，支持一个VPN Gateway接两个通道

- **Cloud Interconnect**：分为两种：Dedicated Interconnect和Partner Interconnect
  - 适用于BigData，和大型Workload
  - 前者就是完全专有（10Gbps，100Gbps），后者要依赖第三方业者（可以从50Mbps开始）
  - 前者是Google专有提供的所以支持SLA水平99.9，后者则不支持

### Cloud Firewall

- 根据IP，Protocol，Port设置，优先顺序用号码定义，越小级别越高，这种明确设置的rule列表叫做静态防火墙
- Firewall Insight是SCC的一个功能组件
- 可以看出每个rule的使用率，识别不被使用的rule和重复的rule用于改善，还可以追迹rule的变更历史

### Cloud Armor

- Layer7的防火墙，强化HTTP/HTTPS的通信保护
- DDoS防御功能：检测异常，确保通信带宽，block特定通信源
- WAF功能，防御OWASP Top10（web应用top10安全风险清单）等一般的脆弱性防御
- IP黑白名单设置
- 机器学习通信Pattern，Block异常通信
- 地域地理区域通信保护和限制

### Cloud Router

- 支持静态路由和动态路由的功能的总称

### Cloud NAT

- 私有IP和公有IP的转换功能
- 高冗长和可用性，统合Operation Suite有监控支持，可以自动灾害恢复
- 对外变换和对内变换，对内变换的时候推介增加使用Firewall过滤

### Cloud Load Balancing

- 不需要Pre-Warming，几秒就可以扩展的高可用性托管型服务
- 分类：三个维度分类
  - Global/Region
  - 内部还是外部（设置中会进行选择，是internet to serverless还是serverless之间）
  - HTTP/S（应用）还是TCP/UDP（网络）
- Proxy（代理模式）和PassThrough（直通模式）：
  - 代理是不通过VM直接和代理连接，但是有Port和Protocol的限制
  - PassThrough是直接和VM连接，需要光缆回线支持，支持全Port和Protocol
  - 代理模式适合需要集中管理和优化流量的场景，如SSL/TLS终止、内容缓存和流量路由等
  - 直通模式则适合对延迟敏感、需要直接与后端服务器通信的场景，特别是在客户端和服务器之间需要端到端加密时
- Health Check功能
  - 针对Layer4：TCP健康检查，UDP健康检查
  - 针对Layer7：HTTP/S健康检查，gRPC健康检查
  - 从而实现MIG（托管型Instance集群）的自动恢复

### Cloud CDN

- CDN的目的：减轻Origin的负担，提高对Client的requests的响应速度
- *Max-age*指标，是刷新CDN内容的时间长度指标，trade-off：信息的新鲜度和Origin服务器的负担
- *Expires*指标，明确指定cache的过期时间，但是优先度上Max-age存在（因为它优先度高）的话，这个指标会被无视
- 通过对*Cache Key*的调整，可以提高*hit率*，降低对Origin服务器的负荷
- *ETag*是内容的version，如果有较老的version内容，它会向Origin服务器要求更新
- *Last-Modifed*表示了内容最后一次更新的时间
- 一般会和*LB负载均衡配合*使用，Origin服务器不会留下CDN的访问日志，但是LB会留下CND的Access Log

### Cloud DNS

- Record：
  - A：IPv4
  - AAAA：IPv6
  - CNAME：别名，应该和AWS一样无法别名不带www的
  - MX：针对mail的
  - SOA（Start of Authority）记录是每个DNS区域（zone）的起点和权威信息来源。它包含该区域的关键管理数据，用于定义域名服务器如何处理区域内的DNS查询和同步。SOA记录是任何DNS区域文件的第一个记录，通常用于指定管理信息。
  - NS：Name Server，指定DNS服务器
- 高性能，高检索能力，100%可用性，自动伸缩

### Apigee

- API的设计，开发，测试服务
- OAuth功能，流量异常检测，API使用情况和性能等分析功能
- HTTP，REST，gRPC，GraphQL等多种接口
- 支持本地环境和混合环境
- API Gateway，Cloud Endpoints等服务可联动使用

## Database

- DBaaS操作简单，高性能，Point-In-Time-Recovery，BuckUp
- 数据库的指标主要是吞吐（Throughput 读写速度）和延迟（Latency 网络延迟）
- **数据库瓶颈原因**主要从软件和硬件分析：
  - 软件：应用构造，DB构造，Query原因
  - 硬件：CPU，RAM，*Storage*（disk容量/IO/HDD/SSD）
- 解决方案：
  - Read Replica：提高读写吞吐
  - In MemoryDB：提高处理速度
  - 分布式处理：数据分散和并行处理比如 Hadoop

### Cloud SQL

- 关系型数据库
- Cloud SQL Auth Proxy：不使用公有IP，而是使用私有IP的代理进行安全的（TLS/SSL）通信转发，不需要新的IP地址，使用IAM可以管理权限
- 创建instance类型后就可以创建database，这里面的database创建=schema创建，一开始会有一些系统schema：sys,information_schema,performance_schema等
- 执行了操作后，都会被记录，在控制台也可以通过operation tab进行确认

### Cloud Spanner

- 支持全球范围的水平伸缩
- 高可用性99.999的SLA
- OLTP强事务处理整合性（global动态伸缩下的高整合性是很难的）
- 托管型，标准SQL使用
- 加密，IAM访问管理，监察log
- ACID属性复习：
  - 原子性（Automicity）：要么成要么不成
  - 一致性（Consistency）：动作前后数字总额一致
  - 独立性（Isolation）：每个处理是独立的不相互干涉，前后顺序当然要正确
  - 耐久性（Durability）：数据会永久保存不会丢失
- Sharding：数据水平分割扩展，使用random的分割方式，更容易均等分割
- 它的workspace写query的地方和BQ的还是挺像的，同时左边有information_schema等信息情报，同时table的情报比较详细，包括key之类的
- 支持从Dataflow的数据流入，和对GCS的数据输出

### BigTable nosqlDB

- 用例和特点：金融数据，IoT数据，时间序列数据，低延迟，高吞吐，实时应用，大数据批处理
- 表由单一的主key和列群（Column-family）构成，自动伸缩
  - 一个record，是一个Partition单位
  - Column-family中都是key-value组合数据，每一个record可以有不同的key-value数据
- Hadoop Eco System互换，相关应用可以立刻移植，HBase互换，支持HBase API
- 支持多种数据源：API连接，Streaming处理的数据，batch处理服务来的数据

### Firebase appDB

- 适合移动应用开发，NoSQL的document数据库
- 数据层级是：Collection（document的集群单位）- Document以filed区分，每一个filed又有自己的Collection单位
- 在线数据是实时同步的，这对应用开发是很重要的
- 多region，高可用性，耐灾害性
- 作为app的后端使用即可
- 两种mode：
  * Native的Firestore：document类型
  * Datastore：key-value类型
  * 这两种都是强整合类型，Native支持Android，IOS，C++，Unity等手机应用库，后者不支持
- 可以设置TTL，数据的生存时间
- 可以设置复合index，这和DynamoDB有点像

### Cloud MemoryStore

- 支持 Redis 和 Memcached 两种开源数据库，RAM 存储所以很快很贵，也因为它的性质，所以最好定期存储到storage中
- Redis 相对于 Memcached 的优点是：
  - Snapshot功能，和AOF选项（Append Only File）利于数据备份和存储恢复
  - 支持事务处理transaction
  - Clustering集群化高可用性
- 实时分析，Messaging，Job Queue等use case
- 全托管，高可用性，自动伸缩

## Data Analytics

- 两个数据仓库
### BigQuery

- 实时数据处理，流数据插入，高可用性，高扩展性，操作简易，标准SQL，可以连动SpreadSheet
- 列存储，降低I/O负担，支持列的压缩，更加降低了I/O的数据量
- 多样性的使用方法：
  - 无缝模型创建训练和预测（BigQueryML）
  - Connected Sheet，可以直接对ss进行查询
  - BQ GIS，可以直接对地理数据进行分析和查询
- 费用包括分析和存储两个部分

### Dataproc

- 使用 Hadoop 和 Spark 服务的服务
- 将存储和计算分离，存储主要使用 GCS
- 支持高速启动和关闭 Hadoop/Spark 集群，可以使用临时集群（长期集群则是固定的不会被删除）降低费用，通过对Yarn内存的监控，进行自动伸缩
- Hadoop系统层级：从上到下：
  - MapReduce（分布式计算框架）
  - YARN（集群的资源管理系统）/HBase（分布式数据库，补全了HDFS的功能，支持碎片化的小的数据处理）
  - HDFS（大规模分布式文件系统）
- 支持大规模的分布式ETL处理或者机器学习，商业报告
- Dataproc Hub是在GCP上的Jupyter Lab笔记本

### Pub/Sub

- 有Queue的非同期处理
- Topic-Subscription
- Publisher-Puller
- Ack通信，受信确认，在被ack之前，信息是会被复制的，确保不丢失
- *At-least-Once*：最少一次配信，配信可能回超过1次，如果确保只一次的话需要其他的功能加持
  - 比如Dataflow的 Deduplication 可以确保 *Exactly-once* 的配信
  - 默认不确保配信顺序，但是顺序和伸缩以及可用性是一种trade-off
  - 再送信可以设置指数退避策略
- 支持Global接收信息，可以将信息保存在地理最近便的地方，当然如果有Region限制的话，也可以设置
- 针对Topic可以创建Snapshot
- 发送的message的形式可以通过schema tab进行具体设置


- 下面的是数据处理服务
### Dataflow

- 大规模分散型数据处理，基于**Apache Beam**，支持Java和Python快速编码，支持batch和流处理
- 组件：Data Source -> I/O Transform -> PCollection ... -> Data Sink
  - PCollection是数据Pipeline
  - Transform是数据处理的node
- Job也有可视化的视图

- 流数据处理中将一定范围的数据分割进行处理的单位是**Window**
  - *滚动窗口（Tumbling Window）*是长度固定且不重叠的窗口。它们将数据流按固定时间段（如每5秒、每分钟等）切分成连续、互不重叠的块。每个事件只属于一个窗口。适用于需要在固定时间间隔内进行统计和计算的场景，如每小时的销售统计、每分钟的流量分析。
  - *滑动窗口（Sliding Window）*与滚动窗口类似，但窗口之间可以重叠。窗口的起始位置根据一个滑动间隔（Slide Interval）来确定，这个间隔可以小于窗口长度，从而导致窗口重叠。适用于需要在重叠时间段内持续进行计算的场景，如实时平均值计算、异常检测等，需要更细粒度地观察数据变化。
  - *会话窗口（Session Window）*根据事件之间的“静默期”（即没有事件发生的间隔）来定义窗口的边界。窗口会在数据活跃时继续延长，如果在设定的静默期内没有新事件到来，窗口就会关闭。会话窗口常用于跟踪用户会话、用户活跃度分析等场景，如电商网站用户的购物会话、网络流量的会话分析。

### Data Fusion

- 可视化GUI功能的数据处理工具，支持*Batch和Spark Streaming*
- 通过Pipeline，Node，DAG表现

### Dataprep

- 数据前处理和清理服务，可以*自动检测数据不整合和欠损*
- 也是一个可以GUI纯操作的服务，在可视化的界面进行数据预处理
- 自动伸缩，和其他服务联动，处理好的数据可以用于数据存储或者机器学习


## AI & ML

- Cloud Vision：图像识别，物体检出，文字检出，实时
- Speech-To-Text/Text-To-Speech：多语言支持的声音文字转换
- Cloud Natural Language：文字内容和情感分析，分类
- Cloud Translation：翻译
- Cloud Video Intelligence：视频中的物检，注释功能
