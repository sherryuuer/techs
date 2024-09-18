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

## Storage

## Network

## Database

## Data Analytics

## AI & ML
