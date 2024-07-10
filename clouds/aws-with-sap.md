## Identity & Federation

### IAM Policies精细控制用户权限
- IAM Role和Resource Based Policies：
  - IAM Role是用户赋予一个Role进行资源操作（这个时候，完全无法使用user原有的权限了）
  - RBP是在资源中进行Polies限制（比如S3 Bucket Policies）（这个时候user不需要放弃自己的其他权限）
  - 后者似乎应用场景更灵活
- IAM Permission Boundary：权限边界，这是一个高级功能，限制了用户的权限边界，对用户定义的任何权限如果超过了这个设置，都是不可用的
  - 三者结合：组织SCP + IAM Permission Boundary + 用户个人权限
- IAM Access Analyzer：分析任何针对资源的*外部用户* -- 若有，通过finding进行记录
  - 设置Zone of Trust：表示对一个资源的信任主体集合，超出这个集合的，都会被finding记录
  - 自动生成Policy的功能构架：通过Cloud Trail记录过去90天的资源的API call记录，来自动生成Policy，这种构架默认你90天内的活动是合规的，但是如何保证是合规的呢，如果有人错误使用了该服务进行了外部连接？
    - 不过说回来，是可以通过人工检查进行细节调整的

### STS认证
- 跨账户的临时用户权限，通过发行token，使得其他账户的用户可以使用自己账户的临时Role进行操作
- 对于第三方账户用户，可以通过*ExternalID*进行精准控制，保证进来的是特定的用户，该ID是秘密的
  - 在没有ExternalID识别的情况下，如果Role的*ARN被泄漏*，就可能造成攻击者擅自进行assume-role，进行访问攻击
- session tag：也是一种控制用户访问权限的方法，在IAM Policy中可以通过该tag设置权限条件
- CLI流程：`assume-role`命令，然后通过得到的临时Credentials，设置环境变量
  - 当有使用MFA的时候，使用[官方手顺](https://repost.aws/knowledge-center/authenticate-mfa-cli)，`get-session-token`参数为账号信息，你的设备名和token code（当时显示的），然后同样得到临时认证信息后，设置环境变量，就可以使用了
- 相比较sts spi，AWS推介使用cognito来控制权限

### Identity Federation

- 如果不想给用户创建AWS user，并且有自己组织的AD等认证系统的情况
- 使用web/mobile应用进行服务连接的情况
- SAML和SSO都是这个范畴
  - AssumeRoleWithSAML
  - SAML的构成图可以看到其认证过程，是先在公司内部得到认证服务器的认证，然后通过AWS的STS获得访问资源的权限
    - 该过程不管是访问资源，还是登陆浏览器的Console界面，还是用ADFS访问公司资源，都是*用户通过STS*，界面访问，返回的是控制台的sign-in URL
  - Custom Identity Broker的工作机制略有不同，他不是认证用户后由用户取得token，而是*broker自己去AWS拿临时role的token或者URL*，发给user使用
- Cognito很高级，很像是user版本的Workload Identity，不再需要用户直接去STS取得token，而是用cognito的token去STS交换token
  - 人称 Token Vending Machine（*TVM*）代币贩卖机！基本就是一个交换token的小机器w
  - 有匿名功能，数据同步功能，和MFA功能

### Micorsoft的AD目录服务回顾

- 组织中的个体是object，包括各种用户和设备
- 他们以tree的形式进行组织，他们的集合是forest
- 域控制器（Domain Controller, DC） 是实现和托管目录服务的服务器，在Microsoft的环境中，它指的是运行Active Directory服务的服务器。
- 微软有自己的ADFS（AD Federation Service）服务，也是一个联合认证的服务
- 支持单点登录SSO
- Sharepoint也是和它的AD服务集成的应用

### AWS的目录服务

- *AWS Managed Microsoft AD*
  - 可以在AWS的云中创建AD服务
  - 和本地AD之间建立Trust连接（AD two-way forest trust），可以同时进行管理
    - 通过 DX 或者 VPN 连接才能建立
    - 这种信任，包括单向和双向，这影响app的访问认证路径
  - 高可用性：可以分布式AZ构架
  - 集成多种AWS资源管理
  - 这种服务本质是一个目录*数据库*，就像是RDS，可以进行backup备份
  - 将本地AD复制到AWS中，以降低服务延迟
    - 这需要在VPC中的EC2中实现，将本地AD直接复制到这个EC2中，同时EC2的AD服务也需要和AWS Managed AD之间建立trust连接
    - 在DX或者VPN不可用的时候，这保证了user可以继续使用AWS服务
- *AD Connector*
  - 和上面的不同，本质上只是一个连接本地AD的Gateway
  - 需要 DX 或者 VPN 连接
  - 没有缓存服务
- *Simple AD*服务则没什么大的功能，就是便宜简单，小规模，不可scale，也不能连接本地AD，不能建立trust连接等

### AWS OU

- 组织管理，是账户的集合，有价格优惠，所有的子账户共享优惠，因为他们被看成一个账户
- 账户的移动需要先脱离原本的OU，再接受新OU的邀请

### SCP

- OU级别管理的黑白名单
- 应用于OU和Account级别
- 它默认not allow anything，所以需要显式允许
- 不起作用：Management Account，服务相关角色（service-linked role）
- 起作用：Users，Roles，Root User
- 每个OU层级的SCP都受制于上面一层OU的SCP

- *权限判断顺序*：
  - 显式Deny则不可用
  - OU SCP
  - Resource-based policies
  - Identity-based policies
  - IAM permissions boundaries
  - Session Policies

- *Tags限制策略*：资源控制policy
  - 当满足tag条件，则可以使用资源
  - 满足一个任何tags（条件ForAnyValue），或者满足所有tags（条件ForAllValues）
- *region限制条件*：RequestedRegion条件
- *Opt-out policies*：限制你的内容被AWS用于AI/ML服务提升
- *Backup Policies*：设置备份计划的策略，在 member 账户中只能浏览不能修改

### IAM Identity Center

- SSO，单点登录服务
- 集成多个账户，单界面登录
- 登录的时候可以集成*AD目录服务的认证功能*，也可以用它*内置的identity store*进行认证管理
- 认证后就可以接到如下服务：
  - AWS资源（*EC2 Linux Instances不支持*，因为Identity Center一般和AD紧密集成，但是Linux一般是非AD环境）
  - Business Cloud Apps：比如Box，365，Slack，salesforce等
  - Custom SAML2.0-enabled Apps
- 连接服务受*Permission Sets*的限制，定义了用户可以使用哪些服务
  - 用户可以被group，然后对不同的group设置不同的Permission Set，比如对某OU的full access权限，对某OU的只读权限等
- *ABAC*（Attribute-based access control）：对用户还可以基于属性进行权限控制

### Control Tower

- 基于OU的服务，快速管理多账户组织
- 安全和合规管理
- 有dashboard
- Account Factory：快速创建账户
- 集成Service Catalog服务管理
- Guardrails（护栏）：
  - 护栏通过 AWS Config 规则和 AWS 服务控制策略（SCPs）来实现。AWS Config 规则检查资源的合规性，而 SCPs 则在 AWS 账户间管理权限。
- Guardrails的三个级别：
  - *Mandatory* 强制性Guardrails：是应用于 AWS Control Tower 下所有账户的不可移除的护栏。它们执行关键的安全和合规要求。
  - *Strongly Recommended* 强烈推介的Guardrails： 默认未启用，但是强烈建议，是基于best practice的
  - *Elective* 可选的Guardrails：允许组织根据其独特的需求定制治理

### RAM（Resource Access Manager）

- 在OU内跨账户共享资源，避免资源重复
- 可以在同一个（分享subnet的情况）subnet内进行账户间的资源共享，各账户 的资源（比如EC2）可以相互通信，但是不能使用，因为他们所属于不同账户
- CIDR（Prefix）list的共享，在高级网络的部分也有提到
- Route53 Outbound Resolver：分享 Forwarding rules 给其他账户的Resolver Outbound Endpoint

## Security

### CloudTrail

- 默认开启
- 主体是AWS account
- 可以记录all region的也可以single region
- 集成CW Logs和S3
- events/API calls活动记录
- 资源被删了先看trail记录

*Events类型*：

- Management Events：
  - 默认开启
  - 所有的账户管理API，资源更改
  - Read Events/Write Events
- Data Events：
  - 默认不开启
  - 可以分为Read和Write Events
  - Lambda Function执行活动（Invoke API）可以显示被执行了多次
- Insights Events：
  - 会持续分析write events，通过正常events来生成一个baseline，然后detect非正常events
  - 异常情况会在CloudTrail的控制台显示，也可以发送到S3桶，基于自动化需要也可以生成EventBridge Event

*Event Retention（保留）*：

- 90天
- 存储用S3，分析用Athena

*Event Bridge 集成构架*：

- 基于特定的 API call logs，生成event，触发 EventBridge 后的操作，比如发送SNS等

### Architectures

- +S3：
  - 每五分钟将logs发送到S3进行dump，之后可以利用S3各种功能（lifecycle，SNS，SQS，Lambda，versioning，lock等
  - 当然CloudTrail自己也可以发送通知到SNS，然后触发SQS，或者Lambda

- +Multi-Account/Multi-Region：
  - 多账户集成logs，设置一个SecurityAccount，S3集成其他账户的Logs
  - 方法一：cross-account role / 方法二：设置bucket policy（需要允许cross-account delivery）

- +CW：
  - CloudTrail通过stream方式，传送logs到CW Logs
  - 后续使用CW的各种功能：Metric Filters，触发CW Alarm，接SNS和Lambda等

- OU Trail：OU级别的Trail聚合
  - 只能创建于Management Account中，然后聚合各个子账户的trails，并存在S3中

- 对Trails事件的快速响应组合服务：
  - EventBridge：是最快的响应方式
  - CW：stream传输，日志分析
  - S3：可以每五分钟传送一次，适合跨账户，长期存储，以及用Athena分析

### KMS

安全专家的[KMS部分内容](aws-with-security.md###KMS)

### Parameter Store

- 集成Cloud Formation
- 有版本管理功能
- 通过IAM Permission进行安全管理
- 通过文件夹形式的层级进行管理
- Parameter Policies可以设置TTL也就是过期时间，集成EventBridge，在过期前通知，或者在该发生变化，而没发生变化的时候通知

### Secrets Manager

- 强制在X天后进行Rotation
- 使用 Lambda 可以自动生成新的 Secrets：需要 Lambda 对 SecretManager 和数据库的访问许可，包括对subnet的网络设置（VPCEndpointorNATGateway）
- 内部使用KMS进行加密，对称加密，发送GenerateDataKeyAPI进行信封加密，需要双方service的权限
- 与各种DB集成用于密码存储和轮换
- 跨区复制功能，用于灾难恢复，跨区DB和跨区应用等
- 可以用Resource-Based policies来控制访问权限

- 集成很多服务，可以从中pull secrets
- ECS（worker/task）也集成使用SecretManager和SSMParameterStore的key进行RDS和API的连接。

*跨账户的Secret分享*：

- 无法使用RAM
- KMS给对方账户 viaService：SecretManager 的 kms：decrypt 解密权限
- SecretManager给对方账户 Resource-based Policies，使得对方账户可以访问

*Parameter Store 和 Secret Manager*：

- PStore更加便宜，但是没有自动rotation，但是可以通过Lambda和EventBridge进行自动rotation的设置
  - 如果设置RDS的密码，SecretManager可以每30天自动触发Lambda修改RDS密码，PStore则需要自己去设置EventBridge每30天触发Lambda更新RDS和PStore的密码
- SecretManager必须用KMS，但是P来说KMS不是必须选项
- 都和CloudFormation集成

### SSL/TLS和CA认证的基本过程

- TLS是SSL的进化，但是人们习惯说SSL，是在传输中加密的方式
- 加密过程非对称加密很贵，对称加密便宜。
- 在SSL传输过程中，一般第一次握手会通过一个非对称加密验证身份：
  - 客户端hello
  - 服务端发送SSL证书（也就是CA，是一种PublicKey）
  - 客户端使用该SSL证书，发送自己的对称加密后的MasterKey
  - 服务端用自己的私钥解密被加密的MasterKey，验证客户端身份
  - 开始对称加密的安全连接

### SSL SNI（Server Name Indication）

- 解决的问题：一个web server加载了多个SSL证书的情况，用于识别是哪个server
- 在ALB/NLB/CloudFront上起作用（CLB不行，它需要多个CLB，每个host一个证书）

### 防止中间人攻击

- 使用HTTPS而不是HTTP
- 使用DNSSEC

### ACM

- 维护SSL证书是非常难的，这个服务集成了AWS服务，并且自动更新，是非常好的服务
- 用于：
  - Load Balancers（包括在Elastic Beanstalk上创建的）
  - CloudFront distributions
  - APIs on API Gateways
- 使用方式：使用CLI上传自己的证书，自己维护和更新，或者使用ACM的，让他帮你自己管理和更新
- Load Balancer会在客户端和服务端之间，终止SSL传输，并验证，然后向服务端开启新的HTTP通信，这将降低EC2解密的CPU负荷
- 公有证书Public CA服务需要开启公有DNS服务，私有证书Private CA服务用于内部服务安全验证
- 它是一个*regional服务*，所以如果用于全球服务，需要在你app发布的每个region设置证书

### CloudHSM（Hardware Security Module）

- 硬件加密，专用硬件，硬件由AWS管理但是*加密是自我管理*，非free，必须*安装客户端软件和HSM进行SSL连接*，管理keys和users。
- 使用SSE-C加密的时候是一个好的选择。
- IAM permission：CRUD an HSM Cluster，意味着只能对Cluster进行创建删除等，但是内部的管理靠你自己
- Multi-AZ，高可用性
- 和KMS的CustomKeyStore集成，所以APIcall可以被CloudTrail捕获
- 和其他Account共享cluster：通过共享其所在的subnet，设置subnet的SG来允许traffic

### Architecture：SSL

- SSL on ALB:
  - client -> HTTPS -> ALB(SSL cert from ACM) -> HTTP -> Auto Scaling Group EC2 Servers
- SSL on web server EC2 Instances:
  - client -> TCP -> NLB -> HTTPS -> Auto Scaling Group EC2(M5) Servers -> (Retrieve SSL private key from) Parameter Store (user data) install cert on EC2
- SSL Offloading:
  - 上述构架中将cert载入EC2，会造成高CPU负荷，所以这里等解决方案，是在后端将cert，offload入HSM进行处理
  - 这种情况SSL的private key就不会离开HSM设备，这提高了安全性
  - 需要在HSM上设置一个加密用户（cryptographic user）（CU），将用户名和密码保存在Parameter Store中，EC2从PStore中获取用户名和密码，以访问HSM中的private key

### S3

- 存储中的数据有各种加密方式
- 传输中的数据（SSL/TLS加密），endpoint推介使用HTTPS
- 客户端加密SSE-C中，HTTPS是强制的
- 强制使用HTTPS可以设置Bucket Policy：aws:SecureTransport

- S3 Event Notification --> SNS/SQS/Lambda

- EventBridge事件驱动：
  - **需要有效化CloudTrail Object level logging on S3 first**
  - 也就是说EventBridge的使用，CloudTrail的log开启是必要条件，会记录data event log（Lambda之类）
  - SNS/SQS/Lambda

- S3 Bucket Policy：
  - SourceIp/VPCSourceIp
  - Source VPC/Source VPC Endpoint
  - CloudFront Origin Identity
  - MFA

- Pre-signed URLs
  - 有期限的URL用于上传和下载内容
  - 会继承，生成该URL的人的权限GET/PUT

- VPC Endpoint Gateway
  - Private Instance --> VPC Endpoint Gateway --> S3
  - 私有连接S3 bucket policy可以设置SourceVpce或者SourceVpc来限制访问源

- S3 Object Lock：防止version被删除，写一次读多次WORM
- Glacier Vault Lock：防止数据未来被修改，也是写一次读多次WORM

- S3 Access Points：
  - 文件夹可以作为一个访问入口
  - 分配不同的 Access Policy 控制访问
  - 每个访问点可以有自己的 DNS 名
  - VPC Origin：限制从自己的VPC的访问，需要*VPC Endpoint*，以及对*Access Point*和*Bucket*双方的访问Policy设置
- S3 Multi-Region Access Point：
  - 在基本功能的基础上，使区域间的bucket双向复制
  - Active/Passive
  - Failover

- S3 Object Lambda：
  - 通过Access Points和Lambda的过滤功能，对特定的object使用者提供不同的（cleansed）数据
  - 比如删除敏感数据，变更文件格式，以及和其他数据库进行数据新处理后进行使用等需求
  - 不需要创建新的bucket，在原有的bucket的基础上进行处理和数据使用即可

### DDoS 防御服务

- AWS Shield
- WAF（based on rules）
  - layer7:ALB/APIGateway/CloudFront/AppSync（保护GraphQL APIs）
  - 它不能直接防DDoS，但他可以设置ACL白名单过滤或者限制访问，甚至有针对BotControl的rule group
- CloudFront & Route53
- AutoScaling
- 分离静态（S3/CloudFront）和动态（EC2/ALB）资源

### Firewall manager

- 组织 Organization 级别的管理，各种防火墙服务在一处管理，Firewall Manager可以一次性统一部署FW在许多账户 accounts。
- 新创建的资源会立刻适用这些rules
- 包括：WAFrules/Shield/SG/NetworkFirewall/Route53ResolverDNSFirewall
- 创建的是Region level的Policy

### Block a IP

- 两种构架，block IP：
- client --> VPC(NACL)(ALB(with WAF ip filter) --> EC2)
- client --> CloudFront(with WAF ip filter) --> VPC(NACL)(ALB --> EC2)
- (组件之间通过SG的白名单允许通过)

### Inspector

- EC2的网络可达性
- 针对*EC2 instance/Container Images/Lambda Function*的网络漏洞分析 -> 会以一个risk score来表达漏洞的优先级

### AWS Config

- 根据Config rules检测*合规与否*以及*设置变化过程*而不阻止变化，无法deny，可以SNS通知
- view CloudTrail API calls if enabled
- region level，需要在每个region都设置
- 跨region和account聚合数据
- 修复：
  - Config -> EventBridge -> Lmabda
  - Config -> SSM Automations

### 各种AWS管理的logs

- Load Balancer access logs -> S3
- CloudTrail logs -> S3/CW Logs
- VPC Flow Logs -> S3/KDF/CW Logs
- Route53 access logs -> CW Logs
- S3 access logs -> S3
- CloudFront access logs -> S3
- AWS Config -> S3

### Guard Duty

[安全专家链接](aws-with-security.md###GuardDuty)

### EC2 Instance Connect

[安全专家链接](aws-with-security.md###EC2instanceconnect（browserbased）)

### AWS Security Hub

[安全专家链接](aws-with-security.md###AWSSecurityHub)

### Detective

[安全专家链接](aws-with-security.md###Detective)

## Compute & Load Balancing

![Architecture](solution-architecture-aws.png)

### EC2 instance types

**image type**

- R(RAM):in-memory caches
- C(CPU):compute&databases
- M(medium/balanced):general/web app
- I(I/O):instance storage/databases
- G(GPU):video rendering/machine learning
- T2/T3:burstable in capacity
- T2/T3 unlimited:unlimited burst

**launch type**

- on demand instance
- spot instance
- reserved
  - 最短一年
- dedicated instances
  - 无人分享你的hardware
- dedicated hosts
  - 预定整个物理服务器

**EC2 included metrics**

- CPU
  - CPU利用率（CPU Utilization）
  - CPU Credit Usage/Balance：适用于T系列突发性能实例，表示使用的CPU积分和剩余的CPU积分，用于在短时间内提高性能
- Network：In/Out
  - 网络流量和带宽使用情况
  - In/Out表示接收和发送的数据量
- Status Check：
  - instance status = check the EC2 VM：实例运行健康状况
  - system status = check the underlying hardware：底层硬件的健康状况
- Disk：Read/Write for Ops/Bytes（only for instance store）
  - 实例存储的IO状况
  - 每秒读写操作数，每秒读写字节数
- 注意，RAM（内存使用情况）不包括在内
  - 内存是计算机系统中至关重要的组成部分，负责临时存储数据和指令，支持程序的高效执行和多任务处理。
  - 内存的性能和容量直接影响计算机系统的整体性能和用户体验。

### EC2 Graviton

- 基于 ARM 架构的高性能、低成本处理器
- 支持Linux各种而不支持windows instances

### EC2 placement group

- EC2的放置策略
  - Cluster：单个AZ中集合的instance群
    - low-latency但是rack损坏则全体不可用
    - 适合大数据低延迟计算，高网络吞吐
  - Spread：在多AZ上进行硬件的隔离分布，高可用性
    - 限制：7 instance per AZ per placement group
  - Partition：跨许多分区（partition）的spread，这些分区实际上坐落于不同racks的集合
    - rack是指物理数据中心的一组服务机架
    - Hadoop/Cassandra/Kafka/HDFS/HBase

### HPC

- 大数据传输服务：DX，Snowball&Snowmobile（数据到云），DataSync（本地到S3/EFS/FSx for Windows）
- 超级计算机是AWS中各种服务组合的运用
  - EC2 Placement Groups比如Cluster
  - instance type比如GPU，Spot instance/Spot Fleets + Auto Scaling
  - EC2 Enhanced Networking（SR-IOV）
  - Elastic Fabric Adapter（EFA）
  - [网络专家跳转链接](aws-with-networking.md###EC2NetworkperformanceOptimization)
  - 存储：EBS/InstanceStore
  - 网络存储：S3/EFS/FSx for Lustre
  - 自动化和编排（Automation&Orchestration）：AWS Batch/AWS ParallelCluster
    - ParallelCluster是开源的HPC管理工具，用textfiles进行设置，自动创建各种服务组件比如VPC，subnet，EC2等

### Auto Scaling

- 通过设置目标值（比如average），或者CW的阈值进行自动伸缩
- Scheduled Actions：如果知道某时段需要增加unit，可以设置schedule
- Predictive Scaling：可预测性伸缩，不断学习和分析，ML技术
- 主要指标：CPU使用率，网络In/Out，RequestCountPertarge和其他自定义指标
- *Spot Fleet support*：支持spot和on-demand实例混合
- *Lifecycle Hooks*：在启动或者关闭一个实例之前可以进行特定的动作的功能，比如cleanUp，抽出日志等
- *Upgrade AMI*：必须update launch configuration/template
  - 然后可以手动关闭旧的实例，或者使用CloudFormation关闭
  - 也可使用*EC2 Instance Refresh*，在启动新的实例的时候会自动使用新的AMI
- 所有Auto Scaling Processes：
  - Launch
  - Terminate
  - HealthCheck：EC2，ELB
  - ReplaceUnhealthy
  - AZRebalance
  - AlarmNotification
  - ScheduledActions
  - AddToLoadBalancer
  - InstanceRefresh

- *AutoScaling 策略构架*
  - 在ALB的同一个target group中发布新的template，和旧的template同时使用，慢慢删除旧template实例，流量会在所有实例中分布
  - 在ALB的不同target group中发布新的template，只分配一小部分流量给新的实例查看效果，慢慢过渡
  - 创建全新的ALB和它的target group，发布新的template，使用*Route53 CNAME weighted record*，分配流量给两个ALB群组进行逐步过渡，以及对新的ALB群组进行手动testing

### Spot Instance & Spot Fleet

- 适合batch作业
- 定义max spot price，当没超过阈值的时候会自动增加spot instance数量，如果超过则自动关闭（在2分钟之内）
- Spot Fleets：set of Spot Instances + (Optional) on-demand Instances
  - 在满足最低价格的限制条件下，自动满足目标capacity需求

### Amazon Fargate work with ECS/EKS

- ECS use case：
  - Run MicroServices
  - Run Batch processing/Scheduled Tasks
  - Migrate Apps to Cloud
- ECS concepts:
  - ECS Cluster
  - ECS Services
  - Task Definitions(json file )
  - ECS Task
  - ECS IAM Roles
- ECS 集成 ALB：使用Dynamic Port Mapping，分配流量：可以在一个EC2实例上发布多个container

- Fargate是一个Serverless的服务，无需构架基础设置，直接添加container

- ECS安全：集成SSM ParameterStore和Secret Manager
- ECS Task Networking：None，host（使用底层host的interface），bridge（虚拟网络），awsvpc（使用自己的ENI和私有IP）

- ECR
  - 支持image漏洞扫描（on push），版本控制和lifecycle
    - 基础漏洞扫描由ECR进行，并触发EventBridge发布通知
    - 高级Enhanced扫描通过Inspector进行，由Inspector触发EventBridge发布通知
  - 通过CodeBuild可以register和push image到ECR
  - 可以cross-region 或者 cross-account 地 replicate image
