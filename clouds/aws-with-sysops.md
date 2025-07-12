## 思考

尝试一下，不要什么都记录，而是记录自己思考的结果，毕竟很多构架图还是需要回去看slides

运维到底是什么，这个考试在说什么。整体在说什么。

预防，保护，检测，修复各个步骤。这很像安全专家考试了。

全体的：
- 管理统筹监控
- 安全认证合规
- 预算管理

普通的服务器：
- 存储
- 计算
- 容器
- 数据库
    数据专属：
    - 分析
    - 数据迁移和传输

- 网络通信

针对应用程序部署的：
- 应用集成（对应于系统设计的应用部分）


系统设计中的各个重要部分包括：

以下是针对系统设计中各个重要部分的 AWS 云服务列表：

1. 服务器架构：
   - Amazon EC2（Elastic Compute Cloud）：提供可扩展的计算能力，支持在云中运行虚拟服务器实例。
   - AWS Auto Scaling：自动调整 EC2 实例的数量，以应对流量变化和负载需求。
   - Amazon Lightsail：提供简单易用的虚拟服务器实例，适用于小型应用和网站的部署。

2. 数据库设计：
   - Amazon RDS（Relational Database Service）：提供托管的关系型数据库服务，支持多种数据库引擎，如 MySQL、PostgreSQL、Oracle 等。
   - Amazon DynamoDB：提供托管的 NoSQL 数据库服务，具有高性能、高可用性和弹性扩展的特点。
   - Amazon Redshift：提供托管的数据仓库服务，用于大规模数据分析和数据处理任务。

3. 网络架构：
   - Amazon VPC（Virtual Private Cloud）：提供私有网络环境，可自定义网络配置、子网划分、路由表等。
   - Amazon Route 53：提供高性能的域名解析服务，支持域名注册、托管区域、负载均衡等功能。
   - AWS Direct Connect：提供专用网络连接服务，将本地数据中心与 AWS 云服务直接连接起来，实现低延迟、高带宽的网络通信。

4. 安全设计：
   - AWS Identity and Access Management（IAM）：提供身份认证和访问控制服务，用于管理用户、组、角色和权限。
   - Amazon Cognito：提供身份认证和用户管理服务，支持用户注册、登录、认证和授权。
   - AWS Certificate Manager：提供 SSL/TLS 数字证书管理服务，用于保护数据传输的安全性和完整性。

5. 性能优化：
   - Amazon CloudFront：提供全球内容分发网络（CDN）服务，加速静态和动态内容的传输，降低延迟和提高响应速度。
   - Amazon ElastiCache：提供托管的内存缓存服务，用于加速数据访问和减轻数据库负载。
   - AWS Lambda：提供无服务器计算服务，按需执行代码，无需管理服务器资源，实现高度弹性和低延迟的计算。

6. 扩展性设计：
   - Amazon SQS（Simple Queue Service）：提供消息队列服务，实现系统组件之间的解耦和异步通信。
   - Amazon S3（Simple Storage Service）：提供高度可扩展的对象存储服务，用于存储和管理大规模的数据和文件。

7. 测试和调试：
   - AWS CloudFormation：提供基础设施即代码（Infrastructure as Code）服务，用于自动化部署和管理 AWS 资源。
   - AWS CodeDeploy：提供持续交付服务，用于自动化部署应用程序到 EC2 实例和其他计算资源上。

8. 部署和运维：
   - Amazon CloudWatch：提供监控和日志管理服务，用于收集和分析系统的指标和日志数据，实现实时监控和自动报警。
   - AWS Systems Manager：提供系统管理服务，用于自动化运维任务、安全补丁管理和配置管理。

9. 用户界面设计：
   - Amazon API Gateway：提供 API 管理和部署服务，用于构建和发布 RESTful 和 WebSocket API。
   - Amazon Sumerian：提供 VR 和 AR 开发平台，用于构建交互式的虚拟现实和增强现实应用程序。


开发工具/
机器学习/
量子技术/
区块链/
物联网/
机器人技术/

主体：人，环境（服务器，数据库，应用），数据，内外

### 管理监控统筹部分（management，monitoring，governance）

**多账户multi-account管理服务**

- Orgnization组织管理很重要。统筹作用。
  - 想删除组织内创建的账户，需要这个账户有自己的支付方式和详细信息。
- ControlTower的landing zone中也可以设置很多关于组织的东西，比如log archive acount，账号监察用的账号，都是关于安全用的，还有各种服务的限制等。它也像是一个统筹的服务。其他还有kms的设定，cloud Trail的设定等。和**安全有关**的很多。
- Service Catalog也是多账户服务，制定可以使用的service范围。规定大家可以用什么不可以用什么。
  - tag option library

**monitoring监控服务**

- CloudWatch
  - 可以做为很多后续步骤的一个trigger。也有自己的可视化仪表盘insight了。alarm，SNS，Eventbridge都可以自由组合。
  - CloudWatch RUM 是一个可以实时关注application性能的功能。
  - CloudWatch ServiceLens 可以可视化分析application的机能。
  - Cloudwatch synthetics 可以监控和测试应用程序的端点。RestAPI，URL，网络内容。
  - 要整合不同region的监控数据，需要cli将数据聚合到cw。
- CloudTrail以人为主体。
- AWSConfig以环境为主体。
  - 可以统筹多账户，多区域的构成config管理。
- AWS Health以时间为线索。

**management统筹管理**

- System Manager好像最重要。
- License Manager就是管理证书是否过期，就是很多其他的软件的。
- Trusted Advisor提供最佳实践的建议。

**Deployment部署服务：**

- CloudFormation超级重要，IaC。重点是基础设施的自动化。变更集功能，可以只更改代码差分的部分并且留有记录，很方便。
- OpsWorks是应用程序的管理服务，也是代码化。Chef（recipes和cookbooks）和Puppet（manifest，应用程序即代码服务，和dataform好像）。

### 安全认证和合规（security，identity，compliance）

**用户和连接管理** Identity and access management

- IAM是对所有在环境中进行活动的用户的管理。
  - 策略中Deny的优先级最高。
  - 混乱代理问题的解决：ExternalID的设置，或者IAMAccessAnalyser用于验证对方身份。（和workloadIdentity好像）
  - ABAC（tags）和RBAC是基于属性和基于role的访问权限。
  - PermissionBoundary权限边界可以限制user的权限。
  - 给人分配一个临时用的角色使用role而不是user。
- STS(security token service)临时的认证。

**验证（Authentication）和授权（Authorization）**

- AWS Directory Service创建管理集成用户目录，然后直接可以使用AWS的资源。
- Amazon Cognito针对应用程序的认证。
  - 针对应用程序的user管理和认证。
  - 针对外部的idp应用的id federation认证。（这个像workloadidentity）
  - user-pool是认证用，id-pool（单独也可以完成认可认证两件事）是认可用。（idp：3rd-part）

**加密和保护** Encryption and protection

- AWS KMS数据加密。
  - key的删除最短要7天，删除后加密的数据无法恢复。
- CouldHSM
  - 符合FIPS 140-2密码学标准。
- AWS Certification Manager加密认证网络通信的数字证书。
- AWS Secrets Manager保护密码。
  - 主要是数据库，然后还有其他的key-value。
- Amazon Security Hub整合下面所有。
- Amazon Macie保护个人信息。
- Amazon GuardDuty针对网络和账号安全。
  - DNS log
  - VPC flow logs
  - Cloud Trail
- Amazon Detective针对GD找到的findings之类的问题进行深入分析。
- Amazon Inspector针对应用程序的安全分析检测。

**网络安全** Network security

- AWS Firewall Manager各种防火墙的集成。好麻烦。他和SecurityHub是一个性质和层级。
- AWS WAF针对网络应用程序的防火墙。
- AWS Shield防DDos的。

### 网络（Networking）
**虚拟网路服务**（VPC：虚拟云，局域网，子网的集合）
- VPC/Subnet
  - route table 是路由表设定。设定IP的跳跃路径。
  - security group 是一种防火墙。
- VPC Flow logs
- VPC Peering
  - 局域网之间的连接。
- VPC Endpoint
  - Gateway:S3/DynamoDB(route table setting) 使用PrivateLink技术。EC2/Lambda -> S3/DynamoDB
  - Interface:50+service(ENI&DNS setting)
- ENI（弹性网卡）
  - ENI 是 Elastic Network Interface 的缩写，是一种虚拟网络接口，允许你在 AWS (Amazon Web Services) 的云计算实例中连接到 VPC (Virtual Private Cloud) 中的一个或多个子网。ENI 可以附加到实例上，充当实例与 VPC 之间的桥梁，允许实例与其他资源进行通信，包括其他实例、互联网和其他 VPC。
  - 当在 AWS EC2 上创建一个新的实例时，默认情况下会分配一个新的 ENI（Elastic Network Interface）。这个 ENI 会附加到实例上，并与所选 VPC 中的一个子网相关联。通常情况下，每个 EC2 实例都会有至少一个默认的 ENI。
  - 这个默认的 ENI 会负责实例的网络通信，并且通常会自动配置一个内部 IP 地址。你可以根据需要配置额外的 IP 地址、安全组、子网等属性。

Egress-Only 在网络领域通常指的是一种网络配置，用于 IPv6 地址。在这种配置下，IPv6 地址只能用于出站连接，而不能接收入站连接。这种配置通常用于提高网络安全性，防止未经授权的入站连接。具体来说，Egress-Only 是一种 IPv6 地址类型，它允许主机向外部网络发出连接请求，但不允许外部网络主动连接到主机。这对于一些服务器和设备来说是一种常见的安全设置，特别是在需要与互联网通信的情况下。

### 存储（Storage）

**数据对象管理** 主要是S3

- S3
  - Amazon S3 Inventory 管理 Amazon S3 存储桶中的对象。它可帮助创建存储桶中对象的完整列表、跟踪对象的更改、满足合规性要求、提高数据治理以及优化存储成本。

**计算机服务的存储**

- EFS：network file system。mount。add by acl or security group。
- EBS
  - 从snapshot恢复的ebs需要一个初始化，一开始读数据很慢。空的不需要初始化。
  - Throughput Optimized HDD (st1):适用于大型、高吞吐量的工作负载，如数据仓库、日志处理等。提供了较低的价格，并支持高吞吐量。最大吞吐量为 500 MB/s。
  - Provisioned IOPS SSD (io1):适用于需要更高 IOPS（每秒输入/输出操作数）的工作负载，如数据库和应用程序。最多支持 64,000 IOPS。每秒1000MB吞吐。
  - 使用DLM服务控制生命周期。
- FSx

**备份服务**

- AWS Backup


### 数据库（database）

**关系型数据库**relational

- RDS
  - RDS Proxy ：用于帮助提高 Amazon RDS 数据库的扩展性、可用性和安全性。它充当应用程序和 RDS 数据库之间的中间层，有效地管理数据库连接。
- Aurora
- Redshift
  - 没有跨区域复制功能（cross-region-replication）。你只能复制snapshot。

**NoSQL**

- DynamoDB
  - global table 可以实现 region 共享。
  - scan 是过滤整个表，query是根据键查询的，更有效率。

**In-memory**

- ElastiCache

### 应用程序集成（Application Integration）

- SQS
- SNS
- SES：适合富文本的电子邮件分发服务。
- EventBridge

### 分析（Analytics）

- OpenSearch service

### 数据转移（Migration and Transfer）

- Snow Family
- Transfer Family
- DataSync

### 预算管理（cost management）

- Cost Explorer
- Cost and Usage Report
- Savings Plans


## EC2

- EC2虚拟服务器
  - *计算存储分离*的思想很重要，所以改变instance类型的操作只适合EBS的实例
  - AMI和instance type的区别是啥：
    - AMI是一个*预配置的虚拟机模板*，包含操作系统、应用程序、配置文件和启动实例所需的软件环境，决定了*跑什么*
    - instance type指实例运行时的*硬件配置*，包括计算能力（CPU）、内存、存储和网络性能，决定了*跑多快*，比如你需要ena增强版网络配置就需要更好的type
  - Placement Group 实例集群物理分布布局，在网络和安全里设置
  - trouble shooting：比如ssh连接，和EC2 connect（AWS API经由），从SG入手
  - 购买plan的区分，还有买断整个服务器host的选项呢，安全需求了算是
  - EC2相关的CW监控指标中RAM，processes，used disk spaces不包括在内的，使用*Unified CloudWatch Agent*可以做到，namespace是CWAgent（需要设置IAM Role）
  - *为什么stop-start一个EC2会有新的ip4的地址？？*是因为每次这样的操作（注意不是reboot）背后是server被migrate到一个新的hardware服务器上
  - EC2的*三层status check问题*：第一层是system层面，主要是底层的host，通过stop和start重启一个新的host，第二层主要是instance层，通过重启或更改配置也许可以解决问题，第三层是EBS的问题，比如可达性和IO操作，通过重启实例或者对EBS进行置换（*这种上层理解蛮重要的！*）
- AMI镜像
  - 可以*加速启动*新的instance，因为预装软件会预先被安装，比如httpd啥的
  - Cross-Account Shared的AMI，需要同时分享你的KMS加密key，对方进行重新的decrypt和用自己的key进行encrypt
  - EC2 Image Builder 自动化创建和测试AMI，只需要付存储费用，哦哦他还可以创建Docker image呢！
- Auto Scaling自动扩张（总是和LB结合使用）
  - step scaling
- Lambda（serverless）小型电脑的感觉

### System Manager

- 需要SSM agent的安装（Amazon Linux AMI已经预先安装了），以及IAM Role，不需要ssh或http/s连接
- SSM 的 Resource Group 功能通过资源的 Tag 实现，还挺方便的，tag多比少好
- *Documents*（yaml） -> Run Command
- *Automation* 运维和部署任务的自动化，Automation Runbook（也是一种Document但是类型是Automation）
- *Parameter Store* 集成 CloudFormation 和 KMS 等很多功能，并且name有层级设置（文件夹那种感觉，所以可以一次获得一整个层级的内容）
- *Inventory*：收集所有的EC2实例的源数据 metadata 的功能，激活后通过 *state manager* 进行 metadata 收集的功能，state manager 是用以维护实例群组保持一个固定状态（比如软件，补丁配置等）的功能，作业单位叫做 *Association*
  - 通过 *Resource data sync* 功能可以将数据导入 s3 用 athena 等其他工具进行分析
- *Patch manager* 自动化升级OS，application，security
- *Session Manager*
  - 无需ssh就可以连接服务器，所以无需security group的设置，只需要有ssm agent
  - IAM Permissions 通过权限限制用户和组，通过tag限制访问的资源
  - session log 可以输出到cloud watch或者s3

## EC2 high scalability & availability

### Load Balancer
- ALB
  - 对于微服务，container-based服务非常好用
  - 可以基于以下进行routing：path in URL，hostname in URL，Query String，Headers
  - target group（背后的对象）：
    - EC2 instance（Auto scaling group）通过 HTTP 等
    - ECS tasks 也通过HTTP
    - Lambda Function：HTTP request会被转换为 json event
    - IP addresses：*必须是private ips*，比如EC2或者一些on-premises的servers
  - health check可以在target group level进行
  - client ip 存在于*X-Forwarded-For*中，servers不会看到真正的client ip，同样的 client 也只能看到alb的hostname，是一个fix的hostname，中间你会看到*elb*就知道那个是alb的网址了
  - Listenr rule（*监听规则*）包括：host header，source ip，Path，http request method 的灵活设置
    - 这个 Path 规则我没有很懂，感受一下：/error设置后可以返回404或者是文字not found等结果

- NLB
  - 一个AZ只能有一个static ip的设置
  - 满足百万级请求超低延迟，因为她是第四层协议，没套那么多消息
  - target group：
    - EC2 instance
    - IP addresses
    - ALB：这个组合很有意思，前面只有一个static ip，后面可以设置很多http监听规则的组合
  - health check 支持TCP，HTTP，HTTPS协议

- GWLB
  - 在网络层面，第三方network virtual appliances流量聚合
  - 比如firewall，IDPS，检测和监控设备
  - 在layer 3 network layer上检测ip packets
  - Transparent Network Gateway + Load Balancer 功能组合
    - *Transparent Network Gateway（透明网络网关）*是一种网络设备或软件配置，其主要特性是在不改变网络设备IP地址或网络架构的情况下，提供网络流量的监控、管理或安全功能。它通常在网络中以“桥接”模式运行，类似于透明地嵌入在网络路径中，而不会干扰正常通信。
  - 协议是 GENEVE 6081
  - 架构：route table - GWLB - target group（3rd party virtual appliances）- 如果ip包没问题就会返回 GWLB 然后流量到 applications 中去
  - target group 中是 ips 并且必须是 private ips

- *sticky session*（session affinity）使用 cookie 实现
- *cross zone load balancer*：跨区域的全server上实现平衡的负载均衡，alb是默认带的功能，不要钱，nlb不默认，要钱
- *SSL certificates*：使用 X.509 证书
  - client 使用 *SNI*（server name indication）协议（一个很新的protocol），识别他们访问的 hostname，这样一来，一个server上就可以共存很多SSL certificates，该协议只能用于ALB，NLB，CloudFront
  - 通过 SNI 才能实现，LB背后有多种 target group
  - 设置方法，是在 LB 的 listener 中设置新的 https 监听规则，然后 forward 到 ACM 中去
- *connection draining*：等待要被从target group中删除的EC2的连接结束的功能，可以设置一个时间比如0～3600s之间，来允许仍在处理的request结束连接
- 备注：trace id 这个东西一般和背后一个集群相关的monitoring吧，比如k8s
- target group可以设置不同的权重 weights，比如用于 blue/green 发布

### Auto Scaling Group

- 可以和*LB结合*使用，功能更加强大呗，创建的时候就可以和LB的 target group 建立连接link
- 可以通过*CW的Alarms*进行规模缩放，比如 *CPUUtilization，RequestCountPerTarget，Average Network In/Out*，或者各种custom metrics
- 可以根据 *schedule* 进行 scaling
- 还有一种 *predictive* scaling，是根据 forecast 进行预测性 scaling
- *cooldown period*：默认300秒，进行scaling后需要instance启动后到一个稳定状态，不能一直检测，为了加速或者减少这个period，最好有一个配置好的AMI用于快速启动
- *lifecycle hooks*：在启动和terminate一个实例的时候进行额外的pre和post操作，是一种脚本执行吧，和*Event bridge*集成，比如invoke一个lambda，或trigger一个SNS或者SQS
- ASG 和 *SQS* 结合：比如用Queue Length作为metric，如果太长代表太多message没有被处理，就trigger scaling out
- 其他ASG服务：EC2 spot fleet requests，ECS，DynamoDB，Aurora

## ElasticBeanstalk

- 方便开发者的managed服务，架构模式可以是single instance，也可以是高可用性的多AZ的ASG模式
- ELB + ASG 的3-tier web架构，或者SQS + ASG的worker架构
- backend：CloudFormation

## CloudFormation

*yaml*:
- 格式来说，冒号：表示 key-value pairs
- 有array表达，就是用 ‘-’ 来表达的，一个元素一个 ‘-’ 前缀，当然每一个元素也可以是 key-value pair

- 关键IaC服务，可描述性构架，并快速实现构架分离
- Cost可控可监是非常重要的
- template是放在S3中的，*这也同样意味着你的template必须是版本管理的*
- template咋做：一个是code editor去网上找模板，一个是infrastructure composer，这个没用过，真的OK吗，深表怀疑
- 最佳实践当然是用yaml文件定义并用CI/CD工具进行自动更新
- 有Rollback功能，自动删除或者恢复到之前的状态。可以选择保留fail情报，手动修复后，用特定api（ContinueUpdateRollback API）继续进行create或update

- 最小权限：User 持有 *iam：PassRole*，CF 持有 *Service Role*
- *CAPABILITY_IAM* or *CAPABILITY_NAMED_IAM*（被命名）权限用于指定CF有权创建IAM相关资源
- *CAPABILITY_AUTO_EXPAND* 权限用于在使用了macros或者nested-stack的时候自动扩展和变更

- *DeletionPolicy* 用于在template被删除或者变更时候*resource的动作*，默认delete，但是S3 bucket如果不为空则会发生错误
  - 针对部分资源保留需要yaml中设置：DeletionPolicy：Retain
  - 删除前创建快照：DeletionPolicy：Snapshot
- *StackPolicy*：一种json的权限定义，主要用于保护个别资源，不被人为定义的错误template意外删除，比如production环境的database等
- *TerminationProtection*：保护stack不被删除的option，GUI设置

- Custom Resources：用于定义AWS尚未支持的资源，On-premise资源，以及3rd-party资源，这个*可以用来自动化在stack删除s3桶之前，删除桶内的资源*

- Components中的注意点：
  - Resources
  - Parameters：dynamic inputs，可以设置 AllowdValues 限制选项
  - Mappings：static variables，比如region，AZ，AWS account，env等，如果你预先知道有哪些用mapping比较好
  - Outputs：创建的资源输出，这个terraform中也有呢，目的是在其他的stack中可以使用，*Outputs：Export：*关键字，使用其他stack的outputs的时候，使用*ImportValue*关键字，在stack之间有了依存关系后，*delete的时候要按照顺序进行*
  - Conditions：资源创建条件，可以使用很多逻辑关键字
- Helpers？：
  - References
  - Functions：*Fn::*
    - *Fn::Base64* 用于保证 UserData 正确的传递 encoded data 给EC2

- Dynamic Reference：支持动态地从 Parameter store 或者 Secrets Manager 取得password之类的资源

- Helper Scripts：cfn-xxx
  - cfn-init：使得EC2的初始配置更可读，我觉得就是user-data的进化方式
  - cfn-signal + WaitCondition

- *Cross Stacks*：使用 Output Export 和 Fn::ImportValue 实现跨 stack 资源复用
- *Nested Stacks*：可以被其他stack复用的嵌套stack，很像是编程里面的module，需要承認那个 CAPABILITY_AUTO_EXPAND 的功能
- *DependsOn* 属性，表达资源之间的依存关系
- 组织OU专用管理账户 + *stacksets*，可以自由在子组织中部署同样的资源

## Lambda

- lambda + EventBridge（cron or rate）
- S3 Event Notification + lambda 是一个异步处理架构
- Lambda 驱动其他服务：*Lambda Excution Role*，其他服务驱动 Lambda：*Resource based policies*（user access or service access），这两个都可以在lambda的权限tag确认到
  - 案例：Lambda + SQS 的架构，是 Lambda 用一个Lambda excution role去确认有没有 message 的情况，而不是被SQS驱动

- 监控：CloudWatch logs，或 X-Ray（设置里面是active tracing）

- RAM设置从128MB到10GB，是和CPU性能绑定的，如果你需要高计算任务，增加RAM
- timeout：3s～900s（15m）
- Excution Context 是lambda运行的临时环境，但是不需要每次运行都启动
  - 在handler中设置db connection是非效率的，因为每次启动都需要连接，最好是在handler之外进行db connection设置
- /tmp 是一个临时工作目录，最大10GB，但是要永久储存还是s3

- *并发concurrecy*
  - 支持高并发但是有 limit，如果超出限制就会被限流叫做 throttle
  - 限流的情况下，同步执行会导致429错误，异步执行会自动retry（指数退避），retry也失败会打入DLQ
  - *值得注意的是*这个limit包括对你所有的lambda functions的限制如果你一个app就吃了所有的限制，那么其他的app就会没法跑，比如alb的应用也有，api gateway的应用也有，还有sdk的invoke，那么如果你alb吃了大部分的limit，其他的就跑不了了
  - reserved & provisioned concurrency
  - 在每个lambda的configuration中可以设置每个lamdba的并发limit

## EBS & EFS

**EBS**
- 是一种network drive，是和AZ绑定的（所以要绑EC2必须要同AZ），就像一个network USB stick
- 创建EC2的时候*delete on termination*选项可以保护EBS不被删除
- EC2 instance store 的速度和性能高，但是在ec2 stop后会被删
- multi-attach：io1/io2 family，同一个AZ，一次顶多16个ec2
- resize一个EBS你需要重新给Linux做分区处理，好麻烦
- 跨区复制用snapshot
- Data Lifecycle Manager
- 从没加密的volume创建的snapshot一定也是没加密的，创建完才能加密

**EFS**
- NFS协议
- 用SG控制权限
- 可以Multi-AZ
- 只能用于Linux系统
- 低延迟需求
- 比较贵，但是有很多storage class帮你减少cost
- *这部分东西如果说没有很好滴理解，那是因为基础部分薄弱，比如Linux等*
- EFS Access Point 限制用户的访问范围，POSIX操作系统接口规范
- EFS 加密不能直接enable必须创建新的然后用 DataSync 转移数据

## S3

- versioning：如果你删除一个文件后他会有个 delete maker，但是你可以删除这个 maker，这相当于重新拿回文件，这很有趣，所以这个对象是一个*操作*
- 处于审计需要，replication操作中，不会复制你*删除特定版本*的操作（因为这看起来就像是恶意删除记录），只会复制delete maker
- S3 Event Notification -> SQS/SNS/Lambda，需要设置相应的IAM Permission，可以通过*json rule*来设置更精细的event过滤，这个没用过
- 性能baseline：取回数据5500request/s，操作和放入数据3500r/s，很高
- Multi-part功能可以使得*上传速度快*（文件超过5G）*实质上是进行并行操作*，那么下载提速用的是*Byte-Range Fetches*，也是一种分割文件数据并行操作的方法
- s3 transfer acceleration支持region之间的高速传送
- *S3 batch operations*是对文件进行批量处理的功能，需要定义文件的metadata，用json或者csv文件
- *S3 inventory*可以输出S3的文件的元数据报告
- Glacier的作为Vault的一系列功能，比如保证不删除，用于审计功能

## Athena

- columer data可以节省cost，比如parquet
- 数据压缩提升小型数据查询效率
- s3数据分区查询提升效率，比如用year，month，day划分文件夹名
- 用large files可以减小overhead，因为大文件容易scan
- Data Source Connector，联合数据查询，比如对关系型数据库，非关系型数据库，object，以及on-premise数据查询

## Advanced Storage

- Snow Family：边缘计算和数据传输
- FSx：lanch第三方高性能file system
- Storage Gateway：Block，FileSystem，Object系统和On-premise连接

## Cloud Front

- Origin不仅可以是S3，也可以是任何Http后端，比如ALB（后面是private的EC2），EC2，S3 website等
- 一般用来全球分布，以降低用户网页延迟
- Allowlist 和 Blocklist 可以用来限制geo地区访问

## Databases

- RDS的存储用的是EBS，根据设定指标可以自动伸缩
- read-replica是异步的，数据具有最终一致性
- read-replica 在 cross-region 的时候是产生 network cost 的
- 灾难恢复上使用multi-AZ即可，可以使用同步复制 sync replication，这一操作不需要停机，是zero-downtime的，只需要modify即可
- *RDS Proxy*：这个功能存在的意义，是其他服务（比如lambda）连接RDS的时候，方便从public subnet进行连接，同时还可防止，过多的connections连接到RDS对数据库造成压力
- *DB Parameter Groups*：是设置DB的参数功能，静态参数立刻生效，动态参数需要reboot。⬅️这个我猜意味着，静态的是参数文件config，动态是要嵌入代码的所以要重启。
  - SSL connection相关的参数：PostgreSQL/SQL Server 的 rds.force_ssl，MySQL/MariaDB 的 require_secure_transport，设置为1
- *RDS Events*：可以将events送达SNS进行订阅/EventBridge

- **Aurora** 具有高可用性和Read的高扩张性，write和read各有自己的endpoint，在read各个节点之间有负载均衡功能。
- 在 Aurora 的所有读写节点之间有 *shared storage volume* 的功能，从10GB可扩展到128TB
- backup 是restore到一个新的DB，backtracking 是inplace的restore，不会创建新的DB，clone使用同一个volume创建一个新的DB

- **ElastiCache**：managed Redis/Memcached
- 作为*RDS的缓存*，存储*session data*（以保持app是stateless的）使用
- redis的数据可以永久存储，memocached则只有临时数据
- 读写分布方式 *Cluster Mode*：
  - 这个mode是enabled的时候，所有的节点node endpoints都有读写功能，当他是disabled的时候，是primiry node+很多read node（endpoints）的方式
- **Redis Scaling** 功能：
  - *Cluster Mode disabled*：
  - 水平扩张主要是针对read replication增加节点
  - 垂直扩张，会内部重新创建 node group，然后进行数据复制（data replication）和DNS更新
  - *Cluster Mode enabled*：
  - 水平扩张的时候，要进行resharding，以及 shard rebalancing操作，支持online/offline两种模式
  - 垂直扩张，是改变node的type，支持online模式

## Monitoring, Auditing and Performance

- *CloudWatch* dashboard，metrics，Logs/insights
- Live Tail是一个debug用的新功能，能对log group实时进行log获取
- CloudWatch Alarm --> EC2/Auto Scaling group/SNS
- Composite Alarms是各种alarms的逻辑集合，可以减少不必要的聒噪预警
- *Synthetics Canary*功能，针对website，API，SAAS应用，模拟用户行为，进行主动监控，发现问题并报错而不是等用户反馈
- *EventBridge*在服务集成上非常重要
  - Event pattern
- *Service Quotas*是一种服务额度监控的功能，是Trusted Adviser的高级平替，进化版，和CW组合可以发布预警
- *CloudTrail*：Compliance，Audit，Governance for SDK/CLI/Console/IAM
  - Management Events，Data Events，Insights Events
  - 他们的存储时间是90天，所以需要的话存储进S3，然后用Athena进行访问和分析
  - 和EventBridge服务组合
  - 日志保护，存储进S3后hash文件，防止日志篡改，同时增加对S3桶的保护，对CloudTrail的保护则使用IAM权限管理
  - 可以统合组织的Trails
- *Config*：针对资源resource的配置进行管理
  - 服务自身没有deny的功能
  - 可以配合 SSM Automation Documents 对不合规的配置进行自动修复，也可以创建自己的Documents来触发Lambda函数进行配置操作等
- *Control Tower*
- *Service Catalog*
- *Billing Alarms*：在CloudWatch中进行设置，针对实际花费
- *Cost Explorer*：cost分析工具
- *AWS Budgets*：结合了上面两个服务的功能，分析cost和alarms消费
- *Cost Allocation Tags*
- *Usage Reports*
- *Compute Optmizer*

## Disaster Recovery

- AWS DataSync：数据移动，数据同步（S3，EFS，FSx）
- AWS Backup：支持cross-region，和cross-acount
- Backup Vault Lock：WORM（write once read many）

## Security and Compliance

- Shared Responsibility Model
- DDos保护服务：AWS Shield / WAF / CloudFront / Route53
- Penetration Testing，有aws的允许和不允许的活动范围
- Inspector：
  - EC2 通过SSM agent实现
  - ECR 是在image被push到ECR的瞬间
  - Lambda Functions：在被deployed的时候
  - 可以集成 Security Hub 和 Event Bridge
- GuardDuty
- Macie：敏感信息PII
- Trusted Advisor：关于系统最佳实践的官方建议，不需要设置
- KMS/KMS Key Policies
- HSM：hard ware
- Artifact：下载合规报告
- ACM 用于传输加密，us-east-1
- Secret Manager 和RDS中各种数据库的集成很好
  - Cloud Trail会记录它所有的API calls

## Identity 认证

- Credential Report
- User -> Last Access
