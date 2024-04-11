## AWS SysOps 甚至统筹全部

---
### 思考

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



### 管理监控统筹部分（management，monitoring，governance）

**多账户multi-account管理服务**

- Orgnization组织管理很重要。统筹作用。
- ControlTower的landing zone中也可以设置很多关于组织的东西，比如log archive acount，账号监察用的账号，都是关于安全用的，还有各种服务的限制等。它也像是一个统筹的服务。其他还有kms的设定，cloud Trail的设定等。和**安全有关**的很多。
- Service Catalog也是多账户服务，制定可以使用的service范围。规定大家可以用什么不可以用什么。

**monitoring监控服务**

- CloudWatch可以做为很多后续步骤的一个trigger。也有自己的可视化仪表盘insight了。alarm，SNS，Eventbridge都可以自由组合。
- CloudTrail以人为主体。
- AWSConfig以环境为主体。
- AWS Health以时间为线索。

**management统筹管理**

- System Manager好像最重要。
- License Manager就是管理证书是否过期，就是很多其他的软件的。
- Trusted Advisor提供最佳实践的建议。

**Deployment部署服务：**

- CloudFormation超级重要，IaC。重点是基础设施的自动化。
- OpsWorks是应用程序的管理服务，也是代码化。Chef（recipes和cookbooks）和Puppet（manifest，应用程序即代码服务，和dataform好像）。

### 安全认证和合规（security，identity，compliance）

**用户和连接管理** Identity and access management

- IAM是对所有在环境中进行活动的用户的管理
- STS(security token service)临时的认证。

**验证（Authentication）和授权（Authorization）**

- AWS Directory Service创建管理集成用户目录，然后直接可以使用AWS的资源。
- Amazon Cognito针对应用程序的认证。

**加密和保护** Encryption and protection

- AWS KMS数据加密。
- AWS Certification Manager加密认证网络通信的数字证书。
- AWS Secrets Manager保护密码。
- Amazon Security Hub整合下面所有。
- Amazon Macie保护个人信息。
- Amazon GuardDuty针对网络和账号安全。
- Amazon Detective针对GD找到的findings之类的问题进行深入分析。
- Amazon Inspector针对应用程序的安全分析检测。

**网络安全** Network security

- AWS Firewall Manager各种防火墙的集成。好麻烦。他和SecurityHub是一个性质和层级。
- AWS WAF针对网络应用程序的防火墙。
- AWS Shield防DDos的。

### 存储（Storage）

**数据对象管理** 主要是S3

- S3

**计算机服务的存储**

- EFS
- EBS
- FSx

**备份服务**

- AWS Backup

### 计算（Compute）

- EC2虚拟服务器
- Auto Scaling自动扩张（总是和LB结合使用）
- Lambda（serverless）小型电脑的感觉。

### 网络和内容传递（networking Content delivery）

**虚拟网路服务**（VPC）
- VPC/Subnet
- VPC Flow logs
- VPC Peering
- VPC Endpoint
- ENI（弹性网卡）

**VPN**

**Gateway**
- AWS Transit Gateway

**Content Delivery Network**
- CloudFront
- Global Accelerator

**DNS**
- Route53

**Load Balancer**
- ELB

### 数据库（database）

**关系型数据库**relational

- RDS
- Aurora

**NoSQL**

- DynamoDB

**In-memory**

- ElastiCache

### 应用程序集成（Application Integration）

- SQS
- SNS
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
