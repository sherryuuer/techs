## AWS安全专家考试相关内容

---
时间戳2024年3月。

安全威胁有哪些？使用哪些工具？构成如何？

"Remediation对安全漏洞的修复和解决" 通常是一个持续的过程，因为安全漏洞和威胁不断变化，需要及时采取相应的措施来应对。安全团队会定期进行漏洞扫描和安全审计，并根据发现的问题制定和执行 remediation 计划。

---

在云服务环境中，安全威胁的类型和形式多种多样，其中一些常见的安全威胁包括：

1. **数据泄露**：云存储和数据库中的敏感数据可能因配置错误、访问控制不当或恶意活动而被泄露，导致机密信息暴露给未经授权的用户或攻击者。

2. **账户和凭证泄露**：云账户凭证、API 密钥或访问令牌可能会因为不慎泄露、社会工程攻击或恶意软件而被窃取，从而导致未经授权的人员访问云资源。

3. **拒绝服务（DoS）攻击**：攻击者可能会发起拒绝服务攻击，通过向云服务提供商的服务器发送大量恶意流量或请求来使服务不可用，影响业务正常运行。

4. **恶意软件和恶意行为**：恶意软件可能会通过各种方式进入云环境，并在其中传播、操纵或损坏数据、应用程序或系统。此外，攻击者可能会在云环境中执行恶意活动，例如挖矿、勒索等。

5. **不安全的 API 和管理接口**：云服务提供商的 API 和管理界面可能会因为设计缺陷、配置错误或漏洞而受到攻击，导致未经授权的访问或恶意操作。

6. **数据隔离问题**：云环境中的虚拟化技术和多租户架构可能会导致数据隔离不严格，使得攻击者能够跨越虚拟边界，访问其他用户或租户的数据。

7. **身份和访问管理（IAM）问题**：不当配置的身份和访问管理策略可能会导致未经授权的用户获得对云资源的访问权限，或者允许攻击者提升其权限级别。

8. **社会工程和钓鱼攻击**：攻击者可能通过欺骗手段诱导云服务用户或管理员泄露凭证、敏感信息或执行恶意操作，例如钓鱼邮件、社交工程技术等。

9. **未经授权的配置更改**：云资源的配置可能会因为人为错误或恶意行为而被更改，导致安全性和合规性风险。

以及其他安全风险。

---

加密货币攻击是什么？

加密货币攻击指的是针对加密货币和区块链技术的各种安全威胁和攻击。由于加密货币和区块链技术的快速发展和广泛应用，这些攻击已成为互联网安全领域的一个重要议题。

以下是一些常见的加密货币攻击类型：

1. **钱包窃取和欺诈**：攻击者通过恶意软件、钓鱼网站或社会工程攻击等手段窃取加密货币钱包的私钥或用户凭证，以获取非法访问和控制加密货币资产的权利。

2. **交易欺诈**：攻击者通过欺诈性交易、双花攻击或恶意操纵交易等手段来欺骗交易对手，非法获取加密货币资产。

3. **挖矿恶意软件**：挖矿恶意软件会在受害者的计算机上运行，利用其计算资源来挖掘加密货币，导致计算机性能下降、能源消耗增加和硬件损坏。

4. **智能合约漏洞**：智能合约是基于区块链的自动执行代码，存在漏洞可能导致合约被攻击、资金被盗取或合约执行失败。

5. **51% 攻击**：当一个恶意实体（或者控制多数计算力的联合实体）控制了区块链网络的多数计算力时，它可以操纵交易、双重花费和阻止交易的确认等，从而破坏网络的正常运行。

6. **网络攻击**：包括分布式拒绝服务（DDoS）攻击、网络分区攻击、Sybil 攻击等，这些攻击可能导致网络拥塞、节点无法访问、交易延迟等问题。

7. **社交工程攻击**：攻击者使用社交工程技术来欺骗用户泄露私钥、访问凭证或其他敏感信息，以获取对加密货币资产的控制权。

加密货币攻击是一个动态和不断发展的领域，随着加密货币和区块链技术的普及和发展，攻击者不断寻找新的攻击方法和漏洞。因此，加密货币用户和相关组织需要采取相应的安全措施，加强对加密货币资产和交易的保护。

---

在 AWS 中，可以使用以下服务和功能来应对各种安全威胁：

1. **数据泄露**：
   - 使用 AWS IAM 和 AWS KMS 管理和控制对存储在云中的数据的访问权限。
   - 使用 Amazon S3 Bucket Policies 和 AWS IAM 来限制对 S3 存储桶中对象的访问。
   - 使用 Amazon Macie 来监控和发现 S3 存储桶中的敏感数据，并实施数据分类和保护策略。

2. **账户和凭证泄露**：
   - 使用 AWS IAM 进行身份和访问管理，强制实施多因素身份验证（MFA）、定期轮换凭证、使用 IAM 角色等安全最佳实践。
   - 使用 AWS Secrets Manager 或 AWS Systems Manager Parameter Store 管理和保护敏感的凭证和密钥。

3. **拒绝服务（DoS）攻击**：
   - 使用 AWS Shield 提供 DDoS 保护，帮助防止和缓解网络层攻击。
   - 使用 AWS WAF 防火墙来过滤和阻止 Web 请求中的恶意流量。

4. **恶意软件和恶意行为**：
   - 使用 AWS WAF 防火墙和 AWS Firewall Manager 来监控和阻止恶意流量。
   - 使用 AWS GuardDuty 和 Amazon Macie 监控和检测恶意行为和异常活动。

5. **不安全的 API 和管理接口**：
   - 使用 AWS IAM 和 AWS Organizations 来限制和管理对 AWS 管理控制台和 API 的访问权限。
   - 使用 AWS Config 和 AWS CloudTrail 记录和监控对 AWS 资源的操作和事件。

6. **数据隔离问题**：
   - 使用 AWS VPC 和安全组来隔离和保护云中的网络流量和资源。
   - 使用 AWS IAM 和资源标签来实现对资源的访问控制和隔离。

7. **身份和访问管理（IAM）问题**：
   - 使用 AWS IAM 进行身份和访问管理，实施最小权限原则，并定期审查和轮换访问凭证。

8. **社会工程和钓鱼攻击**：
   - 提供员工培训以提高安全意识，并教育他们识别和防范社会工程攻击。
   - 使用 AWS IAM、AWS SSO 和 AWS Organizations 实施统一的身份和访问管理策略，减少钓鱼攻击的成功率。

9. **未经授权的配置更改**：
   - 使用 AWS Config 实时监控和评估 AWS 资源的配置，并确保符合安全和合规标准。
   - 使用 AWS CloudFormation 和 AWS Config Rules 自动化和强制实施安全配置。

以上列出的 AWS 服务和功能只是一部分可以用来应对各种安全威胁的示例。实际上，AWS 提供了更多的安全工具和功能，组合使用可以帮助用户建立高度安全的云环境，并保护其数据和应用程序不受安全威胁的影响。

---
安全服务一共包括六个板块：

### 1 - Treat detection and Incident response

**Guard Duty**

提供持续的监视、检测和保护 AWS 帐户和 AWS 已部署的工作负载。它利用机器学习和行为分析技术来检测恶意活动和未经授权的行为。GuardDuty 分析来自 AWS CloudTrail、Amazon VPC 流量日志和 DNS 日志的数据，以便识别与恶意行为相关的模式和异常活动。

**AWS Security Hub**

AWS Security Hub 可集成多种 AWS 服务（例如 Amazon GuardDuty、Amazon Inspector、AWS IAM、Amazon Macie 等）以及第三方安全工具（例如 Check Point、CrowdStrike、Trend Micro 等），从而提供全面的安全性检查和报告。可以跨区域管理，以一个区域为中心进行安全管理。必须激活AWS config服务。

过程：检测-摄取-修复-记录

**Detective**

**Penetration Testing**

### 2 - Security Logging and Monitoring

**Inspector**
**System Manager**：

Parameter store / Inventory / State Manager / Patch Manager / Maintenance Windows / Session Manager

Document 自动执行功能是我觉得亮眼的功能。

**Cloud Watch**

可以使用**Cloud Watch Log insights**进行日志分析。注意这只是一个query引擎而不是一个即时流，如果想要将log流处理需要用**CloudWatch Log Subscriptions**的filter功能将日志不断传入其他服务进行处理，分析，存储。它还可以整合不同区域的日志，将他们聚合起来。那么既然它是一个订阅服务，就需要有起点和目的地账户，并且需要在账户之间设置policy，以供分发和接收。尤其是接收端的权限放行。

可以通过设置metric的阈值（可以进行条件的组合比如AND，OR等，很灵活）发出alarm，可执行的动作，包括对EC2的启动关闭等操作，对Auto Scaling对扩张收缩，以及使用SNS发出通知等。

**Amazon EventBridge**（以前叫做CloudWatch Events）

**Amazon Athena**

**AWS CloudTrail**

包括管理事件，数据事件等，以及CloudTrail Insights，log保留90天，那之后需要存入S3。

QuickSight访问S3的时候如果缺少权限，可能是对KMS进行解码的权限。

可以集成EventBridge在遇到非法API操作的时候，发送SNS通知。和CloudWatch的metric filter一起使用可以更好地过滤日志，联动SNS通知。

可以进行组织organization层面的Trail收集。

**AWS Macie**

通过机器学习查找到S3中的敏感信息主要是个人识别信息PII，然后通过EventBridge进行通知等。

可以自定义正则化表达式进行匹配。还可以设置allow list允许某些信息。适用于组织。

**S3 Notifications**

有两种方式可以实现S3的通知，一种是Event Notifications，可以设置Object动作，以及后续的SNS，SQS，或者Lambda。另一种是Amazon EventBridge，将所有的通知发送到EB以供后续的动作。

**VPC Flow Logs**

**VPC Traffic Mirroring**

可以实时将一个或者多个EC2的网络流量日志，同时进行过滤，和发送到其他的服务比如NLB，以方便分析。

**VPC Network Access Analyzer**

分析网络进出是否合规。可以自定义规则access scope。

**Route53 Resolver Query Logging**

**OpenSearch**

### 3 - Infrastructure Security

**Bastion Hosts**

一种保护私有网络的方法，我的GCP项目中就使用到了这种构架。bastion的意思是堡垒，向外部开放22端口允许公开访问，外部的访问通过bastion访问私有服务器的内容，bastion和私有服务器中也是通过私有IP进行链接，以此保护私有服务器的地址不暴露，私有服务器不对外链接。

**Site to site VPN**

连接云端的VPC和私有数据中心的安全方法。需要设置VGW（虚拟virtual网关gateway）和本地的CGW（custom gateway）之间的连接。（这两个都在VPC服务页面设置），然后在同样的页面设置site-to-site-VPN connection。

- 需要在虚拟网关中开启路由传播功能：Route propagation
- 安全组设置，在VGW的inbound安全组中需要设置icmp协议的开启，不然从本地的服务器无法连接到云端ec2

如果本地的网关有多个，可以通过AWS VPN CloudHub来设置连接，实现一对多的连接。

**AWS Client VPN**

如果你的个人电脑上安装了open的VPN就可以通过设置这项功能，从自己的电脑连接到云端的EC2了，如果你的VPC同时使用了site-to-site-vpn那么也可以从自己的电脑很方便的连接到custom的数据中心。

**VPC Peering**

在同一个账户或者不同账户的不同VPC之间实现连接，仿佛你所有的行为都是在同一个VPC中一样。不可传递，想要连接的VPC必须两两连接。虽然行为看起来是在同一个VPC一样，但是需要更新EC2的路由表route table。

**DNS Resolution**

**VPC Endpoint**

**PrivateLink**

**NACL & SG**

**AWS Transit Gateway**

**CloudFront**

和S3的区域复制功能的区别：静态内容全球覆盖用CF，动态内容在几个区域覆盖用S3。CF可以设置访问国家的黑白名单。

**WAF**

web application firewall.layer 7.用于第七层的漏洞滤网。

**Shield**

防止DDoS攻击的。星际攻击防卫战。

**WAF & Shield & Firewall manager**

将各种防火墙服务在一处管理，可以一次性统一部署在许多账户。

**API Gateway**

包括一个API应有的所有功能。

构架上来说：

- 比如后面可以坐落一个LambdaFunction来被invoke，作为API的后端。
- 可以有一个http端点，这个端点可以是一个本地的服务器，也可以是一个ALB，所有可以成为http端点的东西。
- 或者后面可以坐落很多的AWS服务。比如后面可以有一个kinesis服务，让客户端进行更新推送。所以理论上可以通过API网关暴露你的所有服务。

安全认证方法：

- 内部用户可以使用IAM认证。
- 外部用户可以使用Conito服务。
- 还可以使用custom logic authorizer创建自己的认证方法。

HTTPS安全Custom domain name使用ACM（aws certification manager）：全球边缘优化管理，认证需要使用美国东部的region，如果是region endpoint，需要和api网关同一个区域设置。另外还需要在Route53中设置cname和A-alias。

学习笔记：

通过一个hands-on让我再次重新理解了API网关。它就像是在服务外面套了一层膜，让我们通过一个link的呼出，就能得到那个膜背后的信息。仅此而已。在发布API的时候设置的一个stage可以是开发产品环境的名字，也可以是version的名字，这就是我们经常说的版本号码。

API的安全问题：可以在resource policy中设置access condition来限制公共访问。而私有API必须通过特定的VPC源（VPC interface endpoint）进行访问，这是一个设置的条件。而在**两个VPC环境**中进行私有访问，不需要进行vpc peering，而同样是在一个中设置vpc interface endpoint在另一个中设置policy条件即可。

一些启发：学习云的好处就是可以通过这些服务的学习，作为一些接口，然后理解整个IT的底层原理，只要有心的话，可以通过不断挖掘，深挖知识点，加深对整个系统的理解，这个系统包括计算机系统，网络构架系统等。

**AWS Artifact**

下载安全合规文档和报告等地方。可以说不算是一个服务。

**Route53**dns服务

DNS中毒：是一种IP攻击手段，因为我们使用udp协议从服务器取得网址，udp在安全上比较薄弱，使得黑客可以替换正确的ip导致我们点击错误的网站。

DNSSEC是针对这一问题的安全手段。他是DNS安全扩展，是一种协议。

**AWS Network Firewall**

防火墙是一种对VPC全方位的保护，想象它就在VPC的周围环绕。控制任何的网络进出，包括peering vpc和本地vpn连接。可以进行流量过滤和检查。

一种构架：对防火墙的rule group进行审查可以使用GuardDuty，将finding的结果发送到Security Hub，当发现异常，可以通过Eventbridge启动StepFunction进行后续的操作（对防火墙增加规则，屏蔽IP，SNS通知等）。

防火墙还支持，通过ACM进行的流量加密。

**Amazon SES**

### 4 - Identity and Access Manager

**IAM Policy**

用json格式定义的各种规则。

判定逻辑：首先所有的action是被隐式拒绝的。
```
没有显式deny(no explicit deny) 
--> 组织SCP为allow: allow on organization scps level 
--> 基于资源的许可allow: allow on resource based policy
--> 基于认证的许可allow: allow on identity based policy
--> 边界许可allow: allow on IAM permission boundaries
--> session策略是否许可: allow on session policy
--> 许可: finally allow
```
IAM role：即使是跨账户也可以轻松使用的好功能。是给用户或者应用的临时权限，会随着时间的推移而过期。当要授予一个服务角色的时候，需要先给一个user这种权限，iam:Passrole权限，然后才可以由特定的user进行角色授予。

ABAC：Attribute based access control：基于tag的控制，比如都打有同样标签的user对同样标签的资源进行访问。这可以轻松地扩展资源。基于角色的权限-RBAC（role based access control）在增加资源的时候可能需要手动更新权限。

**STS**:security token service

安全令牌服务，一种临时权限。用户向STS（还有一个服务cognit）发送一个AssumeRoleAPI请求一个临时认证（认证会去IAMpolicy确认可不可以授权）后才可以得到临时的role进行访问。

使用AWSRevokeOlderSessionPolicy可以让老的session在一定时间后失效，这可以让token权限定时过期。

**SCP的相关Policy示例**

限制EC2type的权限：

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "RequireMicroInstanceType",
      "Effect": "Deny",
      "Action": "ec2:RunInstances",
      "Resource": [
        "arn:aws:ec2:*:*:instance/*"
      ],
      "Condition": {
        "StringNotEquals": {
          "ec2:InstanceType": "t2.micro"
        }
      }
    }
  ]
}
```

为全球服务豁免。因为全球服务只在 us-east-1 区域提供，所以当要限制某个区域的服务时，在 NotAction 中列出这些全球服务，就可以正常使用了。

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "DenyAllOutsideEU",
            "Effect": "Deny",
            "NotAction": [
                "a4b:*",
                "acm:*",
                "aws-marketplace-management:*",
                "aws-marketplace:*",
                "aws-portal:*",
                "budgets:*",
                "ce:*",
                "chime:*",
                "cloudfront:*",
                "config:*",
                "cur:*",
                "directconnect:*",
                "ec2:DescribeRegions",
                "ec2:DescribeTransitGateways",
                "ec2:DescribeVpnGateways",
                "fms:*",
                "globalaccelerator:*",
                "health:*",
                "iam:*",
                "importexport:*",
                "kms:*",
                "mobileanalytics:*",
                "networkmanager:*",
                "organizations:*",
                "pricing:*",
                "route53:*",
                "route53domains:*",
                "s3:GetAccountPublic*",
                "s3:ListAllMyBuckets",
                "s3:PutAccountPublic*",
                "shield:*",
                "sts:*",
                "support:*",
                "trustedadvisor:*",
                "waf-regional:*",
                "waf:*",
                "wafv2:*",
                "wellarchitected:*"
            ],
            "Resource": "*",
            "Condition": {
                "StringNotEquals": {
                    "aws:RequestedRegion": [
                        "eu-central-1",
                        "eu-west-1"
                    ]
                },
                "ArnNotLike": {
                    "aws:PrincipalARN": [
                        "arn:aws:iam::*:role/Role1AllowedToBypassThisSCP",
                        "arn:aws:iam::*:role/Role2AllowedToBypassThisSCP"
                    ]
                }
            }
        }
    ]
}
```

**IMDS**

EC2 instance metadata service。提供服务器的元数据。（hostname, instance type, networking settings...）

http://169.254.169.254/latest/meta-data 是每个服务器的endpoint。以键值对的方式，方便进行自动化的设置。这个取得方法是第一个版本，它不使用Token，在 CW 中也可以监控到这个的使用 ：MetadataNoToken 指标。

但是不推介第一个版本，第二个版本更安全，分为两步：

- 首先使用 Header 和 PUT 取得 Session Token。`token = 'cur xxx'; cur xxx -H`
- 然后使用 Session Token call IMDSv2，并且使用 Header。

**S3和对象的跨账户访问权限**

判断一个用户是否可以access一个S3bucket里的object的时候，首先判断user是否有IAM权限，然后是BucketPolicy是否显式拒绝访问，最后是要访问的object是否对用户开放访问（ACL：access control lists，这个现在已经被弃用，但是可以设置开启，但是也可以手动开启：Onwer Enforced setting = Disabled）。

BucketPolicy的访问权限一般是对S3桶资源。Object对访问权限，一般是在arn对url后面有一个*符号表示是bucket内的资源。

跨账户的S3资源访问的方法：一个是使用IAM的user权限和对象账户的BucketPolicy的许可权，另一个是IAM Role访问。

总之S3的访问权涉及一个用户本身的认证和一个Bucket的Policy。原本还有一个ACL是针对桶里的对象的，但是它会让事情变得复杂，比如A自己的桶别人放的文件反而桶的所有者不能访问，需要很复杂的设定，所以ACL被弃用了。

S3的access point：针对桶中的不同文件夹，可以根据文件夹的名字prefix设置不同的访问策略，更灵活地控制访问权。

**Cognito**

**IAM Identity Center**

### 5 - Data Protection

**CloudHSM**

**KMS**

**Secerts Manager**

**S3数据加密**

**Load Balancing**
