## AWS安全专家考试相关内容

---
时间戳2024年3月。

安全威胁有哪些？使用哪些工具？构成如何？

"Remediation对安全漏洞的修复和解决" 通常是一个持续的过程，因为安全漏洞和威胁不断变化，需要及时采取相应的措施来应对。安全团队会定期进行漏洞扫描和安全审计，并根据发现的问题制定和执行 remediation 计划。

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

### 服务分类

在 AWS 中，可以使用以下服务和功能来应对各种安全威胁：

1. **数据泄露**：
   - 使用 AWS IAM 和 AWS KMS 管理和控制对存储在云中的数据的访问权限。
   - 使用 Amazon S3 Bucket Policies 和 AWS IAM 来限制对 S3 存储桶中对象的访问。
   - 使用 Amazon Macie 来监控和发现 S3 存储桶中的敏感数据，并实施数据分类和保护策略。
   - CloudWatch Logs 作为全部数据存储。

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
   - 使用 AWS VPC 和安全组来隔离和保护云中的网络流量和资源。VPC Flow Logs/ELB Access Logs/CloudFront Logs/WAF Logs
   - 使用 AWS IAM 和资源标签来实现对资源的访问控制和隔离。

7. **身份和访问管理（IAM）问题**：
   - 使用 AWS IAM 进行身份和访问管理，实施最小权限原则，并定期审查和轮换访问凭证。

8. **社会工程和钓鱼攻击**：
   - 提供员工培训以提高安全意识，并教育他们识别和防范社会工程攻击。
   - 使用 AWS IAM、AWS SSO 和 AWS Organizations 实施统一的身份和访问管理策略，减少钓鱼攻击的成功率。

9. **未经授权的配置更改**：
   - 使用 AWS Config 实时监控和评估 AWS 资源的配置，并确保符合安全和合规标准。
   - 使用 AWS CloudFormation 和 AWS Config Rules 自动化和强制实施安全配置。

10. **系统漏洞**：
   - Inspector 可以检测 EC2，ECR，lambda 的漏洞和风险。

99. **其他**：
   - 所以存在S3中的log都可以用Athena进行分析。
   - Log的深度存储：Glacier

以上列出的 AWS 服务和功能只是一部分可以用来应对各种安全威胁的示例。实际上，AWS 提供了更多的安全工具和功能，组合使用可以帮助用户建立高度安全的云环境，并保护其数据和应用程序不受安全威胁的影响。

---
安全服务一共包括六个板块：

### 1 - Treat detection and Incident response

**Guard Duty**

- 利用机器学习和行为分析技术来检测恶意活动和未经授权的行为。
- GuardDuty 分析对象来自 AWS CloudTrail、Amazon VPC 流量日志和 DNS 日志的数据，S3 data event，EKS logs等。
- 后面可以加上EventBridge rules进行通知，trigger target: Lambda,SNS。
- 防加密货币攻击。有专用的finding方法。
- 可以在组织 Organization 层级设置一个专用于 GuardDuty 的账户（delegated）用于管理多账户。
- 一种 finding 构架：GuardDuty -> finding -> EventBridge -> SQS/SNS/Lambda -> HTTP(slack)/Email
  - 在lambda后可以进行WAF的ACL调整，block恶意ip，或者对subnet的NACL进行调整，block恶意ip。
  - 也可以将后续动作都包在step functions中。
- 可设置白名单，威胁名单，还有抑制规则（比如你不想看到的ip，已知没威胁的）。
- 注意！GuardDuty对于DNSlogs的解析，只针对default VPC DNS resolver。其他的不出finding结果。

**AWS Security Hub**

- 跨多账户，进行自动安全检测。
- 可集成多种 AWS 服务：Config，GuardDuty，Amazon Inspector，Amazon Macie，IAM access Analyzer，AWS system manager，AWS firewall manager，AWS Health，AWS Partner Network Solutions。
- 集成的 findings 可以被发送到以下服务：Audit Manager, AWS Chatbot, Amazon Detective, Trusted Advisor, SSM Explorer, OpsCenter.
- 还可以针对3rd party工具进行收集和传送。
- 针对上述服务的内容，它自动生成 findings，然后Detective可以进行调查，还可以通过 EventBridge 触发 event：Custom Actions-Security Hub Findings。-->Lambda/SSM Automations/StepFuntions/SNS
- 可以跨区域管理，可以管理多个账户，以一个区域为中心进行安全管理。必须激活AWS config服务。各个区域的Config必须手动确保激活。
- SH支持各种安全标准。Security Standards 可以手动开启。
- ASFF：AWS Security Finding Format。90天自动删除。
- Insights：一群findings的集合。有AWS内置的还可以定制自己的。
- 检测detect - 触发trigger - 补救remediate - log记录

**Detective**

- 机器学习和图算法。
- 深度分析根源问题。
- 生成可视化visualizations和细节details。
- detective检测 - triage问题分类 - scoping问题界定 - response回应

**Penetration Testing**

- 允许的服务：EC2, NAT gateway, ELB, RDS, CloudFront, Aurora, API Gateway, Lambda, Lambda Edge Functions, Lightsail, Elastic Beanstalk.
- 禁止的攻击：DNS区域遍历（zone walking）via Route53 Hosted Zone，DDoS（分布式DoS）攻击，以及以下DoS攻击：Port flooding，Protocol flooding，Request flooding。
- DDoS攻击的测试可以进行：必须通过AWS DDoS test partner，要设置ShieldAdvanced，以及满足速率要求等。
- 如何处置受感染的服务器（Compromised EC2）：（隔离过程可用Lambda自动化）（针对单个container的措施是一样的）
  - capture the metadata
  - enable the termination protection
  - isolate the server, replace the sg with no outbound authorized（隔离：遮断网络进出）
  - detach from ASG
  - deregister from ELB
  - snapshot the EBS for deep analysis
  - tag the EC2 for investigation ticket
- 如何调查受感染的服务器：
  - 关掉instance离线调查，或者snapshot内存memory（可以使用SSM Run Command）和capture network traffic进行在线调查。
- 如何处置受污染的S3：
  - GuardDuty特定目标。
  - CloudTrail和Detective特定恶意活动和API来源。
  - 然后针对S3做出安全设定。
- 如何处置受污染的ECS Cluster：
  - GuardDuty特定目标。
  - 特定污染源是container image还是task。
  - 隔离受影响的task。
  - 评价是否存在恶意活动。（恶意软件）
- 如何处置受污染的RDS：(大概是DB密码泄露)
  - GuardDuty特定目标。
  - 根据情况限制网络通信和可疑用户。restrict the network access and suspected DB user.
  - rotate the suspected DB user password.
  - review DB Audit logs to identify leaked data.
  - Secure DB instance: use Secret Manager（rotate password）,or IAM DB Authentication（manage DB user access without password）.
- 如何处置受污染的AWS凭证：
  - GuardDuty特定目标。
  - rotate更新凭证。
  - attach explicit deny policy, with STS date condition.
  - 检查 CloudTrail logs 看是否有其他非法活动。
  - 查看所有的AWS资源，删除非认可资源。
  - 核实账户信息。
  - 如果受污染的是IAM Role，过程大同小异，但是要加上revoke linked AD（Active Directory） access。
  - 如果受污染的是整个Account，要重置和删除所有的instance上的IAM access keys,IAM user credentials,EC2 key pairs。
  - 替换所有的EC2Instance 中的 key pairs，可以使用SSM Run Command。

**EC2 instance connect（browser based）**

- 运作方式其实不是从浏览器而是通过CLI到该服务的API。
- 会推送一个60秒的临时key到metadata。EC2会去metadata拿到这个key。
- 从AWS固定的IPrange进行ssh连接。18.206.107.24/29
- EC2的SG需要开放该IP的ssh22号端口的inbound许可。
- 由于是API请求，所以该行为都会被CloudTrail记录。

**EC2 Serial Console**

- USE CASE: troubleshoot boot, troubleshoot network configuration, analyze reboot issue
- 默认disable
- Only one active session per EC2
- Must set up OS user and password
- Use with supported Nitro-based EC2

**如果ssh到EC2的accessKey丢了怎么更新EC2内的共钥**

- 一种方法是使用user-data在启动EC2的时候增加一个新的accessKey
- 一种是使用SSM，Document-ResetAccess,key会被存在ParameterStore中，注意必须安装SSMAgent
- 一种是使用EC2 instance connect从浏览器进入，如上这是一种一次性的认证方式，进去之后可以更换key
- 一种是用SerialConsole连接EC2进行修改
- 一种是通过detach根EBS，然后把它attach到其他temporary的EC2上进行key的修改，然后再attach到原来的EC2上
- 如果是window环境的话，比较复杂，根据各个AMI的版本，补救方法不一样，最老的似乎是直接删除，然后有2016版本的使用config文件，他们都是通过EBS的detach和attach实现，另外还有SSM的document方法

**EC2Rescue Tool for Linux**

- 用于收集系统信息和日志：system utilizaiton report,logs and details,还可以检测和补救系统问题
- 需要手动安装或者 run SSM document: AWSSupport-TroubleshootSSH Automation Document
- 可以自动上传结果 results 文件到 AWS Support or an S3 bucket

**IAM Access Analyzer**

- 可以帮助找到哪些资源被外部非法access，定义一个ZoneofTrust，那么它之外的地方都是findings的区域
- 审查policy，Policy Validation
- 可以根据现有的activity活动，比如CloudTrail，生成policy，可能会是一个不错的best practice

**其他概念**

- AUP：AWS Acceptable Use Policy 用户不可以用于非法或者黑客活动，否则将被删除任何相关内容
- AWS Abuse Report向AWS Trust&Safety发起的滥用报告，如果你发现了你可以汇报，如果你收到了类似邮件，那你必须回复问题，不然你账号一天就没了
- IAM Credential Report-account level 可以下载用户们的认证和访问记录
- IAM Access Advisor-user level 可以查看用户被授予的权限，以及最后一次访问是什么时候

### 2 - Security Logging and Monitoring

**定义和术语**

- 漏洞 Vulnerability – 系统漏洞 weakness in a system, system security procedures, internal controls, or implementation that could be exploited
- 漏洞利用 Exploit – 利用软件漏洞和安全缺陷的代码 code that takes advantage of software vulnerability or security flaws
- 有效负载 Payload – 攻击者打算传递给受害者的恶意软件 a malware that the attacker intends to deliver to the victim
- 漏洞扫描工具 Automated Vulnerability Scanner – a tool that run automated scans of an IT environments to detect vulnerabilities, example : nessus scan (note: you don't need to know that tool for the exam)
- 常见漏洞和暴露 Common Vulnerabilities and Exposures (CVE) – a list of publicly disclosed security flaws.
- 通用漏洞评分系统 Common Vulnerability Scoring System (CVSS) – a way to produce a numerical score reflecting the severity of a vulnerability or a security flaws

**Inspector**

- 自动安全评估工具/
- 对EC2通过SSM Agent(这是服务触达EC2的方式，必须设置)检查OS漏洞，以及网络触达network reachability（端口是不是不应该对公网开放之类的）to EC2
- 可以检查ECR中上传的Container Images
- 检查Lambda软件代码和依赖的漏洞，when deployed
- 结果去想：可以发送到SecurityHub中作为报告，也可以trigger EventBridge出发其他内容
- 重点：三种服务（EC2,ECRimages,lambda），针对漏洞，漏洞CVE更新那么他会重新检测，会给出一个risk score风险分数

**System Manager**：

- 免费服务，可以对应EC2也可以对应On-Premises，系统对应Linux和Windows
- 自动patching，增强合规性
- 集成 CloudWatch metrics / dashboard
- 集成 AWS Config
- 重要组成：Resource Group / Document / Automation / Maintenance Windows / Parameter store / Inventory / State Manager / Run Command / Patch Manager / Session Manager
- Document 自动执行功能是我觉得亮眼的功能。
- 必须在server上安装 SSM Agent，AmazonLinux2和一些Ubuntu自带agent，一般出了问题都是因为没agent或者没对EC2授权相应的Role（Role也可以叫做：IAM instance profile，这是在hands on中看到的）
- lanch的新的EC2，比如已经安装了SSM Agent的AmazonLinux2的EC2，会直接出现在SSM的Fleet Manager中，作为一个舰队进行管理。
- 使用 TAGS 对资源进行分组管理：Resource Group，从而进行自动化和cost分配
- Document 你可以定义parameters和actions，用json或者yaml格式的文件。（很像Github Actions或者Cloud Formation，都是IaC），Parameters也可以从 Parameter Store中取得
- Run Command功能是直接跑一个小的命令或者跑一个脚本（document=script），通过resource groups可以直接跑一个server集群，它和IAM还有CloudTrail集成，会被记录，不需要通过ssh连接EC2（而是通过SSM Agent，session manager也是通过Agent），可以控制执行速率rate（一次执行多少，或几个server），和错误控制（几个错误就停止之类），跑命令的结果可以在console表示也可以发送到S3或者CWLogs，可以给SNS发消息，也可以被EventBridge Trigger，甚至可以得到一个生成的 CLI 代码自己拿去控制台执行。

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

**ACM**

**AWS Backup**

### 6 - Management and Security Governance

**AWS Organization**

**Control Tower**

**AWS Config**

**Trusted Advisor**

**Cost Explorer**

**AWS Cost Anomaly Detection**

**CloudFormation**

**Service Catalog**

**RAM: Resource Access Manager**

### others

**CloudShell**

不能直接访问VPC中的EC2和RDS等资源，是用来方便APIcall的。要连接EC2必须设置instance connect之类的。
