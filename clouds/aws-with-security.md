AWS安全专家考试相关内容。

时间戳2024年3月。

## 不同视角的服务启发

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
AWS安全服务一共包括六个板块：

## 1 - Treat detection and Incident response

### Guard Duty

- 可以利用机器学习和行为分析技术来*检测恶意活动和未经授权的行为*。
- GuardDuty 分析对象来自 AWS CloudTrail（针对人）、Amazon VPC 流量日志和 DNS 日志的数据（针对网络），S3 data event（针对数据），EKS logs等。
- 后面可以加上EventBridge rules进行通知，trigger target: Lambda,SNS。
- 防加密货币攻击。有专用的finding方法。
- 可以在组织 Organization 层级设置一个专用于 GuardDuty 的账户（delegated）用于管理多账户。
- 一种 finding 构架：GuardDuty -> finding -> EventBridge -> SQS/SNS/Lambda -> HTTP(slack)/Email
  - 在lambda（自动化处理）后可以进行WAF的ACL调整，block恶意ip，或者对subnet的NACL进行调整，block恶意ip。
  - 也可以将后续动作都包在step functions中。
- 可设置白名单，威胁名单，还有抑制规则（比如你不想看到的ip，已知没威胁的）。
- 注意！GuardDuty对于DNSlogs的解析，只针对default VPC DNS resolver。其他的不出finding结果。

### AWS Security Hub

- 跨多账户，进行自动安全检测。
- 可集成多种 AWS 服务：Config，GuardDuty，Amazon Inspector，Amazon Macie，IAM access Analyzer，AWS system manager，AWS firewall manager，AWS Health，AWS Partner Network Solutions。
- 集成的 findings 可以被发送到以下服务：Audit Manager, AWS Chatbot, Amazon Detective, Trusted Advisor, SSM Explorer, OpsCenter.
- 还可以针对3rd party工具进行收集和传送。
- 针对上述服务的内容，它自动生成 findings，然后 Detective 可以进行调查，还可以通过 EventBridge 触发 event：Custom Actions-Security Hub Findings。-->Lambda/SSM Automations/StepFuntions/SNS
- 可以跨区域管理，可以管理多个账户，以一个区域为中心进行安全管理。必须激活*AWS config*服务。各个区域的Config必须手动确保激活。
- SH支持各种安全标准。Security Standards 可以手动开启。
- ASFF：AWS Security Finding Format。90天自动删除。
- Insights：一群findings的集合。有AWS内置的还可以定制自己的。
- 流程：检测detect - 触发trigger - 补救remediate - log记录

### Detective

- 机器学习和图算法。
- 深度分析根源问题。
- 自动收集和输出事件：VPC Flow Logs，CloudTrail，GuardDuty
- 生成可视化visualizations和细节details。
- 流程：detective检测 - triage问题分类 - scoping问题界定 - response回应

### Penetration Testing

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

### EC2 instance connect

- （browser based）
- 运作方式其实不是从浏览器而是通过CLI到该服务的*API*。
- 会推送一个60秒的临时sshPublickkey到metadata。EC2会去metadata拿到这个key。
- 从AWS固定的IPrange进行ssh连接。18.206.107.24/29
- EC2的SG需要开放该IP的ssh22号端口的inbound许可。
- 由于是API请求，所以该行为都会被CloudTrail记录。

### EC2 Serial Console

- 存在原因：通常情况下，EC2实例的操作是通过SSH、RDP或者其他网络连接方式进行的，但在某些情况下，比如实例操作系统无法启动或网络连接不可用时，就无法通过常规的网络连接方式访问实例。
- EC2 Serial Console允许用户通过串行连接直接访问EC2实例的操作系统控制台，这种连接方式不依赖于实例的网络状态。
- 查看实例的启动日志、调试操作系统问题、修改配置等操作，以便快速诊断和解决实例故障
- USE CASE: troubleshoot boot, troubleshoot network configuration, analyze reboot issue
- 默认disable
- 每个EC2只有一个active session
- 必须设置OS user 和 password
- 使用supported Nitro-based EC2（是一种基于亚马逊自家设计的Nitro系统架构的EC2实例，具有更高的性能、更低的延迟和更好的安全性。）

### 如果ssh到EC2的AccessKey丢了怎么更新EC2内的公钥

- 一种方法是使用user-data在启动EC2的时候增加一个新的accessKey
- 一种是使用SSM，Document-ResetAccess,key会被存在ParameterStore中，注意必须安装SSMAgent
- 一种是使用EC2 instance connect从浏览器进入，如上这是一种一次性的认证方式，进去之后可以更换key
- 一种是用SerialConsole连接EC2进行修改
- 一种是通过detach根EBS，然后把它attach到其他temporary的EC2上进行key的修改，然后再attach到原来的EC2上
- 如果是window环境的话，比较复杂，根据各个AMI的版本，补救方法不一样，最老的似乎是直接删除，然后有2016版本的使用config文件，他们都是通过EBS的detach和attach实现，另外还有SSM的document方法

### EC2Rescue Tool for Linux

- 快速定位和修复Linux EC2实例上的常见故障和性能问题
- 提供了一系列诊断和修复功能，包括文件系统检查、网络故障排除、磁盘空间分析、内存使用情况分析等
- 用于收集系统信息和日志：system utilizaiton report,logs and details,还可以检测和补救系统问题
- 需要手动安装或者 run SSM document: AWSSupport-TroubleshootSSH Automation Document
- 可以自动上传结果 results 文件到 AWS Support or an S3 bucket

### IAM Access Analyzer

- 可以帮助找到哪些资源被外部非法access，定义一个ZoneofTrust，那么它之外的地方都是findings的区域
- 审查policy，Policy Validation
- 可以根据现有的activity活动，比如CloudTrail，生成policy，可能会是一个不错的best practice

### 其他概念

- AUP：AWS Acceptable Use Policy 用户不可以用于非法或者黑客活动，否则将被删除任何相关内容
- AWS Abuse Report向 AWS Trust&Safety 发起的滥用报告，如果你发现了你可以汇报，如果你收到了类似说你滥用了的邮件，那你必须回复问题，不然你账号一天就没了
- IAM Credential Report-account level 可以下载用户们的认证和访问记录
- IAM Access Advisor-user level 可以查看用户被授予的权限，以及最后一次访问是什么时候

## 2 - Security Logging and Monitoring

### 定义和术语

- 漏洞 Vulnerability – 系统漏洞 weakness in a system, system security procedures, internal controls, or implementation that could be exploited
- 漏洞利用 Exploit – 利用软件漏洞和安全缺陷的代码 code that takes advantage of software vulnerability or security flaws
- 有效负载 Payload – 攻击者打算传递给受害者的恶意软件 a malware that the attacker intends to deliver to the victim
- 漏洞扫描工具 Automated Vulnerability Scanner – a tool that run automated scans of an IT environments to detect vulnerabilities, example : nessus scan (note: you don't need to know that tool for the exam)
- 常见漏洞和暴露 Common Vulnerabilities and Exposures (CVE) – a list of publicly disclosed security flaws.
- 通用漏洞评分系统 Common Vulnerability Scoring System (CVSS) – a way to produce a numerical score reflecting the severity of a vulnerability or a security flaws

### Inspector

- 自动（漏洞）安全评估工具
- 对EC2通过SSM Agent(这是服务触达EC2的方式，必须设置)检查OS漏洞，以及网络触达network reachability（端口是不是不应该对公网开放之类的）to EC2
- 可以检查ECR中上传的Container Images
- 检查Lambda软件代码和依赖的漏洞，when deployed
- 结果去向：可以发送到SecurityHub中作为报告，也可以trigger EventBridge出发其他内容
- 重点：三种服务（EC2,ECRimages,lambda），针对漏洞，漏洞CVE更新那么他会重新检测，会给出一个risk score风险分数

### System Manager

- 免费服务，可以对应EC2也可以对应On-Premises，系统对应Linux和Windows
- 自动patching，增强合规性
- 集成 CloudWatch metrics / dashboard
- 集成 AWS Config
- 重要组成：Resource Group / Document / Automation / Maintenance Windows / Parameter store / Inventory / State Manager / Run Command / Patch Manager / Session Manager
- Document 自动执行功能是我觉得亮眼的功能。
- 必须在server上安装 *SSM Agent*，AmazonLinux2和一些Ubuntu自带agent，*一般出了问题都是因为没agent或者没对EC2授权相应的Role（Role也可以叫做：IAM instance profile，这是在hands on中看到的）*
- lanch的新的EC2，比如已经安装了SSM Agent的AmazonLinux2的EC2，会直接出现在SSM的Fleet Manager中，作为一个舰队进行管理。
- 使用 TAGS 对资源进行分组管理：Resource Group，从而进行自动化和cost分配
- **Document** 你可以定义parameters和actions，用json或者yaml格式的文件。（很像Github Actions或者Cloud Formation，都是IaC），Parameters也可以从 Parameter Store中取得
  - 可以在ASG中的EC2被terminating之前发送一个命令，比如跑一个自动的document：需要在ASG中设置一个Lifecycle Hook，将要关闭的EC2设置为一个Terminating:Wait的状态，当EventBridege检测到了这个状态，就可以触发这个document的执行
- **Run Command**功能是直接跑一个小的命令或者跑一个脚本（document=script），通过resource groups可以直接跑一个server集群，它和IAM还有*CloudTrail*集成，会被记录，*不需要通过ssh连接EC2*（而是通过SSM Agent，session manager也是通过Agent），可以控制执行速率rate（一次执行多少，或几个server），和错误控制（几个错误就停止之类），跑命令的结果可以在console表示也可以发送到S3或者CWLogs，可以给SNS发消息，也可以被EventBridge Trigger，甚至可以得到一个生成的 CLI 代码自己拿去控制台执行。
- **Automation**：这个功能也是使用Document进行操作 EC2 等。可以用 EventBridge 触发，进行系统修补等。
- Parameters Store：存参数或者密码，可用KMS加密，集成 IAM Policy，可以用于 CloudFormation 的输入。层级文件夹存储方式。Advance的Parameter设置可以有8K大小，但要付费，可以设置TTL（expiration date，到期可以设置通知notification，也可以通知没变化）。
  * aws ssm get-parameter 通过 with-decrption 可以解密密码，会检查你的KMS权限可否解密，挺酷。
  * 可以通过文件夹层级递归取得 aws ssm get-parameter --path /myapp/dev/ --recursive 你创建parameter的时候名字就是一个path格式就可以了，这很特别，比如name：/myapp/dev/db-password
- *Inventory*：收集EC2元数据，可以通过S3+Athena或者QuickSight探索。（元数据包括：软件，OS，设置，更新时间等）
- State Manager：状态管理器，保证 EC2 满足某一设置状态。
- **Patch Manager**：自动打补丁，可以设置有计划的 Maintenance Windows，可以通过tags，设置 Patch Baseline和Patch Group，进行不同的补丁计划。计划通过Document自动执行，报告可以发送到S3。
  - AWS-RunPatchBaseline applies to both *Windows and Linux*, and AWS-DefaultPatchBaseline is the name of the default Windows patch baseline
- *Session Manager*：不需要ssh的EC2连接，*通过Agent和role*。通过IAM可以限制执行主体和对象EC2（通过tag），甚至可以限制使用的command。访问记录会被CloudTrail记录。可以记录logs（S3，CloudWatch Logs）
- **OpsCenter**：解决操作上的问题：*Operational Issues（被叫做OpsItems）*
  - OpsItems：issues，events，alerts
  - 集合各种info比如config，Cloudtrail日志，CWAlarms，CFstack信息等
  - 解决方式：run automation document
  - 使用EventBridge或者CW Alarms创建OpsItems


### Cloud Watch

- Unified CW Agent 用来发送EC2的RAM，processes，disk space信息，namespace是CWAgent。使用Procstat Plugin。
- EC2要对CW发送log需要相应的Agent权限。创建EC2后安装Agent，会通过漫长的设置创建config.json文件，这个文件可以存储在Parameter Store中，当你想用一个设置管理很多EC2的时候这很有用，但是这需要Agent的Admin权限。如果无法得到metric，troubleshooting找agent的log文件。
- 使用命令行调用config文件，本地调用或者parameter store调用，后者更适合管理很多EC2的情况。
- 安装Agent还可以使用SSM Run Command或者SSM State Manager

- **Alarms**：可以通过CWlogs metrics filter设置，可设置频率，Actions可以对EC2操作，可以对AutoScaling操作，可以发送给SNS服务。**Composite Alarms**是一种针对其他警报的复合警报，可以使用条件语句，比如同时监视CPU和IO。
- **Contributor Insights**：就是找到造成问题根源，是who造成了这些logs，比如一些bad IP地址，URL，host。一种分析手段。

- **CloudWatch Synthetics Canary** 可以帮助用户通过运行自动化脚本（称为“Canaries”）来监控应用程序的可用性、性能和功能。这些脚本模拟用户行为，并*定期*运行，帮助检测应用程序中的潜在问题，使用Node.js或者Python编写，集成CW Alarm

- **Cloud Watch Logs**：
- *来源*：
  - SDK，Logs Agent或者Unified CW Agent
  - ElasticBeanstalk可以收集日志并发送
  - ECS collection of Containers
  - Lambda：function logs
  - VPC Flow Logs
  - API Gateway
  - CloudTrail
  - Route53 Log DNS queries
- *组织方式*：Log group， log stream
- 可以设置过期时间
- KMS加密
- *Send to*：
  - S3：API call：CreateExportTask，形成log后12小时后才能export
  - Kinesis Data Stream
  - Kinesis Data Firehose
  - AWS Lambda
  - OpenSearch
- **Cloud Watch Log insights**：进行可视化日志分析。注意这只是一个query引擎而不是一个即时流，如果想要将log流处理需要用**CloudWatch Log Subscriptions**的filter功能将日志不断传入其他服务进行处理，分析，存储。它还可以整合不同区域的日志，将他们聚合起来（Lambda 实时构架 或者 kinesis Data Firehose近实时构架，发送至OpenSearch或者S3）。那么既然它是一个订阅服务，就需要有起点和目的地账户，并且需要在账户之间设置policy，以供分发和接收。尤其是接收端的权限放行。
- 可以通过设置metric的阈值（可以进行条件的组合比如AND，OR等，很灵活）发出alarm，可执行的动作，包括对EC2的启动关闭等操作，对Auto Scaling对扩张收缩，以及使用SNS发出通知等。
- CW Logs聚合功能：多账户，多区域


### Amazon EventBridge

- 时间cron驱动或者事件event驱动。
- event实际上是一种json格式数据。
  - 可以filter events
  - 源：EC2 Instance，CodeBuild（fail build），S3 Event，Trusted Advisor（new finding），CloudTrai（any API call）
- 可以发送到集成的组织event bus。
- 可以重现replay过去的event。
- event集合：event schema register
- 可以集成到第三方Partner或者自定义Custom的event bus。
- 如果跨账户使用，需要设置resource-based-policy。

### Amazon Athena

- 分析S3中各种格式的文件，使用标准的SQL
- 经常和QuickSight集成，创建dashboard可视化
- 分析各种日志数据很方便，因为他们经常被集成到S3
- 节省数据扫描费用：
  - 使用columnar data，列扫描，使用Apache Parquet or ORC非常省钱。可以用Glue转换数据为Parquet格式。
  - compress data压缩数据
  - 对数据分区，在S3中比如将数据放在如yyyy/mm/dd的文件夹里
  - 使用大文件，large file，比如大于128MB，最小化overhead
- Fedarated（联邦）Query：可以通过 lambda 上跑的Data Source Connector，就可以对各种数据库（关系非关系和cache数据库等）进行SQL查询，于是join各种数据。然后存进S3。确实很强。
- 这个服务本身不存储数据，只是在 query 其他的地方的数据，所以如果 QuickSight 无法对 Athena 数据进行可视化，发生了错误可能是如下两种问题：
  - 对S3数据没有访问权限。
  - 对KMS对解码没有权限，所以无法解码在S3中的数据。
  - 给QS相应的权限即可。

### AWS CloudTrail

- 记录人的行为，是否合规，进行审计，因为人的行为基本是通过API发出的，服务都是API。
- 存储：CW Logs或者S3，或者其他统合服务，在S3就可以用Athena进行Query
- 默认记录management事件，默认不记录data事件（因为操作率太高）
- Cloud Trail Insights服务，持续监测不正常的write操作
- Logs存储90天，想要长期存储需要存入S3。
- 可以集成 EventBridge 在遇到非法API操作的时候，发送SNS通知。和 CloudWatch 的 metric filter 一起使用可以更好地过滤日志，联动SNS通知。
- 可以进行组织 organization 层面的Trail收集。
- Integrity Validation：完整性检查，确保日志未被修改，内部通过一个digest file进行hash算法一致性检验。
- 非实时。

### AWS Macie

- 通过机器学习 ML 查找到S3中的敏感信息主要是个人识别信息PII，然后通过EventBridge进行通知等。
- 可以自定义正则化表达式进行匹配。还可以设置allow list允许某些信息不被检查。适用于组织。
- SecurityHub 可以收集 Macie 的 Findings
  - Findings：它指问题本身，可以进行 findings 查找的服务：GuardDuty，SecurityHub，Inspector，Config，Macie等
  - Macie的Findings包括一个Policy Findings这很意外，很不错因为这个用于监测是否对S3数据权限，加密等进行了违规操作，这是为了确保敏感数据的存储安全，另外敏感数据的Findings肯定是有的。
- 可以有一个Macie专用账户进行多账户管理（专用账户管理是一个很好的实践）

### S3 Notifications

- 有两种方式可以实现S3的通知，一种是Event Notifications，可以设置Object动作，以及后续的SNS，SQS，或者Lambda。
- 另一种是Amazon EventBridge，将所有的通知发送到EB以供后续的动作。

### VPC Flow Logs

- IP，ENI（虚拟网络接口，连接实例和网络的接口，它会有一个接口的ID：interfaceid），子网，虚拟私网通信，ELB以及其他各种网络通信的logs
- 有助于网络通信的trouble shooting，如果进入和出去一个reject一个accept那可能就是NACL的问题，因为它stateless
- 存储和查询于其他无异
- 一些高级的内置既存traffic不会被捕获，比如非自定义的DNS服务器流量，DHCP（动态主机IP配置）流量，EC2元数据网址（196.254.196.254）流量等。

### VPC Traffic Mirroring

- 可进行Traffic mirroring（流量镜像）网络监控技术，用于将网络中的流量复制到监控设备上进行分析和审计，这样将不会对原有的流量产生负载和性能影响。
- 可以有多种构架，比如多账户，多VPC的Peering，使用LB作为接受方方便安全收集数据等。当GuardDuty的Findings被发现有问题，可以通过EventBridge启动Lambda，lanch一个用于monitor的EC2和ENI，用来接受镜像流量和后续分析。（自动化无处不在）
- 可以通过中心化路由Transite Gateway整个多个VPC中的流量镜像进行分析。
- Transit Gateway 是 AWS 提供的一项托管服务，用于简化多个 VPC（Virtual Private Cloud，虚拟私有云）之间的网络连接。它是一个中心化的网络枢纽，允许用户在一个 VPC 中集中管理网络流量，并通过 Transit Gateway 轻松连接多个 VPC、VPN 连接和 AWS Direct Connect 连接。

### VPC Network Access Analyzer

- 分析网络进出是否合规。可以自定义规则access scope，network access requirements等。自定义规则分析等还有config之类。

### Route53 Resolver Query Logging

- 记录通过Route53 Resolver，进行的公共DNS查询log。
- 只用于Public（not for pravite）Hosted Zone（是Route 53中的一种资源记录区域，用于托管公共的DNS记录）
- 日志会记录域名，到各种类型信息，resolverIP地址，client子网地址等的映射信息。
- 只会存在CWLogs中。

### OpenSearch

- 前身是ElasticSearch，目的集中在 Search 上，可以和DynamoDB联合，作为一个搜索增强的部分
- DynamoDB可以进行针对主键和索引的搜索，但是OpenSearch可以针对任何键，并且可以部分匹配
- 不支持原生SQL，需要plugin
- 应用场景：日志分析，监控，网站，知识库搜索，自然语言处理机器学习集成，地理空间搜索，感觉很像GCP的BigTable。
- 经常和流处理结合，比如kinesis数据导入OS
- 公共访问通过access策略，ID，IP，Domain来限制。VPC访问需要更多的设置，包括端点和ENI，以及各种权限等。

## 3 - Infrastructure Security

### Bastion Hosts

一种保护私有网络的方法，我的GCP项目中就使用到了这种构架。bastion的意思是堡垒，向外部开放22端口允许公开访问，外部的访问通过bastion访问私有服务器的内容，bastion和私有服务器中也是通过私有IP进行链接，以此保护私有服务器的地址不暴露，私有服务器不对外链接。

### Site to site VPN

- 连接云端的VPC和私有数据中心的安全方法。需要设置VGW（虚拟virtual网关gateway）和本地的CGW（custom gateway）之间的连接。（这两个都在VPC服务页面设置），然后在同样的页面设置site-to-site-VPN connection。
- 使用ACM发行的私有证书进行传输中的加密。（设置CGW页面有设置项目）
- VGW可以自定义ASN（Autonomous System Number，边界网关协议 BGP 的自治系统号码），它帮助路由器确定最佳路径、建立邻居关系和执行路由策略，从而实现自治系统之间的网络通信。
- CGW有两种方式，一种是本地的publicIP，一种是NAT设备后的privateIP，使用NAT的publicIP。
- 需要在虚拟网关中开启你子网的路由传播功能：Route propagation
- 在VGW的inbound安全组SG中需要设置icmp（许多网络工具和实用程序（如 ping、traceroute）使用 ICMP 协议来检测网络中的主机可达性和路径延迟。）协议的开启，不然从本地的服务器无法ping到云端ec2
- Site-to-Site VPN（使用 *IPSec* 协议加密通信，提供了一种相对简单和成本较低的方式来扩展本地网络到 AWS 云。）适用于临时性或较低带宽要求的场景，而 Direct Connect（一种专线连接服务，允许用户通过专用网络线路将本地数据中心或办公场所直接连接到 AWS 的数据中心。）则适用于对网络性能和可靠性有较高要求的企业级应用场景。

### Direct Connect

- 完全走AWS私有网络：通过AWS Direct Connect Location - 物理中心，所以设置要花几个月的时间。
- VPC端需要一个VGW虚拟网关Private Virtual Interface，或者接到服务资源的Public Virtual Interface。
- AWS Direct Connect Location：
  - AWS Direct Connect Endpoint
  - Customer or Partner Router
- VGW - DirectConnectLocation - DataCenter
- 如果是同时接多个VPC，则需要设置：Direct Connect Gateway，进行枢纽
- 两种类型：Dedicated Connection 是客户直接租用的物理专线连接，而 Hosted Connection 则是通过合作伙伴提供的连接服务来访问 AWS Direct Connect。前者物理上更专有传输速度更快，后者和别的客户共享，设置连接的工程速度更快。
- 不加密，因为是私有连接。如果要加密，需要在上面加一层VPN连接，进行IPsec加密。
- 可以进行高弹性设置，也就是冗余。一个VPC连接两个DC，甚至每个DC中两个物理设备接入点。提高workload的弹性。
- 备份连接：两条DC太贵了，可以用S2SVPN作为备份连接，提高可用性。

### AWSVPN CloudHub

- 多个CGW，通过 AWS 来设置连接，使用VPN CloudHub，可以实现各个CGW之间的连接。走公网，但是是加密的。
- AWS的VGW可以为这些CGW设置动态路由和设置路由表，帮助各个本地中心进行路由。

### AWS Client VPN

- 如果你想访问使用privateIP的EC2，而不是设置它的publicIP，看起来就像你在VPC中：在电脑上安装AWS Client VPN就可以从自己的电脑连接到云端的EC2，如果你的VPC同时使用了site-to-site-vpn那么也可以从自己的电脑很方便的连接到那个custom的数据中心。
- AWS Client VPN -> S2SVPN -> CustomDataCentor
- 注意这是实际上走的公网public internet
- 认证方式：AD认证，MFA，数字证书认证，SSO

### VPC Peering

- 在同一个账户或者不同账户的不同VPC之间实现连接，仿佛你所有的行为都是在同一个VPC中一样。
- 不可传递，想要连接的VPC必须两两连接。
- 虽然行为看起来是在同一个VPC一样，但是需要更新子网的路由表route table。
- 不可以有overlapping的CIDRs：不然会传输错误
- 步骤：设置peering，被设置方VPC接受peering请求，设置双方的路由。

### DNS Resolution

- enableDnsSupport 是 AWS 中用于配置 VPC 的一个功能，它允许 VPC 中的实例使用默认的 DNS 解析服务来解析公共 DNS 域名，从而能够与互联网上的资源进行通信。默认启用，但是关闭了也可以自己设置Custom DNS Server。
- enableDnsHostname 是 AWS VPC 中的一个配置选项，它允许 VPC 内的实例分配具有公开可解析主机名的 DNS 记录，从而使其他网络中的资源可以通过主机名来访问 VPC 中的实例。新VPC默认是禁用的。前置条件是 enableDnsSupport 为 True。
- Private Hosted Zone 是 Amazon Route 53 中的一项服务，用于在VPC中创建和管理私有的 DNS 域名空间。它在 VPC 内部设置专用的域名解析服务，使得 VPC 中的资源可以通过域名进行访问，而无需暴露到公共互联网上。比如 google.demo.com 映射到 www.google.com （CNAME别名）然后通过Route53就可以帮忙解析到真正的google网页，但是你在内部可以用google.demo.com这种自定义的名字来访问。(上面两种属性必须设置为True)
- 域名解析服务就像是将难懂的value，映射到key，然后通过key取得value。

### VPC Endpoint

- 使用Endpoint服务的目的是什么？是为了不通过internet流量提高数据传输的安全性。S3和DynamoDB是自动启用网关的，但是其他服务使用interface需要手动设置。如果不使用interface endpoint，就需要设置internet Gateway，Nat入站，SG和NACL端口和协议开启等设置来通过公有网络访问资源。
- VPC Endpoint Gateway（通过设置固定IP的网络路由实现，内网）：support S3&DynamoDB。不需要访问public internet。要在路由表中设置资源名称比如S3桶。是VPClevel的设置，同时必须开DNSresolution，因为S3和DynamoDB资源是需要解决域名的。注意：每个VPC都必须设置，peering，VPN连接的网络都不管用。
- VPC Endpoint Interface（通过PrivateLink实现，需要进行DNS查询）：support all except DynamoDB。这个选项需要设置 VPC Endpoint Interface（ENI）。依赖SG进行安全设置（因为在EC2上）。需要DNSresolution解决服务的公共域名到私有hostname的映射。注意：这个可以通过 Site-to-site-VPN 和 Direct Connect 进行跨VPC访问。
- 考虑到每一个服务都是一个API（所以他们都需要解决域名，进行IP接口访问！），如果设置interface，有时候需要设置非常多接口，用于多种服务组合。
- troubleshooting：check DNS resolution 和 route table
- 二者区别：Endpoint Gateway 不需要分配专用的 IP 地址，而是使用 VPC 的子网路由表中的目标，通过 Internet Gateway（Internet 网关）进行访问。Endpoint Interface 需要分配一个专用的私有 IP 地址，并将其连接到 VPC 子网中，以提供专用的网络路径进行访问。
- 网关Gateway和网络接口ENI：
  - 网关是一种网络设备，用于连接不同网络之间的通信。在 AWS 中，常见的网关包括 Internet Gateway（Internet 网关）、Virtual Private Gateway（虚拟专用网关，刚刚的VPN连接时候用到的虚拟网关）、NAT Gateway（NAT 网关）等。他们都需要添加路由设置。
  - 网络接口可以附加到 EC2 实例上，以提供与 VPC 和 Internet 的连接，也可以附加到其他 AWS 服务（如 ELB 负载均衡器、RDS 数据库实例等）上，以提供网络功能（这些服务的设置页面都有设置选项）。网络接口可以具有私有 IP 地址、公有 IP 地址、Elastic IP 地址、IPv6 地址等，允许与其他资源进行网络通信。
  - 网关通常是独立的网络设备，而网络接口是附加到实例或其他服务上的虚拟设备。
- 权限和API访问限制：VPC Endpoint Policy。但他的allow不能覆盖IAM和资源权限。
- **LambdaFunction**的网络工作原理：
  - 如何在VPC内设置Lambda：设置VPCID，subnet和SG，在引擎下，Lambda会自动create ENI（需要相应的权限:CreateNetworkInterface on EC2:ENIManagementAccess Policy），就是上面说的interface。这样它才能工作。
  - 那么在VPC中的Lambda中可以接公网吗？？⬇️
  - 一个在PublicSubnet中的lambda无法像里面的EC2那样访问公共网络或者有公共IP。（因为Lambda是无状态的，无服务器的，所以不分配固定IP公有地址，建议通过NAT和外网通信，或者以负载均衡器LB作为接口）
  - 那如何让他有公共访问：放入PrivateSubnet然后设置NatGateway。（这是唯一的方法）。然后在私有子网中和其他内部资源一起工作。
  - 如何访问DynamoDB：通过上面设置的Nat通过InternetGateway在公网访问DynamoDB，或者不用Nat而是通过VPCEndpointGateway访问。（因为你VPC中的所有资源都可以这么访问S3和DynamoDB）

### AWS PrivateLink

-（在GCP中也有同样的暴露方法）
- 目的：expose一个VPC中的应用给其他很多VPC而不需要通过公网，peering，nat等。（AWS也基本这么暴露自己的服务的）
- 高可用性，安全性。
- expose端通过一个NLB，消费端VPC通过ENI。两者进行连接。
- ECS构架：多task多应用 --> ALB --> NLB --> PrivateLink --> ENI-of-VPC/VGW连接的其他VPN网络
- VPC Endpoint构架：VPC Endpoint interface --> PrivateLink --> S3

### NACL & SG

- NACL是stateless的，SG是statefull的
- subset级别防火墙，根据数字排列优先级（越小越高）：100，200 and so on
- defaultNACL对in和out都是all-allow，建议定义customNACL
- 由于有时候客户端需要发送和接收临时端口的请求，所以在设置NACL的时候，端口可能会需要设置为一个范围。
- **Managed Prefix Lists**：方便设置SG和路由表的IP地址栏，是一组CIDR的列表，有自定义（自定义IP）和AWS专属（各种resource的专属CIDR列表）两种。
- SG规则设置为拒绝也无法interrupt切断已经建立的连接，只能等timeout，如果想立刻切断，用NACL立刻deny连接规则就可以。

### AWS Transit Gateway

- 集中式的网络转发设备，可连接多个 VPC、VPN 连接和 Direct Connect 连接，使得跨多个 VPC 和本地网络之间的流量转发更加简单和高效。
- 唯一支持网络多播multicast的服务。
- 支持cross-region
- 需要设置route-table定义VPC之间的通信
- 支持 ECMP（Equal-Cost Multi-Path，等价成本多路径）路由策略，以提高网络的容错性、负载均衡和吞吐量。
- 通过它的转发能力，可以在多账户和多VPC之间，分享一个Direct Connect Gateway（本地数据中心连接）。

### CloudFront

- CDN by edge locations
- 集成Shield web app firewall，有DDos攻击保护功能。
- 数据源可以是S3也可以是自定义（HTTP）的网站。S3通过OAC（OriginAccessControl）和S3 bucket policy进行安全控制。
- 和S3的区域复制功能的区别：*静态内容*全球覆盖用CF，*动态内容*在几个区域覆盖用S3（没有cache）。
- CF可以设置访问国家的黑白名单。出于某些法律上的限制。
- CF Signed URL（一个文件）/Cookies（多个文件）：可以控制用户对特定内容的访问权限，并设置 URL 的有效期限制。可以控制IP，时间范围和请求来源。签名 URL 是通过将访问者的身份信息、访问条件和访问时间戳等信息加密后生成的。使用 AWS 的身份验证密钥（例如访问密钥对）和 CloudFront 提供的 API 或 SDK 来生成签名 URL。相对于S3的署名URL它的控制更精细，来源可以是桶也可以是任何HTTP源。是一种高级的内容控制方式。
- 支持Filed Level Encryption。字段级别的私密信息加密。
- OAC可以设置对kms的访问权限，从而支持SSE-S3，SSE-KMS，实现内容加密。过去的OAI使用Lambda@edge的方式即将（或者已经）废止。
- 通过认证header可以设置很多权限限制：比如ALB：CloudFront Cache Policy
- 集成Cognito，发行JWTtoken，通过Lambda@edge验证身份。（Lambda@Edge 是 AWS Lambda 服务的一种扩展，在 AWS 的全球性内容分发网络 CloudFront 的边缘位置（Edge）上运行 Lambda 函数。使用 Lambda@Edge 可以对请求进行路由、身份验证、内容压缩、图片优化等操作。）

### WAF

- Web application firewall所以是layer 7，用于第七层的漏洞滤网。部署在：*ALB，CloudFront，APIGateway*等
- WAFACL保护内容：*没有DDos*，包括：IP地址，HTTP头部，SQL注入，跨网站脚本攻击（XSS：cross-site-script），地理位置匹配，速率规则等
- 日志发送：CWLogs，S3，kinesisFirehose-others
- 可灵活创建Rule Groups设置自定义的规则，可以设置优先级，以及在CloudWatch中的Metrics字段。

### Shield

- 防止DDoS攻击的。宛如星际攻击防卫战。
- ShieldAdvanced：ALB，CLB，NLB，ElasticIP，edge（CloudFront，Route53，GlobalAccelerator）

### WAF & Shield & Firewall manager

- 组织级别管理，各种防火墙服务在一处管理，Firewall Manager可以一次性统一部署FW在许多账户。
- Region Level
- 三者的使用规则：一般来说WAF足够，但是要组织级别的管理，新增资源立刻适用规则，那么使用Firewall Manager更好，如果要增加DDos保护，那么加上Shield。

### API Gateway

- 包括一个API应有的所有功能。
- 构架上来说：
  - 比如后面可以坐落一个LambdaFunction来被invoke，作为API的后端。
  - 可以有一个http端点，这个端点可以是一个本地的服务器，也可以是一个ALB，所有可以成为http端点的东西。
  - 或者后面可以坐落很多的AWS服务。比如后面可以有一个kinesis服务，让客户端进行更新推送。所以理论上可以通过API网关暴露你的所有服务。
- 安全认证方法：
  - 内部用户可以使用IAM认证。
  - 外部用户可以使用Cognito服务。
  - 还可以使用custom logic authorizer创建自己的认证方法。
  - HTTPS安全Custom domain name使用ACM（aws certification manager）：全球边缘优化管理，认证需要使用美国东部的region，如果是region endpoint，需要和api网关同一个区域设置。另外还需要在Route53中设置cname和A-alias。
- API Gateway就像是在服务外面套了一层膜，让我们通过一个link的呼出，就能得到那个膜背后的信息。
- 在发布API的时候设置的一个stage可以是开发产品环境的名字，也可以是version的名字，这就是我们经常说的版本号码。
- API的安全问题：可以在resource policy中设置access condition来限制公共访问。而私有API必须通过特定的VPC源（VPC interface endpoint）进行访问，这是一个设置的条件。而在**两个VPC环境**中进行私有访问，不需要进行vpc peering，而同样是在一个中设置vpc interface endpoint（之前endpoint的地方有讲）在另一个中设置policy条件即可。

### AWS Artifact

- 下载安全合规文档和报告等地方。可以说不算是一个服务。

### Route53

- DNS Poisoning中毒：是一种IP攻击手段，因为我们使用udp协议从服务器取得网址，udp在安全上比较薄弱，使得黑客可以替换正确的ip导致我们点击错误的网站。
- DNSSEC域名系统安全扩展（Domain Name System Security Extensions）是针对这一问题的安全手段。他是DNS安全扩展，是一种协议。DNSSEC 的主要目标是解决 DNS 投毒攻击（DNS spoofing）和 DNS 缓存投毒攻击等安全漏洞，以确保 DNS 查询的真实性和完整性。
  - 数字签名：DNSSEC 使用数字签名技术来验证 DNS 数据的真实性。在 DNS 数据的传输过程中，DNS 服务器会对 DNS 记录进行数字签名，生成签名数据并附加到 DNS 响应中。
  - 公钥加密：每个 DNS 区域都有一个公钥和私钥对，用于生成和验证数字签名。DNSSEC 使用公钥加密技术来验证数字签名的有效性，客户端可以使用相应的公钥来验证 DNS 响应的签名数据。
  - 链式签名验证：DNSSEC 使用链式签名验证（Chain of Trust）来验证 DNS 数据的完整性。在 DNSSEC 中，每个 DNS 区域的签名数据都包含了前一个区域的公钥指纹，客户端可以通过追溯链式签名来验证整个 DNS 查询路径的真实性。

### AWS Network Firewall

- 是一个对VPC level全方位的保护的服务，网络的layer3～layer7，想象它就在VPC的周围环绕。控制任何的网络进出，包括peering vpc和本地vpn连接。可以进行流量过滤和检查。
- Inspect traffic（Traffic Filtering by rules & Active Flow Inspection）and do Deep Packet Inspection.
- 当流量进入Internet Gateway的时候，被NF捕获和检测。
- 一种构架：对网络防火墙的rule group进行审查可以使用GuardDuty，将finding的结果发送到Security Hub，当发现异常，可以通过Eventbridge启动StepFunction进行后续的操作（对防火墙增加规则，屏蔽IP，SNS通知等）。
- 支持通过ACM进行的流量加密。

### EKS

- 通过pod evnets，node events将log送到CWLogs，他们自身的TTL只有60分钟，长久保存需要CW。

### XRay

- AWS X-Ray 是一项由亚马逊网络服务提供的服务，旨在帮助开发人员分析和调试分布式应用程序。
- 分布式跟踪：X-Ray 可以跟踪分布式应用程序的请求，并记录每个请求通过的每个服务和组件。这使开发人员能够了解请求是如何在不同的微服务、函数、API 等之间传递的。
- 性能分析：X-Ray 可以记录每个请求的性能指标，如响应时间、延迟、错误率等。开发人员可以使用这些指标来识别应用程序中的性能瓶颈，并对其进行优化。
- 故障诊断：X-Ray 可以帮助开发人员诊断分布式应用程序中的错误和故障。它可以显示请求中发生的异常和错误，并帮助开发人员追踪错误的根源。
- 可视化工具：X-Ray 提供了可视化工具，如跟踪图和服务图，帮助开发人员直观地了解应用程序的架构和调用关系。这些工具可以帮助开发人员快速定位和解决问题。
- EC2，ECS 安装XRay agent，Beanstalk，Lambda，API Gateway

### AWS Workspaces

- 对于EC2来说的Security Group，对于Workspaces来说是IP Access Control Group，通过限制IP来限制访问
- Trusted Devices 认证
- Certificate-based 认证（Windows，MacOS，安卓）

### CloudShell

- 可以在浏览器中，用来进行API call，但是不可以访问VPC（EC2，RDS等）不要误解。不要和EC2 Instance Connect弄混了。虽然他们都是浏览器base的。

## 4 - Identity and Access Manager

文档howdoesIAMworks：https://docs.aws.amazon.com/IAM/latest/UserGuide/intro-structure.html

### IAM Policy

- 用json格式定义的各种规则：allow/deny，action/notaction，principal，resource，condition
- PrincipalArn包括：user，role，root，sts federated user session，然后就是SourceArn。
- Calledvia条件：通过某服务进行操作，一般有四个：athena，dynamodb，kms，cloudformation
- NotAction：表示except。
- Global Service：CloudFront，IAM，Route53，Support需要显示的允许，因为他们是us-east-1区域的。如果限制deny一个区域以外的所有服务，要对全球服务开放notaction来排除他们。
- 其他条件：IP/VPC/PrincipalTags/ResourceTags
- 边界许可PermissionBoundaries：规定了User和Role可以有的最大权限。
- 组织的SCP + 边界许可PB + Identity权限 = 有效权限
- 跨账户许可：IAMPolicy，Resource-base-policy，OrgnizationID等都很有帮助
- 判定逻辑：首先所有的action是被隐式拒绝的。
```
没有显式deny(no explicit deny)
--> 组织SCP为allow: allow on organization scps level
--> 基于资源的许可allow: allow on resource based policy
--> 基于认证的许可allow: allow on identity based policy
--> 边界许可allow: allow on IAM permission boundaries
--> session策略是否许可: allow on session policy
--> 许可: finally allow
```
- IAM Role：即使是跨账户也可以轻松使用的好功能。是给用户或者应用的临时权限，会随着时间的推移而过期。当要授予一个服务角色的时候，需要先给一个user这种权限，iam:Passrole权限，然后才可以由特定的user进行角色授予。
- ABAC：Attribute based access control：基于tag的控制，比如都打有同样标签的user对同样标签的资源进行访问。这可以轻松地扩展资源。基于角色的权限是RBAC（role based access control）在增加资源的时候可能需要手动更新权限。
- MFA认证。如果你设置了MFA而没激活，那么你没法删除设备选项，必须请求管理员删除未激活的MFA设备然后重新设置。
- 设置用户的AccessKey的Rotation：IAM Credentials Report可以查看key的期限，但是更好的方法是用AWS Config设置规则然后用SSM自动更新。
- 用户的PassRole权限，是用户可以对resource设置Role的权限。

### STS

- 安全令牌服务，一种临时权限。用户向STS（还有一个服务Cognit）发送一个AssumeRoleAPI请求一个临时认证（认证会去IAMpolicy确认可不可以授权）后才可以得到临时的role进行访问。
- SAML：SSO/WebIdentity：第三方网络服务Google，FB等/AWS推介使用Cognito
- 使用AWSRevokeOlderSessionPolicy可以让老的session在一定时间后失效，这可以让token权限定时过期。
- version1:STS API 是 AWS 提供的全局服务，可以通过 AWS 全局终端节点（global endpoint）进行访问。STS API 提供了一系列 API 操作，如 AssumeRole、AssumeRoleWithSAML、AssumeRoleWithWebIdentity 等，用于生成临时安全凭证（Temporary Security Credentials）。
- version2:AWS STS Regional Endpoints 是在 AWS 全球基础设施中提供的区域级服务，每个 AWS 区域都有一个对应的 STS 终端节点。AWS STS Regional Endpoints 提供了与全局 STS API 相同的 API 操作，但是它们将请求限制在特定的 AWS 区域范围内。
- ExternalID：解决混乱代理问题
- 如果STS的token泄漏了可以通过 AWSRevokeOlderSessionPolicy 立刻ban掉所有的session。一次性的。

### SCP的相关Policy示例

- 限制EC2type的权限：

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

- 为全球服务豁免。因为全球服务只在 us-east-1 区域提供，所以当要限制某个区域的服务时，在 NotAction 中列出这些全球服务，就可以正常使用了。
<details>

<summary>点击展开/折叠</summary>

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
</details>

### IMDS(instance metadate service)

- EC2 instance metadata service。提供服务器的元数据。（hostname, instance type, networking settings...）
- http://169.254.169.254/latest/meta-data 是每个服务器的endpoint。以键值对的方式，方便进行自动化的设置。这个取得方法是第一个版本，它不使用Token，在 CW 中也可以监控到这个的使用 ：MetadataNoToken 指标。
- 但是不推介第一个版本，第二个版本更安全（使用token），分为两步：
  - 首先使用 Header 和 PUT 取得 Session Token。`token = 'cur put xxx -H xxx' token`
  - 然后使用 Session Token call IMDSv2，并且使用 Header。`cur xxx -H with`
- 如何限制对metadata的访问：
  - sudo iptables工具，设置本地防火墙规则。
  - 使用控制台或CLI：HttpEndpoint=disabled

### S3和对象的跨账户访问权限

- 判断一个用户是否可以access一个S3bucket里的object的时候：
  - 首先判断user是否有IAM权限
  - 然后是BucketPolicy是否显式拒绝访问
  - 最后是要访问的object是否对用户开放访问（ACL：access control lists，这个现在已经被弃用，但是可以设置开启，但是也可以手动开启：Onwer Enforced setting = Disabled）。
- 区分桶和object访问权限的policy表记：BucketPolicy的访问权限一般是对S3桶资源。Object对象访问权限，一般是在arn对url后面有一个*符号表示是bucket内的资源。
- 跨账户的S3资源访问的方法
  - IAM Policy + BucketPolicy
  - IAM Policy + ACLs（for objects）：only works when Bucket Owner Enforced setting = Disabled但是默认是Enabled，因为刚刚上面说了，如果一个创建了桶的owner不能对里面的object控制，那么事情会变得复杂。有ACLs的时候需要开放对桶主的访问许可：s3:x-amz-acl:bucket-owner-full-control
  - Cross-account IAM Roles ：bucket policy拥有授予对应账户的stsAssumeRole权限，就可以开放对别的账户Role（临时桶访问）权限了。
- 总之S3的访问权涉及一个用户本身的认证和一个Bucket的Policy。原本还有一个ACL是针对桶里的对象的，但是它会让事情变得复杂，比如A自己的桶别人放的文件反而桶的所有者不能访问，需要很复杂的设定，所以ACL被弃用了。
- 同账户不需要 Bucket Policy
- S3的*Access Point Policy*（服务页面中有这个tab选项）：针对桶中的不同文件夹，可以根据文件夹的名字prefix设置不同的访问策略，更灵活地控制访问权。这是一种很好的层级控制策略。比如不同部门有不同的prefix。
  - 但是相对的，必须设置VPC Endpoint（interface/gateway）来访问这些Access Points。
  - 在bucket policy层级要设置，限制只能通过Access Point进行访问的Policy。
  - 多区域 Access Points 可以设置failover，提高可用性。
- Bucket Policy的其他用法：限制只能用Https访问，限制特定IP源，限制特定user。
- Bucket有一种因为policy设置错误，而被锁locked的情况，只能通过root账户删除bucket的policy然后重新开启。
- CORS：（跨域资源共享）S3 CORS（跨域资源共享）是 Amazon S3（简称 S3）中的一种机制，用于允许客户端网页从不同域名的网站上加载 S3 存储桶中的资源，而无需绕过浏览器的同源策略限制。浏览器的同源策略规定了网页只能请求来自同一源（域名、协议、端口号组合）的资源，这意味着在默认情况下，来自不同域名的网页无法直接访问 S3 存储桶中的资源，以防止跨站点请求伪造（CSRF）等安全问题。通过配置 S3 CORS，您可以允许特定的网站（或者所有网站），或者特定的S3桶，从浏览器中直接加载S3存储桶中的资源，从而实现更灵活的网站设计和更好的用户体验。

以下是一个示例 S3 CORS 配置的 JSON 格式：

```json
{
  "CORSRules": [
    {
      "AllowedOrigins": ["http://example.com"],
      "AllowedMethods": ["GET", "PUT"],
      "AllowedHeaders": ["*"],
      "MaxAgeSeconds": 3000
    }
  ]
}
```

在这个示例中，配置了允许来自 `http://example.com` 域名的网站通过 `GET` 和 `PUT` 方法访问 S3 存储桶中的资源，允许任意标头，并设置了一个最大缓存时间为 3000 秒。Headers的部分可以设置Authorization。这里Origin设置最好是特定的网站或者S3bucket，防止别人重新搞一个网站用你的资源。

Origin=scheme(Protocol)+host(domain)+port
same origin:www.example.com/app1 & www.example.com/app2
diff origin:www.example.com & other.example.com

### Cognito

- Amazon Cognito 是 AWS 提供的一项身份验证和用户管理服务，用于帮助开发者构建安全、可扩展的身份验证和用户管理功能。Cognito 提供了一系列功能，包括用户注册、登录、认证、访问控制、数据同步等，可用于 Web 应用程序、移动应用程序和 IoT 设备等各种场景。
- CUP：用户池（User Pools）：是 Cognito 的核心组件，用于存储用户的身份信息，包括用户名、密码、电子邮件、手机号码等。开发者可以通过用户池管理用户注册、登录、身份验证等流程，并可自定义身份验证方法，包括用户名密码认证、手机短信验证码认证、社交登录认证（如 Google、Facebook、Amazon 登录）、OpenID Connect、SAML 等。
  - CUP 集成 AWS API Gateway，ALB等服务。用户认证成功后会从 CUP 收到一个JWT（json web token），这个JWT用于AWS资源和CUP之间的验证。
  - CUP 内可以设置 User Group 以便更好的控制用户权限。
- 身份池（Identity Pools）以前叫联合认证：身份池用于管理用户的身份凭证，如 AWS Access Key、Token 等，以便访问 AWS 资源。身份池与用户池关联，可以将用户池中的用户身份映射到身份池中的 AWS IAM 角色，从而实现对 AWS 资源的访问控制。
  - 具体来说在用户池给了用户JWT后，通过身份池，可以从STS等服务发行一个AWS资源访问的TempCredentials，使用这个临时身份，就可以像一般的AIAM用户一样临时访问资源了。
  - AWS内的Policy对CognitoUser的权限使用Role。并且通过UserID细化。使用Policy Variables定义用户ID关联的资源。
- 通过这两种服务，可以通过 Cognito 对用户的访问进行细粒度的控制，包括用户群组管理、权限管理、设备管理、会话管理等。
- 数据同步：Cognito 提供了数据同步功能，可用于在多个设备和平台之间同步用户数据、应用程序状态等信息，确保用户在不同设备上的一致性体验。

###  Identity Federation

- IAM Policy使用Policy Variable（类似${resource.user_id}）来控制访问
- SAML2.0
  - IdPs
  - Microsoft AD
  - 需要建立IAM和SAML提供者之间的trust关系
  - UnderTheHood：STS API：AssumeRoleWithSAML
  - 这是一个oldway，更新的方式是用SSO
- Custom Identity Broker
  - 建立方式一样，但是这个只有在上面一种不满足要求的时候使用。
  - 需要自己设置赋予的Role
  - 使用的STSAPI是 AssumeRole or GetFederationToken
- Web Identity Federation with/without Cognito
  - 官方推介Cognito，因为它*支持匿名，MFA，数据同步*。
- SSO
  - 也就是下面一个标题Identity Center

### IAM Identity Center（SSO）

- 前身就是AWS Sigle Sign-On
- 一键登录所有的AWS account，组织，还有其他第三方应用。TB案件中的AWS登录就是这样的。
- Identity Center中的权限控制使用 Permission Sets
- 权限组来自MicroAD或者该服务的built-in Identity Store，通过 group + permission sets 进行管理
- ABAC细度管理

### AWS Directory Services

- AWS Managed Microsoft AD：AWS和On-premises都要设置，双方建立trust关系进行认证，是双方的。支持MFA。要和本地的AD连接需要VPC的VPN连接（Direct Connect或S2S VPN）无缝连接。
- AD Connector：只需要on-premises设置，从AWS认证，中间设置Proxy代理，访问本地AD进行认证，是单方的认证。需要VPC的VPN连接（Direct Connect或S2S VPN）
- SimpleAD：只在AWS内有，非常简单。

- 概念：
- 什么是MicrosoftAD：是一种由微软提供的目录服务，用于在组织内集中管理和存储网络资源和用户身份信息。它是基于LDAP（轻型目录访问协议）的目录服务，旨在提供对组织内用户、计算机、打印机、应用程序等资源的集中管理和身份验证。Object的组织是trees，一组tree是forest。
- 什么是ADFS：是指 Active Directory Federation Services，是由 Microsoft 提供的用于实现单点登录（SSO）和跨组织身份验证的解决方案。它允许用户使用一组凭据（通常是用户名和密码）在不同的应用程序、服务和组织之间进行身份验证，而无需在每个系统中单独登录。ADFS 可以与现有的身份基础设施（如 Active Directory）集成，允许组织利用现有的用户和组织结构信息进行身份验证。

### AWS Verified Access

- 以零信任指导原则为基础构建，在授予访问权限之前验证每个应用程序请求
- Verified Access 消除了对 VPN 的需求，从而简化了终端用户的远程连接体验，并降低了 IT 管理员的管理复杂性。
- Zero Trust Principal，零信任：每个 *访问请求* 都必须经过严格的身份验证和授权，才能授予访问权限。 这有助于降低未经授权访问敏感资源的风险，即使攻击者已经渗透到组织的网络内部。

## 5 - Data Protection

### 加密类型

- 传输中的加密TLS/SSL=HTTPS，存储中的服务器加密Server-Side加密，客户端加密Client-Side加密即信封加密
- 加密类型：对称和非对称，信封加密
- 数字签名：使用非对称加密技术，对消息的摘要进行哈希变换，和私钥加密，对方使用公钥解密验证摘要哈希值是否一致

### CloudHSM

- 硬件加密，专用硬件，硬件由AWS管理但是加密是自我管理，非free，必须安装客户端软件和HSM进行SSL连接，管理keys和users。
- 使用SSE-C加密的时候是一个好的选择。
- IAM permission：CRUD an HSM Cluster（Multi-AZ）高可用性
- 和KMS的CustomKeyStore集成，所以APIcall可以被CloudTrail捕获
- 和其他Account共享cluster：通过共享其所在的subnet，设置subnet的SG来允许traffic

### KMS

- 对称加密（AES256）和非对称加密（RSA）：
  - 对称加密广泛用于内部服务，如果你用客户端信封加密，这个是必须。
  - 无法取得对称加密的密钥，只能通过API使用。
  - 非对称加密可以取得公钥（无法取得私钥）进行数据加密和上传，适用于给无法进行APIcall的user。
- Keys的类型：
  - CustomManagedKeys：客户管理，需要进行信封加密。
  - AWSManagedKey：用于内部服务加密，每年自动更新。
  - AWSOwnedKeys：AWS自己使用的用于某些服务的加密，你啥也摸不到。
- Keys的来源：三种方式
  - KMS创建和管理。外部上传（自制信封加密自己上传管理）。HSMCluster创建的存在CustomKeyStore中的keys。
- Multi-RegionKeys：可用于灾难恢复，主key+复制key，KeyID一样，可以跨区使用，比如一个区加密，一个区解密。一般不推介使用，只有在特殊情况才被推介使用，比如全球服务的DynamoDB GlobalTable或者GlobalAurora，方便客户端在不同区域也可以使用API进行解密。

- 加密过程：
  - 数据<4KB：通过Encrypt API（限制就是4kb）请求Encrypt和Decrypt，KMS确认IAM权限然后进行相应的操作即可。
  - 数据>4KB：通过信封加密：Envelope Encryption == GenerateDataKey API：
  - 信封加密过程
    - client:GenerateDataKey API(请求生成用于加密数据的数据密钥)
    - server:Send plaintext data encrypt key(DEK)(加密数据的key) and Encrypted DEK(被加密的加密数据key)
    - client:Encrypt File with DEK(用加密数据key加密数据)
    - client:Final File(Encrypted datafile and Encrypted DEK)(将加密后的数据和加密后的key放在一起即可)，丢掉没加密的DEK。
    - 那么你想解密的时候就需要请求KMS解密DEK。
  - 信封解密过程
    - client:Decrypt API call
    - server:Send plaintext data encrypt key(DEK)
    - client:Decrypt File with DEK
  - 以上的信封加密和解密都是在客户端发生的，只有DEK的解密会在KMS发生。
  - Encryption SDK：feature-DataKeyCaching 功能可以节省API使用一个key加密多个files（LocalCryptoMaterialsCache），另外，Data Key Cache还可以缓解API call limit的问题。It's a trade-off

- KMS Automatic Key Rotation:
  - 是为了 Customer-Managed CMK not AWS Managed key
  - every 1 year
  - 只改变背后的key，而key-id不变，这样你不用更改各种设置
- 手动更新key（为了提高更新频率之类的目的）：适用于你自己管理的CMK，但是这会改变key的ID，所以最好给key设置别名alias（UpdateAlias API call）。
- Key deletion：
  - KMS生成的key删除会有7-30天冷却时间，期间不能使用，可以恢复，到期会完全被删除。或者可以选择手动disabled，可以随时恢复。
  - ImportedKey：可以设置expiration时间，到期KMS会帮你删除。手动删除该key material，metadata会帮你保留以便重新import。手动disable或者设置到期删除，则会帮你完全删除。
  - AWS自己管理的Key你无法删除。
  - 包含导入材料（是指用于生成加密密钥的随机数据）的密钥：可以设置有效期或者立刻删除。
- Key删除检测：
  - APIcall - CloudTrail - CWLogs - CloudWatchAlarm - SNS。
  - APIcall - CloudTrail - EventBridge - SNS/SSManager(CancelKeyDeletion)
- 删除Multi-Region Key必须先删除复制的keys，并且要过7-30冷却期，主key也要冷却。如果想只删除主key，需要promote其他key为新的主key，才能删除老主key。

- KMS Key Grant：让你将自己的key的使用权，临时赋予给别人，而不会影响整个IAM KMS Policy。没有过期时间需要手动删除。只能通过CLI执行：`aws kms create-grant`（AWS众多服务背后都使用grant，使得他们可以使用KMS加密的内容，在工作结束后会自动删除grant权限 kms:GrantIsForAWSResource）
- 认可流程：显式deny - OUScp - VPCEndpointPolicy - KeyPolicy - Grant - CallerPolicy - same account 之间的组合使用
- 跨账户CrossAccount使用许可：普通的是双方许可，当涉及其他服务比如EBS则需要Grants权限。
- KMS的权限和S3的很像，key的权限，服务的权限，IAM的权限，加上一个授予别人使用的Grant权限。
- ABAC管理，key可以基于tags&aliases管理。

- EBS加密key无法改变，只能通过snapshot重新create的时候使用新的key。（也就是说EBS只有在被创建的时候才能进行加密）
- EBS的加密不是默认的，需要在account-level对加密进行per-region对有效化设置。
- EFS的重新加密，需要创建新的加密EFS，使用DataSync迁移数据。
- SSMParameterStore的加密也是通过KMS，使用对称加密，需要双方权限。两个标准：Standard所有的parameters使用同一个key，Advanced每个parameter使用不同的uniquekey进行信封加密。

### Secrets Manager

- 强制在X天后进行Rotation
- 使用 Lambda 可以自动生成新的 Secrets：需要 Lambda 对 SecretManager 和数据库的访问许可，包括对subnet的网络设置（VPCEndpointorNATGateway）
- 内部使用KMS进行加密，对称加密，发送GenerateDataKeyAPI进行信封加密，需要双方service的权限
- 与各种DB集成用于密码存储和轮换
- 跨区复制功能，用于灾难恢复，跨区DB和跨区应用等

- ECS（worker/task）也集成使用SecretManager和SSMParameterStore的key进行RDS和API的连接。

### S3数据加密

- 四种加密object的方式：（加密方式信息都在uploadAPI的HTTPS的Header部分表示）
  - SSE-S3：AmazonS3管理的server-side加密（AES-256）（default）
  - SSE-KMS：KMS管理的server-side加密，便于自己控制和CloudTrail追踪，相对应的，使用也会受到KMSAPI限制，bucket公开外面也看不到因为没有KMS权限，上传文件的人还需要有GenerateDataKeyAPI权限用来加密数据。
  - SSE-C：客户提供Key，server-side管理，必须使用HTTPS，将key放在Header一起传递给S3进行加密
  - CSE：客户client-side加密上传数据，完全自己管理
- 强制传输中的HTTPS安全，使用BucketPolicy：aws:SecureTransport条件设置，还可以用BucketPolicy设置强制使用什么类型的加密。
- S3-Bucket-Keys（减少对KMS的APIcall，降低费用）：当使用SSE-KMS的时候，可以设置的一种bucket-level的KEY。KMS只生成一次bucket keys，每次上传object，它就会生成新的key用于加密。
- 大文件的multi-part-upload需要的KMS权限：kms:GenerateDataKey,kms:Decrypt。需要对各个部分进行加密，上传后自动被解密合并为一个完整的文件。
- Amazon S3 Inventory 是 Amazon Simple Storage Service (S3) 提供的一项服务，用于提供有关存储桶中对象的详细清单报告。S3 Inventory 可以定期生成清单报告，以列出存储桶中的所有对象及其相关的元数据信息，如对象键 (Key)、大小、最后修改时间等。
- 如何使用S3-Batch批量加密：
  - 需要对S3和KMS的访问权限。
  - 通过S3 Inventory列出和metadata相关的所有object的状态，包括加密状态。
  - 用S3 select&Athena 过滤需要加密的对象。
  - 使用s3:PutObjectAPI进行批量上传加密。
- S3 Glacier Vault Lock：
  - 目的是为了实现 WORM（Write once read many）比如不应该被修改的日志文件。
  - 需要创建一个 Vault lock policy。然后锁定该policy防止被修改。（该锁定过程会发行一个lockID需要再24小时内完成锁定）
  - 然后再通过API存放文件。（没有UI界面）
  - 使用S3 Object Lock也可以达到同样的目的但是比较复杂，需要开启版本保护，以及RetentionMode（保留模式）以及 LegalHold 及其期限和相关权限。
- S3 LifeCycle：各个存储层级的移动。
- S3 Replication：需要开启versioning，非同步复制更新，CRR（跨区复制）和SRR（同区复制）两种。开启复制delete-marker后，删除只会被标记，但是永久删除某版本不会被标志，被复制的文件还是存在（这可以防止恶意删除）。不会发生chaining连锁复制。

### Load Balancing

- 类型：
  - ALB（HTTP/HTTPS/gRPC（谷歌开发的开源远程过程调用框架））是应用层的，静态DNS-URL
  - NLB（TCP/UDP）是传输层，静态IP（ElasticIP）
    - 它的LB目标是EC2或者IP，还可以是ALB（HTTP/HTTPS）。
    - 健康检查支持TCP/HTTP/HTTPS协议。
    - IP Preservation（IP保留功能），默认开启（除了IP地址TCP/TCP/TLS默认关闭）该功能可以在流量进来的时候，获取ClientIP地址，关闭则获得的是NLB的私有IP地址。
    - 现在ALB和NLB都支持绑定SG，这很方便，因为可以更精细的控制流量（比如NLB的SG被EC2的SG允许），SG之间具有控制传递，我的理解SG就像联通门，NACL就像是双向电网。
  - Sticky Session：ALB和NLB支持粘性会话。
    - 两种cookies：Application-based cookies（客户端生成或者LB生成），Duration-based cookies（LB生成Cookies）。
  - GLB（Gateway-IP，GENEVE协议）是网络层，GENEVE（Generic Network Virtualization Encapsulation）协议是一种用于网络虚拟化的封装协议。该LB将流量引向EC2防火墙，有针对 IP Packets 的入侵检测功能。
  - 传输中安全：User - HTTPS - LB - HTTP - PraviteEC2
    - TLS 和 TCP Listeners：TLS监听会在LB就终止443端口安全通信然后解密数据（use SSL certificates key）通过私网传递到target服务器。如果想要端到端的加密，需要TCP监听，从而实现私网中也是加密的状态，但是target服务器需要有解密流量的功能。总之TLS监听的加密在LB结束，TCP的监听方式是全程加密的。
  - LB的的HTTPS安全协议用的是X.509 Certificates(TLS/SSL Certificates)，可以通过ACM设置，也可以自己上传。
  - SNI（Server Name Indication）
    - 是一种 TLS（Transport Layer Security）协议扩展：在单个服务器上托管多个域名的 HTTPS 网站的时候，服务器就可以根据主机名选择合适的证书来建立加密连接，而不再受到 IP 地址的限制。
    - 支持 ALB 和 NLB。
    - 当背后有多个Target Group的时候，就可以使用不同的域名访问了。

### ACM

- 可以发行Public SSL Certificates。
- 支持LB，CloudFront，APIs on APIGateway。
- 自动更新和维护证书。手动上传的需要自己更新和维护。如果自动更新失败了那可能是CNAME没设置。也可以用Email更新但是需要Domain Owner。
- 公有的和私有的，需要对应的CA认证。公有的需要公共DNS（可用Route53设置）。私有只适用于内部的应用。
- 是区域regional服务，如果你服务是全球的你需要各个区域都设置。
- 公有的证书需要设置CNAME进行DNS有效化验证：（私有的不需要验证有效化）（还可以用Email的方式验证）
  - 在验证域名所有权时，AWS ACM 要求你在 DNS 中添加特定的记录，以证明你对该域名的控制。
  - CNAME 记录在这种情况下非常有用，因为它允许你将一个域名指向另一个域名，而不是直接指向 IP 地址。AWS ACM 可以要求你在 DNS 中添加一个特定的 CNAME 记录，将您的域名指向一个由 ACM 控制的特定地址。这样，AWS 就可以验证你对该域名的控制权。
- 证书过期检测：
  - ACM send daily expiration events via EventBridge - SNS/Lambda/SQS
  - AWS Config - acm-certificate-expiration-check via EventBridge - SNS/Lambda/SQS

### AWS Backup

- 对AWS资源的全局备份管理
- 支持，跨区域，跨账户。
- 支持多种备份计划：PITR（point time recovery），tags-based，on-demand，scheduled，to cold storage。
- 支持Vault-lock。write once read only。

### Amazon Data LifeCycle Manager

- 针对EBS快照和EBS-based AMIs的生命周期管理，创建，更新，删除，备份，跨账户copy。
- 可以基于tags的管理。
- 感觉是很限定的服务。EBS only。

### AWS Nitro Enclaves

- 完全隔离的虚拟环境，用来处理敏感信息。
- Cryptographic Attestation（加密认证）技术。
- 会和一般的EC2有一个安全本地通道。

### ECR

- ElasticContainerRegistry
- 公有或者私有image存储。
- in S3
- 只有创建Registry的时候可以选择加密。
- 通过KMS加密。信封加密技术。
- Image Scan：漏洞扫描，basic扫描可以在push时和手动扫描，advanced扫描可以使用Inspector进行（针对OS和编码语言）扫描和通知。

### AWS Signer

- 允许你在构建和部署在AWS上的应用程序时对其（code-signing）进行数字签名。
- 数字签名是一种安全机制，用于验证数据的完整性和来源。通过使用AWS Signer，可以轻松地向应用程序添加数字签名，确保它们在传输和存储过程中不被篡改。这对于确保应用程序在部署过程中的安全性和完整性非常重要，尤其是对于那些涉及敏感数据或重要操作的应用程序。
- AWS Signer支持各种不同的应用程序和开发环境，包括AWS Lambda函数、IoT设备、Docker容器等。

### AWS Glue

- 构架：S3（put file）- Lambda trigger - Glue（turn file to parquet）- Athena analyze
- Glue Data Catalog - Glue Data Crawler 从S3，JDBC，RDS，DynamoDB获取数据，依存于Catalog的服务：Athena，Redshift Spectrum，Amazon EMR
- Glue Job Bookmarks：防止重新处理老数据

### EBS data wiping

- 当你删除一个EBS的时候，AWS会帮你清除数据，把数据全都用0替换，无需你手动删除。

### RDS&Aurora

- 数据库从没加密到加密状态，只能通过snapshot，而且只能是从没加密的snapshot创建一个新的db cluster的时候才能重新加密。其他的操作都不能加密。

## 6 - Management and Security Governance

### AWS Organization

- 统合账户管理，统合支付。
- 可以多层嵌套OU。
- 可以在组织层级设置SCP（Service Control Policy）
  - Management Account不受scp管制。将该账户放在Root组织层级比较好。
  - SCP可以设置黑白名单。
- 可以通过tags进行权限控制。

### Control Tower

- 在组织之上运作的。适用于多账号管理。
- 使用SCP和AWS Config控制人和设置的变化和问题。

### AWS Config

- 指定设置规则，记录资源设置变化（timeline变化记录的方式），检查设置是否合规，dashboard管理。
- 可以储存日志于S3，可以集成，EventBridge，SNS进行通知，以及其他服务。（但是Security Hub是可以自动接收它的评估结果的）
- 是区域服务，但是可以集合各个区域和账户的设置。
- Rules：可以使用AWS的rules，也可以使用Lambda设置CustomRules。
- 无法阻止违规设置，只能overview设置和变化。
- Remediation功能：可以设置用SSM的document进行Automation自动修复。
- Aggregator：整合各个账户的config数据，到一个账户，如果是OU设置下的不需要各个账户设置，如果不是OU则需要被整合的账户授权。只用于整合数据，不用于统一设置，如果要给各个账户统一设置，使用CloudFormation StackSets。

### Trusted Advisor

- 检查和建议。
- 六个方面：Cost-optimization，Performance，Security，FaultTolerance，ServiceLimits，OperationalExcellence。
- 但是 Free Plan 只能用很少的一部分比如安全和服务限制的一部分check，如果要 FullSetCheck 和 supportAPI 的使用，要求 Business&Enterprise Support Plan。

### Cost Explorer

- Resource cost可视化。
- 基于过去数据预测未来12个月的使用量。
- Saving Plan

### AWS Cost Anomaly Detection

- 基于机器学习的异常检测。
- 只需要激活功能，不需要设置。
- 流程：Cost-Monitor，通知警报，根源分析。

### CloudFormation

（一个在各种环节都很重要的服务）

- IaC的所有好处。
- 优化Cost，比如你晚上destory所有环境早上重建。
- 可以通过图表自动生成代码。
- 声明型的代码，因为不需要排序和逻辑。
- 可以复用文档的代码和网上的模板。不要从0造轮子。
- 必须用弗吉尼亚北region。
- 在stack创建界面为CloudFormation赋予角色或者使用user自己的权限。前者更好。
- iam:PassRole：用户可以赋予CloudFormation各种服务角色而自己不拥有对要创建的服务的权限。（符合最小权限原则）
- StackPolicy可以设置，在stack更新的时候，允许和禁止哪些资源更新。用以保护资源。这个Policy默认保护所有资源，所以必须显示设置允许更新。
- 动态索引（DynamicReferences）：
  - reference-name: ssm/ssm-secure 从SSMParameterStore动态取得加密或非加密的参数
  - reference-name: secretmanager 从SecretManager取得密码之类的信息。
  - 格式：{{resolve:service-name:reference-key}}
- Termination Protection：是一种防止stack被误删除的机制，默认无效，手动开启，防user删除。
- Drift Detection：检测是否有通过CF之外的资源设置变更。
- CF-Guard：声明型DSL语言声明你自定义的Policy，然后用内置的testing framework验证你的Policy rules是否*按照预想的方式起作用*。我想这里起作用work是关键。

### Service Catalog

- 工作方式：
  - 管理员设置Portfolio：一组预设的product的组合。和Control：user相对于portfolio的IAM权限。
  - Product：CloudFormation Templates
  - User就可以根据预设好的ProductList，lanch自己想要的服务了。
- 方便user在合规和允许的范围内，使用AWSResource。

### EC2 Image Builder

- 自动构建EC2 AMI，服务本身免费，只需付服务器和AMI存储的费用。
- 过程：定义EC2比如安装内容和test内容 - build EC2 instance - create new AMI - test EC2 instance - 分发distribute到各个region
- Run on schedule / when EC2 updated
- 需要对SSM，ECR，S3等的相对应的访问权限。

### RAM: Resource Access Manager

- 防止资源重复。
- 分享功能：分享资源给其他的账号，可以同组织也可以不同。
- 每个账号只对自己的账号资源负责。
- 主要是VPC，subnets。

### AWS Audit Manager

- 检查你的workload是否有风险和合规。
- 集成多种AWS内部服务，进行连续的证据收集，检查根源，收集报告。

### Well Architected Framework Tool

- 卓越运营 Operational Excellence
- 安全 Security
- 可靠 Reliability
- 性能效率 Performance Efficiency
- 成本优化 Cost Opetimization
- 可持续发展 Sustainability

- 他们是协同合作效果。
- 有专门的WAF工具TOOL（设置workload，回答各种设置问题）帮助检查构架是否符合。
