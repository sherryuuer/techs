SDLC（软件开发生命周期，Software Development Life Cycle）是指软件开发过程中涉及的一系列步骤和阶段。每个阶段都有特定的活动和产出，旨在确保软件项目按时、高质量地交付。典型的SDLC阶段包括：

1. **需求分析（Requirements Analysis）**：收集和分析用户需求，确定系统功能和性能要求。
2. **系统设计（System Design）**：设计系统架构和详细设计，定义软件组件和接口。
3. **实现（Implementation）**：编写代码，将设计转化为实际的软件产品。
4. **测试（Testing）**：进行各种测试（单元测试、集成测试、系统测试、验收测试）以确保软件的质量和功能。
5. **部署（Deployment）**：将软件部署到生产环境，使其可供用户使用。
6. **维护（Maintenance）**：修复缺陷、进行改进和升级，以确保软件的长期稳定和性能。

不同的软件开发方法（如瀑布模型、敏捷开发、DevOps）可能会有不同的阶段和流程，但核心的SDLC概念在大多数方法中都是一致的。

DevOps是一种结合了软件开发（Development）和IT运维（Operations）的实践和文化，旨在通过更紧密的协作和自动化工具来提高软件交付速度和质量。DevOps的核心理念包括以下几个方面：

1. **持续集成和持续交付（CI/CD）**：
   - **持续集成（Continuous Integration, CI）**：开发人员频繁地将代码更改合并到主分支，并通过自动化构建和测试来确保代码质量。
   - **持续交付（Continuous Delivery, CD）**：在持续集成的基础上，代码更改在经过自动化测试后，可以随时部署到生产环境。

2. **自动化**：
   - 使用自动化工具和脚本来减少人为错误，提高效率。包括自动化的构建、测试、部署和监控。

3. **协作与沟通**：
   - 打破开发和运维之间的壁垒，促进团队之间的协作和沟通。通常通过使用跨职能团队来实现这一目标。

4. **监控与反馈**：
   - 实时监控系统性能和用户反馈，快速响应和解决问题。利用日志分析、性能监控工具等来获得持续的反馈。

5. **基础设施即代码（Infrastructure as Code, IaC）**：
   - 使用代码来管理和配置基础设施，使其可以像应用代码一样进行版本控制和自动化部署。

6. **文化转变**：
   - 鼓励一种共享责任和持续改进的文化。开发人员和运维人员共同负责系统的稳定性和性能。

为了实现这些目标，DevOps使用一系列工具来自动化和简化流程，包括但不限于：

- **版本控制系统**：如Git。
- **CI/CD工具**：如Jenkins、CircleCI、GitLab CI。
- **配置管理工具**：如Ansible、Puppet、Chef。
- **容器化工具**：如Docker、Kubernetes。
- **监控和日志工具**：如Prometheus、Grafana、ELK Stack（Elasticsearch、Logstash、Kibana）。

## SDLC Automation

### CI/CD in AWS

- 目的是确保代码*自动化的，正确的，有stages的，被测试的，并且可以在必要时候手动操作的*一种开发方式
- 相关重要服务：CodeCommit，CodeBuild，CodeDeploy，CodePipeline，CodeGuru等
- *CI*持续集成：push代码，通过buildServer（build&test），取得test结果，修改然后继续push，的一个循环
- *CD*持续交付：是CI的下一个步骤，在通过了build和test之后，通过deployServer，自动发布deploy到AppServer生产环境，通过版本控制不断发布
- AWS的发布目的地：EC2，ECS，Lambda，ElasticBeanstalk则包括了自动deploy和provision阶段

### CodeCommit -> Github

- 私有的仓库，Git，版本控制
- 本地的任何Clone仓库都可以随意迁移到另一个仓库
- 认证Authentication：
  * 通过SSHKey：用户通过Console就可以自己设置
  * 通过HTTPS：用户通过AWSCLI设置自己的认证信息
- 认可Authorization：通过IAMPolicy来允许user/role访问仓库
  * 这种控制可以精细到branch级别，这个*Github的分支保护也可以做到，还有强制PR机制，可以给不同分支设置不同的Approve机制*
- 加密Encryption：KMS静态加密，SSH/HTTPS传输加密
- Cross-Account访问：通过本账户的IAM Role和AWS STS（AssumeRoleAPI）进行访问
- 然而可是**2024年7月25日被停止新用户使用了**，请用*Github*
- 集成服务，发布PR信息到EventBridge，通过SNS/Lambda/CodePipeline进行下一步

### CodePipeline

- 可视化CI/CD的Workflow的编排服务，**实质就是一系列的stage的合集**
- Source：不只有Github还有S3，ECRimage仓库和第三方代码仓库等
- Build：CodeBuild，Jenkins，CodeBees，TeamCity
- Test：CodeBuild，AWS Device Farm，和第三方工具
- Deploy：EC2，ECS，S3，*CloudFormation*，ElasticBeanstalk等
- **Invoke**：*Lambda*，*Step Functions*
- ConsistsOfStages：
  * 每一个stage都可以被串行或者并行执行，Github也有类似的功能，增加各种stage
  * **Manual Approval**stage：Owner是AWS，Action是Manual，通过触发一个SNS通知功能，告诉用户进行*代码审查*，审查后Approve代码，用户需要的权限是：*getPipeline*和*putApprovalResult*
  * 可以在每个stage增加手动的Approve动作
  * 集成EventBridge发布failed相关的通知
  * 可以用CloudTrail审查AWS API calls

- 触发管道执行的方式：
  * **Events**事件*触发*Pipeline：AWS中的其他服务 -> 事件触发Pipeline的执行，或者其他第三方服务事件触发，比如merge到了main分支
  * **Webhooks**通过http的webhook进行触发，只需要用任何脚本*推送*payload就可以触发webhook
  * **Polling**的方式，比如Github的仓库为目标，Pipeline进行定期的polling动作，监控仓库变化，取得新的代码

- 集成*CloudFormation*：
  * 支持跨账户跨区域的发布
  * 通过`CREATE_UPDATE`和`DELETE_ONLY`动作进行test环境的发布和删除，以及prod环境的真正发布
  * 它的Action对象可以是 Change Set 也可以是 Stack
  * OverWrite参数Parameter的方式：Json文件/静态文件/动态覆盖

- BestPractice：
  * 总是有一个*pre-prod*环境，进行代码审查，也就是增加*ManualApproval*的stage
  * 并行parallel发布到多个deployment group
  * 并行parallel构建build到多个环境：*RunOrder*
  * 在任何阶段发生了fail等issue，总是用*EventBridge*进行后续触发，比如SNS通知，或者Lambda修复

- **Multi-Region**：支持多区域发布
  * S3的Artifacts stores也应该在对应的区域中设置
  * 除了方法一，Pipeline也可以在build的时候*自动copy*需要的Artifacts，到需要的region，很方便

### CodeBuild

- *Build到底是什么意思呢*，代码的编译，链接第三方各种库，打包，部署的准备等，这相当于**GithubActions中的Workflows定义的build过程**
  * 想起在做React项目的时候通过build命令将代码编译成了可以执行的完整的包的过程
- Source文件是各种仓库，可以用该服务或者上面的Pipeline进行build
- 进行build的指导文件叫做**buildspec.yml**：一般都是yml文件，GithubAction也是，存在于code的*root*目录的地方
- 输出的outputLog，输出到S3和CloudWatchLogs，所以下游可以集成EventBridge进行alarm或者是通知sns等
- 支持的环境：各种预设的代码环境，以及DockerImage

- **How it works**：
  * 从代码仓库取得code文件和*buildspec.yml*文件
  * *pull*需要的*DockerImage*文件用于构建环境等
  * 可设置将一部分文件*cache*到S3中已备下次使用（Optional）
  * *日志*输出到S3或者CWLogs
  * 将构建好的*Artifacts*输出存储到S3，以备发布使用
  * **以上内容和步骤都用buildspec.yml进行具体定义**包括：env，phases，artifacts，cache

- 本地build：出于TroubleShooting的目的，可以在本地跑CodeBuild，通过CodeBuild Agent进行，需要Docker
- 针对VPC内部的资源，需要将CodeBuild创建在*VPC内*，设置相应的subnet，SG等
- 当*CodeBuild Project*被创建好后，就可以集成到CodePipeline中去，作为流程的一个部分（一个**stage**）被执行
- 通过*CodeBuild Service Role*的设置，对AWS中其他服务享有access权限

- **使用Github Actions其实是一样的原理，可以多不同的stage编写不同的yml文件进行Pipeline编写**，比如build.yml，test.yml，deploy.yml文件在不同的on条件下触发

- **环境变量Environment Variables**：多样灵活，Github中的环境变量目前我用过静态存储的
  * 默认环境变量，比如region等
  * 用户自定义静态环境变量，比如环境标识prod等
  * 用户自定义的动态环境变量，比如每次从Secret Manage进行拉取的动态认证信息

- *Build Badges*：在build之后，进行PR的时候，会显示一个*彩色的标识*，提示是否通过了build的stage，并且有链接，可以供代码审查者通过链接查看build的历史记录（可以存在于branch level）

- **Build Trigger**：
  * 可以通过EventBridge和其他AWS服务比如Lambda，进行build触发
  * *Github通过Event*触发CodeBuild的*Webhook*以触发后续流程

- **Validate Pull Request**功能：
  * 在PR通过之后，merge之前，进行评估和测试，保证代码质量，防止代码conflicts等

- 有*可视化的Test Reports*功能：单元unit测试，设置config测试，功能functional测试
  * 需要在*buildspec.yml*文件中增加一个*report Group*设置test report的文件相关信息

### CodeDeploy

- 目的是进行新的版本的部署，**控制部署速度和策略**等
- 部署目标：EC2，本地服务器，Lambda，ECS
- 通过**appspec.yml**文件控制如何发布
- **Deploy到EC2&本地服务器**：
  * 通过*CodeDeploy Agent*操作：Agent可以通过SystemManage进行安装，需要对S3的get和list权限，因为它需要从S3取得新版本的app部署内容
  * in-place发布或者blue/green部署
  * *蓝绿部署*的意思是，同时部署和现有环境一样capacity的另一个新的环境，然后将端点指向新的环境，
    - 必须要有一个LoadBalancer，可以手动也可通过ASG自动化过程
    - 可以设置是否关闭蓝环境，以及多长时间后完全关闭蓝环境，BlueInstanceTerminationOption
    - 该部署方式也可以使用*hook*在各种action前后执行自定义代码
  * Deploy 速度：AllAtOnce/HalfAtATime/OneAtATime/Custom define x%
  * 一些部署策略：比如使用LoadBalancer，ASG，或者*EC2的tags*来锁定部署目标
  * **Deployment Hooks**：意指在部署的整个流程中，根据需要运行一些自定义的代码。
    - 运作方式，是通过*appspec.yml*文件定义hook group，自定义的代码，通过*Agent*进行提前部署
    - hook的含义：
    - *拦截系统事件或函数调用*： 在系统发生某个事件（比如点击按钮、加载页面等）或者调用某个函数时，"hook" 可以拦截这个事件或函数调用，并在原有功能执行之前或之后插入自定义的代码。
    - *扩展系统功能*： 通过"hook"，开发者可以在不修改原有系统代码的情况下，添加新的功能或修改已有功能的行为。
  * Trigger：可以在部署后驱动其他服务比如SNS
- **Deploy到Lambda**：
  * 集成使用SAM的Framework，使用appspec.yml
  * 针对*Lambda alias*进行traffic shift，比如环境的端点指向version1，和version2
  * 也可以使用hooks功能执行自定义代码（lambda）
  * *traffic shift*方式：
    - Linear：线性部署，以一定的速率一点点部署
    - Canary：金丝雀部署，在一开始比如30分钟内，只处理百分之10的流量，觉得没问题了就转移100%的流量到新的版本
    - AllAtOnce：一次转移所有的流量到新的版本，缺点就是不经过观察测试
- **Deploy ECS Task Definition**:
  * 只能使用蓝绿部署，这样以来就是需要并行的两个新旧环境，必须使用LB
  * 部署方式也是*Linear，Canary，AllAtOnce*三种
  * 新Artifact被提前push到*ECR，通过新的image*创建新的green环境，然后通过appspec.yml定义绿色环境发布内容
    - 这个步骤可以通过*CodeBuild*完成
  * 这里也可以用**Deployment Hooks**执行自定义代码，但是这里只能通过**invoke Lambda Function**执行！

- **Redeploy & Rollback功能**
  * Rollback就是重新部署redeploy一个过去的版本
  * Rollback发生的时候，回滚后的也是一个新的版本，而不是restored版本（也就是不是原来的版本号的意思）
  * Rollback可以手动发生也可以自动发生（比如部署失败或者CloudWatch的alarm满足条件的时候）
  * 也可以设置完全不允许rollback

- **TroubleShooting**：
  * InvalidSignatureException：SignatureExpired，<time> is not earlier than <time>：表示你设置的部署时间比EC2上的时间晚，因为在部署的时候需要制定精确的时间
  * 部署失败或者Lifecycle Event被跳过了，这意味着EC2实例出现了问题，健康实例数量不足等，原因包括CodeDeployAgent没有安装 / Service Role或者IAM Instance profile没有充分的权限 / HTTP Proxy的使用需要设置Agent的Proxy uri：parameter / Agent和CodeDeploy之间的Date或time没有匹配
  * AllowTraffic失败的原因，一般原因是ELB的Health Checks设置错误

### CodeArtifact

- *Artifact Management*：存储和获取代码和他们之间的dependencies的中央管理系统，和许多熟知的代码依存管理系统一起工作，比如yarn，npm，pip，PyPI，Maven（Java），Gradle等
- 当各语言的developers获取public artifact repo后，CodeArtifact会存储这些packages，也可以通过手动存储packages进CodeArtifact的方式。
- CodeBuild在进行构建的时候当然也可从Artifact获取（fetch）代码和依存关系
- 集成**EventBridge**：当代码package被created，updated，deleted，会触发事件，比如invoke Lambda，Step Function，SNS，SQS，或者启动一个Pipeline等
- **Domain**：存储的东西放在叫domain的概念中
  * 可以管理*多账户，资产共享，KMS加密*
  * 多个账户有重复资产的时候，只需要被存储一次，然后共享资源
  * 当进行pull package操作的时候，*只有元数据记录metadata record被更新*
  * *Domain-resource based Policy*可以用于限制domain内的账户，资源，使用权限
- *ResourcePolicy*：可以设置基于resource的用户使用权限（制定用户的账户号码，和被访问资源的所属账户号码），来允许跨账户的用户访问，这种访问要么可以读取所有all，要么什么都不可读取none of them
- **Repo树形层级依赖关系**：
  * Repo之间可以有依赖关系，一个repo最多可以有10个upstream的repo作为依赖repo，这样用户可以只用一个endpoint访问所有upstream的代码库（1个repo指向多个upstream的repo）
  * *ExternalConnection*：一个repo最多可以有一个外部连接（外部repo或者public repo比如npm）一个外部连接的repo可以被多个私有repo共享（一个repo被多个downstream的repo指向），外部的package只需cache在连接外部资源的那个repo中即可，中间repo不需要cache

### CodeGuru

- Code Review和Code performance recomendations
- 代码审查部分，组件是，Code Reviewer
  * 审查的是静态代码
  * 依靠机器学习，学习基础是AWS1000多代码库
  * 检测代码问题，安全漏洞，资源泄露，bugs查找，输入内容验证
  * 支持Java和Python
  * Secret Detector：可以检测仓库中是否有硬编码的secret，并提出使用Secrets Manager的相关建议
- 代码性能建议，组件是，Code Profiler
  * 针对的是在runtime中的代码提升建议
  * 识别CPU消耗，最小化app的overhead
  * 可以通过装饰器`codeguru_profiler_agent`（添加到zip文件或者layer）和`@with_lambda_profiler`被集成到Lambda function（也可以直接在lambda function的configuration中enable profiling功能）

### EC2 Image Builder

- 自动创建和维护EC2的VM镜像AMI
- 创建后可以被自动test
- 可以publish到多个regions和多个accounts
- 可以集成到CICD流程
- 可以使用*RAM*（Resource Access Manager）进行资源分享
- **Tracking latest AMIs**构架：
  - *Image Builder* -payload-> *SNS* -payload-> invoke *Lambda* -> store id to *SSM Parameter Store*
  - Use Case：用户可以使用**SSM Parameter Store**中的最新ID，或者CloudFormation可以直接reference该最新ID进行构建

### AWS Amplify

- web&mobile application的快速部署服务，类似于ElasticBeanstalk
- 将各种服务打包：认证，存储，API，CI/CD，Pubsub，分析，AI/ML，Monitoring等
- CI/CD：可以将Github不同分支和不同Amplify的环境connect，当在不同的环境push代码后会直接部署到不同Amplify环境（不同的domain网址）

## Configuration & Management & LaC

### CloudFormation

- IaC的所有好处，Cost上会给每个stack打tag，从而了解每个tag的cost
- *声明性*编程，Terraform也是，所以不需要考虑执行顺序和编排方式
- 无需重新造轮子，可以依赖doc和网上的资源
- **工作方式**：
  - 需要将template上传S3，然后通过template创建stack（resources的合集）
  - 当要更新一个stack的时候，无法修改现有template，而是创建一个新的template使用
  - stacks被name特定
  - 当删除一个stack，它相关的资源都会被删除
  - *手动部署*：使用Application Composer或者Code Editor编辑template，然后在console填写各种需要的parameter，创建stack
  - *自动部署*：编辑yaml文件使用CI/CD方式或者CLI自动部署（推介该方式，terraform也是）
- 创建动态数量的资源使用*Macros&Transform*

- **Building Block**：
  - Template‘s *Components*：
    - AWSTemplateFormatVersion
    - Description
    - Resources：必须有
    - Parameters：动态，当有些参数一开始无法确定，或者之后需要复用的情况 / 使用`!Ref`语法指示 / 可以指定输入类型等 / 有一些Pseudo变量可以直接使用，比如AccountId，Region，StackId等
    - Mappings：静态，硬编码的参数，比如region，AMI，或者环境prod/dev
      * `!FindInMap [RegionMap, !Ref "AWS::Region", HVM64]`
      * 这种方式对匹配不同区域的AMI很有用，因为AMI是地区specific的
    - Outputs：适合stack之间的协作，一个stack输出的参数可以在另一个stack中被使用
      * `Export` - `!ImportValue`
    - Conditions
  - Template‘s *Helper*：
    - References
    - Functions：`!Ref`，`!GetAtt`，`!ImportValue`，`!Base64`，conditions：`!And`，`!Equals`，`!If`，`!Not`，`!Or`

- **Rollbacks**：
  - 当创建和更新失败的时候，删除所有新创建的资源
  - 如果在Rollback的过程中有sb手动修改了资源，这时候Rollback也会失败：需要手动fix资源，然后通过`ContinueUpdateRollbackAPI`重新进行rollback操作
- **ServiceRole**：
  * ServiceRole是服务拥有的IAM Role
  * 当不想给user过多的权力，践行最小权限原则的时候
  * 用户需要`iam:PassRole`Policy
- **CloudFormation Capabilities**：
  * 一些高级扩展能力
  * `CAPABILITY_NAMED_IAM`（特定IAM资源），和`CAPABILITY_IAM`：在CF需要创建和更新一些IAM资源的时候需要的enable功能
  * `CAPABILITY_AUTO_EXPAND`：当需要进行动态转换（dynamic transformation）的时候，如使用macros或者nested stacks的时候
  * 当需要的上述功能不足的时候，会抛出的错误：`InsufficientCapabilitiesException`
- **DeletionPolicy**：
  * default是`DeletionPolicy: Delete`，当stack被删除，资源也会被删除，但是当S3bucket不为空的情况下可能不会删除资源
  * `DeletionPolicy: Retain`，设置在resource中，特定哪些资源在stack被删除后也继续留存
  * `DeletionPolicy: Snapshot`，针对可以snapshot的资源，该设置会留存资源的snapshot
- **StackPolicy**：
  * json表达，policy，allow和deny
  * 当*更新一个stack的时候显式指定哪些资源不允许被更改*（deny）
  * 也需要显式指定哪些资源可以被更改，如allow all deny one，语法组块之间没有顺序要求
- **TerminationProtection**：
  * 保护创建的stack被误删除的功能，在StackActions中进行enable
- **CustomResources**：
  - 用一个例子来说，比如stack删除无法删除不为空的S3桶，通过custom resource（backend lambda functions）可以先运行删除所有S3中资源的function，然后再允许CF删除S3桶
  - *工作方式*：CF自己不会触发Lambda，而是对Lambda发送API请求（包括一个s3 pre-signed url），获得的Lambda的response结果会存储在S3中
- **DynamicReferences**：
  - 动态获取SystemManager的*ParameterStore*（加密和非加密版）或者SecretManager中的用户名和密钥的功能
  - 语法：'{{resolve:service-name:reference-key}}'
  - 服务名service-name包括：ssm，ssm-secure，secretmanager
  - **RDS的动态密码管理的两种方式**：
    * 一种只需要指定：`ManageMasterUserPassword`为true，RDS会自动管理用户密码（在secret manager中）的创建和rotation
      - 获取方式：`!GetAtt MyCluster.MasterUserSecret.SecretArn`
    * 另一种方式就是这里的DynamicReference，通过CF的脚本自动生成，包括三个部分：
      - `AWS::SecretsManager::Secret`：自动生成密码存储至SecretManager
      - `AWS::RDS::DBInstance`：创建RDS，通过resolve语法指向创建的secrets
      - `AWS::SecretsManager::SecretTargetAttachment`：将上述二者关联，自动化rotation
- **EC2 User data**：
  - *简单的*EC2的服务器启动初始化脚本执行：`Fn::Base64 |`
- **cfn-init**：`AWS::CloudFormation::Init`：cfn-init实质是一个命令，在原本的user-data执行的地方，执行该命令，然后在metadata的block中明确记录init的详细内容，详细看一下具体脚本即可：dev-ops/aws/code_v2024-07-29/cloudformation/from-sysops-course/1-cfn-init.yaml
- **cfn-signal & Wait Condition**：在cfn-init后立刻执行的脚本，当达到指定条件，发送完成信号
  - cfn-init和cfn-signal都是在AMI上应当自带的命令，如果没有的话可以自己安装
- **cfn-hup**：
  - helper script，主要用于在 CloudFormation 更新栈后（对元数据metadata的更新），对 EC2 实例进行配置更新
  - 检测元数据变化：它会周期性地（比如1分钟一次）检查 EC2 实例的元数据，一旦发现与 CloudFormation 模板中的配置有差异，就会触发相应的动作
  - 运行用户指定动作：用户可以在 CloudFormation 模板中定义一系列动作，例如重启服务、重新加载配置文件等，cfn-hup 会按照定义的顺序执行这些动作
- **Cross-stack & Nested-stack**:
  * Cross-stack是一个stack的输出被Export，然后被另一个stack用ImportValue语法使用，是一种*值复用*的关系
  * Nested-stack很像是代码中的package，是一种资源复用的方式，是一种*嵌套*关系，通过一个template的URL复用stack
- **DependsOn**：
  * 显式表明两个资源的前后creation关系，必须在DependsOn的对象资源被成功创建后才开始创建
  * 当使用`!Ref`或者`!GetAtt`的时候，内部默认有前后依赖关系

- **StackSets**：堆栈集
  - 跨account跨region地部署资源的方法，比如部署同样的AWS Config设置到不同的账户和地区
  - *Administer账户*创建StackSets，当更改了堆栈集，各个target account和region的*stack instance*都会被updated
  - *Target账户*通过从StackSets进行stack instance的创建，更新，和删除操作
  - 从表述方式来看，管理账户创建的堆栈集更像是一个class类集合，部署到各个账户和region后，创建的是stack instance一种实例
  - IAM Role权限管理方式：
    * Self-managed permission：没有开启OU的情况下，要手动创建和赋予IAM Role，对Administer账户赋予`AWSCloudFormationStackSetAdministrationRole`对target账户赋予`AWSCloudFormationStackSetExcutionRole`
    * Service-managed permission：当有OU功能的情况，需要 enable all feature in AWS Organizations，StackSets会代替你在OU范围内enable trusted access，有助于在新账户添加的时候就被部署stacksets
  - 示例资源：dev-ops/aws/code_v2024-07-29/cloudformation/from-sysops-course/stacksets
- **ChangeSets**：
  - 当需要update一个stack的时候，创建ChangeSet可以了解要发生什么变化，*但它不能告诉你是否这种变化会成功*

- **Drift**：
  - 检查创建和配置的资源是否因*手动操作*等原因，发生了设置偏移，整个stack或者stack中的单个资源
  - *StackSet Drift Detection*：检测stack和他关联的stack instance的变化；但是直接通过CloudFormation对stack instance等的修改不会被认为是drift，但是这不是一个好的实践；该功能也可以关闭

### Service Catalog

- *Admin*设置的一个**Products**，实质是一个*CloudFormation模板*（实质就是CF模板的subset），并对该模版的服务设置了IAM Permission的产品目录
- *User*就可以在该产品列表中lanch产品（tagged）使用，服务会按照CF的设置进行lanch，并让用户在*相应的权限内*使用服务
- governance,compliance,consistency
- 使用CloudFormation StackSets可以设置Product deployment option，设置具体的*限制条件*，比如Account，Region，Permission等
- **Launch Constraints IAM Role**：
  - 用户只拥有对service catalog服务的使用权限，其他需要的权限都在这个launch role上
  - *该role必须有的权限包括*：
    * CloudFormation full access
    * AWS Services in the CloudFormation template
    * Read Access to S3 bucket which contains the template
- 持续部署新的服务，是通过push推送CloudFormation的代码后，invoke一个sync用的Lambda function来将新的变更部署到Service Catalog中

### Elastic Beanstalk

- 以开发为中心的部署application的服务，典型的三层网络构架快速部署的managed服务（ELB+ASG+RDS）
- 包括*各种语言*platform的代码和**Docker**部署
- *Components*：
  * Application：环境，版本，设置的组合
  * Application version：应用的版本迭代
  * Environment
- Beanstalk *Deployment Options* for update：
  * All at Once：快速，一次部署，会有一些downtime
  * Rolling：逐渐部署，可设置批次大小（bucket），低于基本capacity
  * Rolling with additional batches：设置余量的batch，进行逐步部署，保持原有的capacity，对于prod环境的性能维持比较好
  * Immutable：在现有的ASG中添加同样数量的server数量，当新版本正常运行后，关闭旧版本的servers，cost较高，对prod环境友好
  * blue/green（不是内置选项）：重建一个新版本环境，分配一小部分流量比如10%，切换可以（举例）通过Route53切换URL即可
  * Traffic Splitting（Canary）：在负载均衡中分配一小部分流量给新版本的ASG，进行金丝雀测试，方便回滚，当金丝雀测试结束可以直接将流量都引向新版本
- **Web tier和Worker tier分离构架**：是一种普通的*解藕服务*实践方式
  - 如果有运行时间较长的job，比如视频处理，可以通过SQS服务将job推送到worker tier（pull message from SQS）进行处理
  - 通过cron.yaml定义periodic任务（定期运行任务）
- Notification和invoke服务集成：事件集成Eventbridge -> invoke Lambda（发送信息到slack）/ trigger SNS（发送邮件）

### SAM（Serverless Application Model）

- AWS SAM（Serverless Application Model）是一个用于构建和部署无服务器应用的框架。它是一个开源框架，专门设计用于简化*AWS Lambda、API Gateway、DynamoDB、Step Functions*等无服务器资源的定义和管理。
- 使用简化的模板语法（yaml）（基于AWS CloudFormation）来定义无服务器应用。SAM模板是对CloudFormation模板的扩展，提供了特定于无服务器应用的简化语法。
- SAM与AWS CodePipeline、CodeBuild，CodeDeploy等持续集成和持续交付（CI/CD）工具紧密集成，支持自动化构建和部署流程。

### CDK（Cloud Development Kit）

- 使用熟悉的语言JS，Python，Java，.NET等定义基础设施
- 代码中包含的high level components called *constructs*，然后代码会被compile为YAML/JSON格式的CloudFormation代码
- *基础设施代码+runtimeCode*可以同时部署，比如ECS或者Lambda的code
- CDK CLI：
  * `cdk bootstrap`，需要在各个region和account启动cdk应用编译代码
  * `cdk synth`，这会返回被编译的CloudFormation的Yaml代码
  * `cdk deploy`，将编译的代码通过CloudFormation进行部署
  * `cdk destory`，销毁stack
- codes example: dev-ops/aws/code_v2024-07-29/cdk

### Step Functions（base on data engineer note）

- 使用 JSON 格式的*状态机state machine*定义，开发人员可以轻松地编排和管理跨多个 AWS 服务的任务
  * task state：任务单元，比如invoke服务，发送sqs队列，更新db等
  * choice state：条件单元
  * wait：delay状态
  * parallel：增加一些branch并行处理分支
  * map：也是并行的，对dateset里的每一个元素进行map处理
  * pass：传递input到output，或者inject一些固定数据而不做任何处理
  * success，fail状态单元
- 编排Lambda，Batch，ECS，DynamoDB，EMR，SQS，SNS等200多服务
- *可视化workflow*的流程，有点像Airflow
- trigger：Console，SDK，CLI，Lambda，API Gateway，EventBridge，CodePipeline，StepFunctions
- *不能集成Mechanical Turk*服务（在线人工劳动力），这里能集成它的服务是SWF服务
- Standard 和 Express Workflow两种类型：
  - 前者贵后者便宜
  - 前者执行时间可以长达1年，后者最大5分钟
  - 启动rate前者是每秒2000，后者是每秒10万
  - *前者确保执行一次，后者为最少执行一次*
- Express Workflow：包括*同步和异步驱动*，同步会确保得到API返回执行结果，后者只确保启动，同步的更适合编排服务，错误处理，并行处理，重试策略等
- Error Handling：重试或者增加alert功能触发EventBridge-SNS

### AppConfig

- AWS AppConfig 是 AWS Systems Manager 的一项功能，用于管理和部署应用程序配置数据。它帮助开发人员和运维团队在不重启或重新部署应用的情况下**动态地更改应用配置**
- UseCase：
  - 动态特性切换（Feature Toggles）：根据需要在运行时打开或关闭应用的某些功能。
  - 环境特定配置：管理不同部署环境中的配置，如开发、测试和生产环境。
  - A/B 测试：为一部分用户推送新配置以测试新功能，并根据反馈逐步扩展到更多用户。
- config的**sources**包括：Parameter Store，SSM Document，S3 bucket
- 在部署前使用*lambda function或者json schema*进行设置评估**validate**
- *EC2*等会通过**poll**的方式取得*config changes*，然后进行变更，如果中途发生错误，会通过触发**CloudWatch Alarm**进行**rollback**操作

### System Manager

- 免费服务，可以对应EC2也可以对应On-Premises，系统对应Linux和Windows
- 自动patching，增强合规性
- 集成 CloudWatch metrics / dashboard
- 集成 *AWS Config*，根据配置启动 Automation 来对资源进行操作，因为 Config 本身不具有操作资源的能力
- 重要组成：**Resource Group / Document / Automation / Maintenance Windows / Parameter store / Inventory / State Manager / Run Command / Patch Manager / Session Manager**
- Document 自动执行功能是我觉得亮眼的功能。
- 通过自动化管理，降低instance cost，自动化创建 AMI

- 必须在server上安装 **SSM Agent**，AmazonLinux2和一些Ubuntu自带agent，*一般出了问题都是因为没agent或者没对EC2授权相应的Role（Role也可以叫做：IAM instance profile，这是在hands on中看到的）*
- lanch的新的EC2，比如已经安装了SSM Agent的AmazonLinux2的EC2，会直接出现在SSM的Fleet Manager中，作为一个舰队进行管理。

- 使用 **TAGS** 对资源进行分组管理：**Resource Group**，从而进行**自动化**和**cost allocation分配**
- **Document** 你可以定义parameters和actions，用json或者yaml格式的*文件*，实质是一个workflow脚本。（很像Github Actions或者Cloud Formation，都是IaC），Parameters也可以从 Parameter Store中取得
  - 可以在ASG中的EC2被terminating之前发送一个命令，比如跑一个自动的document：需要在ASG中设置一个*Lifecycle Hook*，将要关闭的EC2设置为一个Terminating:Wait的状态，当EventBridege检测到了这个状态，就可以触发这个document的执行

- **Run Command**功能是直接跑一个小的命令或者跑一个脚本（document=script），通过resource groups可以直接跑一个server集群，它和IAM还有*CloudTrail*集成，会被记录，*不需要通过ssh连接EC2*（而是通过SSM Agent，session manager也是通过Agent），可以控制执行速率rate（一次执行多少，或几个server），和错误控制（几个错误就停止之类），跑命令的结果可以在console表示也可以发送到S3或者CWLogs，可以给SNS发消息，也可以被EventBridge Trigger，甚至可以得到一个生成的 CLI 代码自己拿去控制台执行。

- **Automation**：简化的一般性服务器维护和发布作业。这个功能也是使用Document进行操作 EC2 等。可以用 EventBridge 触发，进行系统修补等。
- **Parameters Store**：存参数或者密码，可用KMS加密，集成 IAM Policy，可以用于 CloudFormation 的输入。层级文件夹存储方式。Advance的Parameter设置可以有8K大小，但要付费，可以*设置TTL*（expiration date，到期可以设置通知notification，也可以通知没变化）。
  * aws ssm get-parameter 通过 with-decrption 可以解密密码，会检查你的KMS权限可否解密，挺酷。
  * 可以通过文件夹层级递归取得 aws ssm get-parameter --path /myapp/dev/ --recursive 你创建parameter的时候名字就是一个path格式就可以了，这很特别，比如name：/myapp/dev/db-password
- **Inventory**：收集EC2元数据，可以通过S3+Athena或者QuickSight探索。（元数据包括：软件，OS，设置，更新时间等）
- State Manager：状态管理器，保证 EC2 满足某一设置状态。
- **Patch Manager**：自动打补丁，可以设置有计划的 *Maintenance Windows*，可以通过*tags*，设置 *Patch Baseline* 和 *Patch Group*，进行不同的补丁计划。计划通过Document自动执行，报告可以发送到S3。
  - AWS-RunPatchBaseline applies to both *Windows and Linux*, and AWS-DefaultPatchBaseline is the name of the default Windows patch baseline
- **Session Manager**：不需要ssh的EC2连接，*通过Agent和role*。通过IAM可以限制执行主体和对象EC2（通过tag），甚至可以限制使用的command。访问记录会被CloudTrail记录。可以记录logs（S3，CloudWatch Logs）- `"Action": "ssm:StartSession"`
  - 使用 *VPC Interface Endpoint* 访问在 Private subnets 中的 EC2

- **DHMC: System Manager: Default Host Management Configuration**：提供一个集中化的界面来管理和配置 instances 的默认设置，简化了配置过程，不需要 EC2 instance Profile，EC2 Role 是 *Instance Identity Role*

- **Hybrid Environments**：
  - System Manager 可以管理 On-premise 上的主机，主机名以 *mi-* 开头（云中的是*i-*开头）
- **集成IoT Greengrass**：通过将 SSM Agent 安装在 IoT Greengrass Core 上，管理 devices 集群
  * 需要添加 *Token Excahnge Role* 给 device 用于和 SSM 进行通信许可
  * 支持所有的 SSM 的上述功能
- **OpsCenter**：解决操作上的问题：*Operational Issues（被叫做OpsItems）*
  - OpsItems：issues，events，alerts
  - 集合各种info比如config，Cloudtrail日志，CWAlarms，CFstack信息等
  - 解决方式：run automation document
  - 使用EventBridge或者CW Alarms创建OpsItems
  - 提供 *recommended Runbooks* 来解决issue
  - use case：比如通过 lambda 列出所有较老的 EBS，然后登录到 OpsItems，通过触发 Run Document 来自动删除 EBS snapshot

## Resilient Cloud Solution

- 这部分的关键词就在于resilient，比如复制，冗余，容器，高可用性方面。
- 本地和云的网络连接S2SVPN和DX的冗余构架
- 使用CloudFormation/Beanstalk自动化构架恢复
- *Multi-AZ* 构架，比如三层构架的 failover 高可用性构架
- *Blue-Green* 构架：
  - ALB listener 后的 ASG 群组的蓝绿部署
  - Route53 + 多个 ALB 的 endpoint 的 DNS 流量切换（依存于 client 的 DNS cache 刷新）
  - 多个 API Gateway 的不同 stages 背后坐不同的 Lambda function version 进行部署
  - 一个 API Gateway 后坐不同的 Lambda alias（通过 Lambda 进行切换设置）
- *Multi-Region* 构架：
  - 使用 Route53 的 Health Check 进行 DNS 的自动 failover
  - 在不同的 Region 配置 *ALB + ASG* 的部署，Route53 基于latancy进行 DNS 路由
  - 在不同的 Region 配置 *APIGateway + Lambda* 的部署，Route53 基于latancy进行 DNS 路由
- *Disaster Recovery*：
  - **RPO**：Recovery Point Objective（目标），是你灾难后数据要恢复到的时间节点，这预示着你的data loss有多少
  - **RTO**：Recovery Time Objective，这是你灾难恢复要花费的时间，这预示着你服务downtime是多少
  - 这俩指标你要求的越高，那么就越贵
  - **策略**RTO从慢到快⬇️，从便宜到贵
    - Backup and Restore：有很高的RPO，简单不贵，花费的也就是存储快照的费用
    - Pilot Light：持续地跑一个服务的*轻量版本*（也许只有数据库部分），恢复快
    - Warm Standby：*整个系统*各个组件以*小的scale*在一边同时跑standby
    - Hot Site / Multi Site Approach：*整个系统的复制版本*，都是active的，很贵，RPO和RTO都非常小


### Lambda

- **Version的概念**：
  - *$LATEST*：这是一个可变mutable的变量，存储的是你发布的最新version的代码
  - v1～：version是不可变的immutable，当你发布新版本，这个数字会不断递增，version包括code+configuration
- **Aliases的概念**：是一个版本指针，point版本，但是不能point其他的aliases
  - 可变的mutable，比如定义不同版本为prod，test，dev等，通过aliase指向不同的版本
  - 可以铜通过权重weight设置，进行canary部署
  - Aliases有自己的ARN
- **Environment Variables**：
  - key-value形式的string存储，可以被code引用（`os.getenv("key")`），可以存储secrets，使用KMS或CMK加密

- **Concurrency & Throttling**（并发和限流）:
  - 通过设置*reserved concurrency*限制并发上限
  - *一个账户（中所有的functions）的并发上限是1000*，如果一个function就用完了所有的并发数量，其他functions会被throttle
  - 如果超过了上限的并发执行会引发*throttle*问题：
    * 同步执行synchronous引起限流error429
    * 异步执行asynchronous会自动retry，如果失败则任务会被发送到DLQ
  - 如果需要更高的并发执行上限，可以用*suppor ticket*进行上限申请
- **Cold Starts & Provisioned Concurrency**：
  - invoke一个新的Function实质上是新启动了一个instance，会发生code load，以及init，如果init文件很大，会花费很多时间
  - 使用Provisioned concurrency设置，可以提前进行并发操作，降低延迟

- **FileSystem Mounting**:
  - 同一个VPC中的EFS文件系统可以被mount到Function的local directory
  - 在initialization初始化的时候被mount
  - 依赖*EFS Access Point*
  - IAM + NFS权限管理
  - 其他文件系统的options：
    * 临时存储tmp或者Lambda layer，是最快的，容量较小，就像是EC2的instance store的感觉
    * S3，Object存储，比较快，通过IAM进行权限管理，版本控制上具有原子性
      - 在版本控制中，**原子性**指的是对代码库所进行的一个变更（如提交）要么全部成功，要么完全失败，没有中间状态。换句话说，一个提交（commit）要么完全被记录并应用，要么不会产生任何影响。完整性，一致性，回滚能力。
  - *Cross-Account EFS mounting*需要VPC之间的peering，同时需要EFS file system policy设置对另一个账户的mount，write权限为allow

### API Gateway

- 暴露*HTTP端点*，*Lambda*（REST API），或者*AWS Service*为一个API服务
- 版本控制（stages），认证，流量管理（API keys，Throttles）
- 限制：29秒timeout，最大10MB的payload
- 支持Canary Deployment，创建canary，设置百分比 -> 部署一个新的invoke端点 -> 当测试正常则promote canary即可
- 构架探索：
  - 后面坐一个S3的话，用户上传文件收到10MB的限制，不是一个好的构架，可以*使用Lambda生成一个Pre-Signed URL返回给客户*用于上传大的文件（我预想到了这个构架，很不错，是一种进步）
- *stages*：API Gateway中的变更change，只有通过发布到stage才能影响结果，stage可以根据需要随意取名，新的stage就会有新的URL
- **stage variables**：
  - 可以通过context传递给Lambda functions
  - format语法是：`${stageVariables.variableName}`
  - stage variables可以和*Lambda Aliases*相关联，当指定lambda function的ARN的时候，植入上述语法，API Gateway就可以invoke正确的Lambda function：`${stageVariables.variableName}`指定的stage name和*function的aliase同名*即可

- 部署的Endpoint类型
  - *Edge Optimized*（默认设置）为全球用户：通过*CloudFront Edge locations*，降低延迟，虽然这时候API Gateway还是在一个region中部署的
    - Edge Optimized API Gateway **只接受来自 us-east-1 区域的 ACM 证书**，无论 API Gateway 本身部署在哪个区域。这是 AWS 的一个特定要求。这是因为它是为全局内容分发优化的，它使用 AWS 的全球内容分发网络（CDN）来降低延迟并提高性能。为了简化和集中管理证书，AWS 选择了 us-east-1 作为集中颁发和管理 SSL/TLS 证书的区域。
  - *Regional* 用户，使用regional端点，这种情况的安全证书certificate必须部署在同一个region
  - *Private* 访问只能通过interface VPC endpoint（ENI）在VPC内部进行访问

- import/export *OpenAPI*功能的集成：use API defination as Code
    * *OpenAPI* 规范是一套标准，用于详细描述 RESTful API 的各个方面，包括：
      * 端点（Endpoints）：API 提供的 URL 路径，如 /users、/orders/{id} 等。
      * HTTP 方法：端点支持的操作类型，如 GET、POST、PUT、DELETE 等。
      * 请求参数：需要传递给 API 的参数，包括路径参数、查询参数、请求体等。
      * 响应格式：API 返回的数据结构，包括状态码、响应体、错误信息等。
      * 认证方式：API 所需的身份验证机制，比如 OAuth、API 密钥等。
    - 通过这种规范，API 开发者可以以一致的方式编写和共享 API 定义，从而实现以下功能：
      - 生成 API 文档：工具如 Swagger UI 可以从 OpenAPI 文件中自动生成可视化的 API 文档。
      - 代码生成：根据 OpenAPI 规范，可以生成客户端 SDK 和服务器端代码模板，简化开发工作。
      - 测试工具：可以基于 OpenAPI 文件生成测试用例或模拟服务器，用于测试和调试 API。
      - 因此，OpenAPI 规范为 API 的开发、测试、文档生成等多个环节提供了标准化的支持，极大地提高了开发效率和协作性。

- *Gateway Cache*
  - 0.5GB～237GB
  - 缓存client的response
  - TTL默认300秒（0～3600s）
  - 可以在stage和method层面设置
  - 可以有加密功能的选项

- Error
  - 4xx为client错误
  - 5xx为Server错误，注意APIGateway的回复超时是29秒，会有504错误

- Security和Authentication
  - 支持SSL证书，使用Route53设置CNAME
  - 资源权限限制，比如使用S3的bucketPolicy
  - 资源使用权限，比如对Lambda的IAMrole，用于内部的资源访问
  - CORS：cross-origin resource sharing：基于浏览器，控制哪个domain可以call你的API
  - 三种认证：IAM，Lambda集成第三方认证，Cognito

- 日志和监控
  - CW Logs：
    - 可以在stage层级开启logs
    - 可以发送API Gateway Access Logs
    - 可以直送日志到KinesisDataFirehose
  - CW Metrics：
    - IntegrationLatency，Latency，CacheHitCount，CacheMissCount
  - X-Ray：高级构架 X-Ray API Gateway + AWS Lambda
    - X-Ray 提供了详细的请求分析，包括响应时间、错误率、异常、冷启动等信息
    - 支持与多种 AWS 服务集成，包括 EC2、ECS、Lambda、Elastic Beanstalk、API Gateway 等。它还支持与多种第三方框架和库集成，如 Spring、Express 等

- 将API Gateway作为产品发布给客户的方法：
  - Usage Plan
  - API Keys
  - 429 Too Many Requests（account limit）

- WebSocket API
  - 全双工，双向通信，实时聊天、在线游戏、金融交易
  - 使用Lambda，DynamoDB，或HTTP endpoints
  - URL中使用@connections关键字，用connectionId识别连接

- Private APIs
  - 使用VPC Interface Endpoint连接
  - 每一个VPC Interface Endpoint 都可以用于连接多个Private APIs
  - 对Interface Endpoint可以设置Endpoint Policy
    - aws:SourceVpc/aws:SourceVpce

### ECS

* 容器的优点在于*无需依存于OS*，可以在任何机器上执行可预测行为的任何代码和服务。它是**微服务构架**的基础。
* DockerImage存放于DockerHub（公有），或者AWS的ECR，有公有仓库/私有仓库（自建）
* VM是操作系统级别的隔离，不共享资源，Docker是进程级别的隔离，共享底层设施资源
* build - push/pull - run
- Docker container on AWS = launch ECS tasks on ECS Clusters
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
- **ECS 集成 ALB/NLB**：
  - ECS作为服务expose的时候使用LB
  - 使用Dynamic Port Mapping，分配流量：可以在一个EC2实例上发布多个container
  - ALB是标配，NLB的使用场景：需要高throughput，高性能，或者需要PrivateLink配对创建端点

- **数据存储**：使用EFS文件系统，ECS container可以进行 EFS 数据和文件节点的Mount，结合Fargate就是完全的serverless
  - 适合Multi-AZ的持久性存储分享
  - 注意，*S3不可以mount*为它的文件系统

- ECS安全：集成SSM ParameterStore和Secret Manager
- ECS Task Networking：None，host（使用底层host的interface），bridge（虚拟网络），awsvpc（使用自己的ENI和私有IP）

- **Launch方式1:使用EC2**:
- 也就是说Cluster的集群是通过EC2组成的，只要在EC2中安装*ECS agent*，就可以将他们集成到ECS的Cluster中去
- *ECS Agent*使用*EC2 Instance Profile*定义IAM Role，用于访问ECS，ECR，CloudWatch，SecretManager，SSM Parameter等
- *EC2 Instance Profile*：agent使用，use case：对ECS service进行API call / 发送logs到CW logs / 从ECR拉取镜像 / 从SSM或SecretManager获取敏感数据
- *ECS task Role*：不同的服务需要不同的*ECS Task Role*，task role是在*task definition*中定义的
- **Launch方式2:使用Fargate**：
- 无需管理服务器，只需要定义*task definitions*

- **ECS Auto Scaling**：
  * 使用的是AWS Application Auto Scaling，指标：
    - ECS Service Average CPU Utilization（利用率）
    - ECS Service Average Memory Utilization - scale on RAM
    - ALB Request Count Per Target - metric coming from the ALB
  * *注意*ECS的Auto Scaling不等同于EC2的Auto Scaling，它是ECS tasks层级的，而tasks坐落在EC2中，一个EC2中包含多个tasks
    - Auto Scaling *Group*的Scaling才是基于EC2的Scaling
    - **ECS Cluster Capacity Provider**是和ASG相伴的组件，它基于CPU，RAM利用率自动伸缩EC2 instance的数量
  - **ECS Cluster Capacity Provider（ASG）的自动伸缩的 scope 中嵌套 ECS tasks的自动伸缩！**
  * *Scaling方式三种*：
    - Target Tracking：based on CloudWatch metric
    - Step Scaling：based on CloudWatch Alarm
    - Scheduled Scaling：based on a specified date/time（可预测的变更的场景）
  * Fargate的Auto Scaling的设置很简单，因为它是Serverless的

- 和其他服务的集成：
  - 通过*CloudWatch*的metric进行监控，通过Alarm进行Auto Scaling操作
  - 通过触发*EventBridge*，来进一步触发新的ECS tasks的launch和run
  - 通过EventBridge的*Schedule*触发来间歇性run ECS tasks
  - 通过从*SQS*pull message来触发task执行
  - 通过*EventBridge*发送突然出问题的task的info（发送*EventPattern*）来发送*SNS*给管理员，进一步进行监管和调查

- Logs
  - 通过开启*awslogs log driver*功能发送日志到CWlogs
  - 在task definition中设置 `logConfiguration` parameters
  - Fargate launch类型只需要给Task Execution Role相应的日志写入权限，它支持awslogs，splunk，awsfirelens等log drivers
  - EC2 launch类型需要CW Unified Agent & ECS Container Agent来发送日志，instance需要有相应的权限（profile），需要enable logging使用`/etc/ecs/ecs.config`的`ECS_AVAILABLE_LOGGING_DRIVERS`
  - **Sidecar Container**：是一种专门用于收集其他container的logs的container模块，然后将logs发送到CW logs，感觉就是一种log的集成管理方式

### EKS

- 开源框架
- ELB - AutoScaling - nodes - Pods
- 三种**Nodes托管模式**：*Managed Node Group*和*Self-Managed Nodes*和*with Fargate*（use EFS）：完全托管
- 支持On-demand/Spot Instance
- CloudWatch监控：node上的log存储目录是`/var/log/containers`，使用*CloudWatch Agent*发送日志，*CloudWatch Container Insights*进行仪表盘监控
- **Control Plane Logging**：是一种用于*audit & diagnostic*的控制日志，发送到CW logs
- DataVolumes：使用*CSI（Container Storage Interface）driver*，连接到EBS，EFS，FSx
- Kubernetes中的*Taints和Tolerations*是一种机制，用于控制哪些Pods可以调度到哪些Nodes上。这种机制通过向节点（Node）施加“污点”（Taint）来排斥Pods，然后Pods通过“容忍”（Toleration）这种污点来选择性地允许调度。

### Amazon Fargate work with ECS/EKS

- Fargate是一个Serverless的服务，无需构架基础设置，直接添加container

### ECR
- 支持image*漏洞扫描（on push）*，*版本控制*和*lifecycle*（通过policy rules进行images过期管理，节省存储cost）
  - 基础漏洞扫描由ECR进行，并触发EventBridge发布通知
  - 高级Enhanced扫描通过Inspector进行，由Inspector触发EventBridge发布通知
- 通过*CodeBuild*可以*register和push*Images到ECR
- 可以cross-region 或者 cross-account 地 replicate image
- 私有和公有，公有的叫ECR Public Gallery，后端是S3
- 通过IAM（role）管理访问权限

### ECS/EKS Anywhere

- ECS部署在包括你的**数据中心**的任何地方
- 安装*ECS Container Agent和SSM Agent*
- 使用EXTERNAL的launch type
- 必须要有稳定的AWS Region连接

- EKS则需要安装在数据中心，使用EKS Anywhere Installer
- 使用*EKS Connector*从AWS连接到本地的EKS Cluster
- 以上是有连接的设置
- 无连接的设置需要安装EKS Distro，以及依赖开源工具管理Clusters

### Kinesis

参照Data Engineer部分的Kinesis*全部内容！*和devops的内容基本一致

### Route53

也可参考高级网络笔记，这里只记录重点记忆点！

- 高可用性，域名注册服务，DNSResolver服务
- Hosted Zone分为共有和私有分别解决VPC内外的名称解决问题，它是一个record合集
- Records：CNAME不能用于example.com根域名
- RoutingPolicy：weighted，Latency，Failover（使用health check）

### RDS Read Replicas & Multi-AZ

- **ReadReplica**：
  - 目的在于数据*读取（select）*的可扩展性，减轻主DB的load负担
  - 可以withinAZ，可以跨AZ，也可以跨Region
  - 最多可以有15个replica
  - 注意它是*Async*的，数据是异步不是完全同步的，遵循**最终一致性原则**
  - replica可以升级promote为主DB
  - 一般来说数据在AZ之间传输是产生cost的，但是对于个别managed服务，比如RDS的replica数据同步，只要在**同一个region则不会产生cost，但是跨region则产生cost**
- **Multi-AZ**：
  - 该设置目的是灾难恢复disaster recovery，当然也可以将该被用db用于replica
  - 它不具备扩展目的和功能
  - 数据是*sync*完全同步的，写入主数据库会立刻被同步
  - 具有*相同的DNS*名，自动进行app的failover
  - **将RDS从Single-AZ变更为Multi-AZ**：不会发生downtime，只需要对DB进行*modify*操作即可，背后原理是对主DB创建*Snapshot*，从snapshot创建stand-by DB，然后在两个数据库之间进行同步数据和以备不时之需的failover

### Aurora

- 速度是MySQL的5倍，是PostgreSQL的3倍
- database volume最多可以有128TB
- 最多可以有*15个read replica*
- *Cross-region and zones* read replica复制的是整个database
- 直接从S3，load/offload 数据，*持续备份*到S3
- 30秒内实现master的failover
- 储存中加密使用KMS，传输中加密为SSL

- DB Cluster：
  - 一个Writer Endpoint - MasterDB
  - 多个Reader Endpoint - Read replica - 连接Load balancer，有Auto Scaling功能
  - Optional：分出两个read replica 作为Custom Endpoint比如进行客户自己的分析查询

- **Global Aurora**：
  - Cross-Region read replica：高灾难恢复能力
  - Global Database：最多可以有5个备用region，每个region最多可以有16个read replica
  - Write Forwarding：通过对副DB的写入操作可以，forward到Primary的DB，主DB永远都会先更新，然后再复制到副DB

### ElastiCache

- 托管服务：Redis，Memcached（in-memory db）
- **Redis**：
  - Multi-AZ，高可用性，自动Failover，读取复制，备份和恢复功能，本身很像一个DB
  - *支持 sets 和 sorted sets*
- **Memcached**：
  - Multi-node sharding 数据分区
  - *数据不会persistent*
  - *没有备份和数据恢复功能*
  - 是一个 Multi-threaded 构架
- 低延迟、高吞吐量的数据访问，适合需要快速响应时间的应用程序，如实时分析、游戏、金融交易等
- 构架：
  - *作为RDS的数据库缓存*
  - *User Session Store*，存储客户的登录连接状态

- **Replication：Cluster Mode Disabled**：意味着只有单个shard
  - Primary - Replica node 是非同期数据同步，可以有 0～5 个 replica nodes
  - *Scaling*：支持水平和垂直扩展，水平最多5个node，垂直是提升 node，方法是内部复制一组新的更大的nodes，然后通过DNS切换来升级
- **Replication：Cluster Mode Enabled**：意味着数据在多个shards上进行分区扩展
  - 所谓的集群 Cluster 就是多个 Shards 的集群，在这些Shards上进行分区数据存储
  - 和上面的模式一样，每一个 Shard 都只能有一个Primary + 0 ～ 5个Replica Node
  - 整个Cluster（所有的Shards）的*node上限是500*，你可以有各种组合比如500个单点shards，或者1master1replica，一共250shards也可以，当然还可以500 / 6（代表1个master和5个replica）个shards

- **Redis Auto Scaling**：
  * 自动 scale 需要的*shards*和*replicas*
  * scale policy 支持 **Target Tracking** 或 **Scheduled** 方法
  * Auto Scaling 只针对 Redis 和它的 Cluster Mode Enabled 模式
  * 集成 *CloudWatch Metric* 比如 ElastiCachePrimaryEngineCPUUtilization 等指标进行 Target Tracking，然后通过 alarm 触发scale 动作

- **Redis Connection Endpoint**：
  * 如果是*standalone node*就只有一个endpoint
  * *Cluster Mode Disabled*：
    - Primary endpoind for write
    - Reader endpoint for read（evenly split on all read replicas）
    - Node endpoint for read operations
  * *Cluster Mode Enabled*：
    - Configuration endpoint for all read/write operations
    - Node endpoint for read operations

### DynamoDB

Data Engineer 中记录的更详细！这里关注devops的重点！

- 扩展到很大规模的 Workloads，比如IoT数据，分布式数据库，response性能高速度快，高扩展能力
- Table中感觉都是object，首先要有*Primary Key*，然后每一个row都是一个*item*，有自己的*attributes*（看起来像是column），可以为null，这完全就像是object，每一个item的*size上限是400kb*，*数据类型包括 ScalarType（String/Number/Boolean/Binary），DocumentType（List，Map），SetTypes（StringSet，NumberSet，BinarySet）*等
- 预置 Provisiond Mode & 按需 On-Demand Mode：预置模式也可以设置 Auto Scaling 的上下限 capacity 容量

- **DynamoDB Accelerator（DAX）**
  - 是一种和DynamoDB无缝连接的*cache*功能，是一种*cluster*，它会有一个endpoint，API咯
  - 低延迟，解决HotKey问题，也就是过多reads的问题
  - 默认5分钟的TTL
  - Multi-AZ，推介生产环境最少3个node
  - 安全性高，集成KMS，VPC，IAM，CloudTrail等服务
  - *ElasticCache*可以存储*聚合数据结果*，DAX一般是存储*单个的objects，query/scan*等，可以将两者*结合*，将DDB的结果存储到ElasticCache中，重复使用

- **DynamoDB Streams**
  - 似乎在学习Kinesis的时候看到过，他和KinesisDataStream很像，是通过**Shards**分区数据流的，它自动扩展
  - table中的*有序的，基于item变化（create/update/delete）*的数据流，数据是可以设定的，比如只发送key，或者新的，老的item数据等
  - 可以送到：KinesisDataStreams / Lambda / KinesisClientLibraryApplications
  - 数据可以retention（存留）24小时
  - *无法取得回溯数据*，也就是过去数据，这种性质很像*GA*，也是只能得到新的数据，过去的数据不可重复获得
  - UseCases：实时数据反映，实时分析，OpenSearch服务，*跨区复制*
  - *Lambda*同步数据处理：需要赋予Lambda相应的权限，以及设置*Event Source Mapping*来读取数据流，*Lambda主动*从DDBstream拉取poll数据的方式

- **TTL**：必须使用number类型*Unix Epoch timestamp*作为TTL的时间属性，它实质上是要**手动**创建了一个新的属性，然后设定开启该功能，内部会定期扫描你设置的TTL属性列（column），取得要删除的数据，进行删除操作，它不会消耗WCU

- 用于 Disaster Recovery 的**Backup**：
  * 一种是*持续备份*，使用 Point-in-time Recovery（**PITR**），可以 optional 开启过去35天的持续备份功能
  * 一种是*On-Demand备份* 就是备份全部内容，可以使用AWS Backup服务，会一致被备份，直到显示删除

- 和*S3*的集成：
  * 输出到S3然后可以用Athena查询，必须开启PITR功能
  * 输出到S3的数据经过处理后可以再回到DynamoDB
  * DynamoDB -> S3 file format: Json/ION
  * 从S3可以import文件到DynamoDB，文件格式为：CSV/Json/ION

### DMS

- 数据库迁移服务：Database Migration Service
- 支持同种类数据库迁移，和不同种类，比如Mysql到Aurora的迁移
- 可以进行持续的CDC（Change Data Capture）迁移，随着数据的不断变化进行数据迁移
- 工作原理部分需要一个EC2来跑DMS从而完成迁移工作
- Sourse数据库包括了On-Premise和云中的各种数据库，Targets则不仅包括On-Premise和云中的各种数据库，还包括AWS的各种服务，比如OpenSearch，Kinesis Data Stream，Amazon Kafka等
- **SCT（Schema Conversion Tool）**：当数据库类型不一致的时候使用的schema转换工具，支持OLTP和OLAP的各种数据库之间的转换，同种类的数据库之间则不需要（注意RDS不是数据库引擎，而是一个数据库平台）
- **Multi-AZ Deployment**：是指DMS服务器（比如EC2）部署在不同的AZ，同步更新，是一个stand by replica，提供高数据冗余，低延迟的迁移服务
- *Replication Monitoring*：Task Status（针对task状态监控），Table State（针对迁移的数据的状态的监控）
  - *CloudWatch* Metrics：包括针对 *tasks* 的 metrics，针对 *tables* 的 metrics 和针对 *host*（用于迁移的服务器）的 metrics

### S3 Replication

- CRR跨区域复制，和SRR同区域复制
- 因为是异步复制，所以需要开启version功能
- 支持跨账号的复制，不管是不是跨账号，都必须设置好S3相关的IAM Permission
- *delete marker replication*可以标记被删除的文件，但是如果删除特定的版本，则不会被复制（不会在目标删除特定的版本）

### Storage Gateway

- 是On-premise数据和Cloud数据的桥梁
- *数据类型*：Block 数据（EBS，EC2 instance store），File数据（EFS），Object数据（S3，Glacier）
- *目的*：disaster recovery，backup & restore等
- 三种类型：Volume Gateway，File Gateway，Tapes Gateway
- **RefreshCacheAPI**：当你上传了文件到S3后并不一定能从本地访问File Gateway看到该文件，这时候用到的就是这个API，可以手动invoke，也可以通过触发Lambda事件invoke，文件多的话就会比较贵罢了，或者定期触发Lambda进行invoke
  * 除此之外，File Gateway还有一个功能叫**Automated Cache Refresh**，这个就不需要手动 invoke 上面的 API 了

### Auto Scaling

- **Scaling Policies**
  - Dynamic Scaling：根据CloudWatch的指标分为*Target Tracking* scaling 和 *Simple/Step* scaling，前者是固定维持ASG在一个指标，后者是根据指标进行instances数量的缩放
  - Scheduled Scaling：定期缩放，如果能预知使用量的情况
  - Predictive Scaling：预测性scaling，通过分析历史数据（肯定是机器学习）预测未来的情况来进行提前缩放
- 适合进行Scaling的 *CloudWatch Metric*：CPU 使用率，平均 Network In/Out，RequestCountPerTarget
- **Scaling Cooldowns**：在发生了一次Scaling之后，一般会有一个cooldown时刻，300秒，以进行metirc的更新
  - 可以预先准备好一个 read-to-use AMI，以减少 cooldown 时间
- *Lifecycle Hooks*：在启动或者关闭一个实例之前可以进行特定的动作的功能，比如cleanUp，抽出日志，health checks等，还可以集成其他疏结合服务，比如EventBridge（触发 System Manager Run Command 收集日志），SNS，SQS等
- *SNS/EventBridge通知*集成：SNS通知包括EC2的 launch，terminate，launch error，terminate error，但是 EventBridge 则可以过滤事件，以及发送更多种类的事件通知
- **Termination Policies**：
  - Default：选择拥有最多 instances 的AZ / 关闭使用 oldest 的 Template 的 instance / 如果都是使用相同 template 的 instance，则关闭最接近下次 billing 的 instance（cost考虑）
  - 还可以设置一个或者多个自选 policy，比如最新launch的instance，最老configuration等，可以使用多个policies组合，并且指定 policies 之间的 evaluation 评价顺序
  - 可以用 lambda function 进行custom policy的设置，也就是通过写代码，编辑自己规定的关闭 instance 的规则
- **Warm Pools**：解决 scale out 时候的延迟问题，说白了就是启动 instance 的时间太长
  - 一个预初始化好的instances的pool
  - 它里面的instance的数量，可以设置min，也就是一直维持的最低数量，而max是ASG的最大capacity上限
  - 在Pool中的instance可以有三种状态：Running，Stopped，Hibernated（冬眠）
  - 该Pool中的instance不会影响metric指标收集
  - *instance reuse policy*：默认情况下，scale in 的时候关闭一个 instance，然后 warm pool 会重新创建一个新的instance，但是通过设置，也允许将 scale in 的 instance 重新归还到 warm pool 中
  - 结合*Lifecycle Hooks*在 instance 停止和in-service前，执行一些特定的动作
- AWS中的很多服务比如Aurora，ECS等等，都有**Application Auto Scaling**的功能，针对很多资源，进行内部底层服务器的缩放功能

### ELB

- **ALB** 的*Listen 规则*：
  * 默认按顺序执行收到的 response
  * 支持的 Actions：forward，redirect，fixed-response（比如固定的404回答）
  * Rule Conditions：host-header/http-header/http-request-method/path-pattern/source-ip/query-string
- ALB 的**流量权重控制和蓝绿发布**：
  * ALB 后的 ASG 们可以根据 rule 拥有不同的流量权重，比如当需要进行app发布的时候，可以控制在新旧app群组上的流量分配

- **DualStack Networking**：
  - 允许client与ELB用IPv4或者IPv6进行通信
  - 支持 ALB 和 NLB

- *NLB*可以集成**PrivateLink**服务（VPC Endpoint interface）来 expose 另一个 VPC 中的其他服务

### NAT Gateway

- AWS管理，高可用性，高带宽，以小时付费
- 存在于 AZ scope 中，有绑定一个 Elastic IP，所以要*提高它的可用性，必须设置多个 AZ，多个 NAT Gateway 构架*
- 必须和IGW一起工作不然流量出不去
- 5Gbps 的带宽，最大可以扩张到 100Gbps
- 它不需要 Security Group，是AWS的托管服务
- 注意它和 NAT instance 的区别

## Monitoring & Logging

- Log的种类：应用日志，系统操作日志，访问日志，AWSManagedLogs
- AWS Managed Logs：ELB Access Logs / CloudTrail Logs / VPC Flow Logs / Route53 Access Logs / S3 Access Logs / CloudFront Access Logs

### Cloud Watch

- CW **Metrics** 的单位是 namespace，拥有 Dimenssion（attribute），metric 都有自己的 timestamp，可以穿件 custom metric，比如尚且无法拿到的 RAM 数据
- **Metric Filter**：简单的日志过滤功能，但是好用，下游集成SNS通知功能等，可以使用正则表达式
- **Metric Streams**：可以通过近实时 near-real-time 的方式（可以 *filter metrics*）将 metrics 流发送到 Kinesis Data Firehose，然后发送到 S3/Redshift/OpenSearch 中供其他分析等用途
- **Custom Metrics**：PutMetricData API创建
- **Anomaly Detection**异常检测功能：使用ML算法检测 metric 中的异常值，后续可以发送 Alarm等
- **Lookout for Metrics**：是一个比CW Anomaly detection更高级的功能，可以检测不只是 AWS 自己的服务，还有很多*第三方服务*的 metrics 异常（使用AppFlow服务，这是一个自动化应用程序数据传输的服务），并且可以**自助识别发生异常的 root cause**，后续可以发送alart到 SNS/Lambda/Slack/Webhooks 等
- **CloudWatch logs**：*log groups*层级之下的是一个一个的*log streams*，日志存储，并且可以发送logs到其他服务，S3/OpenSearch/Kinesis/Lambda 等，安全KMS加密
  * 日志源为各种AWS服务，EC2中安装的是**CloudWatch Logs Agent**
  * 更新的 agent 版本是 **CloudWatch Unified Agent**，它可以发送RAM，CPU更详细数据等到Logs，还可以集成SSM服务的各种功能，是一个更新的版本
  * *CloudWatch Logs Insights*：Query 和可视化工具，注意它*不是*一个实时的功能
  * **S3Export**：顾名思义就是输出到S3，有一点要注意，日志需要12小时后才能被export
  * **CloudWatch Logs Subscription**：实时将 logs 数据输出到 Kinesis 服务或者 Lambda 用于分析和处理，有*filter*功能，支持*Multi-account和Multi-region*！需要设置*IAM Role和Destination Resource Access Policy*
- **Live Tail**：debug的时候好用，针对某Log stream进行实时的日志抽取，可以立刻返回最新的日志，实质应该是一个*流数据抽取*服务
- **CloudWatch Alarm**：可以事件驱动和定期驱动，下游可以invoke的服务：EC2，Auto Scaling Group，SNS
- **Composite Alarm**：一般的alarm只能有单一metric指标，composite alarm可以监控其他alarm，使用AND/OR等逻辑表达式，达到复合指标监控功能，减少 alarm noise

- **CloudWatch Synthetics Canary** 是一种用于监控和验证应用程序和服务的工具，它通过*自动化的测试脚本模拟用户交互，比如API，或者URL链接检查*，以便检测潜在的问题，集成Alarm，使用Python/Node.js编写，可以跑一次也可以定期跑，有很多 Blueprints 可以用

### Athena

- 无服务器 Query 引擎
- 将数据load到S3中使用 SQL 进行文件查询，因为要进行数据扫描，并且据此付费，所以使用 Parquet 格式是最省钱的
- 经常下游和 QuickSight 集成使用
- 适合用于分析，S3中的各种数据和日志
- **性能提升方法**：*Columnar数据（Parquet格式按列scan），Compression data（压缩数据为压缩文件，取得数据更快），Partition Dataset（日期分区文件夹），Use Larger Files（防止overhead）*
- **Federated Query**：使用Data Source Connection可以连接到其他的数据进行查询，查询结果还可以存储到S3中，很方便

## Incident & Event Response

### EventBridge

- 事件和时间驱动
- 有**Event Filter**功能，Event会以*Json*的格式发送给下游服务，**Content-filter**功能，可以在*Event Pattern*中提供语法过滤，进行更细致的 Event filter，比如前缀后缀，数字，IP匹配等
- **Input Transformation**：在输出到下游的target的时候，比如CW Logs的时候可以*定义输出格式的转换*，使用Event的json内容中的key-value值，定义变量，输出想要的结果格式转换，然后导入到你的Cloud Watch logs中，比如`$.detail.instance-id`
  * 这是一种很棒的对自己的事件内容，进行自定义log输出格式的功能！
- Partner Event Bus：支持很多第三方服务的bus
- Custom Event Bus：可以为自己的应用创建自定义的bus
- 支持Cross-Account，需要设置*Resource-based Policy*，方便进行Event Bus的中央管理
- *Schema Registry*：分析bus中事件的schema，可以帮助生成你自己app的事件数据，从而提前知道event bus中数据是什么结构

### S3相关

- **S3 Event Notification**
- 针对S3事件进行的通知功能
- 可以过滤Object进行简单的filter
- 下游服务：SQS，SNS，Lambda
- IAM Permission：注意，这里*不定义S3的IAM Role*，而是*定义下游SQS/SNS/Lambda的Resource Policy来允许S3的access*
- *集成 EventBrideg*，将所有的events都发送到EventBridge，就可以发挥更强的功能，下游可以驱动的服务也更多，并且可以同时驱动很多服务

- **S3 Object Integrity**：使用MD5或者Etag算法进行Object的上传验证，如果无法通过则不会被上传，还有很多其他加密算法也可以使用

### Health Dashboard

- 在console的上方的小心脏标志，分为*Service history，Account health 和 Organization health*
  - All region services
- 可以发送Alart和通知，以及问题修复指导，集成**EventBridge**下游可以invoke更多丰富功能
- 可以集成整个Organization的数据

### EC2 instance status check

- 是EC2面板中的一个功能
- **System status check**：
  - 使用Personal Health Dashboard可以检查到instance的硬件和软件问题，比如AWS自己运维的硬件问题，会自动将你的instance进行硬件设备转移（*host migration*）等，都可以确认到 -> 所以有时instance会突然publici ip变动了
  - 它的解决方案就是stop&start服务器
- **Instance status check**：
  - 由于instance的软件和网络问题引起的问题，解决方案就是重启或者更改configuration

- 集成**CW Metrics的自动化修复**
  - 相关metric，以*StatusCheckFail*开头，包括system/instance
  - CloudWatch Alarm适用于针对一个instance进行的通知和修复
  - AutoScaling Group适用于instance集群的自动修复

### CloudTrail

参考安全专家内容

### DLQ of SQS

- SQS**DLQ**：当消息在visibility timeout的时间内多次都没被处理就会被送进dead letter queue，它适合之后的debug
  - Redrive to source：当你的代码问题修复了，可以将这些message重新放回原本的queue进行处理
  - 因为你可能会重复处理message，最好确保你的处理具有幂等性（Idempotency）比如upsert处理
- SNS**DLQ**：SNS的DLQ是和*Subscription-level*绑定的，也就是说一个订阅有一个DLQ，Redirve Policy是通过Json格式，指定对应的SNS资源ARN

### X-RAY

- applications的*分布式*可视化*tracing分析*
- 集成服务：
  * EC2：安装X-ray agent
  * ECS：安装X-ray agent or Docker Container
  * Lambda
  * Beanstalk：agent 是被自动安装的，有*X-ray守护进程daemon*，可以用*config文件enabled该功能*，你的代码中也可以通过x-ray sdk来集成功能
  * API Gateway：对于debug它的error很有帮助
- X-ray agent或者service需要相关的 IAM permission（IAM ROLE） 来访问 X-ray 服务

- **AWS Distro for OpenTelemetry** 是 AWS 提供的一个开源工具，它是对 OpenTelemetry 项目的一个增强版。OpenTelemetry 是一个用于分布式系统的观测工具，用于收集、处理和转发应用程序的跟踪、度量和日志数据
  - 如果想要开源API追踪工具，加上下游发送到多个目的地的话（可以发送到X-ray），可以从X-ray迁移到该服务

## Security & Compliance

全部服务参照安全专家，包括如下重点服务：这里只记录自我简单总结

### AWS Config

- 实质上它背后的违规比较，应该是基于IaC的底层代码，它驱动的修补程序，也就是System Manager则是一种疏结合，System Manager中跑的Document，实质也是一种IaC基础设施重新部署

- 它的*Configuration Recoder*功能，实质用的是CloudFormation StackSet功能记录了一个时间点的资源配置，该配置可以被用于各个account的资源设置

- *Aggregator*用于整合各个账户的config数据，到一个账户，如果是OU设置下的不需要各个账户设置，如果不是OU则需要被整合的账户授权。只用于整合数据，不用于统一设置，如果要给各个账户统一设置，使用CloudFormation StackSets。

- *Conformance Pack*也是一种Yaml文件，是一组Config Rule和Remeddiation actions的合集，实质也是IaC，*CloudFormation of the Config Rules！*因此它可以用Codepipeline和CloudFormation进行自动化部署

### AWS Organization

- SCP很重要，定义的是Users/Roles/RootUser的权限，不定义Management Account的权限，*就算你用SCP限制Management Account的权限，也不会起作用*，Root账户会被影响，root账户不是用来管理整个环境的，他只是一个资产所有者的单位
- 可设置黑白名单list，可以在OU或者Account level上进行设置，OU是有层级的，SCP作用于层级，外层的限制会作用于内层

### Control Tower

- 是一个帮你设置OU的工具服务，一次设置就可以设置起整个OU需要的各种组件
- 基本所有的组件都在*Landing Zone*中
- **Account Factory**：制造各种账号给你的组内成员使用
  - 集成**Identity Center**创建SSO服务 -> 集成On-Premise AD目录
- 使用AWS Service Catalog来预置新的账户
- **Guardrail**：也是一个合规服务，detect & remediate
  - 可阻止性功能的使用SCP进行限制
  - 检测性的只是用Config服务进行检测
  - Guardrails的三个级别：
  - *Mandatory* 强制性Guardrails：是应用于 AWS Control Tower 下所有账户的不可移除的护栏。它们执行关键的安全和合规要求。
  - *Strongly Recommended* 强烈推介的Guardrails： 默认未启用，但是强烈建议，是基于best practice的
  - *Elective* 可选的Guardrails：允许组织根据其独特的需求定制治理

- **Customization for Control Tower**：自动部署服务的框架
  - AWS自创的GitOps自定义框架，帮助使用*CloudFormation和SCP*自定义LandingZone
  - 它和Account Factory Customization的blueprint（只能发布一个模板）不同，它可以不断集成通过CodePipeline持续发布不同的template和SCP策略

- **Account Factory for Terraform**：
  - 有一个官方维护的tf模板，帮助自动部署账户，文件名为 account-request.tf 在[aws-ia](https://github.com/aws-ia)的Github账户中有source，这官方账户全是自动化，神了

### IAM Identity Center（SSO）

- 前身就是AWS Sigle Sign-On
- 登录界面后面是认证的AD池
- 一键登录所有的AWS account，组织，还有其他第三方应用。TB案件中的AWS登录就是这样的。
- Identity Center中的权限控制使用 *Permission Sets*
- 权限组来自MicroAD或者该服务的built-in Identity Store，通过 group + permission sets 进行管理
- ABAC细度管理，根据用户的属性attributes，实质是一种**tag**

**External IdP（外部身份提供者）**指的是一个管理用户身份和认证的第三方服务。例如，Google、Microsoft Azure Active Directory（AD）和Okta等都是常见的身份提供者，它们可以帮助组织管理用户身份和访问权限，而不需要组织自己维护所有的用户账户信息。

在AWS（Amazon Web Services）中，AWS Identity Center（之前称为AWS Single Sign-On，SSO）支持与外部身份提供者集成，以实现统一的身份管理和单点登录。以下是AWS Identity Center如何为外部身份提供者认证的基本流程：

1. **配置身份提供者**：首先，你需要在AWS Identity Center中配置外部身份提供者的连接。这通常包括输入外部身份提供者的相关信息，如其元数据文件、SAML端点URL等。

2. **设置SAML（安全断言标记语言）或OIDC（开放ID连接）**：AWS Identity Center支持通过SAML 2.0或OIDC协议与外部身份提供者进行集成。这些协议使得AWS Identity Center能够与外部身份提供者交换认证信息。

3. **用户登录**：当用户尝试访问AWS服务时，他们会被重定向到外部身份提供者进行认证。这是通过SAML或OIDC协议的单点登录（SSO）实现的。

4. **身份验证**：用户在外部身份提供者那里输入凭据并完成认证后，外部身份提供者会生成一个认证令牌或断言，并将其发送回AWS Identity Center。

5. **获取访问权限**：AWS Identity Center解析认证断言或令牌，验证用户的身份，并根据预配置的访问权限授予用户对AWS资源的访问权限。

6. **单点登录**：一旦认证通过，用户可以无缝访问AWS资源，无需再次登录。

这种集成的主要好处是简化了用户管理和访问控制，同时确保了统一的身份认证和安全策略。如果组织已经使用了外部身份提供者进行身份管理，通过AWS Identity Center集成可以大大提高效率，并确保一致的用户体验。

### WAF

- Layer7
- ALB/API Gateway/CloudFront

### AWS Firewall Manager

- 管理组织中所有的firewall规则
- 新的资源被创建后会立刻适用于这些rules，有利于组织合规性管理
- 有利于WAF*跨账户*的配置，单一账户只需要用WAF就行了
- 可以在各个账户部署Shield服务

### GuardDuty

- 可以利用机器学习和行为分析技术来*检测恶意活动和未经授权的行为*。
- GuardDuty 分析对象来自 AWS CloudTrail（针对人）、Amazon VPC 流量日志和 DNS 日志的数据（针对网络），S3 data event（针对数据），EKS logs等。
- 后面可以加上EventBridge rules进行通知，trigger target: Lambda,SNS。
- 防加密货币攻击。有专用的finding方法。
- 可以在组织 Organization 层级设置一个*专用于 GuardDuty 的账户（delegated）用于管理多账户*。
- 一种 finding *构架*：GuardDuty -> finding -> EventBridge -> SQS/SNS/Lambda -> HTTP(slack)/Email
  - 在lambda（自动化处理）后可以进行WAF的ACL调整，block恶意ip，或者对subnet的NACL进行调整，block恶意ip。
  - 也可以将后续动作都包在step functions中。
- 可设置*白名单，威胁名单，还有抑制规则*（比如你不想看到的ip，已知没威胁的）。
- 注意！GuardDuty对于DNSlogs的解析，只针对default VPC DNS resolver。其他的不出finding结果。

- **Cloudformation 集成issue**：使用CF进行GuardDuty的enabled的时候，如果GuardDuty已经是enabled状态，那么部署就会失败
  - 解决方案：使用CF的*Custom Resource*，使用Lambda构建该资源，判断只有当GuardDuty没有被enable的时候，才对它进行enabled修改和配置

### Detective

- 机器学习和图算法。
- 深度分析根源问题。
- 自动收集和输出事件：VPC Flow Logs，CloudTrail，GuardDuty
- 生成可视化visualizations和细节details。
- 流程：detective检测 - triage问题分类 - scoping问题界定 - response回应

### Inspector

- 自动（漏洞）安全评估工具
- 对EC2通过SSM Agent(这是服务触达EC2的方式，必须设置)检查OS漏洞，以及网络触达*network reachability*（端口是不是不应该对公网开放之类的）to EC2
- 可以检查*ECR*中上传的Container Images
- 检查Lambda软件代码和依赖的漏洞，when deployed
- 结果去向：可以发送到*SecurityHub*中作为报告，也可以trigger EventBridge出发其他内容
- 重点：三种服务（EC2,ECRimages,lambda），针对漏洞，漏洞CVE更新那么他会重新检测，会给出一个risk score风险分数

### Trusted Advisor

- 检查和建议。
- 六个方面：Cost-optimization，Performance，Security，FaultTolerance，ServiceLimits，OperationalExcellence。
- 但是 Free Plan 只能用很少的一部分比如安全和服务限制的一部分check，如果要 FullSetCheck 和 supportAPI 的使用，要求 Business&Enterprise Support Plan。

### Secrets Manager

- 强制在X天后进行Rotation
- 使用 Lambda 可以自动生成新的 Secrets：需要 Lambda 对 SecretManager 和数据库的访问许可，包括对subnet的网络设置（VPCEndpointorNATGateway）
- 内部使用KMS进行加密，对称加密，发送GenerateDataKeyAPI进行信封加密，需要双方service的权限
- 与各种DB集成用于密码存储和轮换
- Multi-region跨区复制功能，用于灾难恢复，跨区DB和跨区应用等：Primary - Read Replica

- ECS（worker/task）也集成使用SecretManager和SSMParameterStore的key进行RDS和API的连接。
