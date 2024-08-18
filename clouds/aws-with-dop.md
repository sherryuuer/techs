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

## Resilient Cloud Solution

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



## Monitoring & Logging

## Incident & Event Response

## Security & Compliance
