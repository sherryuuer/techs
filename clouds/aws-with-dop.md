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

## Configuration & Management & LaC

## Resilient Cloud Solution

## Monitoring & Logging

## Incident & Event Response

## Security & Compliance
