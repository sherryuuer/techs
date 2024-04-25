## 关于云计算服务中IaC和CI/CD服务的总结和比较

### 概念和关系

IaC（Infrastructure as Code）是指将基础设施资源（例如服务器、虚拟机、网络、存储等）的配置和管理编码为文本文件的做法。这些文件通常使用声明性语言编写，例如YAML、JSON或HashiCorp Configuration Language（HCL）。然后，可以使用版本控制系统来跟踪和管理这些文件。

IaC的主要优点包括：

* **可重复性：** IaC使基础设施的配置和管理更加可重复。这意味着可以轻松地在不同的环境中创建相同的基础设施，并且可以更轻松地进行故障排除和恢复。
* **一致性：** IaC有助于确保基础设施在所有环境中都是一致的。这可以减少错误并提高安全性。
* **自动化：** IaC可以用于自动化基础设施的配置和管理。这可以节省时间并提高效率。
* **版本控制：** IaC可以使用版本控制系统进行跟踪和管理。这使您可以查看基础设施配置的更改历史记录，并轻松地回滚到以前的版本。

**CI/CD**（Continuous Integration/Continuous Delivery）是一种软件开发实践，它强调自动化软件开发和测试过程的各个方面。CI/CD管道通常包括以下步骤：

1. **代码提交：** 开发人员将代码提交到版本控制系统。
2. **构建：** 代码被自动构建成可执行文件。
3. **测试：** 可执行文件经过自动测试。
4. **部署：** 如果测试成功，可执行文件将被部署到生产环境。

CI/CD的主要优点包括：

* **更快的更新：** CI/CD可以帮助更快地将新功能和修复程序发布给客户。
* **更高的质量：** CI/CD可以帮助提高软件质量，因为可以更早地发现并修复错误。
* **降低成本：** CI/CD可以帮助降低成本，因为它可以自动化许多手动任务。
* **提高协作：** CI/CD可以帮助提高团队成员之间的协作，因为它可以促进更频繁的代码提交和集成。

**IaC和CI/CD的关系**

IaC和CI/CD密切相关。IaC可用于描述和管理基础设施，而CI/CD可用于自动化基础设施的配置和部署。以下是一些将IaC和CI/CD结合使用的示例：

* 使用CI/CD管道来自动运行IaC模板。这可以用于在每次代码更改时自动更新基础设施。
* 使用IaC来定义要用于测试环境的基础设施。然后，可以使用CI/CD管道自动创建和销毁测试环境。
* 使用IaC来定义要部署到生产环境的基础设施。然后，可以使用CI/CD管道自动将该基础设施部署到生产环境。

将IaC和CI/CD结合使用可以创建更可重复、一致、可扩展和可靠的基础设施。还可以更快地发布新功能和修复程序，并降低成本。

### 云计算服务中的IaC服务总结和比较

**AWS**

- Amazon CloudFormation：一个模板化工具，可用于以声明性方式描述和管理AWS基础设施。CloudFormation模板是JSON或YAML文件，用于定义要创建或更新的AWS资源。同时可以更新stacksets，每次只执行更新的部分，我觉得这是一个方便的功能。
- AWS CloudFormation Registry：这是一项模板注册表，可用于存储和共享CloudFormation模板。CloudFormation Registry让人可以轻松地与他人共享模板，并从他人的模板中学习。我列出这个是因为我对于怎么制作模板还是经验不足，我觉得这种服务很好，很像Github Actions的workflow仓库，给人一些提示。
- AWS OpsWorks：这是一项配置管理服务，可用于自动化AWS资源的配置和管理。OpsWorks使用Chef或Puppet等配置管理工具来执行配置任务。CloudFormation主要用于基础设施资源管理，OpsWorks主要是应用程序配置和管理。
- AWS CodeDeploy：这是一项部署服务，可用于将应用程序部署到AWS基础设施。CodeDeploy支持各种部署方法，包括蓝绿部署和滚动部署。他也会在CI/CD中出现。
- AWS Serverless Application Model（SAM）：这是一组工具和框架，可用于构建、部署和管理无服务器应用程序。SAM使用YAML模板来定义无服务器应用程序的架构。我没用过这个服务，但是根据调查，无服务器应用主要就是指，这种应用程序的背后，是 Lambda 后者 API 端点之类的在提供服务。便于维护和可扩展。
- AWS AWS CDK（Cloud Development Kit）是一种软件开发工具，它用于以编程方式定义基础设施资源和应用程序部署在 AWS 上所需的资源配置。相比于传统的基于模板的部署工具（比如 AWS CloudFormation），CDK 允许开发人员使用常见的编程语言（如 TypeScript、Python、Java 和 .NET 等）来编写基础设施的代码。CDK 提供了一系列的高级抽象和构造器，使得编写和管理基础设施变得更加简单、灵活和可维护。使用 CDK，开发人员可以创建和配置 AWS 资源，如虚拟私有云（VPC）、Amazon S3 存储桶、Amazon EC2 实例、AWS Lambda 函数等。CDK 还支持将多个 AWS 资源组合成堆栈（Stack），以便将相关的资源一起部署和管理。CDK 提供了丰富的库，其中包含了各种 AWS 服务的高级抽象，使得开发人员可以更容易地创建和配置这些服务。一旦基础设施代码编写完成，开发人员可以使用 CDK CLI（命令行界面）将其部署到 AWS 云中。CDK 会将编写的代码转换为 CloudFormation 模板，并通过 CloudFormation 进行资源的创建和管理。这种方式既保留了 CloudFormation 模板的优势，又提供了更灵活和高级的编程接口，使得基础设施的管理更加便捷和可控。
- AWS Application Composer 是一项服务，可帮助轻松地设计、构建和部署无服务器应用程序。它提供了一个可视化界面，可将 AWS 服务拖放到画布上，以创建应用程序架构。然后，Application Composer 会自动生成必要的 AWS CloudFormation 或 AWS Serverless Application Model (SAM) 模板，以便部署应用程序。
- ECS 和 AWS Fargate 等容器服务如果说dockerfile也算是一种环境脚本的话，那他们也算在内了。
- System Manager 中的 RunDocument，我是因为学习安全方面的东西，学到了这个服务才联想到了其他的东西，SSM的Document完全就是一个部署代码，通过在EC2中安装的Agent直接对服务器进行执行。很方便进行统一管理的场景。如果说CloudFormation管理整个AWS的资源，那么SSM就是对EC2集群的方便管理。

**GCP**

- Google Cloud Deployment Manager 也是一种完全托管的 IaC 服务，可用于部署和管理 GCP 资源。它使用 YAML 或 JSON 模板来定义要部署的资源。但是它不像Terraform那么灵活，似乎只支持 GCP 资源。
- GKE 和 CloudRun 相关的容器服务如果也算的话，这里可以顺便列出来。

**Azure**

- 我不熟微软的云服务，所以我简单查了一下，Azure Resource Manager 模板 (ARM 模板)似乎是官方的服务，ARM 模板是一种基于 JSON 的模板语言，用于定义和部署 Azure 资源。它是一种声明性语言，可以描述您想要创建的最终资源状态，Azure Resource Manager 将创建和管理这些资源。可通过 Azure PowerShell、Azure CLI 和 Azure REST API 进行部署。另外似乎有一种叫 Bicep 的语言，是一种基于 Azure Resource Manager 模板的更高层次的抽象语言。它使用类似于 C# 的语法，使其更易于阅读和编写。Bicep 还支持一些高级功能，例如条件逻辑和循环。
- Azure DevOps 是一种云平台，可用于管理软件开发生命周期 (SDLC)。它包括各种工具，用于版本控制、构建、测试和部署。Azure DevOps 可以与 ARM 模板或 Bicep 一起使用来自动化基础设施配置和应用程序部署。这就像是CI/CD的工具了。

**Terraform**：由于它在三个云中都广泛采用，所以单独列出比较好。

一个开源的基础设施即代码工具，可用于以声明性方式描述和管理各种基础设施资源，包括 GCP 资源、AWS 资源、Azure 资源等。它使用 HashiCorp Configuration Language（HCL）来定义基础设施。我所在做的项目，就是对方用 Terraform 进行管理，以前我有过小的学习，但那时候我基础知识有限还无法完全理解很多过程，我会在接下来系统学习一次，相信会有更深的理解。

### 云计算中CI/CD服务的总结和比较

**AWS**：CodePipeline 用于构建、测试和部署应用程序的完全托管服务。易于使用，可扩展，与其他 AWS 服务很好地集成。它内部集成了 CodeCommit，CodeBuild，CodeDeploy服务，以前似乎是有CodeStar但是官网发表，2024年7月开始，他就停止服务了。Pipeline和 Github 很像，但是功能有限，限于在AWS使用。可以拉取 Github 仓库资源进行部署，这一点还是不错的。

**GCP**：Cloud Build 用于构建和测试应用程序的完全托管服务，和其他服务的集成很好。Cloud Deploy 用于部署应用程序的完全托管服务。它支持多种部署策略，并可以与其他 GCP 服务（例如 Cloud Build 和 Cloud Monitoring）集成。Cloud Source Repositories：一种用于托管 Git 存储库的完全托管服务。它可以与其他 GCP 服务（例如 Cloud Build 和 Cloud Deploy）集成。以上这三种服务，就对标AWS的Pipeline了。

**Azure**：Azure DevOps Pipelines。我不太熟，所以暂且列出名字即可。

### 写这篇的契机

我之所以查到这些是因为我想回顾起来以前用过的 Cloud Run 服务，Cloud Run容器服务在 AWS 中有 ECS 和 Fargate 等服务相似。 

关于如何部署 Cloud Run 我以前经验不足的时候进行的一些实践，但是理解很浅，因此想总结一下现在的理解。

它是通过对 Docker file 的部署，在容器中运行应用的一种服务。使用 `docker build` 和 `docker push` 方法就可以将部署好的 image 文件推送到 Register 仓库，然后就可以用 `gcloud run deploy` 命令一键部署了。大部分时候是为了部署应用程序，所以我在网上看到的方法都是关于 Flask 框架的，这对我的理解有很大的偏差影响，但是事实上，跑一些类似 AWS Lambda 程序也可以。

同时如果将 Cloud Build 和 Cloud deploy 和 Cloud Run 集成，就可以在代码更改的时候，出发自动部署，这就很像之前搞的 Github Actions 的workflow了。很方便。但是根据业务需求，也许简单的脚本，并不需要自动部署和版本管理，之需要通过其他的服务，执行对应的 CLI 命令就够了。根据需求的开发最重要。

### Recap

外面最大的开源工具就是 Github，庆幸我每天都在用它。让我对这些服务有了比较初步的理解，而容器服务更加重要，毕竟整个云，都是基于虚拟容易，运作的。我最近开始学习的区块链，更是数字世界以及版本管理的结合体。

无论他们的名字是什么，底层原理都是来自Git工具，和Docker工具，而这两者是DevOps中非常重要的部分，我还在学习基础的Git和Docker，理解还是不太够，但是只要理解了底层知识，相信不管什么云工具都是一样原理。

