## Management

### Resource Manager

- **GCP构架层**：根为组织，下面是folder层，下面是Project层，pj里面是resources
- PJ的ID和Number是不可变的，全球唯一，Name可以随时变更
- **组织Policy**是和AWS的SCP类似的东西，下面的层级会继承它的Policy
- 创建**组织**不能用个人账户，需要*Google Workspace*或者*Cloud Identity*
- 坚守**最小权限原则**，但是也要尽量削减*管理Overhead*（在管理活动中额外消耗的资源或成本，这些成本并不直接产生效益，但为了确保系统或组织的正常运作而必须承担）
- 环境分离：test，staging，prod，最好是组织层级分离
- 使用*会计组织*管理请求书，其他的组织不拥有阅览权

### CDM（Deployment Manager）

- IaC工具组，Yaml格式，CloudFormation也是这个格式
- `gcloud deployment-manager deployments create/update/delete service --config file.yaml`
- 命令行：服务，组件，动作，对象，选项

### Cloud Foundation Toolkit

- 也是IaC，比CDM更好，对应*所有的service*（CDM只能一部分服务），更简单，可用*Terraform*，这么看来CDM很没用啊
- 里面的代码可以用外部的Github进行版本管理，而CDM只能在GUI上创建和保管

### IAM

- 谁，能做什么，针对什么对象
- **Role**是一种很重要的概念，代表一组资源和权限的组合，像是一顶帽子，GCP有predefined role和custom role
- *bindings*：将role和member进行绑定的概念和CEL语法
- RBAC和ABAC，前者基于role后者基于属性，属性有时候是很难定义的，AWS主要通过tag，GCP有自己的属性设置方式
- *Common Expression Language*（CEL）是一种用于定义条件逻辑的表达式语言，主要用于 Google Cloud 中的权限控制和策略管理。它允许用户编写简单的布尔表达式来定义访问控制的条件，例如访问时间、用户角色等，但是AWS的IAM是用json定义的，两者不同
- 基本（Primitive role）的Owner/Editor/Viewer已经被废除，缺乏灵活性和安全性
- IAM的变更日志，可以通logging服务，*输出*到GCS或者BigQuery进行保存和分析
- *权限继承*，组织-folder-PJ，高一层的权限会被低层继承
- **最佳实践原则**
  - 使用单一的Identity Provider
  - 使用特权管理（Cloud Workspace/Cloud Identity）
  - 使用SA进行服务权限管理
  - 经常更新ID认证最新流程
  - 设置SSO单点登录和MFA认证
  - 使用*最小权限原则*进行权限分散的设置
  - 访问监察，比如使用logging，以及AWS的CloudTrail都是该功能的实现
  - 自动化Policy管理，我想应该是用IaC等
  - 设置对resource的访问限制

### Service Account

- 是*应用或者服务作为主体*，进行认证和使用的，IAM是用户为主体
- 用密码或者认证key进行API执行
- 种类：用户自己创建/内置的default的SA
- AWS没有这个概念，而是用IAM Role控制服务主体的权限

### Identity Platform

- 适合开发者和应用程序构建者，用于为面向终端用户的应用实现身份验证和管理，它支持多种身份验证方式，广泛适用于 Web 和移动应用的场景
- AaaS（Authentication as a Service）
- 步骤：选择IdP，追加用户，创建login和认证界面

### Cloud Identity

- 类似于AWS的Identity Center，统合管理
- 实现User管理，Device管理，SSO登录管理，以及各种安全功能（威胁检测，Endpoint管理，多要素认证等）

## Compute

## Storage

## Network

## Database

## Data Analytics

## AI & ML
