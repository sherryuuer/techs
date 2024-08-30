基本上只会在云上部署IaC了现在，所以放入云学习笔记。IaC的好处不言而喻。

- 过程性语言和声明性语言：

Chef和Ansible鼓励采用过程性编程语言，在这种语言中，用户可以编写代码来指示工具如何逐步实现所要达到的最终状态。*Terraform、CloudFormation*、SaltStack、Puppet和OpenStack Heat都鼓励使用声明性编程语言，在这种语言中，用户可以编写代码来指示工具所要达到的最终状态，而IaC工具将负责决定具体的实现步骤。显然过程性语言需要更多思考和考虑，但是指示最终状态的声明性语言，则更好。Terraform的plan命令则可以预览将要发生的事情。

过程性代码很难重用，因为你要考虑历史记录。

- 这次只是测试在GCP的bigquery中的部署，在terraform中有相关的代码

src是一开始理想的有很多文件结构的代码，但是似乎测试不是很，于是简化了版本，给每个dataset一个文件，是src2的代码内容，跑的很好，有以下效果：

1. 代码环境可以分开，dev中merge后可以在开发环境部署，main中merge后可以在生产环境部署
2. 在GithubActions中的的workflow中，使用terraform部署时候的var文件，来动态选择对象project就可以了，环境变量的文件在env文件夹中定义即可

前提条件：在上述的workflow的yml文件中还要动态定义使用的认证信息，使用了workloadidentity，需要在GCP中提前设置好

另外测试了一些关于表格定义的内容有一些注意事项：

- 单纯增加表的col之类的，不会destory表格，所以可以正常运行
- 当删除col或者改变col的定义则会出发destory行为
- 如果删除整个表的定义也是出发destory行为
- 环境默认删除保护选项（deletion_protection）为true，所以需要设置为false
  - 设置方法上，需要在第一次merge中先将删除保护关闭，当系统生效后，再在下一次的merge中对表格进行修改
  - 因为如果同时设置删除保护关闭，和修改表格，这个时候Actions中还没有生效该选项，所以需要两次操作
- 由于上述原因，在运维中就需要确定好步骤：
  - 首先在dev环境进行删除保护关闭，然后通过PR影响生产环境
  - 然后在dev环境进行表格修改，然后通过PR影响生产环境
  - 最后还原所有设置还需要一次操作
  - 前后还需要备份数据和还原数据
- 为什么不能手动删除表格：因为terraform依靠状态文件进行diff管理，所以最好所有的修改都通过基础设置代码
- 最好设置main分支的PR保护，确保运维人员知道修改的内容的严重程度，避免造成数据损失
- 对于经常对表进行操作的项目，用IaC真的好吗，也是一个好考虑的问题，如果团队经常改变表格定义，可能只用DDL会更方便
  - 比如可以用bq的client或者CLI一键创建tables也是一个不错的方法

一些我玩的时候的局限性：

- 因为我用的是公司的GCP环境所以需要认证，而我的WI认证只能在Actions中跑，所以没法本地运行terraform命令

## 学习笔记

- terraform output 命令可以查看输出
- terraform destroy 最好用于删除整个infra，而不是个别的，删除个别资源最好使用修改tf文件的方式
- variables 的设置在 apply 后都是要手动输入的，虽然变量可以放在任何地方，不管放在哪里都可以用 var 呼出
  - 也可以用 -var option 来指定，并且可以多次指定

- 实质上可以在一个文件中写所有的东西，但是进行文件夹组织是一个好的实践，机器看都一样，人看就需要更清晰
- `terraform.tfvars` 是一个用于在 Terraform 中定义变量值的文件。这个文件用于将具体的值赋予在 `variables.tf` 文件中定义的变量。比如
   ```hcl
   project_id = "chaonanwang-sandbox"
   region     = "us-central1"
   ```
   - 当运行 `terraform apply` 时，Terraform 会自动加载 `terraform.tfvars` 中的变量值，并将它们应用到你的 Terraform 配置
   - 可以根据不同的环境创建不同的 `.tfvars` 文件，如 `dev.tfvars`, `prod.tfvars`。在运行 Terraform 时，使用 `-var-file` 参数指定特定的 `.tfvars` 文件：
   ```sh
   terraform apply -var-file="prod.tfvars"
   ```
   - 如果在命令行中使用 `-var` 选项指定了变量值，它将覆盖 `terraform.tfvars` 中的值，它具有最高优先权
   - 似乎terraform都是用双引号，`"Name" = "${var.vpc_name} VPC"`

- Terraform使用*Go语言*开发，进行测试也是使用Go语言进行，对于是否要进行测试，我听到的是藤原老师的提问，这个问题我确实没细想，学习的时候增加思考很重要
- 但是可以将它当作一个*workflow*来使用也是没问题的
- 关于GCP上部署CFn的方式和实践，在项目中做到了，这个很好，内容放在code-store和terraform仓库中
  - 因为只需要部署一个CFn，*如果数量今后增加，最好使用模块化将内容提炼出来，然后再通过source引用*
