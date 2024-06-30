为了使团队成员能够独立管理不同的 dataset，我们可以将每个 dataset 的定义分开，并且保持模块化设计和 GitHub Actions 的自动化部署。以下是一个详细的目录结构和配置示例，展示如何实现分开的 dataset 定义，同时支持团队成员独立管理和 CI/CD 部署。

### 目录结构

```
/project-root
|-- /src/gcp/bigquery
|   |-- /datasets
|   |   |-- /user
|   |   |   |-- main.tf  // 将main文件头部的provider和resource分开放在这里面
|   |   |   |-- usertable.tf
|   |   |   |-- userid.tf
|   |   |-- /account
|   |       |-- accountid.tf
|   |       |-- accountdetail.tf
|   |-- /modules
|   |   |-- /bigquery_table
|   |       |-- main.tf
|   |       |-- variables.tf
|   |       |-- outputs.tf
|   |-- /environments
|   |   |-- development.tfvars
|   |   |-- production.tfvars
|   |-- .github
|       |-- workflows
|           |-- terraform.yml
```

### 文件内容

#### 1. **模块定义（/modules/bigquery_table）**

**`modules/bigquery_table/main.tf`**
```hcl
resource "google_bigquery_table" "table" {
  dataset_id = var.dataset_id
  table_id   = var.table_id
  schema     = var.schema
}
```

**`modules/bigquery_table/variables.tf`**
```hcl
variable "dataset_id" {
  description = "The dataset ID"
  type        = string
}

variable "table_id" {
  description = "The table ID"
  type        = string
}

variable "schema" {
  description = "The schema in JSON format"
  type        = string
}
```

**`modules/bigquery_table/outputs.tf`**
```hcl
output "table_id" {
  description = "The ID of the created table"
  value       = google_bigquery_table.table.table_id
}
```

#### 2. **数据集配置（/datasets/user 和 /datasets/account）**

**`datasets/user/main.tf`**
```hcl
provider "google" {
  project = var.project_id
  region  = var.region
}

resource "google_bigquery_dataset" "user_dataset" {
  dataset_id = "user"
  location   = var.region
}

module "usertable" {
  source     = var.module_source
  dataset_id = google_bigquery_dataset.user_dataset.dataset_id
  table_id   = "usertable"
  schema     = jsonencode([
    {
      "name": "id",
      "type": "INTEGER"
    },
    {
      "name": "username",
      "type": "STRING"
    },
    {
      "name": "email",
      "type": "STRING"
    }
  ])
}

module "userid" {
  source     = "../../../terraform/modules/bigquery_table"
  dataset_id = google_bigquery_dataset.user_dataset.dataset_id
  table_id   = "userid"
  schema     = jsonencode([
    {
      "name": "id",
      "type": "INTEGER"
      "mode": "REQUIRED"
    },
    {
      "name": "user_id",
      "type": "INTEGER"
    },
    {
      "name": "modified",
      "type": "DATETIME",
      "mode": "REQUIRED"
    }
  ])

  time_partitioning = {
    type  = "DAY"
    field = "modified"
  }
  // 使用cluster
  clustering = {
    fields = ["modified"]
  }
  primary_key = ["id"]
}
```

**`datasets/user/variables.tf`**
```hcl
variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "Region for the GCP resources"
  type        = string
}

variable "module_source" {
  default = "../../../terraform/modules/bigquery_table"
}
```

**`datasets/user/outputs.tf`**
```hcl
output "user_table_ids" {
  description = "The IDs of the user tables"
  value       = {
    usertable = module.usertable.table_id
    userid    = module.userid.table_id
  }
}
```

**`datasets/account/main.tf`**
```hcl
provider "google" {
  project = var.project_id
  region  = var.region
}

resource "google_bigquery_dataset" "account_dataset" {
  dataset_id = "account"
  location   = var.region
}

module "accountid" {
  source     = "../../../terraform/modules/bigquery_table"
  dataset_id = google_bigquery_dataset.account_dataset.dataset_id
  table_id   = "accountid"
  schema     = jsonencode([
    {
      "name": "account_id",
      "type": "INTEGER"
    },
    {
      "name": "user_id",
      "type": "INTEGER"
    }
  ])
}

module "accountdetail" {
  source     = "../../../terraform/modules/bigquery_table"
  dataset_id = google_bigquery_dataset.account_dataset.dataset_id
  table_id   = "accountdetail"  # basename(abspath(path.module))
  schema     = jsonencode([
    {
      "name": "account_id",
      "type": "INTEGER"
    },
    {
      "name": "details",
      "type": "STRING"
    }
  ])
}
```

**`datasets/account/variables.tf`**
```hcl
variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "Region for the GCP resources"
  type        = string
}
```

**`datasets/account/outputs.tf`**（option）
```hcl
output "account_table_ids" {
  description = "The IDs of the account tables"
  value       = {
    accountid     = module.accountid.table_id
    accountdetail = module.accountdetail.table_id
  }
}
```

#### 3. **环境变量文件（/environments）**

**`environments/development.tfvars`**
```hcl
project_id = "my-dev-project-id"
region     = "us-central1"
```

**`environments/production.tfvars`**
```hcl
project_id = "my-prod-project-id"
region     = "us-central1"
```

#### 4. **GitHub Actions Workflow（/.github/workflows/terraform.yml）**

设置 GitHub Actions 来自动化部署不同的 dataset。

**`terraform.yml`**
```yaml
name: Terraform

on:
  push:
    branches:
      - main      # 生产环境部署
      - dev       # 开发环境部署

jobs:
  deploy_user:
    name: Deploy User Dataset
    runs-on: ubuntu-latest

    steps:
      - name: Checkout code
        uses: actions/checkout@v2

      - name: Setup Terraform
        uses: hashicorp/setup-terraform@v1

      - name: Terraform Init
        run: cd datasets/user && terraform init

      - name: Terraform Format
        run: cd datasets/user && terraform fmt -check

      - name: Terraform Plan
        id: plan
        run: |
          cd datasets/user
          terraform plan -var-file=../../environments/${{ github.ref == 'refs/heads/main' && 'production.tfvars' || 'development.tfvars' }}

      - name: Terraform Apply
        if: github.ref == 'refs/heads/main' || github.ref == 'refs/heads/dev'
        run: |
          cd datasets/user
          terraform apply -var-file=../../environments/${{ github.ref == 'refs/heads/main' && 'production.tfvars' || 'development.tfvars' }} -auto-approve

  deploy_account:
    name: Deploy Account Dataset
    runs-on: ubuntu-latest

    steps:
      - name: Checkout code
        uses: actions/checkout@v2

      - name: Setup Terraform
        uses: hashicorp/setup-terraform@v1

      - name: Terraform Init
        run: cd datasets/account && terraform init

      - name: Terraform Format
        run: cd datasets/account && terraform fmt -check

      - name: Terraform Plan
        id: plan
        run: |
          cd datasets/account
          terraform plan -var-file=../../environments/${{ github.ref == 'refs/heads/main' && 'production.tfvars' || 'development.tfvars' }}

      - name: Terraform Apply
        if: github.ref == 'refs/heads/main' || github.ref == 'refs/heads/dev'
        run: |
          cd datasets/account
          terraform apply -var-file=../../environments/${{ github.ref == 'refs/heads/main' && 'production.tfvars' || 'development.tfvars' }} -auto-approve
```

### 解读和执行

1. **模块化设计**:
   - 每个 dataset 有自己独立的 Terraform 配置文件夹（`datasets/user` 和 `datasets/account`），这样团队成员可以独立管理各自的 dataset。

2. **顶层配置**:
   - `main.tf` 文件在每个 dataset 目录下定义了与该 dataset 相关的所有资源和模块调用。
   - `variables.tf` 和 `outputs.tf` 文件在每个 dataset 目录下定义了该 dataset 特有的变量和输出。

3. **环境变量文件**:
   - `development.tfvars` 和 `production.tfvars` 文件包含了不同环境的变量配置，位于 `environments` 目录下。

4. **GitHub Actions 工作流**:
   - `terraform.yml` 文件定义了两个独立的工作流，分别用于部署 `user` 和 `account` dataset。
   - 每个工作流在 `main` 和 `dev` 分支上执行时，使用不同的 `.tfvars` 文件来区分开发和生产环境。
   - 使用 `terraform plan`

### 简化工作流步骤

#### 2. **GitHub Actions Workflow**

我们将 `terraform.yml` 工作流更新为一个通用的工作流，自动化部署所有数据集。

**`terraform.yml`**
```yaml
name: Terraform

on:
  push:
    branches:
      - main      # 生产环境部署
      - dev       # 开发环境部署

jobs:
  deploy:
    name: Deploy BigQuery Datasets
    runs-on: ubuntu-latest

    strategy:
      matrix:
        dataset:
          - user
          - account

    steps:
      - name: Checkout code
        uses: actions/checkout@v2

      - name: Setup Terraform
        uses: hashicorp/setup-terraform@v1

      - name: Terraform Init
        run: |
          cd datasets/${{ matrix.dataset }}
          terraform init

      - name: Terraform Format
        run: |
          cd datasets/${{ matrix.dataset }}
          terraform fmt -check

      - name: Terraform Plan
        id: plan
        run: |
          cd datasets/${{ matrix.dataset }}
          terraform plan -var-file=../../environments/${{ github.ref == 'refs/heads/main' && 'production.tfvars' || 'development.tfvars' }}

      - name: Terraform Apply
        if: github.ref == 'refs/heads/main' || github.ref == 'refs/heads/dev'
        run: |
          cd datasets/${{ matrix.dataset }}
          terraform apply -var-file=../../environments/${{ github.ref == 'refs/heads/main' && 'production.tfvars' || 'development.tfvars' }} -auto-approve
```

#### 解读和执行

1. **策略矩阵**:
   - 使用 GitHub Actions 的 `matrix` 功能，为 `user` 和 `account` dataset 分别执行相同的步骤。这避免了为每个 dataset 重复定义工作流。

2. **通用步骤**:
   - 所有 dataset 共享 `checkout`、`init`、`fmt`、`plan` 和 `apply` 的步骤，只是针对不同的 dataset 路径执行。
   - 使用 `matrix.dataset` 动态选择要处理的 dataset 目录（`datasets/user` 或 `datasets/account`）。

3. **环境变量管理**:
   - 通过 `-var-file=../../environments/${{ github.ref == 'refs/heads/main' && 'production.tfvars' || 'development.tfvars' }}`，在 `main` 分支上使用生产环境变量文件，在 `dev` 分支上使用开发环境变量文件。
   - 这种方法确保不同环境的配置文件能够自动应用，避免手动切换配置。

4. **简化工作流**:
   - 一个工作流文件处理所有数据集，避免了为每个 dataset 单独配置工作流的繁琐。
   - 这种方式更易于管理和扩展，当添加新的 dataset 时，只需更新 `matrix` 配置即可。

#### 部署步骤

1. **初始化和格式检查**:
   - `terraform init` 和 `terraform fmt` 确保配置正确初始化且格式化。

2. **计划和应用**:
   - 根据当前分支和策略矩阵，针对每个 dataset，使用相应的环境变量文件执行 `terraform plan` 和 `terraform apply`。

3. **推送代码**:
   - 将代码推送到 `dev` 分支，触发开发环境的部署。
   - 将代码合并到 `main` 分支，触发生产环境的部署。

#### 总结

通过这种方式，我们可以在保持代码分离的同时，使用一个通用的 CI/CD 工作流来管理和部署所有数据集。这减少了工作流配置的重复，提高了管理的简便性和可扩展性。如果将来需要添加新的 dataset，只需在 `matrix` 中添加新的项即可。

### 动态发现dataset的workflow

```yaml
name: Terraform

on:
  push:
    branches:
      - main      # 生产环境部署
      - dev       # 开发环境部署

jobs:
  deploy:
    name: Deploy BigQuery Datasets
    runs-on: ubuntu-latest

    steps:
      - name: Checkout code
        uses: actions/checkout@v2

      - name: Setup Terraform
        uses: hashicorp/setup-terraform@v1

      - name: List Datasets
        id: list_datasets
        run: |
          ls datasets > datasets_list.txt
          cat datasets_list.txt

      - name: Deploy Each Dataset
        run: |
          while read dataset; do
            echo "Processing dataset: $dataset"
            cd datasets/$dataset
            terraform init
            terraform fmt -check
            terraform plan -var-file=../../environments/${{ github.ref == 'refs/heads/main' && 'production.tfvars' || 'development.tfvars' }}
            terraform apply -var-file=../../environments/${{ github.ref == 'refs/heads/main' && 'production.tfvars' || 'development.tfvars' }} -auto-approve
            cd ../../
          done < datasets_list.txt
```

基于src/gcp/bigquery目录的workflow：

```yaml
name: Terraform Deployment

on:
  push:
    branches:
      - main      # 生产环境部署
      - dev       # 开发环境部署

jobs:
  deploy:
    name: Deploy BigQuery Datasets
    runs-on: ubuntu-latest

    steps:
      - name: Checkout code
        uses: actions/checkout@v2

      - name: Setup Terraform
        uses: hashicorp/setup-terraform@v1

      - name: List Datasets
        id: list_datasets
        run: |
          ls src/gcp/bigquery/datasets > datasets_list.txt
          cat datasets_list.txt

      - name: Deploy Each Dataset
        run: |
          while read dataset; do
            echo "Processing dataset: $dataset"
            cd src/gcp/bigquery/datasets/$dataset
            terraform init
            terraform fmt -check
            terraform plan -var-file=../../../environments/${{ github.ref == 'refs/heads/main' && 'production.tfvars' || 'development.tfvars' }}
            terraform apply -var-file=../../../environments/${{ github.ref == 'refs/heads/main' && 'production.tfvars' || 'development.tfvars' }} -auto-approve
            cd ../../../../
          done < datasets_list.txt
```
