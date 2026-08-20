# Google Cloud Consultant 面试经验：如何整理项目经历

这篇是我准备 Google Cloud Consultant / Data Analytics 方向面试后的复盘。它不是标准答案，也不是逐字背诵稿，而是我自己把过往项目整理成 consultant 视角时用到的方法。

我原本的背景更接近 Data Engineer：做过 ETL、DWH、data lake、cloud migration、data quality、reporting 和一些运用改善。准备这个岗位时，我最大的变化不是补了多少新技术，而是重新整理了自己讲项目的方式。

## 核心定位

面试中不要只把自己讲成“会写 pipeline 的人”。Cloud Consultant 更关心的是：

- 你能不能理解客户的业务目标
- 你能不能看懂现有架构的问题
- 你能不能解释技术选型的 trade-off
- 你能不能和上游、下游、平台团队、业务团队一起推进问题
- 你能不能设计一个现实可落地、可运用、可维护的方案

我最后给自己的定位是：

```text
Data Engineer background, moving toward a customer-facing Cloud/Data Consultant.
```

也就是：我有数据工程的实际经验，但不只关注实现，也关注数据质量、运用、架构、成本、维护性和 stakeholder alignment。

## 项目经历的整理方式

我把自己的项目按下面几个角度重新整理了一遍：

| 角度 | 要回答的问题 |
| --- | --- |
| Business goal | 客户或业务真正想解决什么问题 |
| Current architecture | 原来的系统是什么样，有什么限制 |
| My role | 我负责实现、设计、review、协调还是说明 |
| Technical decision | 为什么选这个服务或架构 |
| Trade-off | 这个选择牺牲了什么，避免了什么风险 |
| Data quality | 数据正确性、缺失、重复、不一致如何处理 |
| Operation | 如何监控、重跑、告警、权限管理、版本管理 |
| Result / learning | 项目带来了什么改善，我学到了什么 |

这个整理方式比单纯列技术栈有用很多。面试官不是只想听“我用过 BigQuery / Dataform / Looker / Airflow”，而是想知道我为什么这么设计，以及这种设计在真实项目中解决了什么问题。

## 我重点准备的项目类型

### 1. Cloud migration

最适合作为主项目。因为 migration 天然包含现状分析、目标架构、成本、风险、运用、客户说明和分阶段切换。

我准备时会重点说明：

- 原来的云上分析基盘有什么痛点
- 为什么迁移到 GCP
- BigQuery、Dataform、Looker、Airflow、Pub/Sub 等服务分别解决什么问题
- 哪些处理适合 batch orchestration，哪些处理适合 event-driven
- 如何减少迁移后的运用风险

这个类型的项目非常适合展示 consultant 思维，因为它不是“把 A 服务换成 B 服务”，而是要解释客户为什么需要换、怎么换、风险在哪里、团队以后怎么维护。

### 2. Data lake / data platform from zero to one

这类项目适合展示基础能力和 DataOps 视角。

我会重点讲：

- 数据如何从上游系统进入 data lake
- batch、SFTP、SQL job、workflow、alert、DDL check 如何逐步补齐
- 数据质量问题是如何发现和排查的
- 上游系统、分析团队、业务团队之间如何对齐数据定义
- 为什么数据平台不只是架构图，也包括监控、版本管理和变更通知

这类项目的价值在于，它说明你知道真实数据平台不是一开始就完美的。很多时候是先能跑起来，再一步步补齐可靠性、可维护性和治理能力。

### 3. Lakehouse / modernization

现代化项目适合展示对企业数据基盘演进的理解。

我会避免简单说“旧系统不好”。更合理的表达是：

- 旧系统可能稳定，但在扩展性、成本、开发速度、自动化运用上有改善空间
- 新架构要尊重既有资产和业务连续性
- 数据共享、安全、权限、敏感信息处理和下游利用方式都需要一起考虑
- modernization 不是单纯技术迁移，而是数据利用方式和运用方式的升级

这个表达会比“我们用了新的技术”更成熟。

## 我用的统一 review framework

不管是讲项目、看架构图，还是做 code review，我都尽量用同一套框架：

```text
Data Quality
Security
Scalability
Reliability
Maintainability
Cost
```

这个框架的好处是，面试中即使遇到没准备过的问题，也不会完全没有方向。

例如被问到一个数据 pipeline 设计，我会先想：

- Data Quality: 如何校验 schema、重复、缺失、异常值
- Security: PII、IAM、service account、encryption、access control
- Scalability: 数据量变大后怎么处理，是否需要 partition / clustering / streaming
- Reliability: retry、idempotency、dead letter、monitoring、reprocess
- Maintainability: SQL / DAG / transformation logic 是否容易 review 和变更
- Cost: BigQuery scan cost、storage、job frequency、过度实时化是否有必要

## STAR 不是背稿，而是防止跑题

我没有把所有回答写成完整演讲稿，而是给每个项目准备了 STAR 骨架：

- Situation: 当时是什么背景
- Task: 我需要解决什么问题
- Action: 我实际做了什么
- Result: 结果和学习是什么

真正面试时不会完整照读，但这个结构可以防止回答散掉。尤其是 technical interview 中，讲着讲着很容易陷入服务名和实现细节，STAR 可以提醒自己回到问题、行动和结果。

## 最重要的准备心得

我觉得最有帮助的不是死背 GCP 服务，而是把每个项目改写成“客户问题 -> 架构选择 -> trade-off -> 运用结果”的故事。

Data Engineer 面试更容易被问“你怎么实现”。Consultant 面试则更常看：

- 你是否能解释为什么这样实现
- 你是否能理解非技术方的目标
- 你是否能把复杂问题拆成可讨论的论点
- 你是否知道实际运用会遇到什么问题

如果要准备类似岗位，我会建议先不要急着刷很多新知识点。先把自己的项目重新整理一遍，尤其是每个项目背后的业务目标、架构权衡、数据质量和运用问题。这样回答会自然很多，也更像真实做过项目的人。
