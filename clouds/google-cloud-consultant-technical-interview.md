# Google Cloud Consultant 面试经验：技术与架构讨论

这篇记录我准备 Google Cloud Consultant / Data Analytics 技术面试时的思路。它更偏复盘，不是题库。实际面试中，技术问题并不只是考服务名，而是看你能不能基于业务需求、数据特性和运用约束做判断。

## 技术面试的重点

我准备时把技术面试分成三类：

1. Cloud data architecture
2. Code / SQL review
3. General web, database, system design fundamentals

其中最重要的是第一类和第二类。因为 Cloud Consultant 面试通常不只是问“BigQuery 是什么”，而是问：

- 这个场景应该怎么设计
- 为什么这样设计
- 有什么 trade-off
- 数据质量怎么保证
- 如果失败了怎么重跑
- 成本和权限怎么控制
- 如何向客户解释这个方案

## 我使用的六个 review 维度

不管是架构题还是 code review，我都用同一套框架：

```text
Data Quality
Security
Scalability
Reliability
Maintainability
Cost
```

这个框架很实用。因为面试里不可能提前准备到所有问题，但可以用它快速组织回答。

### Data Quality

关注数据是否正确、完整、一致。

常见点：

- schema validation
- duplicate records
- missing records
- NULL handling
- source / target reconciliation
- business definition alignment
- late-arriving data

数据平台里很多问题表面是 SQL 或 pipeline bug，本质上是数据定义、上游变更、join key、粒度或业务规则没有对齐。

### Security

关注谁能访问什么数据，以及敏感数据如何处理。

常见点：

- IAM / service account
- least privilege
- PII masking
- encryption
- audit logs
- network boundary
- dataset / table / column level access

如果是 customer-facing role，security 不能只说“加权限”。要能说明数据使用范围、角色边界和运用流程。

### Scalability

关注数据量、并发、延迟要求变化后还能不能处理。

常见点：

- BigQuery partitioning / clustering
- batch vs streaming
- horizontal scaling
- Pub/Sub / Dataflow
- query optimization
- file size and small files issue

我会把 performance 也放在 Scalability 下面考虑。

### Reliability

关注失败时系统能不能恢复。

常见点：

- retry
- idempotency
- dead letter queue
- monitoring and alerting
- backfill / reprocess
- checkpoint
- dependency management

数据 pipeline 很少永远不失败，所以面试中能讲清楚恢复方式很重要。

### Maintainability

关注团队以后能不能继续维护。

常见点：

- SQL readability
- modular transformation
- version control
- code review
- naming convention
- documentation
- testability

很多架构不是因为技术不先进而失败，而是因为团队维护不了。

### Cost

关注方案是否过度设计，资源是否浪费。

常见点：

- BigQuery scan cost
- storage lifecycle
- job frequency
- streaming vs batch cost
- right-sizing
- cost monitoring

不要默认所有东西都实时化。很多分析场景 daily batch 已经足够。

## Batch、Streaming 和 Event-driven 的区别

这是我重点准备的一个话题。

### Batch

适合：

- 日次、小时级处理
- 有明确依赖关系的 workflow
- 需要重跑、补数、顺序控制的 pipeline

常见工具：

- Cloud Composer / Airflow
- BigQuery scheduled queries
- Dataform schedules

### Streaming

适合：

- 秒级或分钟级延迟要求
- 持续事件数据
- log、clickstream、IoT、实时监控

常见工具：

- Pub/Sub
- Dataflow
- BigQuery streaming / subscription

### Event-driven

适合：

- 文件上传后触发处理
- 系统事件通知
- 解耦 producer 和 consumer

常见工具：

- Pub/Sub
- Eventarc
- Cloud Run
- Cloud Functions

我准备时特别注意一个点：Pub/Sub 传的是 message / event，不是大型文件本体。如果是 GCS 文件，一般是 GCS event 通知下游，让 Cloud Run、Composer、Dataflow 或 BigQuery load job 去处理文件。

## BigQuery 相关重点

我重点复习了这些：

- BigQuery 是 OLAP，不是 OLTP
- partitioning 是按行把数据分区，常见是日期字段
- clustering 是在 partition 内部按列组织数据，帮助减少扫描
- BigQuery 是 columnar storage，所以不是所有宽表都需要 vertical partitioning
- denormalized table 和 star schema 都常用于 analytics
- query cost 和 scanned bytes 关系很大
- 数据迁移时要做 source / target reconciliation

一个容易混淆的点：

```text
Partitioning is not vertical partitioning.
```

BigQuery partitioning 更接近 horizontal partitioning。Vertical partitioning 是按列拆宽表，通常是为了安全、访问模式或管理原因。

## Code / SQL review 的回答方式

我准备 code review 时不急着指出 bug，而是按这个顺序：

1. 先说明代码在做什么：read -> transform -> write
2. 再从六个维度看风险
3. 最后提出具体改善建议

例如 Python ETL 可以看：

- 是否有 input validation
- API error / retry 是否处理
- 是否 hardcode secret
- 是否一次性读入大文件
- 是否支持 idempotent re-run
- logging 是否足够
- exception 是否吞掉

SQL review 可以看：

- join key 是否正确
- grain 是否一致
- WHERE 条件是否改变 LEFT JOIN 语义
- window function 是否能稳定去重
- incremental load 是否会漏数据
- full outer join 是否可用于 reconciliation
- partition filter 是否存在

这个方式比直接说“这里错了”更像真实 code review，也更适合 consultant 面试。

## 架构题的回答方式

如果被要求设计一个数据平台，我会先问或先声明需求：

- 数据来源是什么
- 数据量和增长速度
- latency 要求
- batch 还是 streaming
- 数据是否包含 PII
- 谁使用数据
- 是否需要 dashboard / self-service analytics
- SLA 和运用团队能力如何
- 成本约束是什么

然后再选择服务。不要一开始就说“用 BigQuery + Dataflow + Looker”。服务选择应该从需求推出来。

## 最重要的心得

技术面试里，知道服务名只是基础。更重要的是能解释：

- 为什么这个服务适合这个场景
- 什么时候不应该用它
- 出错时怎么恢复
- 数据不一致时怎么调查
- 成本和安全如何控制
- 团队以后怎么维护

如果准备时间有限，我会优先练架构解释和 code review，而不是继续背更多零散知识点。
