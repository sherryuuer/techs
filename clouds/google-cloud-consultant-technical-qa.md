# Google Cloud Consultant 面试经验：技术 Q&A 复习清单

这篇是我准备 Google Cloud Consultant / Data Analytics 面试时整理的技术 Q&A 复习清单。它不是完整题库，而是一些我觉得最值得掌握的基础问题和回答方向。

## 复习原则

我没有追求把所有答案背下来，而是要求自己做到：

- 能用简单语言解释概念
- 能说出适用场景
- 能说出 trade-off
- 能联系到数据平台项目
- 能在英文中用短句说明

技术面试中，能把基础讲清楚通常比堆很多术语更重要。

## Web / Network 基础

### 用户输入 URL 后发生什么？

回答线索：

```text
URL -> DNS -> TCP/TLS -> HTTP request -> CDN / Load Balancer -> App Server -> Backend / DB / Cache -> HTTP response -> Browser rendering
```

重点不是背每一步，而是能说明浏览器、网络、服务器和后端服务之间如何协作。

### HTTP 和 TCP/IP 的区别？

HTTP 是应用层协议，定义 request / response 的格式和语义。TCP/IP 是更底层的网络协议族，负责寻址、路由、可靠传输和顺序控制。

简单说：

```text
TCP/IP moves data.
HTTP defines web communication.
```

### 同步和异步处理有什么区别？

同步处理会等待结果，简单但可能阻塞。异步处理提交任务后继续执行，适合长时间、解耦、大量处理，但需要 retry、monitoring、ordering 和 error handling。

数据平台中，batch job、message queue、Pub/Sub event processing 都是常见异步模式。

## Database / SQL 基础

### OLTP 和 OLAP 的区别？

OLTP 面向交易系统，例如订单、支付、账户更新，特点是大量小读写、低延迟、强一致性。

OLAP 面向分析系统，例如 dashboard、reporting、ad-hoc analysis，特点是大量扫描、聚合、复杂查询。

在 GCP 中：

```text
Cloud SQL / AlloyDB -> OLTP
BigQuery -> OLAP
```

### JOIN 有哪些类型？

- INNER JOIN: 只返回两边匹配的记录
- LEFT JOIN: 返回左表全部记录，右表没有匹配时为 NULL
- RIGHT JOIN: 返回右表全部记录
- FULL OUTER JOIN: 返回两边全部记录，不匹配的一侧为 NULL

数据质量检查中，LEFT JOIN 和 FULL OUTER JOIN 很常用，尤其适合查 source / target 之间 missing 或 extra records。

### WHERE 和 HAVING 的区别？

WHERE 在 aggregation 前过滤原始行。HAVING 在 GROUP BY 后过滤聚合结果。

```sql
SELECT
  customer_id,
  SUM(amount) AS total_amount
FROM transactions
WHERE transaction_date >= '2026-01-01'
GROUP BY customer_id
HAVING SUM(amount) > 10000;
```

### Window function 是什么？

Window function 可以在不压缩行数的情况下，对当前行相关的一组数据进行计算。

常见用途：

- deduplication
- ranking
- latest record
- running total
- previous / next value

常见函数：

```text
ROW_NUMBER
RANK
DENSE_RANK
LAG
LEAD
SUM OVER
```

### Transaction 和 ACID 是什么？

Transaction 是一组作为整体执行的数据库操作。ACID 表示：

- Atomicity: all or nothing
- Consistency: 从合法状态到合法状态
- Isolation: 并发事务之间不错误干扰
- Durability: commit 后结果持久保存

在面试中可以顺便说明：强一致性通常更适合在 OLTP 或应用层处理，分析平台通常接受 near-real-time 或 eventual consistency，除非业务明确要求强一致。

## BigQuery / Data Analytics

### BigQuery 为什么适合分析？

BigQuery 是 serverless、columnar、scalable 的数据仓库，适合大规模扫描、聚合和分析查询。使用者不需要管理底层集群，可以把重点放在数据建模、查询优化、权限和成本控制上。

### Partitioning 和 clustering 的区别？

Partitioning 是把表按行分区，常见是按日期字段。查询带 partition filter 时可以减少扫描范围。

Clustering 是在 partition 内部按指定列组织数据，适合经常用于 filter 或 join 的字段。

注意：

```text
BigQuery partitioning is closer to horizontal partitioning, not vertical partitioning.
```

### Data warehouse、data lake、lakehouse 的区别？

Data warehouse 更强调结构化数据、SQL 分析、BI 和数据建模。

Data lake 更适合保存原始或半结构化数据，灵活性高，但如果缺少治理，容易变成难以使用的数据堆。

Lakehouse 尝试结合两者，把 data lake 的开放存储和 warehouse 的表管理、schema、事务能力、治理能力结合起来。

### Batch、streaming、event-driven 如何选择？

Batch 适合定期处理、依赖关系明确、需要重跑和补数的场景。

Streaming 适合持续事件、低延迟和实时分析。

Event-driven 适合文件上传、状态变化、系统事件触发后启动处理。

不要为了“现代化”就把所有东西都做成 streaming。很多业务报表 daily batch 已经足够，而且成本和运用复杂度更低。

### Pub/Sub 适合做什么？

Pub/Sub 适合消息传递、事件通知和 producer / consumer 解耦。

一个重要点：

```text
Pub/Sub carries messages or events, not large file contents.
```

如果是 GCS 文件上传，通常是发送一个事件，里面包含 bucket、object name 等 metadata，然后由 Cloud Run、Cloud Functions、Dataflow、Composer 或 BigQuery load job 去处理文件。

## Data Quality / Migration

### 如何做 source / target reconciliation？

先确认 grain。比较的是订单级、用户级、日汇总，还是明细记录级。

常见方式：

- row count comparison
- primary key comparison
- aggregate comparison
- checksum / hash comparison
- FULL OUTER JOIN 找 missing / extra records
- sample record validation

如果数据不一致，不要先假设是代码 bug。也可能是 source definition、time zone、late arriving data、dedup rule、status mapping 或 join key 的问题。

### Incremental load 要注意什么？

常见风险：

- watermark 选择不正确
- late arriving data 漏处理
- retry 后重复写入
- update / delete 没有反映
- source timestamp 不可靠
- job failed 后无法安全重跑

好的 incremental load 通常需要：

- stable watermark
- idempotent design
- staging table
- MERGE / upsert
- audit table
- reprocess strategy

## Code Review 常见检查点

### Python ETL

- secret 是否 hardcode
- API error 是否处理
- retry / timeout 是否合理
- 是否一次性读取大文件
- input schema 是否校验
- logging 是否足够
- exception 是否被吞掉
- re-run 是否 idempotent

### SQL

- join key 是否正确
- grain 是否一致
- LEFT JOIN 是否被 WHERE 条件意外变成 INNER JOIN
- dedup 规则是否稳定
- partition filter 是否存在
- aggregation 是否重复计算
- NULL 是否按业务规则处理
- incremental condition 是否会漏数据

## 英文快速表达

我给自己准备的英文表达通常很短：

```text
First, I would clarify the business requirement and data freshness requirement.
Then I would check the data volume, data source, security requirements, and operational constraints.
Based on that, I would decide whether batch, streaming, or event-driven processing is appropriate.
```

```text
For data quality, I would check schema, duplicates, missing records, NULL values, and source-to-target reconciliation.
```

```text
For reliability, I would consider retry, idempotency, monitoring, alerting, and reprocessing.
```

## 最后心得

技术 Q&A 不需要准备成百科全书。更重要的是每个概念都能回答三件事：

1. 它是什么
2. 什么时候用
3. 有什么限制或 trade-off

只要能把这三点说清楚，再结合自己的项目经验，回答就会自然很多。
