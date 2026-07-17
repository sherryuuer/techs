# Google Cloud Consultant 技术面试 Q&A

> 用途：保存练习过的具体问题和面试可用答案。
> 格式规则：`简洁答案` 用中文，`Quick notes` 用英文，降低复习负担。

## 0. 使用规则

- 一次练一个问题。
- 先看中文简洁答案，确认逻辑。
- 再用英文 Quick notes 练口头表达。
- 不需要逐字背诵，能用自己的话说清楚即可。

## 1. Web Technologies

### Q1. What happens when a user types a URL?

简洁答案：

用户输入 URL 后，浏览器先通过 DNS 把域名解析成服务器 IP。然后建立网络连接，HTTPS 场景下还会进行 TLS 握手。之后浏览器发送 HTTP request，请求可能经过 CDN、Load Balancer 或 Reverse Proxy，再到达应用服务器。应用服务器处理请求，必要时访问后端服务、缓存或数据库，然后返回 HTTP response。浏览器解析 HTML、CSS、JavaScript 等资源并渲染页面。

Quick notes:

```text
URL -> DNS -> TCP/TLS -> HTTP request -> CDN / Load Balancer -> App Server -> Backend / DB / Cache -> HTTP response -> Browser rendering
```

### Q2. How does data move between frontend and backend?

简洁答案：

前端和后端通常通过 HTTP/HTTPS request 传输数据。前端调用 API，使用 GET、POST、PUT、DELETE 等方法，请求里可能包含 query parameters、headers、cookies、tokens 或 JSON body。后端接收请求后进行认证、输入校验、业务逻辑处理，并读写数据库或调用其他服务。最后后端返回 HTTP response，通常包含 status code 和 JSON 数据。现代 Web 应用通常使用 fetch/AJAX 进行异步通信，避免整页刷新。

Quick notes:

```text
Frontend -> HTTP/HTTPS request -> API/backend -> auth/input validation -> business logic -> DB/service -> HTTP response -> JSON -> frontend update
```

### Q3. What is HTTP? What is TCP/IP?

简洁答案：

HTTP 是应用层协议，用于客户端和服务器之间的 Web 通信。浏览器发送 HTTP request，服务器返回 HTTP response，里面包含 status code、headers 和 body。TCP/IP 是更底层的网络协议族，负责数据如何在网络中传输。IP 负责地址和路由，TCP 负责可靠传输、顺序控制、重传和错误检查。简单说，TCP/IP 负责把数据送到，HTTP 负责定义 Web 请求和响应的格式与含义。

Quick notes:

```text
TCP/IP = how data is transported across the network
HTTP = how client and server structure web requests and responses
HTTP works on top of TCP/IP
```

### Q4. What is the difference between synchronous and asynchronous processing?

简洁答案：

同步处理是调用方等待处理完成后再继续，适合快速、简单、需要立即返回结果的操作。异步处理是调用方提交任务后不必等待完成，任务可以在后台、队列或事件驱动架构中继续执行。同步方式简单但可能阻塞用户或系统；异步方式更适合长时间、大量、解耦的处理，但需要设计 retry、monitoring、ordering 和 error handling。在数据平台里，batch job、Pub/Sub event processing、background ETL 都是常见异步模式。

Quick notes:

```text
Synchronous = wait for result
Asynchronous = submit task and continue
Sync is simple but can block
Async is scalable but needs retry, monitoring, ordering, and error handling
```

### Q5. How would you troubleshoot a slow web application?

简洁答案：

我会先确认慢在哪里：页面加载、API response、数据库查询、前端渲染还是网络传输。然后查看 metrics 和 logs，例如 browser developer tools、API latency、server logs、error rate、database query time 和 infrastructure metrics。如果是前端问题，检查 JS/CSS 大小、请求数量、图片大小、渲染和缓存。如果是后端问题，检查 API 逻辑、外部服务调用、数据库查询、连接池、CPU/内存。如果是基础设施问题，检查 load balancer、autoscaling、网络延迟、CDN/cache hit rate。核心是不猜，先用数据定位瓶颈，再选择优化方式。

Quick notes:

```text
Clarify where slow
-> browser / network / API / backend / DB / infrastructure
-> check metrics and logs
-> identify bottleneck
-> optimize query, cache, payload, pagination, async processing, or scaling
```

### Q6. How are web services scaled?

简洁答案：

Web service 可以从多层扩展。应用层可以通过 load balancer 后面增加 stateless instances 做 horizontal scaling。缓存层可以用 CDN 缓存静态内容，用 cache 减少后端和数据库压力。数据库层可以优化查询、加 index、read replica、partition 或 sharding。长时间任务可以用 queue/message 做异步处理。还需要 monitoring、autoscaling、rate limiting 和 fault tolerance。扩展前应先定位瓶颈，否则可能增加成本但没有解决真正问题。

Quick notes:

```text
Load balancer -> stateless app instances -> CDN / cache -> DB optimization / replicas / partitioning -> async queue -> autoscaling / monitoring / rate limiting -> fault tolerance
Identify bottleneck before scaling
```

## 2. Databases / SQL

### Q1. Relational vs Non-Relational Databases

简洁答案：

关系型数据库用表、行、列和固定 schema 管理结构化数据，支持 SQL、join、constraints 和 transactions，适合一致性要求高、查询关系复杂的场景。NoSQL 数据库的数据模型更灵活，可以是 document、key-value、wide-column 或 graph，适合数据结构变化快、需要高扩展性或访问模式明确的场景。选择时要看数据结构、查询模式、一致性要求、扩展性和运维需求。

Quick notes:

```text
Relational = tables, schema, SQL, joins, transactions, consistency
NoSQL = flexible model, document/key-value/wide-column/graph, scalable, access-pattern driven
Choose based on data structure, query pattern, consistency, scalability, operations
```

### Q2. What are the different types of SQL JOINs?

简洁答案：

JOIN 用来根据 key 合并多个表。INNER JOIN 只返回两边都匹配的记录。LEFT JOIN 返回左表全部记录和右表匹配记录，右边没有匹配时为 NULL。RIGHT JOIN 与 LEFT JOIN 相反。FULL OUTER JOIN 返回两边全部记录，不匹配的一侧为 NULL。在数据质量检查中，LEFT JOIN 和 FULL OUTER JOIN 常用于比较 source 和 target，找 missing 或 unmatched records。

Quick notes:

```text
INNER = matched records only
LEFT = all left + matched right, missing right as NULL
RIGHT = all right + matched left
FULL OUTER = all records from both sides
Validation = LEFT / FULL OUTER to find missing records
```

### Q3. What is the difference between WHERE and HAVING?

简洁答案：

WHERE 在 aggregation 之前过滤原始行，HAVING 在 GROUP BY 之后过滤聚合结果。例如先用 WHERE 过滤 2026 年交易，再按 customer_id 聚合，最后用 HAVING 找总金额大于 10000 的客户。核心区别是执行时机：WHERE 过滤 rows，HAVING 过滤 groups。

SQL example:

```sql
SELECT
  customer_id,
  SUM(amount) AS total_amount
FROM transactions
WHERE transaction_date >= '2026-01-01'
GROUP BY customer_id
HAVING SUM(amount) > 10000;
```

Quick notes:

```text
WHERE = filter rows before aggregation
HAVING = filter groups after aggregation
```

### Q4. What is a window function?

简洁答案：

Window function 可以在不压缩行数的情况下，对当前行相关的一组数据进行计算。GROUP BY 会把每组压成一行，而 window function 保留每一行，同时增加 rank、running total、previous value、latest record 等计算结果。常见函数有 ROW_NUMBER、RANK、DENSE_RANK、LAG、LEAD、SUM OVER。常用于 deduplication、ranking、latest record、running total。

SQL example:

```sql
SELECT
  customer_id,
  transaction_id,
  amount,
  ROW_NUMBER() OVER (
    PARTITION BY customer_id
    ORDER BY transaction_date DESC
  ) AS rn
FROM transactions;
```

Quick notes:

```text
GROUP BY collapses rows
Window function keeps rows and adds calculated values
ROW_NUMBER = unique sequence
RANK = ties and skips numbers
DENSE_RANK = ties without skipping
LAG/LEAD = previous/next row value
```

### Q5. What is an index in a database?

简洁答案：

Index 是数据库额外维护的数据结构，用来加快查找，避免全表扫描。它对 WHERE、JOIN、ORDER BY 常用列有帮助。代价是占用额外存储，并且 insert/update/delete 时需要维护 index，可能降低写入性能。因此 index 应根据实际 query pattern 设计，不是所有列都建。Iceberg 不同于传统行级 index，它更多依靠 metadata、partition、manifest、file-level statistics 和 data skipping 减少扫描量。

Quick notes:

```text
Index = faster lookup
Good for WHERE / JOIN / ORDER BY
Trade-off = extra storage + slower writes
Choose based on query pattern
Iceberg = metadata / partition / manifest / file stats / data skipping
```

### Q6. What is a transaction in a database? What is ACID?

简洁答案：

Transaction 是一组需要作为一个整体执行的数据库操作。例如转账时，扣款和入账必须一起成功或一起失败。ACID 是可靠 transaction 的四个性质：Atomicity 表示 all or nothing；Consistency 表示数据库从一个合法状态到另一个合法状态；Isolation 表示并发事务之间不能错误干扰；Durability 表示 commit 后即使系统故障结果也应保留。

Quick notes:

```text
Transaction = one unit of work
A = Atomicity: all or nothing
C = Consistency: valid state to valid state
I = Isolation: concurrent transactions do not interfere incorrectly
D = Durability: committed data persists
```

### Q7. What is the difference between OLTP and OLAP?

简洁答案：

OLTP 面向日常交易系统，例如订单、支付、账户更新、客户管理，特点是大量小读写、低延迟、强一致性。OLAP 面向分析、报表、BI 和 dashboard，特点是扫描大量数据、聚合、复杂查询。简单说，OLTP 为 transaction 优化，OLAP 为 analysis 优化。Cloud SQL / AlloyDB 更偏 OLTP，BigQuery 更偏 OLAP。

Quick notes:

```text
OLTP = transactional systems, many small reads/writes, low latency, strong consistency
OLAP = analytics/reporting, large scans, aggregation, complex queries
Cloud SQL / AlloyDB = OLTP
BigQuery = OLAP
```

### Q8. What is normalization and denormalization?

简洁答案：

Normalization 是把数据拆成相关表，减少重复，提高一致性和更新便利性，但查询时可能需要更多 join。Denormalization 是把数据合并成更宽的表，允许适度重复，让读取和报表更简单更快。一般 OLTP 更倾向 normalization，OLAP、data mart、reporting 更常用 denormalized table 或 star schema。

Quick notes:

```text
Normalization = split tables, reduce redundancy, better consistency, more joins
Denormalization = wider tables, some duplication, faster/easier reads
OLTP tends to normalize
OLAP/reporting often denormalizes
```

### Q9. What is a primary key and a foreign key?

简洁答案：

Primary key 是唯一标识表中每一行的列或列组合，例如 customer_id。Foreign key 是引用另一张表 primary key 的列，用于表示表之间关系。Primary key 保证唯一性，foreign key 帮助维护 referential integrity。在数据工程中，理解主键和外键对正确 join、deduplication 和 data quality validation 很重要。

Quick notes:

```text
Primary key = uniquely identifies each row
Foreign key = references primary key in another table
Used for relationships, joins, referential integrity, data validation
```

### Q10. How would you check data quality using SQL?

简洁答案：

我会从几个角度检查数据质量。Completeness：必填字段是否 NULL。Uniqueness：主键或业务 key 是否重复。Consistency：foreign key 是否能匹配 master table，status code 是否有效。Reconciliation：source 和 target 的 row count、missing IDs、aggregated value 是否一致。Business rules：金额不能为负、日期不能在未来、转换后的 status code 是否符合 mapping。

SQL examples:

```sql
SELECT *
FROM transactions
WHERE customer_id IS NULL;

SELECT transaction_id, COUNT(*) AS cnt
FROM transactions
GROUP BY transaction_id
HAVING COUNT(*) > 1;

SELECT s.transaction_id
FROM source_transactions s
LEFT JOIN target_transactions t
  ON s.transaction_id = t.transaction_id
WHERE t.transaction_id IS NULL;
```

Quick notes:

```text
Completeness = NULL / missing fields
Uniqueness = duplicate keys
Consistency = foreign key / valid code
Reconciliation = source vs target count / IDs / aggregates
Business rules = amount, date, status mapping
```

### Q11. How would you remove duplicate records while keeping the latest record?

简洁答案：

先定义表示重复的 business key，例如 transaction_id 或 customer_id。然后用 ROW_NUMBER 按 key 分组，并按 updated_at 降序排序，最新记录得到 rn = 1。最后只保留 rn = 1。这个方法适合 ETL 中处理同一业务记录的多个版本。

SQL example:

```sql
WITH ranked AS (
  SELECT
    *,
    ROW_NUMBER() OVER (
      PARTITION BY transaction_id
      ORDER BY updated_at DESC
    ) AS rn
  FROM transactions
)
SELECT *
FROM ranked
WHERE rn = 1;
```

Quick notes:

```text
Define business key
-> ROW_NUMBER() PARTITION BY key ORDER BY updated_at DESC
-> keep rn = 1
```

### Q12. How would you compare source and target datasets after migration?

简洁答案：

我会分层比较 source 和 target。先比较 table-level row count，也按 date 或业务分类比较 partition-level count。然后检查 key coverage，看 source ID 是否都存在 target，以及 target 是否有多余记录。再比较重要聚合指标，例如 total amount、transaction count、active customer count、status count。还要检查 NULL、duplicate、invalid status、date range 等质量规则。最后让业务用户验证关键报表，因为技术 reconciliation 不一定能发现业务定义差异。

Quick notes:

```text
Compare row counts
-> key coverage
-> missing / extra IDs
-> aggregates
-> data quality rules
-> business report validation
```

### Q13. What is the difference between a CTE and a subquery?

简洁答案：

Subquery 是嵌套在另一个 query 中的查询，可以出现在 SELECT、FROM、WHERE 等位置。CTE 是用 WITH 定义的临时命名结果集。两者很多时候能得到同样结果，但 CTE 更适合复杂、多步骤、需要可读性的 SQL。简单一次性逻辑用 subquery 就够；复杂 transformation 或 data quality check 我更倾向 CTE。

Quick notes:

```text
Subquery = nested query
CTE = named temporary result using WITH
CTE is better for readability and multi-step logic
Subquery is fine for simple logic
```

### Q14. What is the difference between UNION and UNION ALL?

简洁答案：

UNION 合并两个查询结果并去重。UNION ALL 合并两个查询结果但保留重复行。因为 UNION 需要去重，通常会有额外处理，可能比 UNION ALL 慢。如果数据集不会重叠，或分析上需要保留重复记录，用 UNION ALL。如果需要 distinct combined result，用 UNION。

Quick notes:

```text
UNION = combine + remove duplicates
UNION ALL = combine + keep duplicates
UNION ALL is usually faster
Choose based on whether duplicates should be removed
```

### Q15. How would you optimize a slow SQL query?

简洁答案：

我会先看 execution plan，确认瓶颈在哪里。然后检查是否扫描了过多数据、join key 是否正确、filter 是否能提前应用。对 BigQuery 这类分析型数据库，重点是减少 bytes scanned：只选必要列、避免 SELECT *、使用 partition filter、利用 clustering、减少不必要中间结果。还要检查 join 逻辑、aggregation 粒度、重复数据和是否能简化步骤。最后必须验证优化前后结果一致。

Quick notes:

```text
Check execution plan
-> reduce scanned data
-> filter early
-> avoid SELECT *
-> optimize joins
-> use index / partition / clustering
-> simplify aggregation
-> validate same result
```

### Q16. Partitioning vs Clustering

简洁答案：

Partitioning 是按某个粗粒度列把大表分成多个部分，常见是按日期。查询带 partition filter 时，BigQuery 可以跳过不相关 partitions。Clustering 是在表或 partition 内按列组织数据，例如 customer_id、status、region，让查询在已选 partition 内减少扫描。简单说，partitioning 决定扫哪些分区，clustering 减少分区内部扫描量。

Quick notes:

```text
Partitioning = split table by coarse-grained column, usually date
Clustering = organize data within partitions by frequently filtered/grouped columns
Partitioning reduces which partitions are scanned
Clustering reduces data scanned inside selected partitions
```

### Q17. How would you optimize JOIN?

简洁答案：

我会先确认 join key 是否正确，数据类型和格式是否一致。然后在 join 前过滤不需要的行，只选择必要列。如果可能，先 deduplicate 或 pre-aggregate，避免直接 join 两个 raw large tables。还要检查 duplicate keys，因为意外的 many-to-many join 会导致 row count 爆炸。在 BigQuery 中，可以通过 partition filter、clustering 和 query plan 检查 scanned bytes / shuffle。很多慢 join 不只是性能问题，也可能是 data modeling 或 data quality 问题。

Quick notes:

```text
Correct join key
-> same data type / format
-> filter before join
-> select needed columns only
-> deduplicate keys
-> avoid many-to-many explosion
-> pre-aggregate if possible
-> use partition / clustering
-> check query plan
```

### Q18. What is a star schema?

简洁答案：

Star schema 是常见 DWH 建模模式，由中心 fact table 和周围多个 dimension tables 构成。Fact table 存储可度量业务事件，例如 transactions、orders、payments、page views。Dimension tables 存储描述性信息，例如 customer、product、date、region。它适合 OLAP、data mart 和 BI，因为业务用户可以按不同维度分析指标。

Quick notes:

```text
Star schema = fact table + dimension tables
Fact = measurable events / metrics
Dimension = descriptive attributes
Good for OLAP / reporting / BI
```

### Q19. What is the difference between fact table and dimension table?

简洁答案：

Fact table 存储可度量业务事件或指标，通常很大，并随着新事件持续增长。Dimension table 存储用于分析上下文的描述性属性，例如 customer_name、region、product_category。分析时通常 join fact 和 dimension，从不同业务维度分析指标，例如 sales by region、product category、customer segment。

Quick notes:

```text
Fact = measurable events / metrics, usually large
Dimension = descriptive context / attributes, usually smaller
Join fact + dimension to analyze metrics by business attributes
```

### Q20. What is slowly changing dimension?

简洁答案：

SCD 用来处理 dimension 数据随时间变化的问题。Type 1 是直接覆盖旧值，简单但不保留历史。Type 2 是每次变化新增一行，通常带 effective_start_date、effective_end_date、current_flag，能保留历史。历史分析重要时用 Type 2，例如要按当时客户所属 region 分析过去销售。

Quick notes:

```text
SCD = handle dimension changes over time
Type 1 = overwrite, no history
Type 2 = keep history with new rows, date range, current flag
Use Type 2 when historical analysis matters
```

### Q21. What is Big Data?

简洁答案：

Big Data 指数据规模、速度或复杂度超出传统单机系统高效处理能力。常用 3V 描述：Volume 是数据量大，Velocity 是产生或处理速度快，Variety 是格式多样，包括 structured、semi-structured、unstructured。实际系统通常需要 distributed storage 和 distributed processing，例如 Hadoop、Spark、Beam、Dataflow、Dataproc、BigQuery。目标不是只存大量数据，而是可靠处理并转化为业务洞察。

Quick notes:

```text
Big Data = volume + velocity + variety
Too large / fast / complex for traditional single-machine processing
Needs distributed storage and processing
Goal = reliable processing + business insights
```

## 3. AI/ML

### Q1. Supervised vs Unsupervised Learning

简洁答案：

Supervised learning 使用有 label 的数据训练模型，常见任务是 classification 和 regression。Unsupervised learning 使用没有 label 的数据，目标是发现隐藏结构或模式，例如 clustering。简单说，supervised learning 是从“带答案的例子”学习，unsupervised learning 是在没有预定义答案的情况下发现模式。

Quick notes:

```text
Supervised = labeled data, learn from examples with answers
Classification = predict category
Regression = predict number
Unsupervised = no labels, discover patterns
Clustering = group similar records
```

### Q2. Classification vs Regression

简洁答案：

Classification 和 regression 都是 supervised learning。Classification 预测类别或标签，例如 fraud / not fraud、churn / not churn。Regression 预测连续数值，例如 sales amount、traffic volume、latency、house price。核心区别是输出类型：classification 输出 label，regression 输出 number。高速道路项目里，预测交通量更像 regression，预测拥堵等级 low/medium/high 更像 classification。

Quick notes:

```text
Classification = predict category / label
Regression = predict continuous number
Fraud yes/no = classification
Traffic volume = regression
Congestion level = classification
```

### Q3. What is overfitting? How would you prevent it?

简洁答案：

Overfitting 是模型在 training data 上表现很好，但在新数据或 validation data 上表现差。原因是模型学到了训练数据里的噪音或过细模式，而不是通用规律。检测方法是比较 training performance 和 validation performance。预防方法包括增加数据、简化模型、regularization、cross-validation、pruning、dropout、early stopping，并检查 data leakage。

Quick notes:

```text
Overfitting = good on training data, poor on unseen data
Detect = train score high, validation score low
Prevent = more data, simpler model, regularization, cross-validation, pruning, dropout, early stopping
Also check data leakage
```

### Q4. What is the bias-variance tradeoff?

简洁答案：

Bias 和 variance 是两类模型误差。Bias 高表示模型太简单，无法捕捉真实模式，容易 underfitting，训练和验证表现都差。Variance 高表示模型对训练数据太敏感，学到了噪音，容易 overfitting，训练表现好但验证表现差。目标是在模型复杂度上取得平衡，既能学习有用模式，又不过度拟合。

Quick notes:

```text
Bias = model too simple -> underfitting
Variance = model too sensitive -> overfitting
High bias = bad train, bad validation
High variance = good train, bad validation
Goal = balance model complexity
```

### Q5. What is feature engineering?

简洁答案：

Feature engineering 是创建、选择或转换输入变量，让模型更容易学习有效模式。例如交通预测中，可以把 timestamp 转成 hour of day、day of week、holiday flag，也可以加入 weather category 或 road segment 信息。它还包括 missing value 处理、categorical encoding、numeric scaling、derived field 和历史聚合。实际项目中，feature quality 往往和 algorithm 一样重要。

Quick notes:

```text
Feature engineering = create / select / transform input variables
Goal = help model learn useful patterns
Examples = time features, holiday flag, weather, category encoding, aggregation
Feature quality can be as important as algorithm
```

### Q6. How do you evaluate model performance?

简洁答案：

模型评价要看问题类型和业务目标。Classification 常用 accuracy、precision、recall、F1、AUC。数据不平衡时 accuracy 可能误导。Precision 适合 false positive 成本高的场景，recall 适合 false negative 成本高的场景。Regression 常用 MAE、MSE、RMSE、R-squared。实际项目还要看结果是否对业务有用，以及 validation data 是否能反映真实未来数据。

Quick notes:

```text
Classification = accuracy, precision, recall, F1, AUC
Regression = MAE, MSE, RMSE, R-squared
Imbalanced data -> accuracy may be misleading
Precision = avoid false positives
Recall = avoid false negatives
Also check business usefulness and validation design
```

### Q7. Precision vs Recall

简洁答案：

Precision 表示模型预测为 positive 的记录中，有多少是真的 positive。Recall 表示实际 positive 的记录中，模型找到了多少。如果 false positive 成本高，例如误判正常交易为 fraud，会影响用户体验，precision 更重要。如果 false negative 成本高，例如漏掉真正 fraud，会带来业务风险，recall 更重要。F1 score 用来平衡 precision 和 recall。

Quick notes:

```text
Precision = of predicted positives, how many are truly positive
Recall = of actual positives, how many did we find
Precision important when false positives are costly
Recall important when false negatives are costly
F1 balances both
```

### Q8. What is a decision tree?

简洁答案：

Decision tree 是 supervised learning model，通过按 feature 条件一步步分裂数据来预测结果。它像 flowchart，每个节点问一个问题，例如 amount 是否大于某个阈值，最后到 leaf node 输出预测。优点是容易理解和解释，适合业务沟通。缺点是单棵树太深时容易 overfit，可以通过限制 depth、设置 min samples 或使用 random forest 缓解。

Quick notes:

```text
Decision tree = flowchart-like supervised model
Splits data by feature conditions
Leaf node gives prediction
Easy to explain
Risk = overfitting if too deep
Prevent = max depth, min samples, random forest
```

### Q9. What is a random forest?

简洁答案：

Random forest 是由多棵 decision trees 组成的 ensemble model。它用不同数据样本和不同 feature subsets 训练多棵树。Classification 时通过投票决定结果，regression 时取平均。相比单棵 decision tree，random forest 通常更稳定、更不容易 overfit。代价是解释性较差，计算成本更高。

Quick notes:

```text
Random forest = many decision trees
Classification = voting
Regression = averaging
More stable, less overfitting than one tree
Trade-off = less interpretable, more compute
```

### Q10. What is SVM?

简洁答案：

SVM 是 supervised learning algorithm，主要用于 classification，也可用于 regression。核心思想是找到一个能最大化类别间 margin 的边界，也叫 hyperplane。靠近边界的数据点叫 support vectors，它们决定了分类边界。SVM 适合中小规模和高维数据，也可通过 kernel 处理非线性边界，但在大数据上可能较慢且调参较难。

Quick notes:

```text
SVM = supervised model
Finds best separating boundary / hyperplane
Maximizes margin between classes
Support vectors define the boundary
Kernel can handle non-linear patterns
```

### Q11. What is clustering?

简洁答案：

Clustering 是 unsupervised learning，用来把相似数据点分到同一组，不需要预定义 label。常见例子是根据购买行为、使用频率、交易金额对客户分群。K-means 是常见 clustering algorithm，根据到 cluster center 的距离把数据分成 K 组。用途包括 customer segmentation、anomaly detection、pattern discovery 和 EDA。

Quick notes:

```text
Clustering = unsupervised grouping
No predefined labels
Example = customer segmentation
K-means = common clustering algorithm
Use cases = segmentation, anomaly detection, pattern discovery
```

### Q12. CNN vs RNN

简洁答案：

CNN 和 RNN 都是 neural networks，但适合不同数据模式。CNN 主要用于 spatial data，例如图像，擅长检测局部模式，如边缘、形状、对象。RNN 主要用于 sequential data，例如 time series、text、speech，可以利用前面步骤的信息理解当前步骤。现在很多 NLP 和 sequence tasks 会用 Transformer，但 CNN/RNN 仍是重要基础概念。

Quick notes:

```text
CNN = spatial patterns, images, local features
RNN = sequential patterns, time series/text/speech
CNN example = image classification
RNN example = sequence prediction
Modern NLP often uses Transformers
```

### Q13. What is collaborative filtering?

简洁答案：

Collaborative filtering 是 recommendation 技术，根据用户行为模式推荐物品。基本想法是：过去行为相似的用户，未来可能喜欢相似物品。User-based filtering 找相似用户，item-based filtering 根据用户交互找相似物品。常用于商品推荐、电影推荐和内容个性化。

Quick notes:

```text
Collaborative filtering = recommendation based on user behavior
Similar users may like similar items
User-based = find similar users
Item-based = find similar items
Use cases = product/movie/content recommendation
```

### Q14. What is Generative AI and what is an LLM?

简洁答案：

Generative AI 是能生成新内容的 AI，例如文本、图片、代码、音频、摘要。LLM 是面向语言理解和文本生成的大语言模型，是 Generative AI 的一种。LLM 可用于 Q&A、summarization、translation、classification、code generation、customer support、knowledge search 等。但企业使用时要注意 hallucination、data privacy、access control、evaluation、monitoring 和 human review。

Quick notes:

```text
Generative AI = creates new content
LLM = generative AI for language/text
Use cases = Q&A, summary, translation, code, support, knowledge search
Risks = hallucination, data privacy, access control, evaluation
Enterprise use needs grounding + monitoring + human review
```

### Q15. What is MLOps?

简洁答案：

MLOps 是用可靠、可重复的方式管理 ML lifecycle 的实践，包括 data preparation、training、validation、deployment、monitoring、retraining。目标是把实验模型稳定地带到 production。重要组成包括 code/data versioning、experiment tracking、model registry、automated pipelines、performance monitoring、drift detection。模型上线后不是结束，还要持续监控数据变化、性能下降和是否需要 retraining。

Quick notes:

```text
MLOps = DevOps for ML lifecycle
Data -> training -> validation -> deployment -> monitoring -> retraining
Needs versioning, pipelines, model registry, monitoring, drift detection
Model is not done after deployment
```

### Q16. What is model drift?

简洁答案：

Model drift 是模型上线后因为现实数据模式变化，导致性能逐渐下降。Data drift 指输入数据分布变化；concept drift 指 feature 和 target 之间的关系变化。例如用户行为、交通模式、fraud pattern、市场环境改变。应通过监控 input data、prediction results、model performance 来发现 drift，并在必要时 retrain 或更新模型。

Quick notes:

```text
Model drift = model performance degrades over time
Data drift = input distribution changes
Concept drift = relationship between features and target changes
Handle = monitor data, predictions, performance, retrain when needed
```

### Q17. What is data leakage in machine learning?

简洁答案：

Data leakage 是训练时使用了预测时本不可能获得的信息。这会让 validation 看起来很好，但 production 表现很差。例如预测客户下月是否 churn，却使用了 churn 发生后才知道的字段；或 time-series model 中使用未来数据。预防方法包括 time-based split、检查 feature availability、避免 future information，并确保每个 feature 在实际预测时可用。

Quick notes:

```text
Data leakage = training uses information unavailable at prediction time
Looks good in validation, bad in production
Examples = future data, post-event fields
Prevent = time-based split, check feature availability, avoid future information
```

## 4. Platforms & Infrastructure

### Q1. What are the main components of cloud infrastructure?

简洁答案：

Cloud infrastructure 主要包括 compute、storage、networking、security 和 operations。Compute 提供计算能力，例如 VM、container、serverless。Storage 存储不同类型的数据，例如 object、block、file、database、DWH。Networking 连接服务和用户，包括 VPC、subnet、routing、firewall、DNS、load balancer。Security 管理身份、权限、加密、secrets、audit logs。Operations 包括 monitoring、logging、alerting、automation、backup、DR。它们共同支撑 scalable、reliable、secure、maintainable system。

Quick notes:

```text
Cloud infrastructure = compute + storage + networking + security + operations
Compute = VM / container / serverless
Storage = object / block / file / DB / DWH
Networking = VPC / subnet / routing / firewall / DNS / load balancer
Operations = monitoring / logging / alerting / automation / backup / DR
```

### Q2. Horizontal scaling vs Vertical scaling

简洁答案：

Vertical scaling 是增强单台机器能力，例如增加 CPU、memory、disk。Horizontal scaling 是增加多台机器或 instances，并通过 load balancer 分发流量。Vertical scaling 简单，但有硬件上限，也可能形成 single point of failure。Horizontal scaling 更适合高流量和高可用系统，但应用通常需要 stateless 或把 state 放到外部共享存储。云架构中常用 load balancer、autoscaling 和 stateless services 实现 horizontal scaling。

Quick notes:

```text
Vertical scaling = bigger machine
Horizontal scaling = more machines
Vertical = simple but limited
Horizontal = scalable and HA, but needs stateless/shared-state design
Cloud = load balancer + autoscaling + stateless services
```

### Q3. What is a load balancer?

简洁答案：

Load balancer 把 incoming traffic 分发到多个 backend servers 或 service instances。目的主要是提高 availability、scalability 和 reliability。如果某个 backend 不健康，load balancer 可以停止向它发送流量，并把请求转发到健康实例。它通常和 health checks、autoscaling、stateless application services 一起使用。

Quick notes:

```text
Load balancer = distributes traffic across backends
Purpose = availability + scalability + reliability
Uses health checks
Stops sending traffic to unhealthy instances
Works with autoscaling and stateless services
```

### Q4. What is a stateless service?

简洁答案：

Stateless service 不把用户 session 或重要 state 存在某个特定 application instance 内。每个请求要么包含处理所需信息，要么 state 存在外部，例如 database、cache、object storage。这样任何 instance 都能处理任何请求，某个 instance 故障时其他 instance 可以继续服务。它非常适合 load balancing、autoscaling 和 failure recovery。

Quick notes:

```text
Stateless = no important state stored inside one app instance
State stored externally = DB / cache / storage
Any instance can handle any request
Good for load balancing, autoscaling, failure recovery
```

### Q5. VM vs Container vs Serverless

简洁答案：

VM 提供完整虚拟机和独立 OS，控制度高，但需要管理 OS patching、capacity 等更多基础设施。Container 把应用和依赖打包，在共享 host OS 上运行，比 VM 更轻，适合一致部署和 microservices。Serverless 让我们无需直接管理服务器，云厂商负责基础设施、扩缩容和大量运维。核心 trade-off 是 control 和 operational simplicity：VM 控制强，container 便携高效，serverless 运维负担低但可能有 runtime、cold start、平台限制。

Quick notes:

```text
VM = full virtual machine, more control, more ops
Container = app + dependencies, lightweight, portable
Serverless = no server management, provider handles scaling/ops
Trade-off = control vs operational simplicity
```

### Q6. Block storage vs File storage vs Object storage

简洁答案：

Block storage 像磁盘一样按 block 存储，通常挂载到 VM，适合 OS、database、低延迟磁盘访问。File storage 用文件夹结构存储，可通过 NFS/SMB 被多个客户端共享，适合 shared file system。Object storage 把数据作为 object 和 metadata 存在扁平 namespace 中，通过 API 访问，扩展性强、成本低，适合 data lake、backup、logs、images、raw data。GCP 中 Persistent Disk 是 block，Filestore 是 file，Cloud Storage 是 object。

Quick notes:

```text
Block = disk-like storage for VMs / DBs
File = shared file system, folders, NFS/SMB
Object = scalable API-based storage with metadata
Persistent Disk = block
Filestore = file
Cloud Storage = object
```

### Q7. What is monitoring, logging, and alerting?

简洁答案：

Monitoring 是收集和观察系统指标，例如 CPU、memory、latency、error rate、throughput、job status。Logging 是收集更详细的事件记录，例如 error logs、request logs、job logs、audit logs。Alerting 是当指标或日志显示异常时通知团队，例如 high error rate、failed jobs、high latency、resource exhaustion。Monitoring 告诉我们发生了什么，logging 帮助解释为什么，alerting 帮助快速响应。

Quick notes:

```text
Monitoring = metrics / system health
Logging = detailed event records
Alerting = notify when something is wrong
Monitoring tells what
Logging helps explain why
Alerting helps respond quickly
```

### Q8. What is distributed system?

简洁答案：

Distributed system 是由多个 computers 或 services 协同工作、对外表现为一个系统的架构。例如 Web 应用可能包含多个 app servers、databases、caches、message queues、storage services。优点是更容易扩展和提高可用性。代价是复杂度更高，需要处理 network failure、partial failure、retry、consistency、latency、monitoring 和服务间协调。云系统通常天然是 distributed，所以要从一开始考虑 failure handling、observability 和 service boundaries。

Quick notes:

```text
Distributed system = multiple machines/services working together
Benefits = scalability + availability
Challenges = network failure, partial failure, retry, consistency, latency, observability
Cloud systems are distributed by design
```

### Q9. What is retry and why can retry be dangerous?

简洁答案：

Retry 是操作失败后再次尝试，适合临时故障，例如网络错误、timeout、短暂服务不可用。但 retry 如果设计不好会很危险。过多 retry 会增加系统负载，让故障更严重，形成 retry storm。对于非 idempotent 操作，retry 还可能造成 duplicate writes。更安全的 retry 需要 retry limit、exponential backoff、jitter、timeout，以及必要时使用 idempotency key。对数据管道来说，还要避免重复加载数据或重复 side effects。

Quick notes:

```text
Retry = try again after failure
Useful for temporary failures
Risks = overload, retry storm, duplicate writes
Safer retry = limit + exponential backoff + jitter + timeout + idempotency
Data pipeline risk = duplicate loading
```

### Q10. What is idempotency?

简洁答案：

Idempotency 指同一个操作执行一次和执行多次，最终结果是一样的。它在 distributed system、API、retry、data pipeline 里非常重要。

比如，一个 API request 因为 timeout 被 retry。如果这个操作不是 idempotent，可能会创建两笔订单或重复写入数据。如果设计成 idempotent，就算 retry 多次，也只会产生一次有效结果。

常见做法是使用 idempotency key、唯一业务 key、去重逻辑，或者在写入前检查目标记录是否已经存在。

在数据管道中，idempotency 很重要，因为 ETL job 可能失败后重跑。如果没有 idempotent 设计，可能造成重复加载、重复聚合或错误数据。

Quick notes:

```text
Idempotency = same operation repeated multiple times has the same final result
Important for retry, APIs, distributed systems, ETL jobs
Risk without idempotency = duplicate orders / duplicate writes / duplicate loads
Solutions = idempotency key, unique business key, deduplication, check-before-write
Data pipeline = rerun should not create duplicate data
```

### Q11. What is the difference between high availability and disaster recovery?

简洁答案：

High Availability, or HA, 是为了让系统在常见故障下仍然持续服务。比如某个 instance、zone、service component 出问题时，系统可以通过 redundancy、load balancing、health check、autoscaling 等方式继续运行。

Disaster Recovery, or DR, 是为了应对更大范围的故障，例如整个 region 故障、严重数据损坏、重大灾害。DR 关注如何恢复系统和数据。

简单说，HA 关注“系统不中断或少中断”，DR 关注“灾难发生后如何恢复”。

常见 DR 指标有 RTO 和 RPO。

RTO, Recovery Time Objective, 表示系统发生故障后，目标是在多长时间内恢复服务。

RPO, Recovery Point Objective, 表示系统发生故障后，最多可以接受丢失多久的数据。

Quick notes:

```text
HA = High Availability, keep system running during common failures
DR = Disaster Recovery, recover system after major disaster
RTO = Recovery Time Objective, how fast to recover
RPO = Recovery Point Objective, how much data loss is acceptable
HA tools = redundancy, load balancing, health checks, autoscaling
DR tools = backup, replication, multi-region, failover
```

### Q12. What is autoscaling?

简洁答案：

Autoscaling 是根据 workload 自动增加或减少计算资源的机制。

比如，当 CPU 使用率、request count、queue length 或其他指标升高时，系统可以自动增加 instances 来处理更多流量。当流量下降时，再减少 instances 来节省成本。

Autoscaling 的好处是提高 scalability 和 cost efficiency。系统可以在高峰期扩容，在低峰期缩容。

但 autoscaling 也需要合理设计。比如 scaling policy、最小/最大实例数、启动时间、健康检查、冷启动、以及是否会影响 stateful workload。

通常 autoscaling 会和 load balancer、stateless services、monitoring 一起使用。

Quick notes:

```text
Autoscaling = automatically add/remove resources based on workload
Scale out when traffic/CPU/queue increases
Scale in when demand decreases
Benefits = scalability + cost efficiency
Needs = scaling policy, min/max instances, health checks, startup time
Works well with load balancer + stateless services + monitoring
```

### Q13. What is infrastructure automation?

简洁答案：

Infrastructure automation 是用代码或自动化工具来创建、配置、部署和管理基础设施，而不是手动操作。

比如可以用 Terraform 管理 cloud resources，用 CI/CD pipeline 部署配置，用脚本或 serverless function 做定期运维任务。

它的好处是减少手工错误，提高可重复性、一致性和可审计性，也方便在不同环境中快速创建相同架构。

但自动化也需要版本管理、review、权限控制和测试，否则错误的自动化可能会快速影响很多资源。

在我的项目里，Lambda 定期处理 EC2 snapshot 或 server 参数更新，也是一种 operations automation 的例子。

Quick notes:

```text
Infrastructure automation = manage infrastructure with code/tools
Examples = Terraform, CI/CD, scripts, serverless operations
Benefits = repeatable, consistent, auditable, fewer manual errors
Needs = version control, review, permission control, testing
Project example = Lambda for EC2 operational automation
```

### Q14. What is a VPC?

简洁答案：

VPC, Virtual Private Cloud, 是在 cloud 里创建的逻辑隔离网络环境。

它让我们可以定义自己的 network range、subnets、routing、firewall rules，并控制哪些资源可以互相通信，哪些资源可以访问外部网络。

在 GCP 里，VPC 可以连接 Compute Engine、GKE、Cloud SQL 等资源。通过 firewall rules、routes、private IP、Cloud NAT、VPN 或 Interconnect，可以控制安全访问和混合云连接。

简单说，VPC 是 cloud resources 的网络边界和通信基础。

Quick notes:

```text
VPC = Virtual Private Cloud
Logical isolated network in cloud
Controls IP ranges, subnets, routes, firewall rules
Connects cloud resources securely
Supports private IP, VPN, Interconnect, Cloud NAT
Network foundation for cloud architecture
```

### Q15. What is the difference between subnet, route, and firewall rule?

简洁答案：

Subnet 是 VPC 里的 IP 地址范围，用来放置 cloud resources，例如 VM 或 GKE nodes。它决定资源属于哪个网络段。

Route 决定网络流量应该往哪里走。它通常根据 destination IP 决定下一跳，例如去 Internet、VPN、NAT、另一个 VPC，还是本地网络。

Firewall rule 决定哪些流量被允许或拒绝。它通常基于 source、destination、port、protocol、direction、target tags 或 service accounts 控制访问。

本质区别是：

```text
Route controls reachability path.
Firewall controls access permission.
```

也就是说，route 决定“怎么走”，firewall 决定“能不能过”。两者都需要。即使 route 存在，如果 firewall deny，流量也不能访问目标；反过来，即使 firewall allow，如果没有 route，流量也到不了目标。

Quick notes:

```text
Subnet = IP range for resources
Route = where traffic should go
Firewall rule = allow or deny traffic
Subnet = placement
Route = path / reachability
Firewall = access control / permission
Route makes destination reachable
Firewall decides whether traffic is allowed
```

### Q16. What is the difference between public IP and private IP?

简洁答案：

Public IP 是可以从 Internet 访问的 IP 地址，通常用于对外公开服务，例如 public web server、load balancer、API endpoint。

Private IP 是只在私有网络内部使用的 IP 地址，不能直接从 Internet 访问。它通常用于 VPC 内部资源之间通信，例如 app server 访问 database。

使用 private IP 可以减少暴露面，提高安全性。对外访问通常通过 load balancer、NAT、VPN、Interconnect 或 bastion 等方式控制。

简单说，public IP 面向外部网络，private IP 面向内部通信。

Quick notes:

```text
Public IP = reachable from Internet
Private IP = internal network only
Public = external-facing service
Private = internal communication
Private IP reduces exposure
External access should be controlled through LB / NAT / VPN / Interconnect / bastion
```

### Q17. What is NAT?

简洁答案：

NAT, Network Address Translation, 是一种把内部 private IP 转换成外部 public IP 访问外部网络的机制。

常见场景是：VM 没有 public IP，但需要访问 Internet 下载 package、调用外部 API 或发送更新请求。这时可以通过 NAT 出站访问 Internet，而不需要给 VM 分配 public IP。

NAT 的好处是减少资源直接暴露在 Internet 上，提高安全性，同时保留 outbound access。

在 GCP 中，Cloud NAT 可以让 private VM 访问 Internet，但外部 Internet 不能直接主动访问这些 VM。

Quick notes:

```text
NAT = Network Address Translation
Private IP -> public IP for outbound access
Allows private resources to access Internet
Does not expose VM directly to inbound Internet traffic
GCP example = Cloud NAT
Use case = private VM downloads packages / calls external API
```

### Q18. What is VPN and Interconnect?

简洁答案：

VPN 和 Interconnect 都可以用来连接 on-premises network 和 cloud network。

VPN, Virtual Private Network, 是通过 Internet 建立加密隧道，把企业内部网络和 cloud VPC 连接起来。它相对容易设置，成本较低，适合中小规模连接、测试环境、backup connection。

Interconnect 是专用网络连接，不走普通 Internet。它提供更稳定、更高带宽、更低延迟的连接，适合大规模生产系统、数据中心和 cloud 之间的大量数据传输。

简单说，VPN 更灵活、成本低，但性能和稳定性受 Internet 影响。Interconnect 更稳定高性能，但成本和设置复杂度更高。

Quick notes:

```text
VPN = encrypted tunnel over Internet
Interconnect = dedicated private connection
VPN = easier, cheaper, flexible
Interconnect = higher bandwidth, lower latency, more stable
Use VPN for smaller / backup / test connections
Use Interconnect for large production workloads and heavy data transfer
```

### Q19. What is the difference between vertical partitioning and horizontal partitioning?

简洁答案：

Horizontal partitioning 是按“行”拆分数据。比如按日期、region、customer_id range 把同一张大表的不同记录分到不同 partition。常用于减少扫描范围、提高查询效率和管理大表。

Vertical partitioning 是按“列”拆分数据。比如把经常访问的基础字段放在一张表，把很少访问的大字段或敏感字段放到另一张表。这样可以减少不必要的列读取，也可以隔离敏感数据。

简单说，horizontal partitioning 是按 rows 拆，vertical partitioning 是按 columns 拆。

在 BigQuery 里，常见 partitioning 多数指 horizontal partitioning，例如按 date partition。选择 partition key 时要看查询过滤条件和数据分布。

Quick notes:

```text
Horizontal partitioning = split by rows
Examples = date, region, customer_id range
Vertical partitioning = split by columns
Examples = frequently used columns vs rarely used / sensitive columns
BigQuery partitioning usually means horizontal partitioning
Choose based on query pattern and data distribution
```

### Q20. What is caching, and when would you use it?

简洁答案：

Caching 是把经常访问或计算成本高的数据临时或预先保存起来，让后续请求或查询可以更快返回，而不必每次都访问原始数据或重新计算。

常见 Web / system cache 包括 browser cache、CDN cache、application cache、database query cache、Redis / Memorystore 这类 in-memory cache。

在数据领域，cache 也很常见。例如 BI dashboard cache、BigQuery query result cache、materialized view、aggregate / summary table、feature store、Lakehouse metadata / file statistics。

例如，把 transaction_detail 预聚合成 daily_customer_summary，本质上就是一种 precomputed result，可以降低查询延迟和计算成本。

缓存适合读取频繁、计算成本高、允许一定 freshness delay 的数据。代价是可能出现 stale data，因此需要设计 TTL、refresh timing、cache invalidation 和 consistency 策略。

Quick notes:

```text
Cache = store frequently used or expensive-to-compute data
Goal = reduce latency and processing cost
Web examples = browser cache, CDN, app cache, Redis/Memorystore
Data examples = BI dashboard cache, query result cache, materialized view, summary table, feature store, lakehouse metadata
Trade-off = faster query + lower cost vs freshness + consistency
Need TTL / refresh / invalidation strategy
```

### Q21. What is the difference between queue and Pub/Sub?

简洁答案：

Queue 通常用于 task processing，也就是“有一个任务需要被处理”。一个 message 通常由一个 worker 消费，多个 worker 可以竞争消费 queue 里的任务。它适合后台任务、异步 job、削峰填谷、控制处理速度和失败 retry。

Pub/Sub 通常用于 event distribution，也就是“某个事件发生了”。Publisher 发布 event，subscriber 订阅 event。一个 event 可以被多个 subscribers 消费，publisher 不需要知道下游有哪些系统。它适合事件驱动架构、系统解耦、多个下游通知、streaming 或 near real-time processing。

简单说，queue 更像“请处理这个任务”，Pub/Sub 更像“这个事件发生了，感兴趣的系统可以各自处理”。

Quick notes:

```text
Queue = task processing
One message -> one worker
Good for background jobs, retry, rate control

Pub/Sub = event distribution
One event -> multiple subscribers
Good for decoupling, event-driven systems, streaming

Queue = do this task
Pub/Sub = this event happened
```

### Q22. What is the difference between throughput and latency?

简洁答案：

Latency 是单个请求或任务从开始到完成需要多长时间，也就是“等多久”。

Throughput 是系统在单位时间内能处理多少请求、任务或数据量，也就是“处理多少”。

例如，一个 API response time 是 200ms，这是 latency。一个系统每秒能处理 1000 个 requests，这是 throughput。

在数据平台里，一个 batch job 跑完需要 30 分钟，这是 latency；系统每小时能处理 1TB 数据，这是 throughput。

优化时要看业务目标。有些场景更关注低 latency，例如实时 API 或 fraud detection。有些场景更关注高 throughput，例如 batch ETL 或大规模数据处理。

Quick notes:

```text
Latency = how long one request/task takes
Throughput = how many requests/tasks/data volume per time unit
API 200ms response = latency
1000 requests/sec = throughput
Batch job duration = latency
1TB/hour processing = throughput
Optimize based on business goal
```

### Q23. What is a protocol? Can you give examples?

简洁答案：

Protocol 是系统之间通信时遵守的一组规则。它定义了数据如何发送、接收、解释，以及双方如何建立连接或处理错误。

常见例子：

- HTTP / HTTPS 用于 Web request 和 response。
- TCP 用于可靠传输，保证顺序、重传和错误检查。
- IP 用于寻址和路由。
- DNS 用于把 domain name 解析成 IP。
- TLS 用于加密通信。
- SSH 用于安全远程登录。
- JDBC / ODBC 用于应用连接数据库。

简单说，protocol 是系统之间“说话的规则”。

Quick notes:

```text
Protocol = communication rules between systems
Defines how data is sent, received, interpreted
HTTP/HTTPS = web communication
TCP = reliable transport
IP = addressing and routing
DNS = domain to IP
TLS = encryption
SSH = secure remote login
JDBC/ODBC = database connectivity
```

### Q24. Cloud Run vs GKE vs Compute Engine

简洁答案：

Compute Engine 是 VM。它给我们最多控制权，可以管理 OS、runtime、network、installed software，适合需要自定义环境、传统应用、或者需要完整 VM 控制的 workload。但运维负担也最大。

GKE 是 managed Kubernetes。它适合运行 containerized applications、microservices，以及需要复杂 orchestration、service discovery、rolling deployment、autoscaling 的场景。它比 VM 更适合现代容器平台，但也需要 Kubernetes 运维知识。

Cloud Run 是 serverless container platform。我们只需要提供 container image，Cloud Run 负责运行、扩缩容和大部分基础设施管理。它适合 stateless service、API、event-driven service、轻量 backend。运维负担最低，但控制度比 GKE 和 VM 少。

选择时可以这样判断：

```text
Need full OS control -> Compute Engine
Need Kubernetes / complex microservices orchestration -> GKE
Need simple stateless container with low operations -> Cloud Run
```

Quick notes:

```text
Compute Engine = VM, most control, most ops
GKE = managed Kubernetes, container orchestration, microservices
Cloud Run = serverless container, stateless, low ops
Full OS control -> Compute Engine
Complex Kubernetes platform -> GKE
Simple stateless service/API -> Cloud Run
```

### Q25. How would you design monitoring for a data pipeline?

简洁答案：

我会从 pipeline 的业务目标和 SLA 开始设计 monitoring。首先要监控 job 是否成功执行，例如 success / failure、retry count、execution time、schedule delay。

然后监控数据量和数据质量，例如 input row count、output row count、null count、duplicate count、schema change、source-target reconciliation。

还要监控性能和资源，例如 processing time、queue backlog、CPU / memory、Dataflow worker status、BigQuery slot usage 或 query cost。

对于告警，我会区分 severity。比如 job failed、数据量突然为 0、关键字段 NULL 激增、延迟超过 SLA，应该触发 alert。

最后，monitoring 不只是发现失败，还要帮助定位问题，所以需要 structured logs、error message、run ID、source file name、table name、execution timestamp 等信息。

Quick notes:

```text
Pipeline monitoring:
Job status = success / failure / retry / duration / delay
Data quality = row count / null / duplicate / schema / reconciliation
Performance = processing time / backlog / CPU / memory / cost
Alert = failed job / zero rows / SLA delay / quality anomaly
Logs = run ID / error message / source file / table / timestamp
Goal = detect issue + diagnose root cause
```

项目连接：

```text
Furusato = Slack alarm, DDL diff check, DAG management
Gaming = Airflow workflow monitoring
PayPay = data validation and downstream feedback loop
```

### Q26. How would you handle failure in a distributed system?

简洁答案：

在 distributed system 里，failure 是正常情况，不是例外。所以设计时要假设 network、service、database、message queue、worker 都可能失败。

首先要做 failure detection，例如 health check、timeout、monitoring、logging、alerting。

然后要做 isolation 和 graceful degradation。比如某个非核心服务失败时，不应该拖垮整个系统，可以返回 fallback、降级功能或暂时跳过非关键处理。

对于临时失败，可以使用 retry，但要加 retry limit、exponential backoff、jitter，避免 retry storm。对于写入操作，要保证 idempotency，避免重复写入。

对于异步系统，要使用 queue 或 Pub/Sub 做 decoupling，并设计 dead-letter queue 来保存无法处理的消息。

最后要有 observability 和 recovery plan，包括 logs、metrics、tracing、runbook、backup、failover 和 disaster recovery。

Quick notes:

```text
Assume failure will happen
Detect = health check / timeout / monitoring / alerting
Isolate = avoid cascading failure
Degrade = fallback / skip non-critical function
Retry = limit + backoff + jitter
Write safety = idempotency
Async = queue / Pub/Sub / dead-letter queue
Recovery = logs / metrics / runbook / backup / failover / DR
```

## 5. Application Modernization

### Q1. What is application modernization?

简洁答案：

Application modernization 是把 legacy application 或 legacy system 改造成更适合当前业务和技术需求的架构。

它不只是把旧系统搬到 cloud，也不只是换一个技术栈。更重要的是改善系统的 scalability、reliability、maintainability、security、development speed 和 operational efficiency。

Modernization 可以包括很多方式，例如 rehost、replatform、refactor、拆分 monolith、引入 containers、serverless、CI/CD、observability，或者把传统 batch / DWH 迁移到现代 cloud data platform。

但 modernization 一定要结合业务目标和风险控制。不是所有系统都需要彻底重写，有些系统只需要逐步迁移或优化。

面试核心句：

```text
Modernization is not just replacing old technology. It is about improving business agility, scalability, reliability, maintainability, and operations while controlling migration risk.
```

Quick notes:

```text
Application modernization = improve legacy systems for current business/technical needs
Not just lift-and-shift
Goals = scalability, reliability, maintainability, security, development speed, operational efficiency
Methods = rehost, replatform, refactor, containers, serverless, CI/CD, observability
Need business goal + risk control
```

### Q2. Why do companies modernize legacy systems?

简洁答案：

公司 modernize legacy systems 通常不是因为旧系统“不能用”，而是因为它们逐渐无法满足新的业务和技术需求。

常见原因包括：扩展性不足、运维成本高、开发速度慢、系统依赖复杂、人才难找、监控和自动化不足、安全和合规要求提高，以及难以支持新的数据分析或 AI 用例。

对客户来说，modernization 的目标通常是降低长期成本、提高可靠性、提升开发效率、加快业务变化响应速度，并让数据和系统更容易被利用。

但 modernization 也有风险，例如业务中断、数据不一致、迁移成本、用户影响、旧逻辑理解不足。所以需要分阶段推进，而不是一次性全部替换。

Quick notes:

```text
Why modernize:
scalability limits
high operation cost
slow development
complex dependencies
hard-to-maintain legacy skills
weak monitoring / automation
security / compliance needs
new analytics / AI requirements

Goal = lower long-term cost, better reliability, faster delivery, better data usage
Risk = downtime, data inconsistency, migration cost, user impact
Need phased approach
```

### Q3. What is the difference between rehost, replatform, and refactor?

简洁答案：

这三个是 cloud migration / modernization 的常见方式，改动程度不同。

Rehost 也叫 lift-and-shift，就是尽量不改应用架构，把系统搬到 cloud 上运行。例如把 on-prem VM 搬到 cloud VM。优点是快、风险相对低；缺点是无法充分利用 cloud-native 能力。

Replatform 是做一些有限改造，让系统更适合 cloud，但不彻底重写。例如把自建数据库换成 managed database，或者把 batch job 调整到 managed workflow 上。它比 rehost 更能提升运维效率。

Refactor 是更深度地修改应用架构或代码，例如拆分 monolith、改成 microservices、serverless、event-driven architecture。它收益最大，但成本、时间和风险也最高。

选择时要看业务目标、时间、预算、风险、系统复杂度和团队能力。

Quick notes:

```text
Rehost = lift-and-shift, minimal change, fast but limited cloud benefit
Replatform = limited changes, use managed services, better operations
Refactor = redesign architecture/code, cloud-native, high benefit but high cost/risk
Choose based on business goal, risk, time, budget, complexity, team skill
```

### Q4. How would you migrate a legacy system to the cloud?

简洁答案：

我会先从业务目标和现状调查开始，而不是直接选择 cloud service。

第一步是确认业务目标：为什么要迁移，是为了降低成本、提高扩展性、改善运维、提升开发速度，还是支持数据分析。

第二步是盘点现有系统：application、database、batch job、interfaces、dependencies、users、SLA、security requirement、data flow。

第三步是识别风险和约束，例如 downtime 允许时间、数据一致性、legacy logic、性能、合规、预算和团队技能。

第四步是设计 target architecture，并选择迁移方式：rehost、replatform、refactor，或者组合使用。

第五步是分阶段迁移，先从低风险模块或非关键 workload 开始，进行 parallel run、data validation、user acceptance test。

最后再逐步 cutover，并准备 rollback plan、monitoring、runbook 和 post-migration optimization。

Quick notes:

```text
Migration approach:
1. Clarify business goal
2. Assess current system
3. Identify dependencies, risks, constraints
4. Design target architecture
5. Choose rehost / replatform / refactor
6. Migrate in phases
7. Parallel run + validation + UAT
8. Cutover with rollback plan
9. Monitoring + post-migration optimization
```

### Q5. How would you reduce migration risk?

简洁答案：

降低 migration risk 的核心是不要一次性全部切换，而是分阶段、可验证、可回滚地迁移。

首先要充分理解现有系统，包括 dependencies、batch schedule、data flow、business logic、SLA、downstream users 和关键 report。

然后按风险拆分迁移范围，从低风险或非核心 workload 开始，逐步扩大范围。

迁移过程中要做 parallel run，让旧系统和新系统同时运行一段时间，并比较结果，例如 row count、key coverage、aggregated metrics、business report output。

还要准备 rollback plan。如果 cutover 后出现严重问题，需要能快速切回旧系统或暂停迁移。

最后要有 monitoring 和 stakeholder communication，确保技术团队和业务用户都知道迁移状态、风险、验证结果和 cutover plan。

Quick notes:

```text
Reduce migration risk:
understand current system
identify dependencies
start with low-risk scope
phased migration
parallel run
data validation
business user validation
rollback plan
monitoring
stakeholder communication
```

### Q6. What is phased migration?

简洁答案：

Phased migration 是把迁移拆成多个阶段，而不是一次性全部迁移。

比如可以先迁移非核心系统、低风险 data mart、某个业务 domain、只读 report，或者一部分用户。验证稳定后，再逐步扩大范围。

它的好处是降低风险，因为每个阶段的影响范围较小，问题更容易定位和修复。也可以让团队逐步积累经验，减少一次性 cutover 的压力。

在 data migration 中，phased migration 常和 parallel run、data validation、business user confirmation 一起使用。

缺点是迁移周期可能更长，并且在一段时间内需要维护新旧系统并行。

Quick notes:

```text
Phased migration = migrate in multiple stages
Start with low-risk scope
Examples = non-critical system, one data mart, one domain, read-only reports
Benefits = lower risk, easier validation, easier rollback, team learning
Trade-off = longer timeline, temporary dual maintenance
Works with parallel run + validation + user confirmation
```

### Q7. What is parallel run?

简洁答案：

Parallel run 是在迁移或系统切换期间，让旧系统和新系统同时运行一段时间。

目的不是长期维护两套系统，而是验证新系统是否能产生和旧系统一致或可接受的结果。

在数据迁移中，parallel run 可以比较 row count、key coverage、aggregated metrics、data quality rules 和关键 report output。

如果结果一致，业务用户也确认没问题，就可以逐步 cutover 到新系统。

Parallel run 的好处是降低切换风险。缺点是短期内运维成本更高，因为需要同时维护新旧系统，并处理结果差异分析。

Quick notes:

```text
Parallel run = old and new systems run at the same time
Goal = validate new system before cutover
Compare = row count, keys, aggregates, data quality, reports
Benefit = reduce cutover risk
Trade-off = temporary dual maintenance and reconciliation effort
```

### Q8. What is a rollback plan?

简洁答案：

Rollback plan 是在 migration、deployment 或 cutover 出现严重问题时，如何安全回到之前稳定状态的计划。

它应该在上线前准备好，而不是出问题后再想。

一个好的 rollback plan 需要明确：什么情况下触发 rollback，谁来做决策，具体回滚步骤是什么，数据如何恢复，如何通知 stakeholders，以及回滚后如何验证系统恢复正常。

在数据迁移中，rollback plan 还要考虑数据一致性。如果新系统已经写入或处理了部分数据，需要决定是丢弃、回补、重新处理，还是同步回旧系统。

Rollback plan 的目标不是鼓励回滚，而是控制风险，让团队知道最坏情况下如何恢复服务。

Quick notes:

```text
Rollback plan = how to return to previous stable state
Prepare before cutover
Define trigger, owner, steps, data recovery, communication, validation
Data migration risk = partial writes / inconsistent data
Goal = control risk and restore service safely
```

### Q9. What is monolith vs microservices?

简洁答案：

Monolith 是把多个功能模块放在一个整体应用里，一起开发、部署和扩展。它的优点是架构简单、开发和调试相对容易，适合小团队、早期产品或业务复杂度不高的系统。

Microservices 是把系统拆成多个小服务，每个服务负责一个相对独立的业务能力，可以独立开发、部署和扩展。它适合大型系统、多个团队协作、不同模块有不同扩展需求的场景。

但 microservices 不是一定更好。它会增加分布式系统复杂度，例如 service communication、observability、deployment、data consistency、network failure、API versioning。

所以选择时要看业务规模、团队能力、系统复杂度和运维成熟度。

Quick notes:

```text
Monolith = one application, one deployment unit
Pros = simple, easier development/debugging
Good for small teams / simple systems

Microservices = many independent services
Pros = independent deployment/scaling, team ownership
Good for large systems / complex domains

Trade-off = distributed complexity, observability, data consistency, network failure
Choose based on scale, team, complexity, operations maturity
```

### Q10. When would you choose microservices, and when would you avoid them?

简洁答案：

我会在系统规模较大、业务 domain 边界清楚、多个团队需要独立开发和部署、不同模块有不同扩展需求时考虑 microservices。

比如支付、用户、订单、通知、分析这些模块如果责任边界清楚，而且团队也能独立负责，就比较适合拆成服务。

但如果系统还比较小、业务还在快速变化、团队规模不大、DevOps 和 observability 不成熟，我会避免过早拆成 microservices。

因为 microservices 会带来额外复杂度，包括服务间通信、distributed transaction、data consistency、monitoring、deployment、API versioning 和故障排查。

所以不是为了“现代化”就一定要拆微服务，而是要看业务和组织是否真的需要、是否有能力运维。

Quick notes:

```text
Choose microservices when:
clear domain boundaries
large system
multiple teams
independent deployment needed
different scaling needs

Avoid microservices when:
small system
unclear domain
small team
low DevOps maturity
weak observability

Risk = distributed complexity
Do not split just because it sounds modern
```

### Q11. What is CI/CD?

简洁答案：

CI/CD 是现代软件交付中的自动化实践。

CI, Continuous Integration, 是开发者频繁合并代码后，系统自动运行 build、test、lint、code check，尽早发现问题。

CD 可以指 Continuous Delivery 或 Continuous Deployment。Continuous Delivery 是代码通过测试后，随时可以部署到生产环境，但通常需要人工批准。Continuous Deployment 是通过测试后自动部署到生产环境。

CI/CD 的目标是提高交付速度、减少手工错误、提高质量和可重复性。

在 cloud migration 或 modernization 项目中，CI/CD 很重要，因为它可以让应用、数据 pipeline、DAG、SQL transformation 和 infrastructure change 更可控、更容易 review 和 rollback。

Quick notes:

```text
CI = Continuous Integration
Auto build / test / lint / code check after code merge

CD = Continuous Delivery or Continuous Deployment
Delivery = ready to deploy, may need approval
Deployment = automatically deploy to production

Goal = faster delivery, fewer manual errors, better quality, repeatability
Useful for app code, data pipelines, DAGs, SQL transformations, infrastructure
```

### Q12. What is DevOps?

简洁答案：

DevOps 是一种把 development 和 operations 更紧密结合的文化和实践，目标是更快、更稳定地交付系统。

它不是单一工具，而是一组实践，包括 CI/CD、infrastructure as code、automated testing、monitoring、logging、incident response、feedback loop。

传统模式下，开发团队只负责写代码，运维团队负责部署和运行，容易出现交接和责任边界问题。DevOps 强调共同负责系统从开发到运行的整个 lifecycle。

在 modernization 项目中，DevOps 很重要，因为 cloud-native system 需要自动化、可观测性、快速发布和快速恢复能力。

Quick notes:

```text
DevOps = development + operations collaboration
Goal = faster and more reliable delivery
Practices = CI/CD, IaC, testing, monitoring, logging, incident response
Shared responsibility across lifecycle
Important for cloud-native modernization
```

### Q13. What is SRE?

简洁答案：

SRE, Site Reliability Engineering, 是把 software engineering 的方法应用到 operations 上，用工程化方式提高系统可靠性。

SRE 关注系统是否可靠运行，而不只是能不能部署。它会使用 SLI、SLO、error budget、monitoring、alerting、incident response、automation 等方法来管理 reliability。

和传统运维相比，SRE 更强调自动化、可观测性、故障复盘和用数据衡量可靠性。

在 cloud 或 data platform 中，SRE 思维很重要，因为系统不仅要能运行，还要能稳定运行、快速发现问题、快速恢复。

Quick notes:

```text
SRE = Site Reliability Engineering
Apply software engineering to operations
Focus = reliability, automation, observability, incident response
Key concepts = SLI, SLO, error budget
Goal = reliable system with measurable targets
```

### Q14. What is observability?

简洁答案：

Observability 是系统通过外部信号让我们理解内部状态的能力。

它不只是 monitoring。Monitoring 通常告诉我们“系统是否正常”，observability 更关注“为什么不正常”。

常见三大信号是 metrics、logs、traces。

Metrics 是数值指标，例如 latency、error rate、CPU、job duration。Logs 是事件记录，例如 error message、request log、job log。Traces 用来追踪一个请求或任务经过多个服务的完整路径。

在 distributed system 或 data pipeline 中，observability 很重要，因为问题可能发生在多个服务、多个 job 或多个数据处理阶段之间。没有 observability，就很难定位 root cause。

Quick notes:

```text
Observability = understand internal state from external signals
Monitoring tells if something is wrong
Observability helps explain why
Three pillars = metrics, logs, traces
Useful for distributed systems and data pipelines
Goal = faster root cause analysis
```

### Q15. What is SLA, SLO, and SLI?

简洁答案：

SLI, Service Level Indicator, 是实际测量的指标，例如 availability、latency、error rate、job success rate。

SLO, Service Level Objective, 是基于 SLI 设定的目标值。例如 99.9% availability，或者 95% requests 在 300ms 内完成。

SLA, Service Level Agreement, 是对客户或用户的正式承诺，通常包含违约后果，例如赔偿或服务信用。

简单说：

SLI 是测量值。  
SLO 是内部或服务目标。  
SLA 是对外承诺。

在 data pipeline 中，也可以有类似概念。例如 job success rate、data freshness、pipeline completion time 都可以作为 SLI，目标值可以定义成 SLO。

Quick notes:

```text
SLI = Service Level Indicator, measured metric
SLO = Service Level Objective, target based on SLI
SLA = Service Level Agreement, external commitment
SLI = what we measure
SLO = what we aim for
SLA = what we promise
Data pipeline examples = job success rate, freshness, completion time
```

### Q16. What is blue-green deployment?

简洁答案：

Blue-green deployment 是一种降低发布风险的部署方式。

它准备两套环境：blue 是当前生产环境，green 是新版本环境。新版本先部署到 green，并进行测试和验证。确认没有问题后，把流量从 blue 切到 green。

如果 green 出现问题，可以快速把流量切回 blue，实现 rollback。

它的优点是切换快、回滚简单、对用户影响小。缺点是需要维护两套环境，成本较高，并且要注意数据库 schema change 和数据兼容性。

Quick notes:

```text
Blue-green = two environments
Blue = current production
Green = new version
Test green first, then switch traffic
Rollback = switch traffic back to blue
Pros = quick cutover, easy rollback
Cons = double environment cost, DB compatibility issues
```

### Q17. What is canary deployment?

简洁答案：

Canary deployment 是一种逐步发布新版本的方式。

它不会一次性把所有用户切到新版本，而是先让一小部分流量或用户使用新版本，比如 1%、5%、10%。如果 metrics、logs、error rate、latency 都正常，再逐步扩大流量。

如果发现问题，可以停止扩展或回滚，只影响少量用户。

它的优点是风险低，可以在真实生产流量下验证新版本。缺点是需要更好的 monitoring、traffic control 和版本兼容性管理。

Quick notes:

```text
Canary = gradual rollout
Start with small traffic percentage
Monitor error rate, latency, logs, business metrics
Increase traffic if healthy
Rollback if issues appear
Pros = lower risk, real production validation
Needs = monitoring, traffic control, compatibility
```

### Q18. How would you design a highly available system?

简洁答案：

设计 highly available system 的目标是避免单点故障，让系统在部分组件失败时仍然可以继续服务。

首先要识别关键组件，例如 application servers、database、network、load balancer、message queue、storage。

然后为关键组件设计 redundancy，例如多个 instances、multiple zones、replication、managed services。

应用层通常使用 stateless services + load balancer + health checks。如果某个 instance 不健康，流量可以自动转到健康实例。

数据层要考虑 replication、backup、failover，以及 RTO / RPO 要求。

还需要 monitoring、alerting、autoscaling、rollback plan 和 incident response。高可用不是只靠部署多个机器，还要能发现问题、隔离问题、恢复服务。

Quick notes:

```text
HA goal = avoid single point of failure
Identify critical components
Use redundancy = multiple instances, zones, replication
App layer = stateless services + load balancer + health checks
Data layer = replication + backup + failover + RTO/RPO
Ops = monitoring, alerting, autoscaling, rollback, incident response
HA = continue service during partial failure
```

### Q19. How would you design a resilient system?

简洁答案：

Resilient system 是指系统不仅能减少故障，还能在故障发生时限制影响、自动恢复或快速恢复。

我会从几个方面设计。

第一，假设 failure 一定会发生，包括 network failure、service failure、database failure、message delay、partial failure。

第二，使用 isolation，避免一个组件失败导致整个系统崩溃。例如用 service boundary、queue、circuit breaker、timeout。

第三，使用 retry，但要加 retry limit、exponential backoff、jitter，并保证 idempotency，避免 retry storm 和重复写入。

第四，设计 graceful degradation。如果非核心功能失败，核心功能仍然可用。

第五，建立 observability，包括 metrics、logs、traces、alerts，让团队能快速发现和定位问题。

最后，需要 backup、failover、runbook、incident response 和 postmortem，不断改进系统。

Quick notes:

```text
Resilience = limit impact and recover from failure
Assume failure will happen
Use isolation = service boundary, queue, circuit breaker, timeout
Retry safely = limit, backoff, jitter, idempotency
Graceful degradation = keep core function available
Observability = metrics, logs, traces, alerts
Recovery = backup, failover, runbook, incident response, postmortem
```

### Q20. How would you modernize a legacy DWH or batch system?

简洁答案：

我会先确认业务目标和现状，而不是直接把旧系统替换成新服务。

第一步是盘点当前 DWH、batch jobs、data sources、downstream reports、users、SLA、data quality issues 和运维流程。特别要理解 legacy system 里的业务逻辑，因为很多规则可能隐藏在 SQL、batch job 或手工运维里。

第二步是识别痛点，例如成本高、扩展性不足、开发速度慢、运维依赖人工、监控不足、数据质量问题、下游报表难维护。

第三步是设计 target architecture。例如使用 Cloud Storage 作为 landing zone，BigQuery 作为 analytical DWH，Dataform 管理 ELT 和 data marts，Looker 做 reporting / semantic layer，Cloud Composer 管理 batch workflow，必要时使用 Pub/Sub / Dataflow 做 event-driven 或 streaming 处理。

第四步是分阶段迁移。可以先迁移低风险 data mart 或 read-only reporting，和旧系统 parallel run，比较 row count、key coverage、aggregated metrics、report output，并让业务用户确认。

最后再逐步 cutover，同时准备 rollback plan、monitoring、data validation、access control 和 operation runbook。

Quick notes:

```text
Modernize legacy DWH/batch:
1. Understand business goal and current architecture
2. Inventory DWH, batch jobs, reports, users, SLA, data quality
3. Identify pain points = cost, scalability, slow development, manual ops
4. Target = Cloud Storage + BigQuery + Dataform + Looker + Composer
5. Event-driven/streaming = Pub/Sub + Dataflow if needed
6. Phased migration
7. Parallel run + validation + business confirmation
8. Cutover + rollback + monitoring + runbook
```

### Q21. How would you explain modernization trade-offs to a customer?

简洁答案：

我会先避免只从技术角度说“新架构更好”，而是把 trade-off 和客户的业务目标、风险、成本、团队能力联系起来说明。

例如，rehost 迁移速度快、风险较低，但 cloud-native 收益有限。refactor 可以提高扩展性、可维护性和开发速度，但成本高、周期长、风险也更大。

对于 batch 和 streaming，也是 trade-off。Batch 简单、成本低、运维容易，但实时性差。Streaming 延迟低，但架构、监控、错误处理和成本管理更复杂。

对于 microservices，也不是越拆越好。它能支持独立部署和扩展，但会增加分布式系统复杂度。

所以我会用几个维度和客户一起比较：business value、migration risk、cost、time、operation complexity、team skill、future scalability。

最后给出 phased recommendation，而不是一次性追求最理想架构。

Quick notes:

```text
Explain trade-offs with business context
Do not say new technology is always better

Compare:
business value
migration risk
cost
timeline
operation complexity
team skill
future scalability

Examples:
rehost = fast, lower risk, limited benefit
refactor = high benefit, higher cost/risk
batch = simple, cheaper, less real-time
streaming = low latency, more complex
microservices = flexible, but distributed complexity

Recommend phased approach
```

### Q22. How would you decide between batch, event-driven, and streaming architecture?

简洁答案：

我会先确认业务对 latency 的要求，而不是一开始就选择 streaming。

如果业务只需要日报、周报、定时 report，batch 通常就足够。Batch 架构简单、成本较低、运维容易，适合大多数 reporting 和定期分析场景。

如果需要在某个事件发生后触发处理，例如文件上传、订单创建、状态变化，可以用 event-driven architecture。它适合系统解耦和异步处理。

如果业务需要持续低延迟处理大量数据，例如实时监控、fraud detection、实时推荐、IoT 数据处理，就考虑 streaming。但 streaming 对监控、错误处理、状态管理、成本控制要求更高。

所以判断维度包括：latency requirement、data volume、event pattern、cost、operation complexity、failure handling 和业务是否真的需要实时性。

Quick notes:

```text
Start with latency requirement

Batch:
scheduled, simple, cheaper
good for daily reports and periodic analytics

Event-driven:
triggered by events
good for file upload, order created, status change
decouples systems

Streaming:
continuous low-latency processing
good for fraud, monitoring, IoT, real-time recommendation
more complex operations

Choose based on:
latency, volume, event pattern, cost, ops complexity, failure handling
```

### Q23. How would you handle stakeholder alignment in a modernization project?

简洁答案：

在 modernization project 中，stakeholder alignment 很重要，因为不同团队关注点不同。

业务团队关心 report、业务连续性和数据是否可信。技术团队关心架构、依赖、性能和运维。安全或合规团队关心访问控制、敏感数据和审计。管理层关心成本、风险和交付时间。

我会先识别主要 stakeholders，并明确每个团队的目标、concerns、decision owner 和 dependency。

然后把讨论从“技术选择”转成“业务目标、风险和 trade-off”。例如不是直接说要迁移到 BigQuery，而是说明它如何改善扩展性、运维、开发速度，同时有哪些迁移风险。

对于数据项目，我会特别对齐 data definition、quality criteria、validation method、cutover plan 和 rollback plan。

最后要保持透明沟通，例如定期分享 migration status、open issues、risk、decision log 和 validation results。

Quick notes:

```text
Stakeholders have different priorities:
business = continuity, reports, trusted data
tech = architecture, dependency, performance, operations
security = access, sensitive data, audit
management = cost, risk, timeline

Alignment approach:
identify stakeholders
clarify goals, concerns, owners, dependencies
discuss business goal + trade-offs
align data definition and validation criteria
share status, risks, decisions, validation results
```

## 6. Data

### Q1. What is the difference between DWH, Data Lake, and Lakehouse?

简洁答案：

DWH, Data Warehouse, 是面向分析和报表的结构化数据平台。它通常保存清洗、建模后的 structured data，适合 SQL 查询、BI、dashboard 和 business reporting。优点是数据质量和治理较强，缺点是对 raw data、半结构化数据和灵活探索不如 data lake。

Data Lake 是用于保存大量 raw data 的平台，可以存 structured、semi-structured、unstructured data，例如 CSV、JSON、logs、images。优点是灵活、成本低、适合大规模存储和探索。缺点是如果没有治理，容易变成 data swamp，数据质量和可发现性会下降。

Lakehouse 是结合 Data Lake 和 DWH 思想的架构。它在低成本 storage 上保存数据，同时通过 table format 和 metadata 管理提供 schema、transaction、time travel、data governance 和多引擎访问能力。例如 Iceberg、Delta Lake、Hudi 都属于 Lakehouse 常见技术。

简单说：

DWH 强在治理和分析。  
Data Lake 强在灵活和低成本存储。  
Lakehouse 试图同时获得 data lake 的灵活性和 DWH 的管理能力。

Quick notes:

```text
DWH = structured, curated, SQL analytics, BI/reporting
Data Lake = raw data, flexible, low-cost storage, all data types
Lakehouse = data lake storage + table management + governance

DWH strength = quality, governance, performance for BI
Data Lake strength = flexibility, raw/semi/unstructured data
Lakehouse strength = flexibility + schema/table/transaction management

Examples:
DWH = BigQuery / Teradata / Snowflake
Data Lake = Cloud Storage / S3
Lakehouse = Iceberg / Delta Lake / Hudi
```

### Q2. What is ETL vs ELT?

简洁答案：

ETL 是 Extract, Transform, Load。数据先从 source 抽取出来，在进入目标系统前先完成清洗、转换、聚合，然后再 load 到 DWH 或数据库。

ELT 是 Extract, Load, Transform。数据先抽取并 load 到目标平台，例如 BigQuery，然后在目标平台里用 SQL 或工具进行转换。

传统 DWH 或外部 ETL 工具常用 ETL，因为目标系统计算能力有限，或者数据进入前需要先处理。现代 cloud DWH，例如 BigQuery，通常更适合 ELT，因为它本身有强大的计算能力，可以直接在 DWH 内做 transformation。

ELT 的优点是保留 raw data，转换逻辑更透明，也更适合 Dataform 这类 SQL-based transformation 工具。ETL 的优点是可以在 load 前处理敏感数据、减少目标系统负担，或适应某些复杂外部处理。

选择 ETL 还是 ELT，要看数据敏感性、目标平台能力、转换复杂度、成本、治理和运维方式。

Quick notes:

```text
ETL = Extract -> Transform -> Load
Transform before loading into target

ELT = Extract -> Load -> Transform
Load raw data first, transform inside DWH

ETL good for:
pre-load cleansing
sensitive data handling
external complex processing

ELT good for:
cloud DWH like BigQuery
raw data retention
SQL-based transformation
Dataform-style workflows

Choose based on:
security, platform capability, cost, complexity, governance
```

### Q3. What is batch processing vs streaming processing?

简洁答案：

Batch processing 是定期或一次性处理一批数据，例如每天晚上处理一天的交易数据，生成报表或 data mart。它适合日报、月报、定期分析、历史数据处理等场景。

Streaming processing 是数据一产生就持续处理，适合低延迟场景，例如 fraud detection、实时监控、IoT、实时 dashboard。

Batch 的优点是架构相对简单、成本较低、运维容易，也更容易重跑和校验。缺点是 latency 较高，不能实时反映最新状态。

Streaming 的优点是低延迟、实时性强。缺点是设计更复杂，需要处理状态管理、late events、ordering、retry、monitoring 和成本控制。

选择时先问业务是否真的需要实时。如果只是 reporting，batch 通常足够。如果业务价值依赖实时响应，再考虑 streaming。

Quick notes:

```text
Batch = process data in scheduled groups
Good for reports, data marts, historical processing
Pros = simple, cheaper, easier validation/retry
Cons = higher latency

Streaming = process data continuously
Good for fraud, monitoring, IoT, real-time dashboard
Pros = low latency
Cons = complex state, late events, ordering, monitoring, cost

Key question = does business really need real-time?
```

### Q4. What is Dataflow vs Dataproc?

简洁答案：

Dataflow 和 Dataproc 都可以做大规模数据处理，但定位不同。

Dataflow 是 Google Cloud 的 fully managed data processing service，基于 Apache Beam，可以处理 batch 和 streaming。它更 serverless，运维负担低，适合新建的数据处理 pipeline，特别是需要 streaming、autoscaling、低运维的场景。

Dataproc 是 managed Spark / Hadoop service。它适合运行现有 Spark、Hadoop、Hive、Pig 等生态的 workload。如果客户已经有大量 Spark job，或者团队熟悉 Spark，Dataproc 可以降低迁移成本。

简单说，Dataflow 更适合 cloud-native managed pipeline，Dataproc 更适合迁移或运行 Spark / Hadoop ecosystem。

选择时要看现有资产、团队技能、处理模式、运维要求和迁移成本。

Quick notes:

```text
Dataflow = managed Apache Beam
Batch + streaming
Serverless-like, low ops
Good for new cloud-native pipelines and streaming

Dataproc = managed Spark / Hadoop
Good for existing Spark/Hadoop workloads
Useful when team already uses Spark ecosystem

Dataflow = cloud-native managed processing
Dataproc = Spark/Hadoop migration or compatibility
Choose based on existing assets, team skill, ops, cost
```

### Q5. What is BigQuery, and why use it for analytics?

简洁答案：

BigQuery 是 Google Cloud 的 serverless data warehouse，主要用于大规模数据分析。

它适合 analytics 的原因是：不需要自己管理服务器，计算和存储分离，可以处理很大的数据量，并且支持标准 SQL。对于报表、data mart、日志分析、迁移后的 DWH，都很适合。

BigQuery 的优势包括 scalability、低运维、和 GCP 服务集成好，比如 Cloud Storage、Dataflow、Dataform、Looker、Pub/Sub。

但使用 BigQuery 时也要注意成本和性能。比如避免 `SELECT *`，使用 partitioning 和 clustering，控制扫描数据量，合理设计表结构。

Quick notes:

```text
BigQuery = serverless cloud data warehouse
Good for large-scale analytics

Why use it:
no server management
separate storage and compute
standard SQL
scalable
good integration with GCP services

Common use cases:
DWH modernization
data marts
BI reporting
log analysis
migration from legacy EDW

Need to manage:
cost
query performance
table design
partitioning / clustering
avoid SELECT *
```

### Q6. How would you design a modern data analytics platform on GCP?

简洁答案：

我不会一开始就决定所有组件，而是先确认业务需求和约束。

首先我会确认：数据来源是什么，数据量多大，需要 batch 还是 near real-time，主要使用者是谁，BI、ML、ad-hoc analysis 哪个更重要，数据敏感度如何，SLA、成本和运维能力有什么要求。

如果只是定期报表和分析，架构可以简单一些：source data 通过 batch ingestion 进入 Cloud Storage 或 BigQuery，用 Dataform / BigQuery SQL 做 transformation，最后通过 Looker 或 BI 工具提供报表。

如果需要实时分析或事件驱动处理，可以加入 Pub/Sub 和 Dataflow。

如果客户已有 Spark / Hadoop 资产，可以考虑 Dataproc；如果主要是 SQL-based analytics，则优先 BigQuery + Dataform。

所以现代数据平台不是固定组合，而是根据 business objective、latency、data volume、governance、team skill 和 cost 来选择合适的架构。

Quick notes:

```text
Do not start with tools.
Start with requirements.

Clarify:
business goal
data sources
data volume
batch or real-time
users and use cases
BI / ML / ad-hoc analysis
security and governance
SLA
cost
team skill

Simple reporting:
batch ingestion
Cloud Storage / BigQuery
BigQuery SQL / Dataform
Looker / BI

Real-time:
Pub/Sub + Dataflow

Existing Spark:
Dataproc

SQL analytics:
BigQuery + Dataform

Architecture should fit requirements, not include every service.
```

### Q7. How would you migrate a legacy EDW such as Teradata to BigQuery?

简洁答案：

我会先做 assessment，而不是马上迁移。

首先确认现有 Teradata 的表、数据量、SQL、batch jobs、依赖关系、SLA、报表和用户。然后按业务优先级把 workload 分组，选择低风险、价值高的部分先迁移。

目标架构通常是：数据进入 Cloud Storage 或 BigQuery，核心 DWH 放在 BigQuery，transformation 用 BigQuery SQL / Dataform，调度用 Airflow / Composer，BI 用 Looker 或现有 BI 工具。

迁移时要分阶段进行：schema conversion、SQL rewrite、data migration、parallel run、data validation、performance tuning，最后逐步 cutover。

风险包括 SQL 差异、数据不一致、性能变化、成本失控、下游报表影响。所以需要 validation rules、reconciliation、rollback plan 和 stakeholder alignment。

Quick notes:

```text
Start with assessment.

Check:
tables
data volume
SQL logic
batch jobs
dependencies
SLA
reports
users

Migration steps:
prioritize workloads
schema conversion
SQL rewrite
data migration
parallel run
validation
performance tuning
cutover

Target:
BigQuery DWH
Dataform / BigQuery SQL
Composer / Airflow
Looker / BI

Risks:
SQL differences
data mismatch
performance
cost
downstream impact

Need:
validation
reconciliation
rollback plan
stakeholder alignment
```

### Q8. How would you migrate an AWS analytics platform to GCP?

简洁答案：

我会先确认现有 AWS 架构和迁移目标，例如 Redshift、Glue、Lambda、S3、QuickSight 分别承担什么角色，以及客户为什么要迁移到 GCP。

一般可以这样映射：Redshift 迁移到 BigQuery，Glue ETL 可以迁移到 Dataflow、Dataproc 或 BigQuery SQL，Glue Catalog / metadata 需要重新整理，Lambda 的事件处理可以用 Cloud Functions、Cloud Run 或 Pub/Sub + Dataflow，QuickSight 报表可以迁移到 Looker 或继续连接 BigQuery。

迁移时不能只替换服务名，要重新评估数据模型、SQL、调度、权限、成本和运维方式。

我会采用 phased migration：先迁移低风险数据集，建立 BigQuery DWH 和 Dataform transformation，再做报表验证和 parallel run，最后逐步切换用户。

Quick notes:

```text
Start with current AWS architecture and business goal.

Typical mapping:
Redshift -> BigQuery
Glue ETL -> Dataflow / Dataproc / BigQuery SQL
Lambda -> Cloud Functions / Cloud Run / Pub/Sub
S3 -> Cloud Storage
QuickSight -> Looker / BI connected to BigQuery

Do not only map services.
Review:
data model
SQL
orchestration
permissions
cost
operations

Migration approach:
phase by workload
build BigQuery DWH
use Dataform for transformation
validate reports
parallel run
gradual cutover
```

### Q9. What is EDW modernization?

简洁答案：

EDW modernization 是把传统 Enterprise Data Warehouse 升级成更灵活、可扩展、低运维的数据平台。

它不只是把 Teradata、Oracle、Redshift 迁移到 BigQuery，而是重新整理数据架构、处理方式、数据模型、治理、成本和使用体验。

传统 EDW 常见问题是扩展困难、成本高、batch 复杂、数据孤岛、变更慢、报表依赖不清楚。

现代化之后，通常会使用 cloud data warehouse、data lake / lakehouse、ELT、自动化调度、data quality、lineage、BI semantic layer，让业务更快使用数据。

重点是业务目标：降低成本、提升性能、提高数据可用性、加快分析速度、减少运维负担。

Quick notes:

```text
EDW modernization = upgrade legacy enterprise DWH

Not only migration.
Also improve:
architecture
data model
processing
governance
cost
user experience

Legacy problems:
hard to scale
high cost
complex batch
data silos
slow changes
unclear report dependencies

Modern platform:
cloud DWH
data lake / lakehouse
ELT
orchestration
data quality
lineage
BI semantic layer

Business goals:
lower cost
better performance
faster analytics
less operations
better data usability
```

### Q10. What is Lakehouse migration?

简洁答案：

Lakehouse migration 是把原来分散在 data lake、DWH、Spark 平台或对象存储里的数据和处理流程，迁移到更统一的 lakehouse 架构。

Lakehouse 的目标是结合 data lake 的低成本、开放格式和 DWH 的 SQL 分析、治理、性能能力。

迁移时要先确认现有数据格式、表格式、计算引擎、SQL / Spark jobs、权限、下游报表和数据质量要求。

如果客户已有 Iceberg、Delta、Hudi 或 Spark 资产，不能只看 BigQuery，也要考虑开放表格式、metadata、catalog、transaction support 和 compute engine 的兼容性。

在 GCP 上可以考虑 Cloud Storage + BigLake / Iceberg + BigQuery，或者根据 workload 搭配 Dataproc、Dataflow、Dataform。

Quick notes:

```text
Lakehouse migration = move data lake / DWH / Spark workloads
to a unified lakehouse architecture

Goal:
data lake flexibility
DWH-like SQL analytics
governance
performance
open table formats

Check:
data format
table format
compute engine
Spark / SQL jobs
permissions
downstream reports
data quality

Important:
Iceberg / Delta / Hudi
metadata catalog
transactions
schema evolution
engine compatibility

GCP options:
Cloud Storage
BigLake / Iceberg
BigQuery
Dataproc
Dataflow
Dataform
```

### Q11. What is Apache Iceberg?

简洁答案：

Apache Iceberg 是一种 open table format，用在 data lake / lakehouse 里管理大规模表数据。

普通文件放在 object storage 上时，只是一堆 Parquet / ORC / Avro 文件，很难像数据库表一样管理。Iceberg 在这些文件之上增加了 table metadata，让数据湖里的文件可以像表一样被查询和管理。

Iceberg 支持 schema evolution、partition evolution、time travel、snapshot、ACID-like table operations，并且可以被不同引擎读取，比如 Spark、Flink、Trino、BigQuery 等。

它的价值是开放、可扩展、避免被单一计算引擎绑定，适合 lakehouse 架构。

Quick notes:

```text
Iceberg = open table format for lakehouse

It adds table metadata on top of files in object storage.

Supports:
schema evolution
partition evolution
snapshots
time travel
ACID-like table operations

Works with:
Spark
Flink
Trino
BigQuery and other engines

Value:
open format
engine interoperability
large-scale table management
avoid vendor / engine lock-in
```

### Q12. What is Databricks, and how is it related to Lakehouse?

简洁答案：

Databricks 是一个基于 Spark 生态的数据和 AI 平台，主要用于 data engineering、analytics、machine learning 和 lakehouse。

它提出并推广了 Lakehouse 架构，把 data lake 的低成本存储和 DWH 的数据管理、SQL 分析能力结合起来。

Databricks 常用 Delta Lake 作为 table format，而不是 Iceberg。Delta Lake 也支持 transaction、schema evolution、time travel 等 lakehouse 能力。

面试里可以这样说：Databricks 是一种 lakehouse 平台，BigQuery / BigLake / Iceberg 是 GCP 上实现 lakehouse 或开放数据分析架构的其他选择。选型要看客户已有技术、团队技能、开放性、成本和 GCP 集成需求。

Quick notes:

```text
Databricks = data + AI platform based on Spark ecosystem

Used for:
data engineering
analytics
machine learning
lakehouse

Lakehouse idea:
data lake storage
DWH-like management and SQL analytics

Databricks often uses Delta Lake.
Delta Lake supports:
transactions
schema evolution
time travel

Positioning:
Databricks = lakehouse platform
BigQuery / BigLake / Iceberg = GCP-side options

Choose based on:
existing stack
team skill
openness
cost
GCP integration
```

### Q13. Snowflake vs BigQuery?

简洁答案：

Snowflake 和 BigQuery 都是 cloud data warehouse，都适合大规模分析，但设计理念不同。

BigQuery 是 Google Cloud 的 serverless DWH，用户不需要管理 warehouse 或 cluster，按扫描量或 capacity 计费，和 GCP 生态集成很好。

Snowflake 是跨云的数据平台，用户通常需要选择 warehouse size，计算资源更显式，可以在 AWS、Azure、GCP 上运行，对 multi-cloud 场景比较友好。

如果客户主要在 GCP 上，想降低运维、和 Dataflow、Pub/Sub、Looker、Vertex AI 集成，BigQuery 很自然。

如果客户强调跨云、已有 Snowflake 资产、或者希望更显式地隔离 compute warehouse，Snowflake 也可能合适。

Quick notes:

```text
Both = cloud data warehouse

BigQuery:
serverless
no cluster / warehouse management
strong GCP integration
good for GCP-native analytics
pricing by scan or capacity

Snowflake:
cross-cloud platform
explicit virtual warehouses
good compute isolation
strong multi-cloud story

Choose based on:
cloud strategy
existing assets
operations model
cost model
integration needs
team skill
```

### Q14. What is Dataform?

简洁答案：

Dataform 是 Google Cloud 上用于管理 BigQuery SQL transformation 的工具。

它可以把 SQL transformation 像代码一样管理，包括依赖关系、版本控制、测试、文档和执行顺序。简单说，它让 BigQuery 里的 ELT 流程更工程化。

在数据平台里，Dataform 常用于从 raw / staging tables 生成 data marts 或 reporting tables。

相比手动执行 SQL，Dataform 的优点是可维护性更好，依赖更清楚，也更适合团队协作和 code review。

Quick notes:

```text
Dataform = SQL workflow tool for BigQuery

Used for:
ELT transformation
data marts
reporting tables
staging -> curated layers

Benefits:
SQL as code
dependency management
version control
tests
documentation
code review
maintainability

Good for:
BigQuery-based analytics platform
team-managed transformation logic
```

### Q15. What is Looker semantic layer?

简洁答案：

Looker 的 semantic layer 是在数据库表和业务用户之间定义统一业务口径的一层。

它通常用 LookML 定义维度、指标、join 关系、权限和业务逻辑。这样用户做报表时，不需要每个人自己写 SQL，也不容易出现“同一个指标不同部门算出来不一样”的问题。

比如 revenue、active user、conversion rate 这些指标，可以在 semantic layer 里统一定义。

它的价值是提高数据一致性、复用性和治理能力，让 BI 不只是 dashboard，而是有统一业务模型的数据服务层。

Quick notes:

```text
Looker semantic layer = shared business logic layer

Defines:
dimensions
measures
joins
metrics
permissions
business rules

Why useful:
consistent metrics
less duplicated SQL
better governance
reusable data model
self-service BI

Example:
revenue
active users
conversion rate

Value:
one definition of business metrics
```

### Q16. What is Cloud Composer / Airflow used for?

简洁答案：

Cloud Composer 是 Google Cloud 上的 managed Apache Airflow，用来编排 data pipeline 和 workflow。

它不负责真正处理大量数据，而是负责调度和控制任务顺序。比如先抽取数据，再跑 BigQuery transformation，再做 data quality check，最后刷新报表。

Airflow 的核心概念是 DAG，也就是有依赖关系的任务图。它适合 batch workflow、复杂依赖、定时任务、重试、失败通知和跨系统编排。

但如果只是简单事件驱动或实时消息处理，Pub/Sub + Cloud Run / Dataflow 可能更合适，不一定需要 Airflow。

Quick notes:

```text
Cloud Composer = managed Apache Airflow

Used for:
workflow orchestration
data pipeline scheduling
task dependency management
retry
failure notification
cross-system coordination

Airflow concept:
DAG = directed acyclic graph

Example:
extract data
run BigQuery SQL
run data quality checks
refresh BI tables

Not for:
heavy data processing itself

For event-driven simple workflows:
Pub/Sub / Cloud Run / Dataflow may be better
```

### Q17. What is Pub/Sub used for in data architecture?

简洁答案：

Pub/Sub 是 Google Cloud 的 messaging service，用来做系统之间的异步解耦和事件传递。

在数据架构里，它常用于 event-driven ingestion。比如上游系统产生订单事件、文件到达事件、job 完成事件后，把消息发到 Pub/Sub，下游的 Dataflow、Cloud Run 或 Cloud Functions 再消费这些消息。

它的价值是解耦生产者和消费者，提高扩展性和可靠性。上游不需要知道下游是谁，下游也可以独立扩展。

但 Pub/Sub 本身不是数据库，也不是 ETL 引擎。它主要负责消息传递，后续处理、状态管理、数据落地通常交给 Dataflow、Cloud Run、BigQuery 等服务。

Quick notes:

```text
Pub/Sub = messaging service

Used for:
async communication
event-driven ingestion
decoupling systems
triggering downstream processing

Examples:
order event
file arrival event
job completion event
data change event

Consumers:
Dataflow
Cloud Run
Cloud Functions

Value:
decoupling
scalability
reliability

Not:
database
ETL engine
long-term storage
```

### Q18. How would you design data quality checks?

简洁答案：

我会先从业务规则开始，而不是只做技术检查。

常见 data quality checks 包括：required fields 是否为空、primary key 是否重复、数据类型是否正确、取值范围是否合法、金额是否为负、日期是否异常、source 和 target 的 row count / total amount 是否一致。

对于迁移项目，还要做 reconciliation，比如比较旧 DWH 和新 BigQuery 的记录数、关键指标、抽样数据和报表结果。

Data quality checks 可以放在 pipeline 中间或加载之后执行。如果检查失败，应该记录日志、发 alert、阻止下游使用错误数据，或者进入人工确认流程。

重点是和业务方确认什么叫“正确数据”。

Quick notes:

```text
Start from business rules.

Common checks:
required fields
duplicate primary key
data type
valid range
negative amount
abnormal date
row count
total amount

Migration validation:
source vs target row count
key metrics
sample records
report result comparison

When failed:
log
alert
stop downstream
quarantine bad records
manual review

Key point:
define correctness with business users
```

### Q19. How would you handle schema changes in a data pipeline?

简洁答案：

我会先判断 schema change 的类型和影响范围。

如果是新增 nullable column，通常影响较小，可以让 pipeline 和下游表兼容新增字段。

如果是删除字段、字段改名、类型变化、含义变化，就风险更高，因为可能影响 transformation、data quality checks、BI 报表和下游用户。

处理方式包括：建立 schema change notification 流程，版本管理 schema，在 pipeline 中做 schema validation，提前测试 transformation 和报表，必要时保留旧字段一段时间。

对于重要系统，我会避免直接 breaking change，而是做 backward-compatible change，再逐步通知和切换下游。

Quick notes:

```text
First classify schema change.

Low risk:
add nullable column

High risk:
drop column
rename column
type change
meaning change

Impacts:
pipeline
transformation
data quality checks
BI reports
downstream users

How to handle:
schema change notification
schema versioning
schema validation
test transformations
test reports
keep old field temporarily

Prefer:
backward-compatible changes
gradual downstream migration
```

### Q20. How would you validate migration results?

简洁答案：

我会从多个层次验证，而不是只看 row count。

第一层是 technical validation，比如 source 和 target 的 row count、null count、duplicate count、schema、partition range 是否一致。

第二层是 business validation，比如关键指标是否一致，例如 total sales、transaction count、active users、balance、daily revenue。

第三层是 report validation，比较旧系统和新系统的报表结果是否一致。

第四层是 sampling validation，抽样检查具体记录，确认 key fields、status、amount、date 等是否正确。

如果发现差异，要记录差异类型，判断是数据问题、转换逻辑问题、时间窗口问题，还是旧系统本身的历史问题。

Quick notes:

```text
Do not only check row count.

Validation layers:

1. Technical validation
row count
null count
duplicate count
schema
partition range

2. Business validation
total sales
transaction count
active users
balance
daily revenue

3. Report validation
old report vs new report

4. Sampling validation
compare key records
status
amount
date

If mismatch:
classify root cause
data issue
transformation issue
time window issue
legacy system issue
```

### Q21. What is data governance?

简洁答案：

Data governance 是管理数据如何被定义、访问、使用、保护和维护的一套规则和流程。

它不只是安全权限，也包括数据所有者、业务定义、数据质量、lineage、metadata、访问控制、合规要求和变更管理。

在数据平台里，如果没有 governance，常见问题是：指标口径不一致、没人负责数据质量、敏感数据被错误访问、下游不知道数据来源、schema 变化没有通知。

好的 governance 目标是让数据可信、可控、可追踪，并且能被业务安全地使用。

Quick notes:

```text
Data governance = rules and processes for managing data

Covers:
data ownership
business definitions
metadata
data quality
lineage
access control
compliance
change management

Problems without governance:
inconsistent metrics
unclear ownership
poor data quality
wrong access to sensitive data
unknown data source
schema changes without notice

Goal:
trusted data
controlled access
traceable usage
safe business use
```

### Q22. How would you protect sensitive data in a data platform?

简洁答案：

我会先做 data classification，识别哪些是 sensitive data，例如 PII、payment data、financial data、customer data。

然后按最小权限原则设计访问控制。用户和服务账号只拿到完成工作需要的权限，使用 IAM、dataset / table / column-level access control，必要时加 row-level security。

对于敏感字段，可以使用 masking、tokenization、hashing 或 encryption。数据在传输中和存储时都应该加密。

同时需要 audit logging，记录谁访问了什么数据，并定期 review 权限。

重点是安全不能只靠一个功能，要结合分类、权限、加密、审计和流程管理。

Quick notes:

```text
Start with data classification.

Sensitive data:
PII
payment data
financial data
customer data

Controls:
least privilege
IAM
dataset / table access
column-level access
row-level security

Protection:
masking
tokenization
hashing
encryption at rest
encryption in transit

Operations:
audit logging
access review
approval process

Security = classification + access control + protection + audit
```

### Q23. What is data lineage?

简洁答案：

Data lineage 是记录数据从哪里来、经过哪些处理、最后流向哪里的过程。

它可以回答几个问题：这个报表的数据来源是什么？这个字段是怎么计算出来的？如果上游表变化，会影响哪些下游表和报表？如果数据出错，应该从哪里开始排查？

在数据平台里，lineage 对 troubleshooting、impact analysis、governance、合规和迁移都很重要。

比如迁移项目中，如果不知道报表依赖哪些表和 SQL，就很难安全切换系统。

Quick notes:

```text
Data lineage = data flow and dependency tracking

Answers:
where data comes from
how a field is calculated
which downstream tables use this data
which reports are impacted
where to debug data issues

Useful for:
troubleshooting
impact analysis
governance
compliance
migration planning

Migration example:
know report dependencies before cutover
```

### Q24. What is a data catalog?

简洁答案：

Data catalog 是数据资产的目录，用来帮助用户发现、理解和管理数据。

它通常包含表名、字段说明、数据 owner、业务定义、数据分类、敏感级别、更新时间、数据质量信息和 lineage。

没有 data catalog 时，用户常常不知道哪个表能用、字段是什么意思、数据是否可信、出问题该找谁。

在 GCP 上，可以用 Dataplex / Data Catalog 相关能力来管理 metadata、分类、治理和发现。

Quick notes:

```text
Data catalog = inventory of data assets

Contains:
tables
columns
business definitions
data owner
metadata
data classification
sensitivity level
update time
data quality info
lineage

Why useful:
data discovery
understanding data
trust
governance
ownership

GCP:
Dataplex / Data Catalog capabilities
```

### Q25. Transactional database vs analytical database?

简洁答案：

Transactional database 主要用于业务系统的日常交易处理，也就是 OLTP。它关注快速写入、更新、一致性和单笔交易，比如订单、支付、用户状态更新。

Analytical database 主要用于分析，也就是 OLAP。它关注大规模读取、聚合、扫描和报表，比如统计销售额、用户行为分析、月度报表。

MySQL、PostgreSQL、Cloud SQL、AlloyDB 更偏 transactional database。BigQuery、Redshift、Snowflake 更偏 analytical database。

简单说：业务系统用 transactional DB，分析和报表用 analytical DB。不要把大量 BI 查询直接打到业务数据库上。

Quick notes:

```text
Transactional DB = OLTP
For daily business transactions

Focus:
fast writes
updates
consistency
single-record operations

Examples:
orders
payments
user status

Analytical DB = OLAP
For analytics and reporting

Focus:
large scans
aggregation
read performance
historical analysis

Examples:
BigQuery
Redshift
Snowflake

Rule:
OLTP for operations
OLAP for analytics
Avoid heavy BI queries on production DB
```

### Q26. How do you choose Cloud SQL, AlloyDB, Spanner, or BigQuery?

简洁答案：

我会先看 workload 是 transactional 还是 analytical。

如果是普通业务系统，需要 MySQL、PostgreSQL 或 SQL Server 兼容性，可以选 Cloud SQL。它适合中小规模 OLTP 系统，迁移成本低。

如果是 PostgreSQL workload，但需要更高性能和可用性，可以考虑 AlloyDB。

如果是全球分布式、高并发、强一致性、需要水平扩展的 OLTP 系统，可以考虑 Spanner。

如果是分析、报表、大规模扫描和聚合，不应该选 OLTP 数据库，而应该选 BigQuery。

Quick notes:

```text
Start with workload:
OLTP or OLAP?

Cloud SQL:
managed MySQL / PostgreSQL / SQL Server
good for normal business apps
lower migration cost

AlloyDB:
PostgreSQL-compatible
higher performance / availability
for demanding PostgreSQL workloads

Spanner:
globally distributed OLTP
horizontal scale
strong consistency
high availability

BigQuery:
OLAP analytics
large scans
aggregation
BI reporting

Rule:
business transactions -> Cloud SQL / AlloyDB / Spanner
analytics -> BigQuery
```

### Q27. What is a data mart?

简洁答案：

Data mart 是面向某个业务领域或使用场景整理好的分析数据集。

比如销售 data mart、财务 data mart、用户行为 data mart、运营报表 data mart。它通常从 DWH 或 raw/staging 数据加工而来，结构更接近业务用户需要。

Data mart 的目标是让报表和分析更容易、更快、更一致。用户不需要理解复杂的原始表和中间表，只需要使用已经整理好的业务数据。

在 BigQuery 项目里，常见做法是用 Dataform 或 SQL 从 staging / core tables 生成 data marts，再给 Looker 或 BI 工具使用。

Quick notes:

```text
Data mart = business-focused analytical dataset

Examples:
sales data mart
finance data mart
user behavior data mart
operations reporting mart

Source:
DWH
raw / staging tables
core tables

Purpose:
easier analysis
faster reporting
consistent metrics
hide raw data complexity

Common pattern:
staging / core -> Dataform SQL -> data mart -> Looker / BI
```

### Q28. How do you optimize BigQuery cost and performance?

简洁答案：

BigQuery 优化的核心是减少不必要的数据扫描，并让查询更容易执行。

常见方法包括：不要 `SELECT *`，只选需要的列；使用 partitioning 限制扫描日期范围；使用 clustering 优化按高频字段过滤或 join；避免重复计算，把常用结果做成 materialized view 或 summary table。

SQL 上要注意：先过滤再 join，避免不必要的 cross join，减少复杂 UDF，检查 query plan，避免对 partition column 做函数导致 partition pruning 失效。

成本方面，可以设置预算和 alert，监控高成本 query，必要时使用 reservation / slot capacity。

Quick notes:

```text
Goal:
reduce scanned data
make query execution efficient

Query practices:
avoid SELECT *
select only needed columns
filter early
avoid unnecessary cross join
check query plan
avoid function on partition column
reduce expensive UDFs

Table design:
partitioning
clustering
materialized views
summary tables

Cost control:
budgets
alerts
monitor expensive queries
reservation / slot capacity if needed
```

### Q29. How do you handle late-arriving data?

简洁答案：

Late-arriving data 是指数据在业务时间已经发生，但到达数据平台的时间比较晚。

比如订单是 7 月 1 日发生的，但因为系统延迟或重传，7 月 3 日才进入 pipeline。如果只处理当天数据，就可能漏掉这条记录。

处理方式包括：按 event time 而不是 ingestion time 做业务统计；保留一定的 backfill window，比如每天重算最近 3 天或 7 天；使用 upsert / merge 更新目标表；设计幂等处理，避免重复写入。

对于 streaming，还要考虑 watermark、allowed lateness 和重复事件处理。Watermark 用来判断某个 event time 之前的数据大概率已经到齐了，从而决定什么时候关闭窗口并输出结果。

如果 watermark 之后又来了属于旧窗口的数据，就是 late data。系统要根据配置决定丢弃、更新之前结果，或者送到 side output / dead letter。

Quick notes:

```text
Late-arriving data = event happened earlier
but arrives later in the data platform

Example:
order date = July 1
ingestion date = July 3

How to handle:
use event time for business logic
keep backfill window
reprocess recent days
upsert / merge target tables
idempotent pipeline
deduplication

Streaming:
watermark
allowed lateness
duplicate handling

Watermark = progress of event time
Used to decide when a window is complete

Late data after watermark:
drop it
update previous result
send to side output / dead letter
```

### Q30. What trade-offs do you consider when designing a data architecture?

简洁答案：

设计 data architecture 时，我不会只追求技术上最先进，而是看业务目标和约束。

常见 trade-offs 包括：batch 和 streaming 的实时性与复杂度；BigQuery、Dataflow、Dataproc 等服务之间的运维成本和灵活性；normalized model 和 denormalized model 的一致性与查询性能；成本和性能；开放格式和托管服务；短期迁移速度和长期可维护性。

比如，如果业务只是日报，batch 可能比 streaming 更简单可靠。如果客户已有大量 Spark job，Dataproc 可能比完全重写到 Dataflow 更现实。如果 BI 查询很多，适当 denormalize 或做 data mart 可以提升使用体验。

好的架构不是组件最多，而是在成本、性能、可靠性、治理、团队能力和业务价值之间做平衡。

Quick notes:

```text
Architecture = trade-offs, not maximum tools

Common trade-offs:
batch vs streaming
managed service vs flexibility
BigQuery vs Dataflow vs Dataproc
normalized vs denormalized model
cost vs performance
open format vs managed platform
fast migration vs long-term maintainability
real-time value vs complexity

Examples:
daily reports -> batch may be enough
existing Spark jobs -> Dataproc may be practical
heavy BI usage -> data marts / denormalization

Good design balances:
business value
cost
performance
reliability
governance
team skill
maintainability
```

## 7. Security

### Q1. What are the main areas of cloud security?

简洁答案：

Cloud security 不是只设置权限，而是要从多个层面保护系统和数据。

主要包括：identity and access management、network security、data protection、application security、logging and monitoring、compliance、incident response。

在数据平台里，我会特别关注 IAM 最小权限、敏感数据分类、加密、BigQuery / storage 的访问控制、audit logs、以及谁可以访问哪些 dataset、table、column。

面试时可以说：安全设计要从一开始进入架构，而不是系统完成后再补。

Quick notes:

```text
Cloud security areas:

IAM
network security
data protection
application security
logging and monitoring
compliance
incident response

For data platform:
least privilege
data classification
encryption
dataset / table / column access
audit logs
sensitive data protection

Key message:
security should be designed from the beginning
not added later
```

### Q2. What is IAM and least privilege?

简洁答案：

IAM 是 Identity and Access Management，用来控制谁可以访问什么资源，以及可以做什么操作。

Least privilege 是最小权限原则，意思是用户或 service account 只应该拥有完成工作所需的最小权限，不应该给过大的权限。

比如数据分析用户可能只需要读取某些 BigQuery dataset，不需要修改表结构或管理 IAM。ETL service account 可能需要写入目标表，但不需要访问所有项目资源。

设计 IAM 时，我会按角色、团队、数据敏感度和操作类型来分配权限，并定期 review 权限，避免长期保留不必要的 access。

Quick notes:

```text
IAM = Identity and Access Management

Controls:
who can access
which resource
what action

Least privilege:
only grant minimum required permissions

Examples:
analyst -> read specific BigQuery datasets
ETL service account -> write target tables
avoid project-wide admin roles

Good practice:
role-based access
service accounts
separate duties
regular access review
avoid over-permission
```

### Q3. What is Zero Trust?

简洁答案：

Zero Trust 是一种安全理念：不要因为用户在公司网络里，或者服务在内部网络里，就自动信任它。

每次访问都应该根据 identity、device、location、context、risk level 来验证和授权。

传统安全更像是“内网可信，外网不可信”。Zero Trust 更强调“never trust, always verify”。

在数据平台里，Zero Trust 意味着即使用户在公司内部，也不能默认访问所有数据。仍然需要 IAM、MFA、device check、least privilege、audit logging 和 sensitive data protection。

Quick notes:

```text
Zero Trust = never trust, always verify

Do not automatically trust:
internal network
corporate device
internal service

Verify based on:
identity
device
location
context
risk level

For data platform:
IAM
MFA
device validation
least privilege
audit logs
sensitive data protection

Key idea:
access should be continuously verified
```

### Q4. How do you protect data at rest and in transit?

简洁答案：

Data at rest 是存储中的数据，比如 BigQuery、Cloud Storage、database 里的数据。Data in transit 是网络传输中的数据，比如 API 调用、客户端到服务端、服务之间传输的数据。

At rest 通常通过 encryption、IAM、访问控制、key management 来保护。

In transit 通常通过 TLS / HTTPS、secure APIs、private network、VPN / Interconnect 来保护。

在数据平台里，我会确保敏感数据存储时加密，传输时使用安全协议，同时限制谁可以读取、导出或共享数据。

Quick notes:

```text
Data at rest = stored data
Examples:
BigQuery
Cloud Storage
databases

Protect at rest:
encryption
IAM
access control
key management

Data in transit = data moving over network
Examples:
API calls
client to server
service to service

Protect in transit:
TLS / HTTPS
secure APIs
private network
VPN / Interconnect

Key:
encrypt data
control access
protect transfer path
```

### Q5. What is encryption key management?

简洁答案：

Encryption key management 是管理加密密钥的生命周期，包括创建、存储、使用、轮换、权限控制和废弃。

加密本身不够，关键是密钥由谁管理、谁能使用、是否可以审计、是否能定期轮换。

在 GCP 上，常见选择有 Google-managed keys 和 customer-managed keys，也就是 CMEK。Google-managed keys 运维简单，CMEK 给客户更多控制权，适合合规要求更高的场景。

如果客户有严格安全或合规要求，我会讨论是否需要 CMEK、key rotation、separation of duties 和 audit logging。

Quick notes:

```text
Key management = manage encryption keys lifecycle

Includes:
create
store
use
rotate
access control
audit
destroy / disable

Options:
Google-managed keys
CMEK = customer-managed encryption keys

Google-managed:
simpler operations

CMEK:
more customer control
better for strict compliance

Consider:
key rotation
who can use keys
who can manage keys
audit logging
separation of duties
```

### Q6. How do you secure sensitive data in BigQuery?

简洁答案：

我会先识别敏感数据，比如 PII、payment data、financial data，然后按数据敏感度设计访问控制。

在 BigQuery 里，可以用 dataset / table 权限控制谁能访问数据；用 column-level security 限制敏感字段；用 row-level security 限制不同用户能看到哪些行。

对于特别敏感的字段，可以做 masking、hashing、tokenization，或者只在 authorized view 里暴露脱敏后的数据。

同时要开启 audit logs，监控谁访问了哪些表，并定期 review 权限。

Quick notes:

```text
Start with data classification.

Sensitive data:
PII
payment data
financial data
customer data

BigQuery controls:
dataset access
table access
column-level security
row-level security
authorized views

Data protection:
masking
hashing
tokenization

Operations:
audit logs
access review
least privilege

Goal:
right people access right data only
```

### Q7. What are audit logs used for?

简洁答案：

Audit logs 用来记录谁在什么时候对什么资源做了什么操作。

它对 security、compliance、troubleshooting 和 incident investigation 都很重要。

比如在数据平台里，我们可以通过 audit logs 查看谁查询了敏感表、谁修改了 IAM、谁删除了数据、哪个 service account 执行了 pipeline。

Audit logs 不是只为了事后调查，也可以用于异常监控，比如发现异常访问模式或高风险操作时触发 alert。

Quick notes:

```text
Audit logs record:
who
when
which resource
what action

Used for:
security
compliance
troubleshooting
incident investigation

Data platform examples:
who queried sensitive tables
who changed IAM
who deleted data
which service account ran pipeline

Also useful for:
anomaly detection
alerting on risky actions
access review
```

### Q8. How do you design secure access to a data platform?

简洁答案：

我会先按用户类型和数据敏感度来设计访问，而不是给所有人一样的权限。

常见用户包括 data engineer、analyst、business user、service account、external partner。不同角色需要不同权限。

比如 data engineer 可以管理 pipeline 和部分表；analyst 可以查询授权的数据集；business user 可能只通过 Looker 看报表；service account 只允许执行指定 pipeline。

技术上可以用 IAM、BigQuery dataset/table/column/row-level access、authorized views、service accounts、audit logs 和定期权限 review。

重点是 least privilege、separation of duties，以及让业务能用数据但不能越权访问敏感数据。

Quick notes:

```text
Start with:
user types
data sensitivity
business need

User types:
data engineer
analyst
business user
service account
external partner

Access examples:
engineer -> manage pipelines
analyst -> query authorized datasets
business user -> Looker reports
service account -> specific pipeline actions

Controls:
IAM
dataset / table access
column-level security
row-level security
authorized views
audit logs
access review

Principles:
least privilege
separation of duties
safe data access
```

### Q9. What is network security in cloud?

简洁答案：

Cloud network security 是控制服务之间、用户和服务之间、云和本地环境之间的网络访问。

核心是限制不必要的网络暴露，让系统只开放必须的路径。

常见手段包括 VPC、subnet、firewall rules、private IP、NAT、VPN / Interconnect、load balancer、TLS、private service access。

在数据平台里，比如数据库或内部 API 不应该直接暴露到公网；ETL job 可以通过 private network 访问数据源；外部连接需要明确的认证、加密和访问控制。

Quick notes:

```text
Network security = control network access

Protect communication between:
users and services
services and services
cloud and on-prem

Controls:
VPC
subnet
firewall rules
private IP
NAT
VPN / Interconnect
load balancer
TLS
private service access

Data platform examples:
do not expose database publicly
ETL uses private network
secure external connections
authenticate and encrypt traffic

Goal:
only required network paths are open
```

### Q10. What is VPC Service Controls?

简洁答案：

VPC Service Controls 是 GCP 里用来降低数据外泄风险的安全机制。

它可以为 Google Cloud 服务设置 service perimeter，比如 BigQuery、Cloud Storage、Pub/Sub 等。即使用户有 IAM 权限，如果请求来自 perimeter 外部，或者想把数据传到 perimeter 外部，也可能被阻止。

它的重点不是替代 IAM，而是在 IAM 之外增加一层边界保护，防止数据被错误或恶意地移动到不受控的位置。

在数据平台里，它适合保护敏感 dataset、storage bucket，减少从受保护环境向外部 project、账号或网络泄露数据的风险。

Quick notes:

```text
VPC Service Controls = security perimeter for GCP services

Purpose:
reduce data exfiltration risk

Can protect:
BigQuery
Cloud Storage
Pub/Sub
other supported GCP services

Key idea:
IAM controls who can access
VPC-SC controls from where and to where data can move

Use case:
protect sensitive datasets
prevent data movement outside trusted boundary
reduce accidental or malicious exfiltration

Not a replacement for IAM
It adds another security layer
```

### Q11. What is compliance, such as PCI DSS or GDPR?

简洁答案：

Compliance 是指系统和流程要满足法律、行业标准或公司内部规定。它不是只选择某个云服务，而是要落实到架构、权限、加密、审计、数据保留和运维流程里。

PCI DSS 全称是 Payment Card Industry Data Security Standard，是支付卡行业的数据安全标准，主要保护信用卡、借记卡等 cardholder data。

PCI DSS 关注 card number、expiration date、cardholder name、service code，以及更敏感的 authentication data，比如 CVV 和 PIN。对于数据平台来说，要避免随便存储完整卡号，CVV 原则上不应存储，并使用 masking、tokenization、encryption、least privilege 和 audit logs。

GDPR 全称是 General Data Protection Regulation，是欧盟的个人数据保护法规，关注 personal data 和个人隐私权利。

GDPR 保护 name、email、phone number、address、IP address、user ID、location data、behavior data 等个人数据。数据平台需要考虑 data minimization、purpose limitation、retention policy、right to access、right to erasure、masking / anonymization、access control 和 audit logs。

一句话区别：PCI DSS 保护支付卡数据，GDPR 保护个人数据和隐私权利。

Quick notes:

```text
Compliance = legal / industry / company requirements
It requires architecture + process + operations

PCI DSS = Payment Card Industry Data Security Standard
Focus:
payment card data security
cardholder data
card number
expiration date
cardholder name
service code
CVV / PIN are highly sensitive

For data platform:
do not store CVV
avoid storing full card number if possible
masking
tokenization
encryption
least privilege
audit logs
access review

GDPR = General Data Protection Regulation
Focus:
personal data protection
privacy rights

Personal data:
name
email
phone number
address
IP address
user ID
location data
behavior data

For data platform:
data minimization
purpose limitation
retention policy
right to access
right to erasure
masking / anonymization
access control
audit logs

Difference:
PCI DSS protects payment card data.
GDPR protects personal data and privacy rights.
```

### Q12. How do you handle security incidents?

简洁答案：

Security incident 是已经发生或可能发生的安全问题，比如异常访问、数据泄露、权限误配置、账号被盗用、恶意操作等。

处理时我会按流程来：先 detect，确认异常；然后 contain，限制影响范围；再 investigate，分析原因和影响；接着 remediate，修复问题；最后做 postmortem，防止再次发生。

在数据平台里，比如发现某个用户异常查询敏感表，我会先查看 audit logs，确认访问范围，临时收紧权限，通知相关负责人，然后评估是否有数据泄露风险。

重点是不要只修复表面问题，还要改进 IAM、监控、alert、权限 review 和操作流程。

Quick notes:

```text
Security incident examples:
abnormal access
data leakage
wrong IAM setting
compromised account
malicious operation

Incident response steps:
detect
contain
investigate
remediate
postmortem

Data platform example:
check audit logs
identify accessed data
limit permissions
notify owners
assess data leakage risk
fix root cause

Improve:
IAM
monitoring
alerts
access review
operation process
```

### Q13. How do you secure APIs and applications?

简洁答案：

API 和 application security 的重点是确认谁能调用、传输是否安全、输入是否可信、后端权限是否合理。

常见措施包括 authentication、authorization、HTTPS / TLS、input validation、rate limiting、logging、secret management、以及不要把敏感信息写进代码或日志。

对于 API，要明确 caller 身份，限制每个 caller 能访问的数据和操作。对于 service-to-service 调用，也要使用 service account 或 token，而不是默认信任内网。

如果是容器或 serverless 应用，还要注意最小权限、镜像漏洞扫描、环境变量和 secret 管理。

Quick notes:

```text
API / application security focuses on:
who can call
what they can do
secure transport
trusted input
proper backend permissions

Controls:
authentication
authorization
HTTPS / TLS
input validation
rate limiting
logging
secret management

Avoid:
hardcoded secrets
sensitive data in logs
over-permissioned service accounts
trusting internal network by default

For container / serverless:
least privilege
image scanning
secure env vars
secret management
```

### Q14. How do you explain security trade-offs to a customer?

简洁答案：

我会先说明安全设计通常是在 security、usability、cost、operation complexity 之间做平衡。

比如权限越严格，数据越安全，但申请和使用可能更复杂；VPC Service Controls 可以降低数据外泄风险，但也可能影响开发和跨项目访问；CMEK 给客户更多控制权，但会增加 key management 的运维责任。

我不会只说“越安全越好”，而是根据数据敏感度、合规要求、业务影响和团队运维能力来推荐合适的安全级别。

对客户解释时，我会把风险、收益、成本和运维影响讲清楚，然后给出分阶段方案，比如先保护最敏感的数据，再逐步扩大范围。

Quick notes:

```text
Security trade-offs:
security
usability
cost
operation complexity

Examples:
strict IAM -> safer but more approval work
VPC-SC -> reduces exfiltration risk but adds access complexity
CMEK -> more control but more key management responsibility

Do not say:
more security is always better

Consider:
data sensitivity
compliance
business impact
team operations capability

Explain:
risk
benefit
cost
operational impact

Approach:
start with most sensitive data
phase rollout
review and improve
```
