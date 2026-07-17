# Google Cloud Consultant 技术面试准备

> 目的：准备 Google Cloud Consultant, Data Analytics 后续技术面试。
> 本文件保存面试结构、学习路线、主题要点和项目映射。
> 具体练习问答保存到 `google-cloud-consultant-technical-qa.md`。

## 0. 文档分工

- 主技术文档：`google-cloud-consultant-technical-interview.md`
  - 面试结构。
  - 主题目录。
  - Sally 的准备优先级。
  - 各领域核心知识点。
  - 项目映射。
- 技术 Q&A 文档：`google-cloud-consultant-technical-qa.md`
  - 已练习问题。
  - 简洁面试答案。
  - 速记版。
  - SQL / architecture 示例。

之后练习规则：

```text
概念框架 / 学习路线 -> 写入本文件
具体问题答案 -> 写入 technical-qa.md
```

## 1. 面试结构

后续技术部分包含两个面试方向：

1. `Domain-specific skills`
2. `Code evaluation and systems solutioning (non-Cloud) / architectural patterns (Cloud)`

## 2. Domain-Specific Skills 总览

这个面试不是单纯问 Google Cloud 服务。它会按 domain 检查基础知识、项目经验、问题分析能力，以及能否把技术解释给客户。

可以分成两大类：

1. `Non-Cloud domain-specific skills`
2. `Cloud domain-specific skills`

Sally 的主线应该放在：

- `Databases / SQL`
- `Data`
- `Platforms & Infrastructure`
- `Application Modernization`
- `Security`

其他领域如 Web Technologies、AI/ML 也要准备基础理解，但不是主打。

## 3. Non-Cloud Domain-Specific Skills

### 3.1 Web Technologies

这个方向不会要求 coding，但可能考察 web / internet 的理论知识。

需要理解：

- Frontend / backend data transfer。
- HTTP request / response。
- TCP/IP 基础。
- Browser 与 server 之间如何通信。
- REST API。
- JSON。
- CORS。
- Cookie / session / token。
- Asynchronous processing。
- Troubleshooting / debugging。
- Compatibility / UI / UX。
- Scaling web services。

高频问题：

- What happens when you type a URL in a browser?
- How does data move between frontend and backend?
- What is HTTP? What is TCP/IP?
- What is the difference between synchronous and asynchronous processing?
- How would you troubleshoot a slow web application?
- How are web services scaled?

Sally 的准备定位：

- 不是主战场，回答保持基础清楚即可。
- 可以连接到 API / upstream / downstream integration。
- 可以连接到 batch vs event-driven。
- 可以连接到 client-facing troubleshooting。

### 3.2 Databases / SQL

这是 Sally 的强项之一，必须重点准备。

需要准备：

- SELECT / WHERE / GROUP BY / HAVING / ORDER BY。
- JOIN: INNER / LEFT / RIGHT / FULL。
- Subquery。
- CTE。
- Window function。
- Aggregation。
- CASE WHEN。
- UNION / UNION ALL。
- NULL handling。
- Deduplication。
- Ranking。
- Date functions。
- Transaction basics。
- Index basics。
- Relational vs non-relational databases。
- OLTP vs OLAP。
- Star schema。
- Fact / dimension。
- SCD。
- Big Data / Data Analysis。

可能被要求写 SQL：

- 多表 join。
- 每组最新记录。
- Top N per group。
- 去重。
- 累计值。
- moving average。
- conversion rate。
- retention。
- missing data check。
- duplicate check。
- data quality validation SQL。
- source vs target migration validation。

Sally 的项目连接：

- Wholesale：MySQL -> Glue / Spark SQL -> Redshift。
- Pharmaceutical：SQL Server scheduled SQL -> Tableau。
- Furusato：join 后 NULL、key mismatch、validation。
- Gaming：BigQuery / Dataform data mart。
- PayPay：data consistency / downstream validation。

回答核心：

```text
SQL 不是只写查询，而是把业务定义、数据粒度、join key、aggregation logic、data quality 一起整理清楚。
```

## 4. Cloud Domain-Specific Skills

### 4.1 AI/ML

Sally 不要包装成 ML specialist。定位为：

```text
I have practical exposure to ML projects, especially data preparation, feature understanding, model validation, and collaboration with ML specialists.
```

需要理解：

- Supervised learning。
- Unsupervised learning。
- Regression / classification。
- Linear regression。
- Logistic regression。
- Decision tree。
- Random forest。
- SVM。
- Clustering。
- Feature engineering。
- Train / validation / test split。
- Overfitting。
- Bias-variance tradeoff。
- Model performance。
- Precision / recall / F1。
- Deep learning 基础。
- CNN / RNN 基础概念。
- Recommendation / collaborative filtering。
- Generative AI / LLM basics。

项目连接：

- Highway AI Traffic Prediction。
- Redshift data。
- SageMaker / Python。
- Historical traffic data。
- Weather data。
- Road segment information。
- scikit-learn initial trial。
- SARIMA / SARIMAX with another team。

回答核心：

```text
The model itself is important, but the quality of input data, feature design, time-series characteristics, and business definition are also critical.
```

### 4.2 Platforms & Infrastructure

这个方向会讨论 distributed systems infrastructure。

需要理解：

- Compute。
- Storage。
- Networking。
- Distributed systems。
- Monitoring。
- Logging。
- Load balancing。
- Protocols。
- Virtualization。
- Containerization。
- PaaS。
- Automation。
- Tooling。
- Big data infrastructure。
- File storage / distributed storage。

Google Cloud 相关关键词：

- Compute Engine。
- Cloud Storage。
- BigQuery。
- VPC。
- Load Balancing。
- Cloud Monitoring。
- Cloud Logging。
- GKE。
- Cloud Run。
- Cloud Functions。
- Pub/Sub。
- Cloud Composer。
- Dataflow。
- Dataproc。

项目连接：

- Highway Cloud：Lambda, EC2 snapshot, CloudWatch alarm, JP1。
- Gaming：Airflow, Pub/Sub, CI/CD。
- PayPay：Lakehouse operations, data integration。
- Furusato：Cloud Composer, Slack alarm, GitHub DAG management。

回答核心：

```text
Infrastructure is not only about creating resources. It also includes automation, monitoring, operational visibility, failure handling, and maintainability.
```

### 4.3 Application Modernization

这个方向关注 migration 和 architecture。

需要理解：

- Cloud migration。
- Application architecture。
- Monolith vs microservices。
- Containers。
- Serverless。
- CI/CD。
- DevOps。
- Platform operations。
- SRE / observability。
- API strategy。
- Hybrid / multi-cloud / edge。
- High availability。
- Resilience。

项目连接：

- Gaming：AWS analytics platform -> GCP。
- PayPay：legacy Teradata / A-Auto -> Lakehouse modernization。
- Furusato：manual / immature operations -> DataOps foundation。

回答核心：

```text
Modernization is not just replacing old technology. It requires understanding existing business logic, operational constraints, risk, cost, and phased migration.
```

### 4.4 Data

这是最重要领域。

需要理解：

- Distributed data processing。
- Hadoop。
- Spark。
- Beam。
- Batch processing。
- Streaming processing。
- EDW modernization。
- Teradata。
- Snowflake。
- Databricks。
- Lakehouse。
- Data lake。
- DWH。
- Database migration。
- PostgreSQL。
- MySQL。
- SQL Server。
- AlloyDB。
- Transactional database。

Google Cloud 相关关键词：

- BigQuery。
- Cloud Storage。
- Dataflow。
- Dataproc。
- Datastream。
- Database Migration Service。
- Cloud SQL。
- AlloyDB。
- Spanner。
- Pub/Sub。
- Dataform。
- Looker。
- Cloud Composer。
- Dataplex。
- Data Catalog。

项目连接：

- PayPay：Teradata / A-Auto, Iceberg, BigQuery, Lakehouse。
- Gaming：Redshift / Glue / Lambda / QuickSight -> BigQuery / Dataform / Looker / Airflow / Pub/Sub。
- Furusato：GCP data lake, SFTP, batch, Dataform, data quality。
- Wholesale：MySQL, Glue, Spark SQL, Redshift。
- Pharmaceutical：SQL Server, Tableau。

回答核心：

```text
For data architecture, I first clarify business usage, current data flow, data quality, latency requirement, governance, and operation model. Then I choose DWH, data lake, lakehouse, batch, or streaming based on those requirements.
```

### 4.5 Security

Security 不一定是主领域，但 Cloud Consultant 一定会被看基础意识。

需要理解：

- Compliance。
- PCI DSS。
- HIPAA。
- GDPR。
- FedRAMP。
- Zero Trust。
- Identity verification。
- Device validation。
- Access control。
- Encryption at rest。
- Encryption in transit。
- Encryption in use。
- Key management。
- Data classification。
- PII / sensitive data。
- Application security。
- API security。
- Container security。
- Serverless security。
- Logging / monitoring。
- Incident response。

Google Cloud 相关关键词：

- IAM。
- Service Account。
- VPC Service Controls。
- Cloud KMS。
- Secret Manager。
- Cloud Armor。
- Cloud Audit Logs。
- Security Command Center。
- Data Loss Prevention。
- BigQuery policy tags。
- Row-level security。
- Column-level security。

项目连接：

- PayPay：sensitive data exclusion / transformation, user consent, secure DLH。
- Furusato：data quality and upstream/downstream controls。
- Gaming：migration risk and operational control。

回答核心：

```text
Security should be designed from the beginning, especially around identity, access control, data classification, encryption, audit logs, and usage purpose.
```

## 5. Sally 的优先级

必须重点准备：

1. Databases / SQL。
2. Data。
3. Application Modernization。
4. Platforms & Infrastructure。
5. Security basics。

需要基础准备：

1. Web Technologies。
2. AI/ML。

## 6. 项目映射表

| Topic | Best project | Keywords |
| --- | --- | --- |
| SQL | Wholesale / Pharma / Gaming / Furusato | joins, aggregation, validation, Dataform |
| Data architecture | PayPay / Gaming / Furusato | DWH, data lake, lakehouse, BigQuery |
| EDW modernization | PayPay / Gaming | Teradata, Redshift, BigQuery |
| Lakehouse | PayPay | Iceberg, secure DLH, data sharing |
| Batch vs event-driven | Gaming | Airflow, Pub/Sub |
| Data quality | Furusato / PayPay | NULL, key mismatch, validation |
| Infrastructure | Highway Cloud / Furusato | Lambda, CloudWatch, JP1, Composer |
| AI/ML | Highway AI | SageMaker, Redshift, scikit-learn, SARIMA |
| Security | PayPay | sensitive data, consent, access scope |

## 7. Code Evaluation and Systems Solutioning / Architectural Patterns

这个面试不是 hands-on coding，但会看三件事：

1. 能不能理解并评估代码。
2. 能不能设计 end-to-end system。
3. 能不能清楚解释思考过程、提出 trade-off，并接受 interviewer poke holes。

Cloud role 不一定只问 GCP，但能自然提到 Google Cloud 产品会加分。

## 8. 第二个技术面试的核心评价点

面试官在看：

- 是否理解问题。
- 是否主动问 clarifying questions。
- 是否能把 abstract problem 转成 system design。
- 是否能清楚、简洁地解释思路。
- 是否能识别 constraints。
- 是否能讨论 trade-offs。
- 是否能设计 robust system。
- 是否理解 limitations。
- 是否考虑 resource estimation。
- 是否能评价 code 的 bug / complexity / optimization。

Sally 的回答姿势：

```text
Before jumping into the design, I would like to clarify the goal, users, data volume, latency requirement, availability requirement, and operational constraints.
```

日语：

```text
すぐに設計に入る前に、まず目的、利用者、データ量、latency、可用性、運用制約を確認したいです。
```

## 9. System Design Answer Framework

系统设计题统一按这个顺序：

```text
1. Clarify requirements
2. Define scope
3. Identify users and use cases
4. Estimate scale
5. Design high-level architecture
6. Define interfaces / APIs
7. Design data model
8. Discuss processing flow
9. Discuss reliability / scalability / security
10. Explain trade-offs and limitations
```

Clarifying questions：

- What is the business goal?
- Who are the users?
- What is the expected traffic or data volume?
- Is this batch, real-time, or near-real-time?
- What is the required latency?
- What availability is required?
- What are the security / compliance requirements?
- What systems do we need to integrate with?
- What is in scope and out of scope?

## 10. Architecture Discussion Checklist

设计时要覆盖：

- Feature set。
- Interfaces。
- API design。
- Data model。
- Class / component responsibility。
- Distributed system boundary。
- Storage choice。
- Compute choice。
- Sync vs async。
- Batch vs streaming。
- Reliability。
- Scalability。
- Observability。
- Security。
- Cost。
- Simplicity。
- Known limitations。

核心句：

```text
I would start with a simple design that satisfies the core requirements, then add scalability, reliability, and operational controls where the requirements justify the complexity.
```

## 11. Code Evaluation

虽然不是 coding interview，但可能给一段代码让你评价。

需要看：

- Correctness。
- Edge cases。
- Null / empty input。
- Time complexity。
- Space complexity。
- Readability。
- Maintainability。
- Error handling。
- Input validation。
- Security issue。
- Concurrency issue。
- Resource usage。

回答顺序：

```text
1. First, I would confirm what the code is expected to do.
2. Then I would check correctness and edge cases.
3. After that, I would look at time and space complexity.
4. Finally, I would suggest improvements for readability, maintainability, and robustness.
```

## 12. Data Analytics System Design Example

想定问题：

```text
Design a system for a customer who wants to collect event data, process it, and provide analytics dashboards.
```

Simple architecture：

```text
Event source
-> Pub/Sub
-> Dataflow
-> BigQuery raw / curated tables
-> Dataform for marts
-> Looker dashboards
-> Cloud Monitoring / Logging
```

Trade-offs：

- Pub/Sub + Dataflow gives near-real-time processing, but adds operational and cost complexity。
- Batch ingestion is simpler and cheaper if daily reporting is enough。
- BigQuery is good for analytical queries, but not for high-QPS transactional workloads。
- Looker semantic layer helps standardize metrics, but requires governance around metric definitions。

## 13. Migration System Design Example

想定问题：

```text
Design a migration approach from a legacy DWH to a modern cloud data platform.
```

Approach：

- Understand current DWH, jobs, reports, and users。
- Identify critical data domains。
- Define target architecture。
- Start with low-risk data marts。
- Run old and new systems in parallel。
- Compare row counts and aggregated results。
- Validate with business users。
- Cut over gradually。
- Monitor cost, latency, quality, and adoption。

Risks：

- Data definition mismatch。
- Batch dependency。
- Performance regression。
- Cost surprise。
- Downstream report impact。
- User adoption。

## 14. Cloud Architectural Patterns

需要会讨论：

- Load balancing。
- Horizontal scaling。
- Stateless service。
- Async messaging。
- Queue-based load leveling。
- Retry with backoff。
- Idempotency。
- Circuit breaker。
- Blue-green deployment。
- Canary release。
- Data partitioning。
- Caching。
- Multi-region / DR。
- Observability。

Google Cloud 可连接：

- Cloud Load Balancing。
- Managed Instance Groups。
- Cloud Run。
- GKE。
- Pub/Sub。
- Cloud Tasks。
- Cloud Storage。
- BigQuery。
- Memorystore。
- Cloud Monitoring。
- Cloud Logging。

## 15. 面试中的互动方式

如果 interviewer 挑战你的设计，不要防御。

可以说：

```text
That's a good point. If the latency requirement is stricter, I would change the design from batch to event-driven processing.
```

```text
If cost is the main constraint, I would start with a simpler batch architecture and only introduce streaming where the business really needs it.
```

```text
I made that assumption for the initial design. If the data volume is much larger, I would revisit partitioning, processing framework, and monitoring.
```

日语：

```text
ご指摘の通りです。その前提であれば、設計を少し変える必要があります。
```

```text
もし latency がより厳しい要件であれば、batch ではなく event-driven / streaming の構成を検討します。
```

## 16. Sally 的准备重点

优先准备：

1. 解释自己的设计思路。
2. Clarifying questions。
3. Data analytics system design。
4. Migration system design。
5. Batch vs streaming trade-off。
6. Code evaluation checklist。
7. Reliability / scalability / security。
8. 用自己的项目作为 evidence。

不需要准备成算法竞赛。重点是：

```text
Can Sally understand an open-ended technical problem, structure it, explain trade-offs, and design a practical system?
```

备考：提前联系doc的写入方式，练习画图等
