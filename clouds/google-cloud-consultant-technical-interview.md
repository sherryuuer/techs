# Google Cloud Consultant 技术面试准备

> 目的：准备 Google Cloud Consultant, Data Analytics 的非编码技术面试。
> 定位：不是纯开发者，而是从 Data Engineer 走向 Cloud/Data Consultant。

## 0. 回答总框架

技术面试不要只回答“用什么服务”。尽量按下面顺序组织：

```text
Business objective
-> Current architecture
-> Pain points
-> Constraints
-> Target architecture
-> Technology choices
-> Trade-offs
-> Migration plan
-> Risks
-> Success metrics
```

日语核心句：

```text
単にサービスを置き換えるのではなく、まずビジネス目的、既存構成、運用上の課題、データ利用者への影響を整理したうえで、段階的に移行することが重要だと考えます。
```

## 1. Legacy DWH Migration to GCP

### 想定質問

```text
If a customer wants to migrate a legacy DWH such as Teradata with batch jobs to Google Cloud, how would you approach it?
```

### 回答要点

- 先确认 business objective。
- 不要直接说 Teradata -> BigQuery。
- 盘点 existing DWH / batch / reports / users / SLA。
- 找出 pain points：cost, scalability, performance, operation, development speed。
- Target：Cloud Storage + BigQuery + Dataform + Looker。
- Workflow：Cloud Composer / Airflow。
- Event-driven：Pub/Sub / Dataflow。
- 分阶段迁移，parallel run。
- 做 data validation。
- 成功指标：cost, performance, data quality, user adoption。

### 日语回答骨架

```text
まず、Teradata を BigQuery に単純に置き換えるのではなく、ビジネス目的と現在の利用状況を整理します。

現在の DWH、batch job、重要な report、下流ユーザー、SLA、データ品質課題を確認します。そのうえで、コスト、性能、運用負荷、開発スピード、既存業務への影響を整理します。

ターゲットアーキテクチャとしては、Cloud Storage を landing zone とし、BigQuery を中心に DWH / data mart を構築し、Dataform で ELT と mart 管理、Looker で reporting / semantic layer を整備する構成が考えられます。batch workflow は Cloud Composer、event-driven な処理は Pub/Sub や Dataflow を検討します。

移行は一括ではなく、重要度の低い領域や特定の data mart から段階的に進め、既存システムとの並行稼働、件数比較、集計結果比較、業務ユーザー確認を行います。

主なリスクは、データ定義の違い、batch 依存関係、性能、コスト、下流 report への影響です。そのため、stakeholder alignment、data validation、運用設計を含めて進めることが重要です。
```

## 2. Target GCP Data Architecture

### 基本構成

```text
Source systems
-> Cloud Storage landing zone
-> Dataflow / batch ingestion / transfer
-> BigQuery raw / staging / mart
-> Dataform for ELT and data mart management
-> Looker for reporting and semantic layer
-> Cloud Composer for workflow orchestration
-> Cloud Logging / Monitoring for operation
```

### 技术选择理由

| 组件 | 用途 | 说明 |
| --- | --- | --- |
| Cloud Storage | landing / raw data | cheap, scalable, decoupled storage |
| BigQuery | DWH / analytics | serverless, scalable, low operation |
| Dataform | ELT / data mart | SQL-based, version control friendly |
| Looker | BI / semantic layer | KPI, dimension, measure standardization |
| Cloud Composer | batch workflow | dependency, schedule, retry, visibility |
| Pub/Sub | messaging | event-driven ingestion |
| Dataflow | stream / batch processing | managed Apache Beam |
| Dataproc | Spark / Hadoop workloads | use when existing Spark ecosystem is needed |

## 3. Airflow / Cloud Composer vs Pub/Sub

### 判断标准

- Airflow / Cloud Composer：定期 batch、依赖关系、顺序控制、retry、可视化。
- Pub/Sub：event-driven、异步消息、解耦、需要 near real-time 的场景。
- 不是二选一，经常组合使用。

### 日语回答骨架

```text
Cloud Composer は、依存関係や実行順序が明確な batch workflow に向いています。たとえば日次処理、複数 step の ETL、失敗時の retry、実行状況の確認が必要な場合です。

一方で Pub/Sub は、ファイル到着やイベント発生をきっかけに非同期で処理を開始したい場合に向いています。システム間を疎結合にできる点がメリットです。

そのため、すべてを event-driven にするのではなく、定期処理は Composer、即時性が必要なイベント処理は Pub/Sub というように、用途に応じて使い分けます。
```

## 4. Dataflow vs Dataproc

### 判断标准

- Dataflow：managed, serverless, streaming/batch, Apache Beam, lower ops。
- Dataproc：managed Spark/Hadoop, existing Spark jobs, more control, migration of Spark workloads。

### 日语回答骨架

```text
Dataflow は Apache Beam ベースの managed service で、batch と streaming の両方に対応でき、運用負荷を抑えたい場合に向いています。

Dataproc は Spark / Hadoop workload を GCP 上で実行したい場合に向いています。既存の Spark job や Hadoop ecosystem を活かしたい場合には Dataproc が選択肢になります。

新規設計で運用負荷を抑えたい場合は Dataflow を優先し、既存 Spark 資産の移行や Spark 固有の処理を活かす場合は Dataproc を検討します。
```

## 5. Batch vs Streaming

### 判断标准

- Batch：daily / hourly, cost efficient, simpler operation, report use cases。
- Streaming：low latency, event-driven, fraud / monitoring / real-time dashboard。
- 先问业务是否真的需要 real-time。

### 日语回答骨架

```text
まず、本当に real-time が必要かを確認します。多くの reporting や daily analytics では batch で十分な場合があります。

Batch は構成が比較的シンプルで、コストや運用を管理しやすいです。一方で streaming は低遅延で処理できますが、設計、監視、障害対応、コスト管理が複雑になります。

そのため、業務上必要な latency、データ量、障害時の影響、運用体制を確認したうえで選択します。
```

## 6. ETL vs ELT

### 判断标准

- ETL：load 前に変換。外部処理、複雑変換、機微情報制御。
- ELT：先に DWH に load、その後 SQL で変換。BigQuery + Dataform と相性が良い。

### 日语回答骨架

```text
BigQuery のような scalable DWH を使う場合、まず raw data を取り込み、その後 BigQuery 上で ELT として変換する設計が有効な場合が多いです。

Dataform を使うことで、SQL ベースで data mart の変換ロジックを管理でき、version control や review もしやすくなります。

ただし、機微情報を load 前に除外する必要がある場合や、外部システム側でしかできない処理がある場合は ETL を選ぶこともあります。
```

## 7. DWH vs Data Lake vs Lakehouse

### 简单定义

- DWH：structured data, analytics, SQL, performance, governed。
- Data Lake：raw/semi-structured/unstructured, cheap storage, flexible。
- Lakehouse：data lake storage + table management + analytics governance。

### 日语回答骨架

```text
DWH は構造化データを分析しやすい形で管理する基盤で、reporting や business analytics に向いています。

Data Lake は raw data や多様な形式のデータを柔軟に保存できる一方で、管理ルールが弱いと data swamp になるリスクがあります。

Lakehouse は data lake の柔軟性を活かしながら、Iceberg や Delta Lake のような table format によって schema evolution、snapshot、複数 engine からの利用をしやすくする考え方です。
```

## 8. Data Quality and Validation

### 确认点

- row count。
- null count。
- duplicate。
- key consistency。
- aggregation result comparison。
- schema changes。
- business definition。
- upstream/downstream confirmation。

### 日语回答骨架

```text
Data quality は技術的な validation だけでなく、業務上の定義確認も重要です。

移行時には、件数比較、NULL、重複、key consistency、集計結果の比較、schema 差分などを確認します。

また、上流仕様、変換ロジック、下流 report の解釈がずれている場合もあるため、関係者とデータ定義を確認し、変更通知や確認ルールを整備することが重要です。
```

## 9. Data Governance

### 关键词

- data ownership。
- access control。
- PII / sensitive data。
- consent。
- lineage。
- data catalog。
- retention。
- audit。
- metric definition。

### 日语回答骨架

```text
Data governance では、誰が data owner なのか、どのデータを誰が利用できるのか、機微情報をどう扱うのかを明確にする必要があります。

特に個人情報や user consent が関係する場合、access control、masking、監査、利用目的の管理が重要です。

また、Looker の semantic layer や Data Catalog を活用して、KPI やデータ定義を標準化し、利用者ごとの解釈のばらつきを減らすことも重要だと考えます。
```

## 10. Project Mapping

| 技术问题 | 使用项目 | 重点 |
| --- | --- | --- |
| Legacy DWH migration | PayPay / Gaming | Teradata, BigQuery, phased migration |
| GCP architecture | Gaming | BigQuery, Dataform, Looker, Airflow, Pub/Sub |
| Airflow vs Pub/Sub | Gaming | batch vs event-driven |
| Data quality | Furusato / PayPay | NULL, key mismatch, validation |
| Governance | PayPay | sensitive data, consent, sharing scope |
| Data Lake / Lakehouse | PayPay / Furusato | Iceberg, GCP data lake |
| AWS to GCP migration | Gaming | Redshift/Glue/Lambda to GCP |

