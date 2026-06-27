# Cloud Data Analytics Consultant 咨询 Playbook

> 来源：`cloud-consultant-data-analytics.md`
> 目的：把之前练习过的内容整理成一份可以在咨询、Mock 面试、方案讨论中按步骤使用的执行型文档。

## 使用方式

客户提出数据分析、BI、AI、实时化、数据平台相关需求时，不要直接进入技术方案。按照 8 个 Step 依次推进：

1. 开场与目标确认
2. 现状 As-Is 盘点
3. To-Be 与范围定义
4. 数据与架构方案设计
5. Governance 与风险控制
6. Roadmap 与 MVP 提案
7. 成功指标与期待效果
8. 收尾确认与下一步

每个 Step 下面都包含：

- 这个阶段的目标。
- 应该确认的问题。
- 推荐日语表达。
- 应该产出的内容。
- 与这个阶段直接相关的知识点。

## Step 1. 开场与目标确认

### 1.1 目标

先确认客户真正想解决的业务问题，而不是马上讨论工具、架构或云服务。

核心原则：

- 不只问「想看什么数据」，而要问「想支持什么业务决策」。
- 面对「实时」「AI」「自动化」等需求，不直接接受字面要求，要确认业务目的。
- 提问不要一次太多，先控制在 3 个左右。

### 1.2 需要确认的问题

- 为什么现在要做这个项目。
- 想改善哪个业务指标或运营问题。
- 主要使用者是谁。
- 最终要支持什么决策或行动。
- 成功状态在业务上如何判断。

### 1.3 推荐提问

1. 「まず、このプロジェクトで一番解決したい業務課題を教えていただけますか？」
2. 「このデータやダッシュボードは、誰が、どのような意思決定に使う想定でしょうか？」
3. 「成功した状態を、業務面ではどのように判断されますか？」

### 1.4 输出物

- Business Objective
- Primary Users
- Decision / Action
- Success Criteria

### 1.5 咨询思维框架

Issue / Cause / Solution：

- Issue：现在发生了什么问题。
- Cause：为什么会发生。
- Solution：用什么方式解决。

As-Is / To-Be / Gap / Approach：

- As-Is：现状是什么。
- To-Be：目标状态是什么。
- Gap：差距在哪里。
- Approach：如何分阶段缩小差距。

### 1.6 表达提醒

- 不说「遅い気がする」，改成「鮮度が不足している可能性がある」。
- 不说「保留」「以后再说」，改成「次フェーズの拡張候補」。
- 阶段性总结后，要确认客户方向：「この方向性はいかがでしょうか？」

## Step 2. 现状 As-Is 盘点

### 2.1 目标

确认当前数据、系统、流程、组织责任和痛点。这个阶段决定后续方案是不是贴合现实。

### 2.2 需要确认的问题

- 数据源：业务系统、DB、文件、日志、IoT、外部数据。
- 当前数据流程：手工、Batch、CDC、Streaming、Excel 汇总。
- 当前使用工具：DWH、BI、ETL、Catalog、权限管理。
- 当前痛点：慢、不准、重复、口径不一致、权限不清、成本高。
- 相关角色：业务 Owner、IT、Data Platform、Security、Compliance、BI 用户。

### 2.3 推荐提问

1. 「現在、必要なデータはどのシステムにあり、どのように集計されていますか？」
2. 「現状で一番困っている点は、データの鮮度、品質、定義の不一致、運用負荷のどれに近いでしょうか？」
3. 「データの Owner や承認者は、現時点で明確になっていますか？」

### 2.4 输出物

- Source List
- Current Data Flow
- Current Pain Points
- Stakeholder Map
- Constraints

### 2.5 当前数据流程的判断点

Data Ingestion 比 ETL 更广，包含 Batch、CDC、Streaming 等数据采集方式。

| 当前情况 | 可能问题 | 后续方向 |
| --- | --- | --- |
| Excel 手工汇总 | 工数高、错误多、更新慢 | 自动化 Ingestion + DWH + BI |
| 多系统数字不一致 | KPI 定义和数据来源不统一 | KPI 定义书 + DWH 统一口径 |
| 报表慢 | 查询模型、扫描量、并发或 Mart 设计问题 | Partition / Clustering / Data Mart |
| 权限靠人工管理 | 审计困难、风险高 | Access Matrix + 定期 Review |
| 数据质量不稳定 | 缺失、重复、延迟、异常未被检测 | DQ Rule + Error Handling |

### 2.6 Stakeholder 视角

数据项目通常不只是 IT 项目，需要确认：

- Business Owner：定义业务目标和 KPI。
- Data Owner：负责数据定义和质量。
- Platform Team：建设和运维数据基盘。
- Security / Compliance：确认权限、审计、隐私要求。
- BI Users：实际使用 Dashboard 或数据产品的人。

## Step 3. To-Be 与范围定义

### 3.1 目标

把客户的理想状态转成可交付、可分阶段推进的范围。

### 3.2 需要确认的问题

- 目标数据产品：Dashboard、Data Mart、DWH、Lakehouse、AI Model、Alert。
- 使用者优先级：经营层、部门 Manager、现场员工、Data Scientist。
- 数据新鲜度：日次、小时级、分钟级、秒级。
- MVP 范围：先做哪些数据、KPI、报表、业务场景。
- 下一阶段扩展：AI、实时化、更多部门、更多数据源。

### 3.3 推荐提问

1. 「最初の MVP では、どの業務領域に絞ると一番早く価値を出せそうでしょうか？」
2. 「正式な数値と速報値を分ける必要はありますか？」
3. 「AI やリアルタイム化は、初期スコープか次フェーズの拡張候補かを整理してもよろしいでしょうか？」

### 3.4 输出物

- To-Be Vision
- MVP Scope
- Out of Scope / Next Phase
- Freshness Requirement
- KPI Definition Draft

### 3.5 「实时」需求的拆解

客户说「リアルタイム」时，要拆成：

- 哪些指标需要实时。
- 谁会看。
- 看完之后会采取什么行动。
- 多久以内行动才有业务价值。
- 正式值和速報值是否需要分开。

判断例：

| 场景 | 推荐方式 | 判断理由 |
| --- | --- | --- |
| 日次销售报表 | Batch | 正式值通常日次确认即可 |
| 库存变化 | CDC 或短周期 Batch | 需要较新数据，但未必秒级 |
| EC 点击日志 | Streaming 或 Batch | 取决于是否需要实时推荐或监控 |
| 不正检测 | Streaming | 秒到分钟级响应更有价值 |

### 3.6 AI 需求的拆解

客户说「AI を使いたい」时，要确认：

- 预测或分类的目标是什么。
- 谁使用 AI 结果。
- AI 结果会触发什么行动。
- 是否有足够历史数据。
- 数据质量、标签、特征、反馈闭环是否准备好。

推荐表达：

> 「AI を後回しにするというより、AI が成功するために必要なデータ品質、定義、履歴データを MVP1 で整備する進め方がよいと考えます。」

## Step 4. 数据与架构方案设计

### 4.1 目标

根据业务需求选择合适的数据架构，不为了技术而技术。

### 4.2 架构思考顺序

1. Source：数据从哪里来。
2. Ingestion：Batch、CDC、Streaming 怎么选。
3. Storage：Data Lake / DWH / Lakehouse 如何分工。
4. Processing：ETL / ELT / Stream Processing。
5. Serving：Data Mart、BI、API、AI Feature。
6. Governance：Owner、Quality、Catalog、Access、Lineage。
7. Operations：Monitoring、Alert、Cost、SLA。

推荐表达：

> 「まずは業務上必要な鮮度に応じて、Batch、CDC、Streaming を使い分ける方針がよいと考えます。すべてをリアルタイムにするのではなく、正式値と速報値を分けて設計します。」

### 4.3 输出物

- High-Level Architecture
- Data Flow
- Ingestion Pattern
- Storage / DWH / Mart Design
- BI / AI Consumption Design
- Non-Functional Requirements

### 4.4 Cloud Data Analytics 典型分层

Source -> Ingestion -> Data Lake -> DWH -> Data Mart -> BI / AI -> Governance

各层职责：

- Source：业务系统、日志、IoT、外部数据等原始来源。
- Ingestion：数据采集与同步，包含 Batch、CDC、Streaming。
- Data Lake：灵活保存 Raw 数据。
- DWH：整合、标准化、沉淀可信 KPI 的分析基础。
- Data Mart：面向 BI、分析、AI 的用途别数据集。
- BI / AI：可视化、经营分析、预测、异常检测等消费层。
- Governance：数据责任、质量、目录、权限、血缘、审计。

### 4.5 Batch / CDC / Streaming 选择

- Batch：适合日次、周次等低即时性分析。
- CDC：适合高效捕捉业务数据库变更。
- Streaming：适合点击流、不正检测、IoT、实时监控等。

判断原则：

- 不要把所有东西都做成实时。
- 按业务行动需要的数据新鲜度选择方式。
- 正式值和速報值可以分开设计。

### 4.6 AWS / Azure / GCP 服务映射

| 能力 | AWS | Azure | GCP |
| --- | --- | --- | --- |
| Data Lake | S3 | ADLS Gen2 | Cloud Storage |
| DWH | Redshift | Synapse Analytics | BigQuery |
| ETL / Data Integration | Glue | Data Factory / Synapse Pipelines | Dataflow / Dataproc / Data Fusion |
| Streaming | Kinesis | Event Hubs | Pub/Sub |
| BI | QuickSight | Power BI | Looker / Looker Studio |
| ML | SageMaker | Azure ML | Vertex AI |
| Governance / Catalog | Glue Data Catalog / Lake Formation | Microsoft Purview | Dataplex / Data Catalog / IAM / Policy Tags |

云选型观察点：

- 现有云环境。
- 现有 BI 工具。
- 数据源位置。
- 团队技能。
- 治理与合规要求。
- 成本模型。
- AI / ML 活用方针。

### 4.7 Databricks / Snowflake / BigQuery / Redshift 对比

| 产品 | 强项 | 成本模型重点 |
| --- | --- | --- |
| Databricks | Lakehouse、Spark、AI / ML、Streaming | Cluster size、运行时间、DBU、云基础设施费用 |
| Snowflake | Cloud DWH、SQL、数据共享、多云 | Virtual Warehouse 大小和运行时间 + Storage |
| BigQuery | GCP Native、Serverless DWH、大规模 SQL 分析 | On-demand 扫描量，或 Slot / Capacity + Storage |
| Redshift | AWS Native DWH、S3 联动、AWS 集成 | Provisioned node 或 Serverless RPU + Storage |

Snowflake Warehouse：

- 不是数据保存位置，而是执行 SQL 的计算资源。
- 可以理解为 Snowflake 抽象化提供的计算层。
- 通过 Warehouse Size、Auto Suspend、Auto Resume、Multi-cluster 等控制。

BigQuery Slot：

- BigQuery 执行查询的虚拟计算单位。
- 不是 CPU 1 个或 VM 1 台，而是内部抽象化的并行处理能力单位。
- 当 BI 和 Batch 查询较稳定时，Slot 预约可以稳定性能和成本。

## Step 5. Governance 与风险控制

### 5.1 目标

让方案可信、可运营、可审计，而不是只把数据连起来。

### 5.2 必须检查的治理点

- Data Ownership：谁负责定义和质量。
- Data Quality：缺失、重复、异常、延迟、照合。
- Catalog / Metadata：业务定义、Owner、更新频率、机密等级。
- Access Control：Role、Row、Column、Masking。
- Lineage：Source 到 BI / AI 的流向。
- Change Management：KPI 或字段定义变更时如何管理影响。

### 5.3 推荐提问

1. 「この KPI の公式定義は、どの部門が Owner として承認する想定でしょうか？」
2. 「個人情報や機密情報について、閲覧可能な範囲は役割ごとに分かれていますか？」
3. 「データ品質エラーが発生した場合、誰に通知し、どのように再処理する運用が必要でしょうか？」

### 5.4 输出物

- Governance Policy
- Data Quality Rules
- Access Matrix
- Metadata / Catalog Items
- Lineage / Impact Analysis Plan

### 5.5 Data Governance 五大要素

Data Ownership：

- 明确谁对数据负责。
- Owner 不只是 IT，通常也包括业务部门。

Data Quality：

- 管理缺失、重复、异常值、主数据不一致、更新延迟等问题。

Data Catalog / Metadata：

- 管理表名、字段定义、业务含义、Owner、更新频率、机密等级、使用场景。

Access Control / Security：

- 使用 Role、Row-Level Security、Column-Level Security、Data Masking 等方式控制访问。

Data Lineage：

- 可视化数据从 Source 到 Raw、DWH、Mart、BI / AI 的流动和转换。

### 5.6 Data Quality Rule

记忆方式：

- 空？必填项是否为 NULL。
- 变？值是否异常。
- 矛盾？是否与主数据或其他数据冲突。
- 重复？ID 或业务键是否重复。
- 照合？是否与上游系统的件数、合计一致。
- 延迟？是否在需要的时间前更新。

DQ 异常处理流程：

检测 -> 隔离 -> 影响确认 -> 通知 Owner -> 修正 -> 再处理 -> 通知使用者 -> 再发防止

### 5.7 Metadata / Catalog 示例

以 `sales_transaction` 为例：

- 表说明：整合 POS 和 EC 的销售明细交易数据。
- Owner：销售部门 / 销售管理部门。
- 更新频率：每天凌晨 2 点同步，上午 9 点前反映到 DWH。
- 机密等级：Confidential / 社内機密。
- 主要用途：经营日报、销售分析、店铺 / 商品 KPI 分析、需求预测。

字段 `sales_amount` 的业务定义示例：

- 反映退货和折扣后的税前销售金额。
- 不包含运费和积分使用金额。

### 5.8 Access Control 示例

典型角色与控制方式：

- 经营层：可查看全公司 KPI 和经营日报，但原则上不需要个人识别信息。
- 店铺经理：通过 Row-Level Security 只能查看自己店铺的数据。
- Marketing：可查看客户分群和购买趋势，但姓名、邮箱、电话等通过 Column-Level Security 或 Masking 控制。

核心原则：

- 最小权限。
- 按角色、业务需要和数据敏感度设计访问。
- 技术控制需要配合审批、审计和定期复核。

访问治理需要建立：

- 权限矩阵。
- 申请、审批、定期审查流程。
- 离职或异动时的权限删除。
- 审计日志。
- Data Owner、Security、Compliance、Platform Team、业务 Manager 的职责分工。

### 5.9 Lineage 与 Impact Analysis

Data Lineage：

- 可视化 Source -> Raw -> DWH -> Mart -> BI / AI 的数据流。

Impact Analysis：

- 当字段、定义、逻辑发生变化时，确认影响哪些 DWH、Mart、BI、AI 模型和 KPI。

例：`sales_amount` 从含税改为税前时，需要确认：

- 受影响的 DWH fact 表。
- 经营 KPI Mart。
- 销售报表。
- 销售额、毛利率、客单价、前年比等 KPI。
- 需要通知的对象：销售、财务、经营层、BI 使用者。
- 是否重新计算历史数据，或只从变更日之后适用。

## Step 6. Roadmap 与 MVP 提案

### 6.1 目标

把方案拆成客户能接受、团队能执行的阶段。

### 6.2 标准 Roadmap

- Phase 1：Assessment / 现状调查
- Phase 2：Foundation / 数据基盘构筑
- Phase 3：BI Enablement / 可视化
- Phase 4：Governance / 治理整备
- Phase 5：AI Expansion / AI 活用扩张

### 6.3 MVP 提案模板

- 业务范围：先聚焦一个高价值领域。
- 数据范围：选择关键数据源和 Master。
- KPI 范围：先统一核心 KPI。
- 用户范围：先服务最关键使用者。
- 技术范围：先建立可扩展的数据流和 DWH / Mart。
- 后续扩展：Streaming、AI、更多数据源、更多用户。

推荐表达：

> 「MVP1 では早期に業務価値を出すため、対象データと利用者を絞ります。一方で、後続の AI 活用やリアルタイム化に拡張できるよう、データ定義、品質管理、基盤設計は最初から意識します。」

### 6.4 输出物

- Phase Plan
- MVP Deliverables
- Timeline
- Dependencies
- Next Phase Candidates

### 6.5 提案书结构

Cloud Data Analytics 平台建设提案可按以下结构组织：

1. 背景・目的
2. 现状课题
3. 目标状态 / To-Be
4. 解决方针 / Approach
5. 整体架构
6. Data Governance 方针
7. 执行 Roadmap
8. 期待效果
9. 风险与应对
10. Executive Summary

### 6.6 MVP 的正确理解

MVP 是 Minimum Viable Product，不是粗糙版本，而是在最小范围内交付业务价值的初始版本。

在 DWH / Lakehouse 项目中，可以先聚焦销售、库存、商品主数据等关键数据，尽早提供经营 Dashboard。AI 不是被放弃，而是在 MVP1 中先整备 AI 成功所需的数据基础。

## Step 7. 成功指标与期待效果

### 7.1 目标

同时定义交付成果和业务成果。技术完成不等于项目成功。

### 7.2 推荐提问

1. 「技術的な完成だけでなく、業務効果として何を測定すべきでしょうか？」
2. 「現在のレポート作成時間や意思決定リードタイムはどの程度でしょうか？」
3. 「導入後、どの指標が改善すれば成功と言えますか？」

### 7.3 Delivery KPI

- 数据连携成功率。
- Dashboard 更新频率。
- Data Mart 完成数量。
- DQ Check 覆盖率。
- 权限设置完成率。

### 7.4 Business KPI

- 报表制作时间减少。
- 从周次分析改善为当日分析。
- 决策速度提升。
- 缺货、故障、异常的早期发现。
- 用户使用率提升。

### 7.5 输出物

- Delivery KPI
- Business KPI
- Measurement Method
- Baseline / Target

## Step 8. 收尾确认与下一步

### 8.1 目标

确认客户是否认同方向，并明确下一步行动。

### 8.2 收尾结构

1. 复述业务目标。
2. 总结现状课题。
3. 提出方案方向。
4. 说明 MVP 和后续扩展。
5. 确认客户反馈。
6. 明确下一步。

### 8.3 推荐表达

> 「本日の内容を踏まえると、まずは経営層向けの主要 KPI と重要データに絞って MVP を作り、その上でガバナンスとデータ品質を整備しながら、次フェーズで AI やリアルタイム化に拡張する進め方がよいと考えます。この方向性はいかがでしょうか？」

### 8.4 输出物

- Agreed Direction
- Open Questions
- Next Actions
- Required Stakeholders
- Required Data / Documents

## Step 9. 实战模板与案例库

### 9.1 快速检查清单

咨询中至少确认以下问题：

- 业务目标是什么。
- 谁使用结果。
- 用来做什么决策或行动。
- 当前数据在哪里。
- 当前流程有什么痛点。
- 需要多新鲜的数据。
- 哪些 KPI 需要统一定义。
- 谁是 Data Owner。
- 哪些数据敏感，需要权限控制。
- MVP 先做什么，后续扩展什么。
- 成功指标如何衡量。

### 9.2 客户需求到方案的转换表

| 客户说法 | 需要追问 | 可能的咨询判断 |
| --- | --- | --- |
| 想做实时 Dashboard | 哪些指标真的需要实时，谁会采取行动 | 区分正式值和速報值，Batch / CDC / Streaming 组合 |
| 想用 AI 预测 | 预测什么，谁使用，历史数据是否足够 | 先整备数据质量、特征、业务反馈闭环 |
| 报表数字不一致 | KPI 定义是否统一，数据来源是否一致 | 建立 DWH、KPI 定义书、Data Owner、Lineage |
| Excel 作业太多 | 哪些流程手工，频率多高，错误多不多 | 自动化 Ingestion、DWH、BI、DQ Check |
| 数据不能随便看 | 有哪些角色和敏感字段 | Access Matrix、RLS、CLS、Masking、审计 |
| 系统慢或成本高 | 查询模式、数据量、扫描量、并发 | Partition、Clustering、Mart、成本监控 |

### 9.3 一页式回答模板

当面试官或客户给出一个 Case 时，可以按下面顺序回答：

1. 「まず業務目的と利用者を確認します。」
2. 「次に、現状のデータソース、データフロー、課題を整理します。」
3. 「その上で、必要な鮮度に応じて Batch / CDC / Streaming を使い分けます。」
4. 「DWH / Data Mart では KPI 定義を統一し、BI や AI で使いやすい形に整備します。」
5. 「同時に、Data Owner、Quality、Catalog、Access Control、Lineage を設計します。」
6. 「初期 MVP では対象データと利用者を絞り、早期に業務価値を出します。」
7. 「最後に、Delivery KPI と Business KPI の両方で成功を測定します。」
8. 「この方向性について、お客様の優先順位と制約を確認したいです。」

### 9.4 已练习过的 Retail Mock

客户需求：

> 「リアルタイム経営ダッシュボードを作りたい。AIで需要予測もしたい。」

整理后的应对思路：

- 先确认业务目的、当前课题、使用者和数据新鲜度要求。
- 将「实时 Dashboard」和「AI 需求预测」拆成两个论点。
- 当前状态：POS 销售、库存、EC 日志分散，Excel 周次汇总。
- 第一用户：经营层。店铺经理可作为后续阶段。
- 「实时」需要拆成正式值和速報值。
- POS 销售夜间确认，因此正式值日次即可。
- 店铺别销售和库存的速報值可考虑营业中每小时更新。
- MVP1 聚焦 POS 销售、库存、商品 Master、店铺 Master。
- EC 日志和 AI 需求预测作为 MVP2 以后扩展。
- MVP1 不是不做 AI，而是先整备 AI 所需的数据基础。

MVP1 成果物：

- 数据连携处理。
- DWH / Data Mart。
- KPI 定义书。
- 基础 Data Quality Check。
- 面向经营层的 BI Dashboard。

成功指标分为两类：

- Delivery KPI：数据连携成功率、Dashboard 更新频率、Data Mart 完成度等。
- Business KPI：从周次可视化改善为当天可视化、Excel 工数减少、提前发现销售下滑店铺或缺货风险、经营层使用率。

当 KPI 定义因部门不同而冲突：

- 盘点各部门定义。
- 对经营 Dashboard 的官方 KPI 达成共识。
- 必要时保留用途别 KPI。
- 创建 KPI 定义书。
- 反映到 DWH / ETL / Mart / BI。
- 明确 Data Owner 和变更管理流程。

### 9.5 行业别典型项目与论点

| 行业 | 典型项目 | 面试重点论点 |
| --- | --- | --- |
| 零售 | 经营 Dashboard、需求预测、库存分析 | KPI 定义、鮮度、Batch / CDC、MVP、BI |
| 金融 | 客户 360、不正检测、信用风险、AML | 安全、审计、权限、实时检测 |
| 制造 | IoT Sensor 分析、予知保全、质量异常检测 | Streaming、时序数据、异常检测、边缘处理、保存成本 |
| 通信 | 网络故障检测、解约预测、通信质量分析 | 大量日志、实时监控、客户体验 |
| 医疗 | 患者数据分析、医院经营 Dashboard、临床试验数据 | 隐私、匿名化、严格权限、审计 |
| 媒体 / 广告 | 广告效果、用户行为、A/B Test、推荐 | 事件日志、Streaming、大量数据、同意管理 |
| 物流 | 配送路线优化、延迟预测、GPS Tracking | 实时位置、外部数据、延迟 Alert |

### 9.6 Architecture Mock 入口

首个推荐 Case：

> 製造業の顧客が、工場のセンサーデータを使って、設備故障の予兆検知と生産状況ダッシュボードを作りたい。

练习目标：

- 要件确认。
- 数据源识别。
- Batch / Streaming 的取舍。
- Data Lakehouse 架构。
- 时序数据处理。
- 异常检测。
- BI。
- Governance。
- Cost / Monitoring / Scalability。
- Trade-off。

建议开场问题：

1. 「まず、このプロジェクトで一番改善したい業務課題は、設備停止時間の削減、生産状況の可視化、品質改善のどれに近いでしょうか？」
2. 「センサーデータは現在どの頻度で取得され、どこに保存されていますか？」
3. 「予兆検知の結果は、誰が、どのタイミングで、どのようなアクションに使う想定でしょうか？」

### 9.7 Behavioral / Leadership 扩展方向

后续可补充 STAR 故事库：

- Stakeholder conflict：业务方要求实时或 AI，但数据和预算不成熟。
- Ambiguity：需求模糊时如何拆解问题。
- Leadership：推动跨部门 KPI 定义达成一致。
- Failure / Learning：数据质量问题影响报表后的修复与再发防止。
- Collaboration：与 Data Owner、Security、Compliance、Platform Team 协作。

每个故事建议按以下结构准备：

- Situation：背景。
- Task：你的责任。
- Action：你如何拆解、沟通、执行。
- Result：业务结果和可量化影响。
- Reflection：学到什么，下次如何改进。

### 9.8 后续练习方式

当前偏好的练习方式：

- 使用日语进行会话式 Mock，不需要英语练习。
- 你扮演 Consultant，助手扮演客户或面试官。
- 每轮回答后，希望得到良かった点、改善点、補足、振る舞い、より良い回答例。

推荐练习顺序：

1. Architecture Mock
2. Code Evaluation
3. Behavioral / Leadership
4. Full Mock Interview

以后继续练习时，可以直接指定：

- 「从 Architecture Mock 的制造业 Case 开始，用日语会话式练习。」
- 「从 SQL Code Evaluation 中级题开始。」
- 「帮我把 Retail Mock 用更自然的日语 Consultant 口吻重新回答。」
- 「基于这个整理版，扩展 Behavioral STAR 故事库。」
- 「针对 GCP Cloud Data Analytics Consultant，帮我做一轮 Full Mock。」

## Step 10. Code Evaluation 独立练习

这一部分与咨询主流程相关，但更适合作为单独训练模块。

### 10.1 练习目的

从零写代码不是重点。重点是阅读 SQL / Python / API / 数据处理代码，识别问题、Bug、边界情况、性能、可靠性、安全性和可维护性。

### 10.2 练习流程

1. 助手给出问题 SQL 或代码。
2. 你进行 Review。
3. 助手反馈良かった点、見落とした点、Bug / Edge Case、性能改善、可靠性 / 安全性、より良い回答例。
4. 进入下一题。

### 10.3 重点主题

- SQL 聚合错误。
- JOIN 造成重复。
- NULL 处理。
- 日期与时区。
- 重复记录。
- Window Function。
- Partition / Clustering。
- `SELECT *` 和全表扫描。
- Python 数据处理异常处理。
- API Response、Retry、Rate Limit。
- 数据质量检查。

推荐从中级 SQL Code Evaluation 开始。
