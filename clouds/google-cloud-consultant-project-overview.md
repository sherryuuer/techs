# Google Cloud Consultant 面试用项目一览

> 目的：把 Sally 过去的数据基盘 / 云迁移 / Lakehouse 项目整理成面试时可以快速调用的项目地图。
> 主线：从 Data Engineer 进化到 Cloud/Data Consultant。

## 0. 面试中的整体定位

不要把自己讲成「只会写 pipeline 的人」。核心定位是：

```text
我有数据工程和云数据基盘的实际经验，不只做实现，也参与需求整理、架构设计、数据质量、运用改善和 stakeholder 调整。今后希望站在更接近客户的位置，帮助客户整理数据基盘问题并设计可落地的云数据方案。
```

日语核心句：

```text
これまでデータエンジニアとして、複数のクラウドデータ基盤プロジェクトに関わってきました。特に、単なる実装だけでなく、上流・下流の関係者との調整、データ品質、運用設計、アーキテクチャ改善にも関わってきました。今後はこの経験を活かして、よりお客様に近い立場で、クラウドデータ基盤の課題整理とソリューション提案に挑戦したいと考えています。
```

面试时每个项目都尽量连接到 6 个点：

1. 业务问题
2. 数据架构
3. stakeholder 调整
4. trade-off
5. 数据质量
6. 运用改善

## 1. 项目总览

| 项目 | 公司 / 期间 | 角色 | 技术 | 面试定位 |
| --- | --- | --- | --- | --- |
| PayPay Card Data Lakehouse | PayPay Card / Jun 2025-present | Data Engineer | AWS S3, Iceberg, GCP BigQuery, Teradata, A-Auto | 最新项目。强调 Lakehouse 现代化、AWS 到 GCP 数据连携、跨团队 stakeholder 调整、数据质量 |
| Gaming Company GCP Migration | ZEAL / Mar 2024-May 2025 | Tech Lead | AWS, Redshift, Glue, Lambda, QuickSight, GCP, BigQuery, Dataform, Looker, Airflow, Pub/Sub, CI/CD | 主打项目。强调从 AWS 分析基盘迁移到 GCP、架构设计、客户说明、Tech Lead |
| Furusato Nozei IT Startup Data Lake | ZEAL / Oct 2020-Mar 2024 | Data Engineer / SE | GCP, Data Lake, ETL, cron, Airflow, SQL, PHP, GA4 | 主打项目。强调 end-to-end 数据基盘、数据质量问题、上游/下游协调、治理改善 |
| Wholesale AWS Glue Data Lake | ZEAL / May 2020-Sep 2020 | Data Engineer / SE | AWS S3, Glue, ETL | 辅助项目。强调早期 AWS data lake / Glue / ETL 基础经验 |
| Pharmaceutical Report Development | ZEAL / Feb 2020-Apr 2020 | Data Engineer / SE | ETL, reporting, SQL | 辅助项目。强调 ETL 到 downstream analysis / report 的基础经验 |
| Highway AI Traffic Prediction | ZEAL / Dec 2019-Feb 2020 | Data Engineer / SE | AWS SageMaker, Redshift, ML model | 辅助项目。强调 AI / ML on AWS、预测模型、分析基盘利用 |
| Highway Cloud Infrastructure | ZEAL / Sep 2019-Dec 2019 | Data Engineer / SE | AWS, ETL, cloud infrastructure | 辅助项目。强调 AWS cloud infrastructure 和 ETL pipeline 基础 |
| Manufacturing System Development | ZEAL / Sep 2018-Aug 2019 | System Engineer | DB2, Oracle, MicroStrategy | 辅助项目。强调传统系统、DB、BI/report、系统开发基础 |
| Satake Foods Finance Department | Satake Foods / Oct 2013-May 2018 | Finance | Accounting, budgeting, data analysis, stakeholder reporting | 职业起点。强调业务理解、数字分析、stakeholder 说明能力 |

## 2. 项目 1：PayPay Card Data Lakehouse

### 一句话定位

现代化中的企业数据基盘项目：在尊重既存 Teradata / batch 资产的前提下，以 Iceberg / Lakehouse 为基础，把 secure DLH 的数据转换为非敏感数据后，提供给 BigQuery、Databricks Delta Sharing、Grid 共通基盘、未来 Snowflake 等多个分析利用场景。

### STAR

**S：Situation**

- PayPay Card では、既存データ基盤のモダナイゼーションを目的として、Data Lakehouse プラットフォームの構築を進めています。
- 従来の Teradata と A-Auto を中心とした DWH / batch job 構成は安定している一方で、コスト、拡張性、開発スピード、クラウドネイティブな運用の面で改善余地がありました。
- 現在のプロジェクトでは、secure DLH 上の Iceberg tables のデータを internal DLH や GCP BigQuery 環境へ連携する取り組みがあります。
- 機微情報を除外・変換したデータを、Databricks Delta Sharing や PayPay グループ 4 社の Grid 共通基盤など、複数の分析基盤で利用できる形に整備していく取り組みも含まれています。

**T：Task**

- Data Engineer として、データ連携 workflow の設計、実装、運用改善を担当しています。
- AWS platform team、GCP 基盤側、上流システム、下流のデータ利用部門など、複数の stakeholder と連携する必要がありました。
- データ要件、連携方式、変換ルール、品質基準、優先度を整理する必要がありました。
- 単にデータを移動するだけでなく、安全性、共有範囲、利用目的、運用性を考慮した設計が求められました。

**A：Action**

- data transfer / transformation / validation の流れを設計しました。
- AWS 側のデータ構造と、GCP BigQuery 側での利用要件を確認しました。
- 下流部門からデータ不一致の指摘があった場合、変換前後のデータを比較し、原因を切り分けました。
- 問題が上流仕様、変換ロジック、または下流側の解釈の違いによるものかを整理しました。
- 関係部門とデータ定義、status code の意味、更新頻度、品質基準を確認しました。

**R：Result**

- より信頼性が高く、運用しやすく、拡張性のある Lakehouse 型のデータ連携基盤づくりに貢献しました。
- Iceberg を利用することで、データレイク上のデータを単なるファイルではなく table として管理でき、schema evolution、snapshot 管理、複数分析基盤からの利用といった面で将来的な拡張性を高められると理解しています。
- この基盤は BigQuery だけでなく、機微情報を除外・変換した internal DLH、Databricks Delta Sharing の外部 table、PayPay グループ 4 社で user consent に基づいて分析する Grid 共通基盤、将来的な Snowflake などの分析基盤の土台にもなり得ると考えています。
- この経験から、データ基盤のモダナイゼーションは単なる技術移行ではなく、データの安全性、利用目的、共有範囲、運用性を整理しながら進めることが重要だと学びました。

### 顾问视角卖点

- 不是简单把数据从 AWS 搬到 BigQuery，而是要把 secure DLH、internal DLH、非敏感数据转换、Delta Sharing、Grid 共通基盘、未来 Snowflake 利用等整体数据利用路径整理清楚。
- 旧系统不能简单说「古いからダメ」。要承认它的稳定性，同时说明云原生角度的改善空间。
- 面试中可以用来回答 stakeholder coordination、data quality、legacy modernization、cloud migration。

### 日语核心句

```text
従来の Teradata と A-Auto を中心とした構成は、非常に安定している一方で、クラウドネイティブな観点では、拡張性、コスト最適化、開発スピード、運用自動化の面で課題があると理解しています。
```

```text
ステークホルダー調整では、単に会議を設定することではなく、各チームの目的・制約・データの使われ方を理解し、論点を整理して、意思決定しやすい状態にすることが重要だと考えています。
```

## 3. 项目 2：Gaming Company GCP Migration

### 一句话定位

从 AWS 分析基盘迁移到 GCP 的大型数据平台项目。适合主打 Cloud Consultant 面试，因为它同时包含云迁移、架构设计、客户说明、Tech Lead 和运用风险改善。

### STAR

**S：Situation**

- お客様は AWS 上で Redshift、Glue、Lambda、QuickSight を利用した分析基盤を運用していました。
- 過去に S3 trigger と Lambda を組み合わせた event-driven な構成により、意図しないループ実行が発生し、不要なコストや運用リスクが課題になっていました。
- 既存の分析基盤には、運用制御、コスト管理、保守性、拡張性の面で改善余地がありました。
- そのため、より安定していて、制御しやすく、運用しやすい GCP ベースのデータ分析基盤への移行を検討していました。

**T：Task**

- Tech Lead として、AWS から GCP への分析基盤移行におけるアーキテクチャ設計を担当しました。
- BigQuery、Dataform、Looker、Airflow、Pub/Sub などを組み合わせ、用途に応じた構成を整理する必要がありました。
- コア機能の実装、code review、メンバー支援を担当しました。
- 顧客向けに技術選定理由、構成方針、運用上のメリットを説明する必要がありました。

**A：Action**

- BigQuery を中心とした DWH を設計しました。
- Dataform を利用して data mart を構築し、分析チームが SQL ベースで mart のロジックを管理しやすい構成にしました。
- Looker を reporting / dashboard 基盤として設計しました。
- Airflow を使って scheduled batch DAG、依存関係、実行順序を管理しました。
- 不定期ファイルアップロード後にすぐ DWH へ反映したいケースでは、Pub/Sub を活用しました。
- CI/CD の整備を進め、チームメンバーの code review や顧客向けの技術説明も行いました。

**R：Result**

- より安定していて、運用しやすく、拡張性のある GCP データ分析基盤を構築しました。
- Airflow と Pub/Sub の役割を分けることで、event-driven 処理による運用リスクを抑えつつ、一部の業務で必要な即時性にも対応できる構成にしました。
- Dataform により、BigQuery 上の data mart ロジックを SQL ベースで管理しやすくしました。
- Looker / LookML により、KPI、dimension、measure、join logic などを semantic layer として整理し、指標定義のばらつきを減らす方向性を作りました。
- この経験から、クラウド移行では単にサービスを置き換えるのではなく、既存運用の課題、コスト、保守性、チームの運用しやすさを考慮して設計することが重要だと学びました。

### 技术选择逻辑

| 技术 | 选择理由 |
| --- | --- |
| BigQuery | Serverless、scalable、运维负荷低，适合 GCP 上的数据分析基盘 |
| Dataform | SQL-based、与 BigQuery 亲和，分析团队更容易维护 data mart 逻辑 |
| Airflow | 适合定期 batch、依赖关系、执行顺序、失败重试和可视化运维 |
| Pub/Sub | 适合不定期文件上传等需要即时反映的 event-driven 场景 |
| Looker | 支持 dashboard/reporting，并通过 LookML 作为 semantic layer 统一 KPI、dimension、measure 的定义 |

### 顾问视角卖点

- 不是「AWS 不好，所以换 GCP」，而是基于客户现有环境、运用风险、成本、扩展性和团队维护能力做设计。
- Airflow vs Pub/Sub 的回答可以体现 trade-off：定期处理用 workflow orchestration，即时事件用 messaging。
- Dataform 的回答可以体现 enablement：让分析团队更自走，减少所有修改都依赖工程师。

### 日语核心句

```text
すべてをイベントドリブンにするのではなく、定期処理や依存関係が明確な処理は Airflow で管理し、即時性が必要な不定期ファイル連携については Pub/Sub を使う、というように用途を分けて設計しました。
```

```text
Dataform を使うことで、BigQuery 上のデータマート定義を SQL ベースで管理でき、分析チームもロジックを理解・改善しやすくなると考えました。
```

```text
Looker は単なる dashboard ツールではなく、LookML を使って KPI、dimension、measure、join logic などを semantic layer として管理できる点が重要だと理解しています。Dataform で BigQuery 上の data mart を整備し、Looker でその上の業務指標の意味を統一することで、SQL の属人化や数値定義のばらつきを減らせると考えています。
```

## 4. 项目 3：Furusato Nozei IT Startup Data Lake

### 一句话定位

客户数据成熟度和开发运维流程都还不成熟的环境中，从零开始构建 GCP data lake，并逐步补齐 batch 处理、SFTP 文件连携、Dataform 定期执行、DDL 差分检查、Slack 告警、GitHub 版本管理等基础能力的 end-to-end 数据基盘项目。

### STAR

**S：Situation**

- ふるさと納税関連の企業では、統一された分析基盤がなく、データが複数のシステムに分散していました。
- 対象データには、寄附、自治体、返礼品、ユーザー、注文、配送、決済、GA4 行動データなどがありました。
- プロジェクト開始時点では、データ連携や運用の仕組みもまだ発展途上の状態でした。
- 上流システムから PHP でファイルを作成してもらい、こちら側で SFTP 経由で取得し、新しく構築した GCP 基盤で日次 batch 処理を行うところから始まりました。

**T：Task**

- GCP ベースの data lake をゼロから構築し、複数の業務データを統合する必要がありました。
- 日次 batch、レポーティング、業務分析、自治体向けのメール通知 job を安定して運用できる状態にする必要がありました。
- データ連携、job 管理、監視、バージョン管理など、基盤運用に必要な仕組みを段階的に整備する必要がありました。
- 上流システム担当者、分析チーム、業務側と連携し、データ仕様、処理タイミング、品質確認方法を整理する必要がありました。

**A：Action**

- GCP data lake の設計と実装に参画しました。
- 上流システムから連携されるファイルを SFTP 経由で取得し、GCP 基盤上で日次 batch 処理する workflow を整備しました。
- ETL、job management、cron による定期実行設定、Dataform の定期実行、運用サポートを担当しました。
- 上流システム側の PHP ファイル生成処理について、保守や調整も一部支援しました。
- 分析チームが作成した SQL を定期実行 job として設定しました。
- 自治体向けにメールを送信する Airflow DAG を実装しました。
- 上流と下流の DDL 差分を検知する運用ツールや、Slack への alarm 通知など、運用改善のための仕組みを開発しました。
- 当初は GitHub による DAG 管理が十分ではありませんでしたが、運用上のリスクを踏まえて、Cloud Composer の DAG を GitHub で管理する形へ改善しました。
- テーブル結合後に大量の NULL が発生した際、分析チームと上流システム担当者と一緒に原因を調査しました。
- その結果、暗号化方式やデータ仕様に対する認識がずれており、key が正しく join できていないことが分かりました。
- その後、過去データの再処理を行い、単体テスト、結合テスト、分析側の確認ルール、上流仕様変更時の通知ルールを強化しました。

**R：Result**

- データ基盤は、複数部門の分析や自治体向け業務を支える基盤へと発展しました。
- DDL 差分検知、Slack 通知、Dataform 定期実行、GitHub による DAG 管理などを整備することで、運用の属人化や手作業リスクを減らすことができました。
- データ品質問題をきっかけに、検証、仕様確認、変更通知のプロセスを改善しました。
- データ活用や運用プロセスがまだ発展途上の環境でも、段階的に信頼性を高めていく経験を得ました。
- この経験から、データ基盤ではアーキテクチャだけでなく、運用、監視、バージョン管理、データ品質チェックを少しずつ仕組み化していくことが重要だと学びました。

### 顾问视角卖点

- 这个项目最适合回答 data maturity 低的客户如何从零建设数据基盘。
- 可以强调：不是一开始就有成熟 DevOps / DataOps，而是在项目中逐步补齐 SFTP、batch、Dataform 定期执行、告警、DDL 差分检测、GitHub 管理。
- 和 PayPay 的区别：PayPay 是 enterprise Lakehouse modernization；Furusato 是 immature environment 里的 zero-to-one data platform + operation maturity。
- 可以作为 failure / lesson learned：运维风险暴露后，推动 GitHub 管理 DAG，减少手工操作和属人化。

### 日语核心句

```text
このプロジェクトでは、データ活用や運用プロセスがまだ成熟していない環境で、GCP ベースの data lake をゼロから構築し、日次 batch、SFTP 連携、Dataform 実行、Slack 通知、GitHub による DAG 管理などを段階的に整備しました。
```

```text
この経験から、データ基盤ではアーキテクチャだけでなく、運用、監視、バージョン管理、データ品質チェックを少しずつ仕組み化していくことが重要だと学びました。
```

## 5. 项目 4：Wholesale AWS Glue Data Lake

### 一句话定位

批发行业客户的 AWS ETL / reporting 基盘项目。上游数据来自 MySQL，通过 Glue connection / crawler 取得数据结构，并用 Glue job / Spark SQL 进行加工后写入 Redshift，供后续报表分析使用。

### STAR

**S：Situation**

- 卸売業界のお客様向けに、販売データや業務データを分析・レポートに活用するための AWS ベースのデータ処理基盤を構築する必要がありました。
- 上流データは主に MySQL に格納されており、そのデータを AWS Glue で取得・加工し、Redshift に連携する構成でした。
- Redshift に格納された加工済みデータは、後続の report / analytics 用途で利用される想定でした。
- Sally にとっては、AWS Glue、Spark SQL、Redshift を組み合わせた ETL 処理を実務で経験した初期プロジェクトでした。

**T：Task**

- Glue connection を作成し、MySQL のデータ構造を Glue crawler / Data Catalog 側で扱えるようにする必要がありました。
- Glue job を開発し、Spark SQL で上流データを加工して Redshift にロードする必要がありました。
- 最終的に report で必要となるデータ項目を整理し、どのテーブルからどのように取得・結合・集計するかを設計する必要がありました。
- チーム内に Spark や SQL に慣れていないメンバーもいたため、主要な SQL 設計や Glue job 作成をリードする必要がありました。
- 必要なデータ定義や集計条件について、顧客と確認しながら進める必要がありました。

**A：Action**

- Glue connection を設定し、MySQL 上のデータ構造を Glue から参照できるようにしました。
- Glue job を作成し、Spark SQL を使ってデータ抽出、結合、加工、集計処理を実装しました。
- report で最終的に必要となるデータを取得するために、長い SQL の設計と実装を担当しました。
- 各項目をどの上流テーブルから取得するか、どの条件で結合・集計するかを整理しました。
- 顧客と必要なデータ、集計条件、report での使われ方を確認しながら、SQL ロジックを調整しました。
- Spark / SQL に不慣れなチームメンバーを支援し、主要な Glue job 作成をリードしました。

**R：Result**

- MySQL から Glue job を通じて Redshift にデータを連携し、report / analytics に利用できる形へ整備しました。
- 複雑な Spark SQL を設計・実装し、顧客と確認しながら最終的に必要なデータを作成することができました。
- AWS Glue、Spark SQL、Redshift を組み合わせた ETL 処理の実務経験を積みました。
- この経験から、ETL 開発ではツールの理解だけでなく、最終的に必要なデータ項目、業務上の定義、SQL ロジックを顧客と確認しながら整理することが重要だと学びました。
- その後のより大規模な GCP、DWH、Lakehouse プロジェクトに取り組むうえでの土台になりました。

## 6. 项目 5：Report Development for Pharmaceutical Industry

### 一句话定位

制药行业的 SQL Server / Tableau reporting 支援项目。主要处理药品销售数据和药局数据，通过定期执行 SQL 从 SQL Server 抽取数据，并集成到 Tableau BI 报表中。

### STAR

**S：Situation**

- 製薬業界のお客様向けに、医薬品の販売データや薬局データを活用した reporting / BI 基盤を支援する必要がありました。
- 下流では Tableau を利用しており、SQL Server 上のデータを BI で利用できる形に連携する必要がありました。
- SQL Server の maintenance と、report / BI へのデータ連携を支援するプロジェクトでした。
- Sally にとって、SQL Server、定期 SQL 実行、Tableau 連携を通じて、ETL と downstream analysis のつながりを経験した初期プロジェクトでした。

**T：Task**

- SQL Server 上の販売データや薬局データを抽出し、Tableau BI で利用できるようにする必要がありました。
- 定期実行される SQL を用いて、必要なデータを抽出・加工する ETL 処理を整備する必要がありました。
- SQL Server の maintenance を行いながら、report 側で必要なデータが正しく連携されるように確認する必要がありました。
- データ項目、抽出条件、加工ロジック、Tableau 側での利用方法を意識して作業する必要がありました。

**A：Action**

- SQL Server の maintenance を担当しました。
- 定期実行される SQL を作成・調整し、販売データや薬局データを抽出しました。
- 抽出したデータが Tableau BI で利用できるように、report 連携に必要なデータ形式や項目を確認しました。
- BI / report 側で想定通りに利用できるか、データ抽出結果や連携結果を確認しました。
- 必要に応じて SQL の抽出条件や加工ロジックを調整しました。

**R：Result**

- SQL Server 上の医薬品販売データや薬局データを、Tableau BI / report で利用できる形に連携しました。
- 定期 SQL 実行によるデータ抽出と BI 連携の基本的な流れを経験しました。
- ETL は単なるデータ抽出ではなく、最終的に report で使われる項目や分析目的を意識して設計・確認することが重要だと理解しました。
- その後の DWH、data lake、BI / reporting 関連プロジェクトに取り組むうえでの基礎になりました。

## 7. 项目 6：AI Traffic Prediction for Highway Industry

### 一句话定位

高速道路業界向けに、AWS SageMaker と Redshift を利用して交通渋滞予測モデルを検証・開発した AI / ML 分析プロジェクト。履歴データ、天気、道路区間情報などを使い、Python / scikit-learn で初期検証し、後に時系列モデルを扱う他チームとも連携した。

### STAR

**S：Situation**

- 高速道路業界のお客様向けに、交通データを活用して将来の交通渋滞状況を予測する必要がありました。
- 予測には、過去の交通履歴データ、天気データ、道路区間情報などを利用していました。
- 既存データを分析基盤に蓄積し、予測モデルの学習や評価に利用できる形にする必要がありました。
- AWS 上で Redshift と SageMaker を利用した分析・機械学習の仕組みを構築するプロジェクトでした。

**T：Task**

- AWS SageMaker と Redshift を利用して、traffic congestion prediction model を検証・開発することが求められました。
- 予測モデルに必要な履歴データ、天気、道路区間情報などを準備し、学習・評価に利用できる形へ整備する必要がありました。
- SageMaker 上で Python code を作成し、モデル検証を行う必要がありました。
- 初期段階では scikit-learn の基本的なモデルを試しましたが、交通渋滞のような時系列性のあるデータに対しては精度面で課題がありました。
- 自分たちだけではモデル選定の知識が十分ではなかったため、他チームとも連携しながら適切なモデルアプローチを検討する必要がありました。
- モデル開発だけでなく、データ基盤と AI / ML 利用のつながりを理解する必要がありました。

**A：Action**

- Redshift 上のデータを利用し、交通渋滞予測モデルに必要なデータ準備を行いました。
- AWS SageMaker 上で Python code を作成し、scikit-learn の基本的なモデルを使って初期検証を行いました。
- 履歴データ、天気、道路区間情報などをモデルの入力として利用できるように整備しました。
- モデルの入力データ、特徴量、予測結果を確認し、分析目的に合う形かを検証しました。
- 初期モデルでは十分な結果が出なかったため、他チームと連携し、SARIMA / SARIMAX などの時系列モデルを含むアプローチを検討しました。

**R：Result**

- AWS 上で data warehouse と machine learning service を組み合わせた分析プロジェクトを経験しました。
- Redshift のデータを SageMaker で利用し、交通渋滞予測に向けたデータ準備、Python 実装、モデル検証の流れを経験しました。
- scikit-learn の一般的なモデルだけでは時系列性のある交通データに十分対応できない場合があることを学びました。
- AI / ML の成果はモデルだけでなく、前段のデータ品質、特徴量設計、時系列性の理解、分析目的の整理に大きく依存することを学びました。
- 自分だけで判断しきれない領域では、専門性のある他チームと連携しながら進めることの重要性を理解しました。
- その後、data analytics や cloud data platform を考える際に、AI / ML 利用を見据えたデータ整備の重要性を意識するようになりました。

## 8. 项目 7：Cloud Infrastructure for Highway Industry

### 一句话定位

高速道路業界向けの AWS cloud infrastructure / operations / ETL pipeline 整備プロジェクト。同じ顧客の後続 AI traffic prediction project に参加する前に、Lambda による EC2 運用自動化、CloudWatch alarm、ETL 実装、JP1 job 運用に関わった。

### STAR

**S：Situation**

- 高速道路業界のお客様向けに、AWS 上でデータ処理や分析に必要な cloud infrastructure を整備する必要がありました。
- このプロジェクトは、後続の AI traffic prediction project と同じ顧客向けの前段となる基盤整備プロジェクトでした。
- EC2 などの server 運用を安定させるため、定期的な snapshot 取得や server parameter 更新などの運用自動化が必要でした。
- システム監視のために CloudWatch alarm を整備する必要がありました。
- データを収集・加工し、後続の分析や業務利用につなげる ETL pipeline も必要でした。
- job 管理には JP1 も利用されており、クラウドサービスと従来型の運用ツールが混在する環境でした。

**T：Task**

- AWS cloud infrastructure の setup と運用自動化に参画することが求められました。
- Lambda を利用して、EC2 運用に関わる定期処理を自動化する必要がありました。
- CloudWatch alarm を設定・開発し、異常や運用イベントを検知できるようにする必要がありました。
- データ処理に必要な ETL pipeline を実装する必要がありました。
- JP1 を含む既存の job 運用と AWS 上の処理を理解しながら、安定して動く構成にする必要がありました。

**A：Action**

- AWS 上の cloud infrastructure setup に参画しました。
- Lambda を作成し、EC2 の定期 snapshot 取得や server parameter 更新など、運用に関わる処理の自動化を支援しました。
- CloudWatch alarm の設定・開発に参画し、監視や異常検知の仕組みを整備しました。
- ETL pipeline の実装に関わり、データの抽出・加工・連携処理を整備しました。
- JP1 を利用した job 運用にも関わり、処理の実行タイミングや運用フローを確認しました。
- 処理結果や実行状態を確認し、後続処理で利用できるように調整しました。

**R：Result**

- AWS 上での cloud infrastructure、Lambda 運用自動化、CloudWatch monitoring、ETL pipeline の基本的な構成を実務で経験しました。
- クラウド基盤では、計算処理だけでなく、運用自動化、監視、job 管理、処理順序、実行状態の確認が重要であることを学びました。
- JP1 のような従来型の job scheduler と AWS cloud service が混在する環境で、既存運用を理解しながらクラウド基盤を整備する経験を得ました。
- この経験が、後続の AI traffic prediction project や、その後の AWS Glue、GCP data lake、cloud migration プロジェクトにつながる基礎になりました。

## 9. 项目 8：System Development for Manufacturing Industry

### 一句话定位

製造業向けのシステム刷新・データ基盤・BI report 開発プロジェクト。伝統的な DB / BI / system development の基礎経験。

### STAR

**S：Situation**

- 製造業のお客様向けに、既存システムの infrastructure を見直し、データ活用に必要な基盤を整備する必要がありました。
- DB2 や Oracle などの traditional database を利用したシステム環境でした。
- 業務ユーザー向けに MicroStrategy による report 開発も必要でした。

**T：Task**

- infrastructure の刷新と data platform 構築に参画することが求められました。
- DB2、Oracle を利用したデータ処理や管理を理解する必要がありました。
- MicroStrategy を利用して、業務ユーザー向けの report を開発する必要がありました。

**A：Action**

- DB2、Oracle を利用したデータ基盤・システム開発に参画しました。
- 既存 infrastructure の見直しや data platform 構築を支援しました。
- MicroStrategy を利用して、業務 report の開発を行いました。

**R：Result**

- traditional database、BI report、system development の基礎経験を積みました。
- クラウド以前のシステム構成や既存 DB の考え方を理解したことで、その後の cloud migration や legacy modernization を考える土台になりました。
- 業務ユーザーが必要とする report を意識し、データ基盤と利用者体験をつなげて考えるきっかけになりました。

## 10. 项目 9：Satake Foods Finance Department

### 一句话定位

会計・予算管理・業務データ分析の経験。Data / Cloud Consultant としての業務理解、数字感覚、stakeholder reporting の土台。

### STAR

**S：Situation**

- Satake Foods の Finance Department で、会計処理、financial reporting、budget management を担当していました。
- 業務上の数字を正確に扱い、関係者に分かりやすく報告する必要がありました。
- また、イベント企画や実施後の振り返りにおいて、データ分析を活用する場面もありました。

**T：Task**

- 会計処理、予算管理、財務報告を正確に行う必要がありました。
- イベントや業務改善に関して、データをもとに actionable insights を整理する必要がありました。
- 分析結果や実績を stakeholder に分かりやすく説明し、次の改善につなげる必要がありました。

**A：Action**

- financial reporting や budget management など、会計業務を担当しました。
- イベント計画や実施後の評価にデータ分析を活用しました。
- 分析結果や post-event report を stakeholder に共有し、改善点を整理しました。

**R：Result**

- 数字の正確性、業務理解、stakeholder への説明力を身につけました。
- IT 転向後も、単に技術を見るだけでなく、ビジネス目的や利用者の意思決定を意識する姿勢につながっています。
- Cloud / Data Consultant として、技術と業務課題をつなげて考えるうえでの基礎になっています。

## 11. 面试问题与推荐项目

| 面试问题 | 优先使用项目 | 回答重点 |
| --- | --- | --- |
| Tell me about yourself | 全体 | 日本经验、Data Engineer 背景、云数据基盘、从实现走向顾问 |
| Cloud migration experience | Gaming / PayPay | AWS to GCP、BigQuery、Lakehouse、迁移不仅是搬数据 |
| Customer-facing / stakeholder experience | Gaming / PayPay / Furusato | 客户说明、部门协调、数据定义和优先级整理 |
| Data quality issue | Furusato / PayPay | NULL、key、规格差异、validation、feedback loop |
| Architecture design | Gaming / PayPay | BigQuery、Dataform、Airflow、Pub/Sub、Iceberg、trade-off |
| Trade-off decision | Gaming | Airflow vs Pub/Sub、batch vs event-driven、稳定性 vs 即时性 |
| Legacy modernization | PayPay / Gaming | 尊重旧系统稳定性，同时改善成本、扩展性、运用性 |
| Leadership | Gaming | Tech Lead、code review、成员指导、客户说明 |
| GCP data analytics knowledge | Gaming / Furusato | BigQuery、Dataform、Looker、Airflow、数据湖 |
| Failure / lesson learned | Furusato | 上游规格认知差异、历史数据修正、测试和通知流程改善 |
| AWS experience | Gaming / Wholesale / Highway AI / Highway Cloud | Redshift、Glue、Lambda、SageMaker、AWS infrastructure |
| AI / ML data use case | Highway AI | SageMaker、Redshift、traffic prediction、データ準備 |
| BI / reporting experience | Gaming / Manufacturing / Pharmaceutical / Satake | Looker、MicroStrategy、reporting、業務ユーザー向け分析 |
| Traditional system / legacy background | Manufacturing / PayPay | DB2、Oracle、Teradata、legacy modernization |
| Business / accounting background | Satake Foods | 会計、予算管理、数字分析、stakeholder reporting |

## 12. 简历项目 bullet 草案

### PayPay Card

```text
• 既存データ基盤のモダナイゼーションを目的とした Data Lakehouse プラットフォーム開発プロジェクトに参画。
• AWS 上の secure Iceberg tables から社内 GCP BigQuery 環境へのデータ連携ワークフローを設計・実装。
• データ転送、変換、検証プロセスを整備し、下流の分析利用に向けた安定的なデータ連携を支援。
• 複数部門の stakeholder と連携し、データ要件の整理、データ不整合の調査、優先度調整を実施。
• downstream users からのフィードバックをもとに、データ品質改善、データガバナンス、運用プロセス改善を推進。
```

### Gaming Company

```text
• AWS ベースの分析基盤から GCP への移行プロジェクトをリードし、BigQuery を中心とした DWH、Dataform によるデータマート、Looker によるレポーティング基盤を設計。
• Airflow と Pub/Sub を活用し、定期バッチ処理とイベントドリブン処理の両方に対応するデータパイプラインおよび orchestration workflow を設計・実装。
• Tech Lead として、アーキテクチャ設計、コア機能の実装、code review、メンバー支援、顧客向けの技術説明を担当。
```

### Furusato Nozei IT Startup

```text
• 寄附、自治体、返礼品、注文、配送、決済、GA4 行動データなど、複数の業務データを統合する GCP ベースの data lake 基盤を設計・実装。
• ETL workflow、定期実行 job、データ品質検証、運用サポートを担当し、業務部門および自治体向けのレポーティング・分析活用を支援。
• 分析チームおよび上流システム担当者と連携し、データ検証、結合テスト、仕様変更通知プロセスを改善。
```

## 13. 准备优先级

1. 自我介绍：把 13 年日本经验、会计到 IT、Data Engineer 到 Consultant 的主线讲顺。
2. Gaming 项目：准备成最完整的 cloud migration / architecture / Tech Lead 故事。
3. PayPay 项目：准备成最新的 Lakehouse / legacy modernization / stakeholder coordination 故事。
4. Furusato 项目：准备成 data quality / governance / end-to-end data platform 故事。
5. 技术补强：BigQuery、Dataform、Airflow、Pub/Sub、Iceberg、SQL vs NoSQL、ETL vs ELT、star schema、data governance。

## 14. 项目速查总结

### 项目 1：PayPay Card Data Lakehouse

- 最新项目。
- Enterprise Lakehouse modernization。
- 既存 Teradata / A-Auto。
- Secure DLH / Iceberg tables。
- Internal DLH / BigQuery 连携。
- Sensitive data 除外・変換。
- Delta Sharing / Grid 共通基盤。
- Data quality / validation。
- Stakeholder coordination。
- 重点：modernization 不是单纯迁移。

### 项目 2：Gaming Company GCP Migration

- 主打项目。
- AWS analytics platform to GCP。
- Redshift / Glue / Lambda / QuickSight。
- BigQuery DWH。
- Dataform data marts。
- Looker reporting / semantic layer。
- Airflow for scheduled batch。
- Pub/Sub for event-driven。
- Tech Lead / code review / client explanation。
- 重点：architecture trade-off。

### 项目 3：Furusato Nozei IT Startup Data Lake

- 主打项目。
- Zero-to-one GCP data lake。
- 数据活用・运用流程発展途上。
- Donation / municipality / order / delivery / GA4。
- SFTP daily batch。
- ETL / job management / Dataform。
- Slack alarm / DDL diff check。
- Cloud Composer DAG to GitHub。
- NULL join incident / data quality。
- 重点：DataOps maturity improvement。

### 项目 4：Wholesale AWS Glue Data Lake

- AWS ETL / reporting。
- Wholesale / distribution。
- Sales and business data。
- MySQL source。
- Glue connection / crawler。
- Glue Data Catalog。
- Glue job / Spark SQL。
- Redshift target。
- Complex SQL design。
- 重点：client requirement clarification。

### 项目 5：Pharmaceutical Report Development

- Pharmaceutical industry。
- Sales data。
- Pharmacy data。
- SQL Server maintenance。
- Scheduled SQL extraction。
- Tableau BI。
- Report integration。
- Data availability validation。
- 重点：ETL to downstream BI。

### 项目 6：AI Traffic Prediction for Highway Industry

- Highway customer。
- Traffic congestion prediction。
- Redshift data。
- SageMaker / Python。
- Historical traffic data。
- Weather data。
- Road segment information。
- scikit-learn initial trial。
- SARIMA / SARIMAX with another team。
- 重点：time-series model suitability。

### 项目 7：Cloud Infrastructure for Highway Industry

- Same highway customer。
- AI project 前段基盤。
- AWS cloud infrastructure。
- Lambda operations automation。
- EC2 scheduled snapshots。
- Server parameter updates。
- CloudWatch alarm。
- ETL implementation。
- JP1 job operations。
- 重点：cloud + legacy operations。

### 项目 8：System Development for Manufacturing Industry

- Manufacturing industry。
- System infrastructure renewal。
- Traditional database。
- DB2 / Oracle。
- Data platform foundation。
- MicroStrategy reports。
- BI for business users。
- Legacy system experience。
- 重点：traditional DB / BI background。

### 项目 9：Satake Foods Finance Department

- Career starting point。
- Accounting operations。
- Financial reporting。
- Budget management。
- Data analysis for events。
- Post-event reports。
- Stakeholder reporting。
- Business / numbers understanding。
- 重点：business perspective foundation。
