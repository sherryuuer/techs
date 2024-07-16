* 流程和组件：存储 - 数据迁移 - 底层计算和容器选择 - 分析和应用集成（ETL和编排服务）
* 周边护城河：安全和合规 / 网络构架 / 管理和组织 / 机器学习（特殊领域在ML流开发中继续深入）/ 开发工具和CICD

## 🚩数据工程基础

### 数据类型

- 结构化数据：传统数据库，CSV文件等
- 非结构化数据：音像，文字等，邮件，需要处理，无法立刻使用
- 半结构化数据（semi）：有层级和标签，比结构化数据更灵活，但是不像非结构化数据那么乱，比如XML，json，*日志文件*等，日志文件很重要，可能是数据工程等的处理对象

### 数据特性（properties）

- Volume：数据量，关系到处理方式和规模
- Velocity：数据产生速度，关系到批处理还是流处理的选择
- Variety：数据类型，这涉及到数据的处理方式

### 数据仓库和数据湖

**数据仓库**

- 传统关系型数据库，star，或者snowflake schema，优化比较复杂的处理
- 云仓库案例：Amazon Redshift，Google Bigquery，Azure SQL Data Warehouse
- 上游数据：比如我的GA点击流数据，事务数据，catalog数据
- 下游数据：会计用的DM，分析数据用的DM，机器学习的DM等，还有一些view

**数据湖**

- 不需要事先准备schema，load什么就是什么，各种类型的数据混杂
- 云湖示例：Amazon S3，GCS，Azure Data Lake Storage，HDFS
- 构架比如：S3 - AWS Glue - Athena分析
- 适合数据分析，数据发现，机器学习

**二者比较**

- 数据仓库 - 数据湖
- *数据类型*：结构化数据 - 各种类型都可以
- *schema和处理方式*：ETL - ELT，数据湖的数据load下来就是它的schema，当你用的时候才会去想要处理
- *灵活性*：小 - 大
- *费用*：昂贵（为复杂的query而设计）- 便宜
- 现在很多情况是结合二者使用的，即满足大数据需要，又满足一般的处理需要

### Data Lakehouse

- 混合数据构架，结合数据湖和数据仓库的优点
- 基本建立在云和分布式构架之上
- 受益于Delta Lake，它将ACID事务处理带向了Big data领域
- 示例：
  - AWS Lake Formation（with S3 and Redshift Spectrum）
  - Delta Lake（Apache Spark）
  - Databricks Lakehouse Platform
  - Azure Synapse Analytics（以前称为Azure SQL Data Warehouse）是Microsoft Azure的一种分析服务。它结合了大规模数据仓库、大数据分析和数据集成的能力，提供了统一的分析体验

### Delta Lake核心原理

- *ACID事务特性*通过使用WAL（Write-Ahead Logging，预写日志）和存储层的快照机制来实现
- *Delta Log*是Delta Lake的核心组件，它是一个事务日志，记录了所有对数据表的操作，包括插入、更新和删除操作，Delta Log是通过一系列JSON文件来实现的，这些文件记录了数据表的每个版本的元数据和操作历史
- Delta Lake强制执行数据的架构（*Schema Enforcement*），确保数据符合预定义的架构，它还支持架构演变（*Schema Evolution*），允许在数据写入过程中动态更新架构，而不会破坏现有数据
- 使用*基于Parquet*的存储格式，结合索引和数据分区技术，提高数据查询和处理的性能
- 支持*版本控制和时间旅行*功能，通过事务日志和快照机制，用户可以查看和查询数据在不同时间点的状态
- *分布式存储和计算框架*（如Apache Spark）实现高可用性和容错能力，故障恢复通过*事务日志和快照机制*，快速恢复到一致性状态

### Data Mesh

- 是一种*数据管理和构架方式*
- 通过*去中心化*的方式，将*数据治理和数据管理职责*分散到不同的*业务域（domain）*中，以实现更高的可扩展性和敏捷性
- Domain-based data management
- 这种方法借鉴了*微服务架构*的思想，将数据视为*产品*，由各个*业务团队*来管理和提供服务
  - 现在的bsp项目应该就是这样的，不同的公司治理不同的数据域
- Data Mesh采用*联邦式数据治理*模式，在保证数据治理一致性的同时，允许各个域根据自身需求灵活地管理和使用数据

## Storage

## Database

## Migration & Transfer

## Compute

## Container

## Analytics

## Application Integration
## ⬆️数据工程的各种组件⬇️全面统筹和ML加持
## Security & Identity & Compliance

## Networking & Content Delivery

## Management & Governance

## Machine Learning

## Developer Tools
