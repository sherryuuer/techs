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

### ETL和ELT

- 传统的数据存储是ETL，但是数据糊ELT则是先load，在需要的时候再转换
- 可以是各种源，数据库，文件，API等，可以是批处理，实时，或者近实时处理
- 自动化编排：
  - Glue/Glue Workflows
  - Lambda
  - Step Functions
  - MWAA
  - EventBridge

### Data Sources和Format

- JDBC，ODBC，APIs，RawLogs，Streams
- CSV：最常见文件格式，tsv也是
- Json：软件设置文件，web服务器和客户端等交互文件，嵌套格式的数据
- Avro：二进制格式，大数据，实时处理系统
- Parquet：列式数据，IO和存储优化，分布式系统

### 相关概念

- 数据建模（Data Modeling）是指设计和定义数据结构和关系的过程，以便在数据库系统中有效存储、管理和检索数据。
- 模式演变（Schema Evolution）是指在数据库或数据仓库中随着业务需求变化对数据模式进行修改和扩展的过程。
- 数据血缘（Data Lineage）是指追踪数据从源头到最终目的地的流动路径及其所经历的所有处理步骤的过程。
- 数据采样（Data Sampling）是从一个大数据集中选取子集以便进行分析或建模的方法，其常见的方法包括随机采样、系统采样、分层采样和聚类采样等。
- 数据倾斜机制（Data Skew Mechanisms） 是指在分布式计算或数据存储中，数据不均匀分布导致的负载不平衡问题，其主要机制包括分区键选择不当、数据倾斜以及资源分配不均衡等。应当进行数据分布的调查。*salting*是一个很有趣的方法，比如有的key非常热门，可以给他们加上随机的修饰，将他们打散。
- 数据验证（Data Validation）通过一系列规则和约束条件来检查和确保数据的*完整性，一致性，准确性和有效性*。其目的是保证数据符合预期的格式和业务规则。
- 数据分析（Data Profiling）是一种通过*统计和分析技术*来检查和总结数据集特征的过程。它帮助识别数据的结构、内容和质量问题。

### 数据库性能提升方法

- indexing：进行索引
- Partitioning：分区，并行计算
- Compression：列式压缩

- *SQL回顾*：
  - aggregation聚合函数：count，sum，avg，max，min，case
  - grouping，nested grouping and sorting：聚合之外的都是group健
  - pivoting：可以使用CASE语句或 PIVOT函数实现，在不支持Pivot函数的数据中用case可以实现，但是如果列很长的话写起来就会很痛苦
  - inner join，left outer join，right outer join，full outer join，cross outer join
  - Regular Expressions：like语句

### Git也是数据工程的重要工具！

## Storage

### S3

- **Object**存储，网站hosting，数据湖
- globally unique name
- *key*是除了bucket名之外的**full path**，其中，文件名之外的都是*prefix*，他只是看起来很长的key而已
  - *key = prefix + object_name*
- Object最大*5TB*，超了就需要*multi-part upload*功能进行上传
- *Version*ID，版本控制，没开启这个功能前，id只是null，删除了特定最新版本后，会变成前一个版本
- Metadata功能，存储key-value
- tags功能
- **为什么你可以打开（open按钮）自己上传的文件**：因为那是一个pre-signed URL，它识别了你是文件的上传者
- **安全Security**：
  - User-Based：IAM Policies控制
  - Resource-Based：三种
    - Bucket Policies，是json控制，比如控制匿名用户的*数据访问范围*
    - Object Access Control List又叫ACL，这个太细致了似乎没有被用
    - Bucket Access Control List也是ACL，是桶级别的访问控制
  - 其他AWS资源的访问，使用*IAM Roles*
  - **Cross-Account**的访问，使用*Bucket Policy*
  - 防止数据泄漏，AWS设置了*block public access*
  - 能访问对象的*条件*是，你拥有任何一个base的权限，并且没有被显式deny
  - 数据加密，使用加密key进行加密
- **Replication**：复制，两种，CRR跨区，和SRR同区
  - 必须开启版本控制功能，复制规则可以细致设置
  - 可以是不同账户之间的复制
  - 是一种非同步复制
  - 必须给予bucket相应的IAM权限进行object的读取等操作
  - 应用场景：低延迟，跨区合规审查，日志聚合等
  - 只有你开启功能后的新object会被复制，对于没有被复制成功的，和老的项目，可以用*S3 Batch Replication*功能进行复制
  - *可以复制已经删除的版本markers*，比如你想保留一些记录，注意，当删除origin的对象后，会被复制，但是**如果你删除origin的特定version，则不会被复制**，这说明delete marker是一般的对象删除操作，而对特定version的删除，则有些违规了，这不会被复制！
  - *没有*桶之间的连锁chaining复制功能
  - **注意：桶的区域选择不是从右上角而是在桶自己的页面。**
- **S3 Lifecycle**设置：Class之间的数据移动
  - Glacier（after60days）Deep Archive（after180days）/Intelligent-Tiering/One-Zone（easyToBeCreatedObjects）
  - Rules：Transition Actions：用于设置tier之间的object移动
  - Rules：Expiration Actions：用于设置object合适被删除，包括versioning的/可删除未完成的multi-part文件上传
  - 可以设置于特定的prefix对象，也可用于对象tags
- **S3 Analytics**：分析class使用，建议lifecycle rules，适用于Standard的class，开启后需要一两天生效
- **S3 Event Notification**：对象事件通知，需要通知的对象服务，通过服务的ResourcePolicy开启S3的对自己的访问权限，比如S3->invoke/send/push->Lambda/SNS/SQS
  - 联动**EventBridge**，就可以用EB的各种高级功能，比如objects过滤（用JSONrules）和多目的地（StepFunctions，Kinesis等）触发
- BaselinePerformance：高性能读写
  - prefixs数量无上限
  - 3500PUT/COPY/POST/DELETE
  - 5500GET/HEAD(若均匀分布，可达22000)perSecond/perPrefix
  - HEAD请求方式是只返回头部，可用于确认object是否存在
- **Multi-Part Upload**：推介100MB以上使用，5GB以上文件必须使用
- **S3 Transfer Acceleration**：传输加速使用的是edge location，并且是私有传输
- **S3 Byte-Range Fetches**：说白了就是支持部分下载，只得到想要的部分数据
- **S3 Select&Glacier Select**：也是一种变相的部分取得数据的方式，使用SQL进行Object的过滤，降低CPU和网络负荷
- **Encryption**：
  - *存储中的加密：*
  - SSE-S3：AES-256
  - SSE-KMS：因为是API，可以被CloudTrail记录，GenerateDataKeyAPI/DecryptAPI
  - DSSE-KMS：dual的D，是双层加密的意思，为了符合一些公司的安全标准的一种高级加密方式，反正就是两层嵌套加密
  - SSE-C：必须使用HTTPS，EncryptionKey必须加在每个HTTP请求上，AWS不会保存你的EncryptionKey，你要自己加密自己key，必须使用CLI，GUI中没有相关选项
  - CSE：完全自己管理，上传（*强制使用HTTPS*）前加密，拿到后自己解密
  - *传输中的加密：*
  - 两种端点HTTP/HTTPS
  - Policy设置HTTPS：aws：SecureTransport：true/false，另外还可以使用Policy强制加密
- **Access Points**：
  - 文件夹级别的访问点，可以设置自己的DNS源名和Policy
  - 可以在*VPCEndpoint*的Policy中设置只能访问某个AccessPoint
  - 可以对*同一个*桶中的*不同AccessPoint*，使用*LambdaFunctions*进行数据加工处理后返回给用户，比如数据过滤和数据丰富enrich等

### EBS Volume

（可参考SAP内容）

- 块存储的实质是attach在你的实例上的网络驱动*Network Drive*，想象成一个*network USB stick*！
- 一次只能mount一个实例，instance对EBS是一对多的关系
- 绑定（locked）于*一个AZone*，跨区复制你需要snapshot它
- 每个月30GB免费SSD存储
- 可以设置是否在删除instance的时候保留卷，可以选择只保留root卷
- *Elastic Volumes*：不需要detach就可以改变他们的参数比如size，type等

### EFS - Elastic File System

（可参考SAP内容）

- 是一种Network File System，可以被mount到*许多instances*
- 可以跨*多个AZ*和EC2一起工作
- Protocol：NFSv4.1
- 使用*SG*进行访问控制
- 适用于**Linux based AMI**（POSIX：UNIX系统接口标准规范），而*不是windows*
- 使用KMS进行存储中的加密
- **Storage Class**：
  - Standard：频繁访问
  - IA：不频繁访问（Infrequent access）
  - Archive
  - 有*LifeCycle Policy*可以设置

- 创建EFS的时候重要是你创建自己的*SG*，用来控制访问，attach在EFS上的*SG*会允许来自mount的EC2的访问
- 可以在创建EC2的时候就mount你创建的EFS系统，必须*先选择subnet*，从而可以选择相应AZ的EFS
- 这个很像是sharepoint，大家共享一个文件系统，高级

### AWS Backup

- 全自动备份的集中管理，托管型服务
- 支持*跨区域，跨账户*备份
- 服务：EC2/EBS/S3/RDS/Aurora/DynamoDB/DocumentDB/Neptune/EFS/FSx/StorageGateway
- PITR（Point to time Recovery）
- On-Demand/Scheduled 备份支持
- Tag-based备份Policies（叫做Backup Plans）
- 最后还是备份到了S3中
- *Vault Lock*：支持WORM（write once read many）防止被删除，甚至root用户也不可删

## Database

### DynamoDB

- NoSQL*非关系型*数据库，*分布式*数据库，其他比如MongoDB
- scale主要通过*水平扩展*（传统关系型数据库可以提升CPU和内存等垂直扩展方式）
- 需要的数据都以行*ROW*出现
- *多AZ*复制，高可用性
- 低延迟，高速读取和存储
- 集成**IAM**的认证和安全，集成**WebIdentityFederation**或者**CognitoIdentityPools**的认证发放功能
  - IAM的Condition设置可以进行更细致的API权限管理
  - Condition：*LeadingKeys*：可以限制只访问PrimaryKey
  - Condition：*Attributes*：可以特定限制用户能看到的attributes
- 可以用DatabaseMigrationService来进行数据迁移，从other到DDB
- 支持通过**DynamoDB Streams**的事件驱动编程作业（event driven programming）
- Table Class：Standard/Infrequent Access（IA）
- **Partitions**：数据存储于内部分区
  - 通过hash算法进行数据的分布选择
  - 计算分区有多少：取capacity和size中最大的：
    - partitionsCapacity = （RCUs/3000） + （WCUs/1000）
    - partitionsSize = totalSize / 10GB
- **PrimaryKey**：在创建table的时候添加
  - PartitionKey（Hash）：必须具有唯一性，必须种类多diverse以利于数据分布
  - PartitionKey+SortKey（Hash+Range）：该组合必须具有唯一性（比如P相同S不同），数据被PartitionKey集群（Grouped）
- Row/item：attributes可以在之后添加，每一个item的大小上限是400KB
- 支持的数据类型：
  - ScalarTypes：String/Number/Binary（图片文件等也是）/Boolean/Null
  - DocumentTypes：List/Map
  - SetTypes：StringSet/NumberSet/BinarySet
- 大数据UseCase：
  - 游戏，手机应用，即时投票，日志提取
  - *S3对象metadata管理，用于S3的对象索引数据库，可以通过invoke Lambda来写入数据*
  - 不适合传统数据库的复杂join管理，不适合大量IO率的对象存储

- **强一致性读取（Strongly Consistent Read）和最终一致性读取（Eventually Consistent Read）**
  - *最终*一致性读取是默认选项，每次写入后有可能不一致
  - *强*一致性在每次写入后都可以得到正确读取结果，用API设置参数*ConsistentRead*为True，这个模式会消费两倍的RCU

- Read/Write Capacity Modes：（两种，可变更，变更时间24hours）
  - **Provisioned Mode（default）**
    - 读取容量单元RCU-ReadCapacityUnit/写入容量单元WCU-WriteCapacityUnit
    - 突然高读写会使用Burst Capacity模式，这个模式下也被消费了，则get**ProvisionedThroughputExceededException**：这说明WCU和RCU被用完了，可能的*原因*有：
      - Hot Keys：一个分区键被读了太多次
      - Hot Partitions：某个分区负载太高
      - Very Large items：因为读写依存于item的大小
    - 建议使用指数退避策略*Exponential Backoff*，或者尽量多分区
    - 如果是RCU问题，使用*DynamoDB Accelerator（DAX）*解决问题（后面单独说）
  - **On-Demand Mode**
    - 这个模式是上面那个的2.5倍
    - 通过RRU（ReadRequestUnits）和WRU（WriteRequestUnites）进行付费（他们和RCU，WCU一样的）
    - 自动伸缩，无计划瓶颈，无限WCU/RCU
    - 对于你也无法预测的workload，比较适合这个
  - **WCU**：是指每秒（每个item为1KB单位）需要的写入单位，比如每秒写入10个items，每个项目2kb，那么每秒所需WCU就是20WCUs（10*2/1）
  - **RCU**：是指每秒（每个item为4KB单位）需要的读取单位，比如每秒读取10个items，每个项目4kb，那么每秒所需RCU就是10RCUs（10*4/4）**注意这里的4kb，如果不是4的倍数，则加上去一个成为整数**，这个是强一致性读取模式，如果是最终一致性模式，则只需要一半的RCU，除以2，为5RCUs

- **APIs**：
- 数据写入：PutItem/UpdateItem（这个也可以写入新item，可利用无锁的*AtomicCounters*方法）/ConditionalWrites
- 数据读取：GetItem：通过PrimaryKey读取，默认最终一致性读取，可设置强一致性，可以通过**ProjectionExpresion**来取得特定item的特定attributes
  ```python
  import boto3
  # Create a DynamoDB resource
  dynamodb = boto3.resource('dynamodb')
  # Select your table
  table = dynamodb.Table('Movies')
  # Perform a query with a ProjectionExpression
  response = table.query(
      KeyConditionExpression=boto3.dynamodb.conditions.Key('Title').eq('Inception'),
      ProjectionExpression='Title, Year'
  )
  # Print the items retrieved
  for item in response['Items']:
      print(item)
  ```
- 使用**Query**读取数据：
  - *KeyConditionExpression*是基于key的条件，key=xxx或者sortkey><=between等
  - *FilterExpression*这个必须是非key的attri，不能用hash或range键的attri进行过滤，过滤key条件用第一个
  - *Limit*，和sql一样可以限制items的数量
  - 支持*pagination*，结果分页表示，以节省流量
- 使用**Scan**读取数据：
  - 一种效率不高的方式，因为要先扫描整个table然后过滤数据
  - 最多返回1MB数据，然后用分页*pagination*方法持续读取
  - 提高速度可以用多个workers进行*Parallel Scan*
  - 可以使用*ProjectionExpression* & *FilterExpression（不消耗RCU）*
- 数据删除：DelteItem（支持Conditional delete）/ DeleteTable（想删除所有item的话用这个比较快）

- **Batch Operations**：
  - 一种比较省钱的处理方式，降低API的call数量
  - 并行处理parallel，效率比较高，但是可能会有个别处理失败，需要重试
  - *BatchWriteItem*：不能update，只能put和delete，失败的operations是*UnprocessedItems*
  - *BatchGetItem*：可以从多个table提取，并行提取减少延迟，失败的operations是*UnprocessedKeys*
  - 失败对应策略：增加RCU或者指数退避策略

- **PartiQL**：
  - 一种和SQL适配的语言，和SQL很像，基本就是SQL
  - select/insert/update/delete
  - 可以跨table操作
  - Run方式：Console / NoSQL Workbench for DynamoDB / DynamoDB APIs / CLI / SDK

- **LSI（LocalSecondaryIndex） & GSI（GlobalSecondaryIndex）**：
  - LSI是本地二级索引，在*同一分区键*下，使用不同的排序键进行查询，必须使用与主表相同的分区键，只能指定不同的排序键
  - GSI是全局二级索引，可以使用*与主表不同的分区键和排序键*，也就是允许在表的任意属性上进行查询
  - LSI必须在表创建的时候指定，GSI则可以在表创建后添加或删除
  - LSI最多一个表5个，GSI最多一个表20个
  - 支持*强一致性*的只有*LSI*，两者都支持最终一致性
  - *GSI*的吞吐限流throttling会影响主表，要特别注意

- **DynamoDB Accelerator（DAX）**
  - 是一种和DynamoDB无缝连接的*cache*功能
  - 低延迟，解决HotKey问题，也就是过多reads的问题
  - 默认5分钟的TTL
  - Multi-AZ，推介生产环境最少3个node
  - 安全性高，集成KMS，VPC，IAM，CloudTrail等服务
  - *ElasticCache*可以存储*聚合数据结果*，DAX一般是存储*单个的objects，query/scan*等

- **DynamoDB Streams**
  - 似乎在学习Kinesis的时候看到过，他和KinesisDataStream很像，是通过**Shards**分区数据流的，它自动扩展
  - table中的*有序的，基于item变化（create/update/delete）*的数据流，数据是可以设定的，比如只发送key，或者新的，老的item数据等
  - 可以送到：KinesisDataStreams / Lambda / KinesisClientLibraryApplications
  - 数据可以retention（存留）24小时
  - UseCases：实时数据反映，实时分析，OpenSearch服务，*跨区复制*
  - *Lambda*同步数据处理：需要赋予Lambda相应的权限，以及设置*Event Source Mapping*来读取数据流

### RDS

- 托管型传统关系型数据库，Not For BigData


## Migration & Transfer

## Compute

## Container

## Analytics

## Application Integration
## ⬆️数据工程的各种组件⬇️全面统筹和ML加持
## Security & Identity & Compliance

## Networking & Content Delivery

- 主要参考网络专家内容，只回顾关键想法
- *路由表*决定流量，决定子网的public还是private性质
- VPCFlowLogs记录的是所有IP的进出流量，比如ENI网卡
- AWS PrivateLink用于服务暴露，包括AWS的还有你自己的，为AWS或者你自己的服务创建ENI
  - 使用ELB作为服务的暴露端点
- 其他重要services：
  * VPCPeering，VPCEndpoint，VPN，DX
  * Route53
  * CloudFront

## Management & Governance

## Machine Learning

## Developer Tools
