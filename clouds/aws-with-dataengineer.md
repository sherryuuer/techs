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

### 数据保护
- **Data Mask**：掩盖一部分敏感数据：替换，打散，加密，hash，或者一开始就不要import敏感数据
- **Key Salting**：加一个random的数字之后，给数据加密，TB案件就是这样，可以防止同样的数据hash成一样的数据，最好是每条数据都有不同的salt，但是TB是共享一个salt不太好
- **Keep data where it belongs**：出于合规原因，将数据保存在它因该在的地理region，所以在数据库replica的时候要格外小心。使用OU service control和S3的region设置都是有效手段

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
- **TTL**：必须使用number类型*Unix Epoch timestamp*作为TTL的时间属性，它实质上是要**手动**创建了一个新的属性，然后设定开启该功能，内部会定期扫描你设置的TTL属性列（column），取得要删除的数据，进行删除操作，它不会消耗WCU
- 大数据UseCase：
  - 游戏，手机应用，即时投票，日志提取
  - *S3对象metadata管理，用于S3的对象索引数据库，可以通过invoke Lambda来写入DynamoDB数据*
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
  - 使用它的PartiQL查询的时候，也就只能用设定的key们查询

- **LSI（LocalSecondaryIndex） & GSI（GlobalSecondaryIndex）**：
  - LSI是本地二级索引，在*同一分区键*下，使用不同的排序键进行查询，必须使用与主表相同的分区键，只能指定不同的排序键
  - GSI是全局二级索引，可以使用*与主表不同的分区键和排序键*，也就是允许在表的任意属性上进行查询
  - LSI必须在表创建的时候指定，GSI则可以在表创建后添加或删除
  - LSI最多一个表5个，GSI最多一个表20个
  - 支持*强一致性*的只有*LSI*，两者都支持最终一致性
  - *GSI*的吞吐限流throttling会影响主表，要特别注意

- **DynamoDB Accelerator（DAX）**
  - 是一种和DynamoDB无缝连接的*cache*功能，是一种*cluster*，它会有一个endpoint，API咯
  - 低延迟，解决HotKey问题，也就是过多reads的问题
  - 默认5分钟的TTL
  - Multi-AZ，推介生产环境最少3个node
  - 安全性高，集成KMS，VPC，IAM，CloudTrail等服务
  - *ElasticCache*可以存储*聚合数据结果*，DAX一般是存储*单个的objects，query/scan*等，可以将两者*结合*，将DDB的结果存储到ElasticCache中，重复使用

- **DynamoDB Streams**
  - 似乎在学习Kinesis的时候看到过，他和KinesisDataStream很像，是通过**Shards**分区数据流的，它自动扩展
  - table中的*有序的，基于item变化（create/update/delete）*的数据流，数据是可以设定的，比如只发送key，或者新的，老的item数据等
  - 可以送到：KinesisDataStreams / Lambda / KinesisClientLibraryApplications
  - 数据可以retention（存留）24小时
  - *无法取得回溯数据*，也就是过去数据，这种性质很像*GA*，也是只能得到新的数据，过去的数据不可重复获得
  - UseCases：实时数据反映，实时分析，OpenSearch服务，*跨区复制*
  - *Lambda*同步数据处理：需要赋予Lambda相应的权限，以及设置*Event Source Mapping*来读取数据流，*Lambda主动*从DDBstream拉取poll数据的方式

### RDS

- 托管型传统关系型数据库，各种关系型数据库的引擎，Not For BigData
- 满足ACID：原子性，一致性，隔离性，耐久性
- 使用CW进行监控CPU，memory，storage，replica lag（滞后）等是一个好的实践
- VPC + PrivateSubnet + SG + EBS
- 自动备份，point-to-time 恢复
- Snapshots：Cross-region复制
- RDS Events：操作和过期的通知 - SNS
- failover：Multi-AZ 实例复制和同步
- Read Replicas：Cross-region读取操作
- 分布式read构架：
  - 通过Route53的weighted record set设置，给不同的read备份不同的权重用于用户访问权重控制
- RDS Proxy：
  - 数据库连接管理
  - 和IAM认证集成
  - 自动清理非活跃的连接，提高应用访问性能
- *APIGateway*：如果db坐在它后面，它的*rate limits*可以用于保护数据库

- Security：
  - KMS加密
  - Transparent Data Encryption（for Oracle and SQL Server）透明加密
  - SSL传输功能
  - CloudTrail无法trackRDS中的queries，因为不是API操作吧
  - EC2到RDS的IAM Role访问，依靠对RDS的API call，获取一个Auth Token

- Cross-Region Failover构架
  - 数据read replica
  - 通过Health Check和CW Alarm驱动CW Event
  - 驱动Lambda更新DNS，指向备份的read replica

- **Query Optimizations**
  - 使用indexes加速搜索
  - 避免全表scan
  - 简化where语句
  - 使用 `analyze table`：更新数据库的统计信息，这些统计信息对于数据库查询优化器（Query Optimizer）生成高效的查询执行计划至关重要
  - 控制表的大小，使用足够的RAM来存放indexes，表的数量不要太多，10000个就太多了
  - 对于PostgreSQL，在loading数据的时候，关闭数据库backup和multi-az功能，使用*autovacuum*，执行清理任务以回收这些死元组所占用的存储空间，并更新表的统计信息以优化查询性能

- **Lock command**：为了保持数据一致性的SQL语句
  - 用于数据共享的*Shared Locks*：防止同时写入，但是允许读取 -> `for share`
  - 用于数据更新的*Exclusive（排他性） Locks*：写入和读取都无法同时进行，以进行数据更新 -> `for update`
  - `lock tables table_name write;`，`unlock tables;`

### Aurora

（突出关键特性）

- 速度是MySQL的5倍，是PostgreSQL的3倍
- database volume最多可以有128TB
- 最多可以有*15个read replica*
- *Cross-region and zones* read replica复制的是整个database
- 直接从S3，load/offload 数据，*持续备份*到S3
- 30秒内实现master的failover
- 储存中加密使用KMS，传输中加密为SSL

- DB Cluster：
  - 一个Writer Endpoint - MasterDB
  - 多个Reader Endpoint - Read replica - 连接Load balancer，有Auto Scaling功能
  - Optional：分出两个read replica 作为Custom Endpoint比如进行客户自己的分析查询

- Troubleshooting
  - Performance Insights：可视化查询等待时间和用户使用情况
  - CloudWatch Metrics：CPU，Memory，Swap Usage
  - Enhanced Monitoring Metrics：at host level，process view，per-second metric
  - Slow Query logs

- Data API
  - 一种安全的用HTTPS端点，运行SQL的途径
  - 没有JDBC连接，没有持续的数据库连接
  - 必须赋予权限；Data API和Secret Manager

- RDS Proxy可以只对Read Replica进行端点连接

- Global Aurora：
  - Cross-Region read replica：高灾难恢复能力
  - Global Database：最多可以有5个备用region，每个region最多可以有16个read replica
  - Write Forwarding：通过对副DB的写入操作可以，forward到Primary的DB，主DB永远都会先更新，然后再复制到副DB

- 将RDS转换为Aurora：
  - 使用RDS的Snapshot可以重新Aurora数据库实例
  - 或者使用RDS可以直接创建Aurora的Read Replica然后直接升级为Aurora DB实例

### DocumentDB

- Aurora是AWS自己implement的替代PostgreSQL/Mysql的数据库的话，*DocumentDB*也是针对于*MongoDB*的一种同样的存在
- 用于存储，查询，和索引JSON数据
- 设计理念和Aurora是一样的，高可用性，跨3个AZ冗余复制
- DocumentDB的存储容量会自动以10GB速度增长
- 自动扩展workloads到每秒百万级响应规模

### Amazon MemoryDB for Redis

- NoSQL键值对数据库，高耐久，in-memory数据库
- 耐久性来自于，Multi-AZ的事务性处理日志log
- 可以无缝扩展，从10sGB规模扩展到100sTB规模的存储
- Use Case：网络和手机app，在线游戏，流媒体

### Amazon Keyspaces（for Apache Cassandra）

- 开源的分布式NoSQL数据库管理系统
- 高度可扩展性和无单点故障的特点，Cassandra采用了去中心化的对等架构（peer-to-peer architecture），所有节点都具有相同的权重，避免了传统主从架构中的单点故障问题
- 使用Cassandra Query Language（CQL）
- Capacity：On-demand mode或者Provisioned mode with auto-scaling
- 安全加密，backup功能，和PITR（Point-in-time Recovery）35days
- Use Case：存储IoT设备数据，time-series数据

### Amazon Neptune

- 全托管的Graph数据库，为高度互联的数据设计
- 罗马神话中的海神尼普顿，象征着海量数据之间的深层关系和连接
- 高可用性来自于跨3个AZ，15个read replicas，Aurora也是这样
- 构建和运行，需要*高度连接性数据集*的应用
- 存储上亿级别的关系和查询，延迟则为毫秒milliseconds级别
- Use Case：社交网络分析、推荐引擎、欺诈检测和知识图谱等

### Amazon Timestream

- 全托管型可扩展的时间序列数据库
- 每天可以存储和分析*兆*数量级别的events数据
- 在时间序列数据的处理上，速度是关系型数据库的1000s倍，价格则是1/10
- 适配SQL，可以schedule执行
- 最近的数据会存在memory中，历史数据则会存储于价格优化的storage中
- 内置时间序列数据分析functions，可以近实时地进行分析
- 数据当然也是加密的
- Use Case：IoT/Kinesis/ApacheFlink数据，实时时间序列数据分析
- 下游服务：QuickSight/Sagemaker/Grafana/AnyJDBC连接

### Redshift

- 全托管，PB级别的DataWarehouse
- 面向OLAP的数据库（col索引），而不是OLTP（row索引）
- SQL，ODBC，JDBC连接
- 很快很便宜，官方说很厉害，适合分析
- 内置replication和backup功能
- Monitoring：CloudWatch，CloudTrail
- **构架**：Leader Node - Compute Node x N - Node Slides （每个Node有自己的计算资源和存储）
- MPP：Massively Parallel Processing
- 列式存储，列式数据压缩
- *RA3 nodes类型*：SSD-based，有*跨region数据分享的能力*
- 支持空间数据（Spatial Data）是指与地理位置和形状相关的数据，进行地理相关的分析
- 支持*数据糊输出Data lake export*，特点就是unload到S3的时候是*Apache Parquet*格式（更快更省空间），高速且适配Spectrum，Athena，EMR，SageMaker等，*自动分区*
- **权限管理**使用Grant和Revoke命令
- 安全和加密*使用HSM*，硬件安全模型，但是很麻烦哦
  - 需要建立Redshift和HSM之间的*client-server*认证的*信任连接（trusted connection）*
  - 如果migrate没加密的数据到加密的数据库需要重建cluster，然后转移数据

- **Redshift Serverless**：
  - 自动伸缩，优化成本和性能，更容易搭建开发环境，使用机器学习进行内部优化
  - 会通过一个endpoint和JDBC/ODBC连接，或者只是控制台，进行Query操作等
  - 需要设置：IAM Role，Database Name，管理者认证， VPC，加密设置，Audit Logging
  - 在创建后，也可以管理snapshots / recovery points
  - 内部通过RPU（*Redshift Processing Units*）进行性能扩展，按per second计费，可以设置其max-limit来限制成本
  - 不支持Maintenance Windows / version tracks，所以有时候会掉线
  - 必须通过VPC访问
  - monitoring views是通过`SYS_`开头的view进行的
  - 支持CloudWatch logs，`/aws/reshift/serverless/`中，支持各种metrics，似乎都以Query开头

- **Redshift Spectrum**：
  - 可以处理*EB*（PBx1024）级别的S3中的非结构化数据，它和Athena的理念很像，但是Athena有自己的界面，但是Spectrum看起来只是Redshift中的一个表格
  - 无限制的*并行处理能力*，*水平扩展能力*，使用的是区分开的存储和计算资源
  - 支持数据压缩Gzip，Snappy压缩，以及各种数据格式

- **Durability & Scaling**：耐久性何来
  - Cluster内的replication
  - 持续性地S3备份，可以*非同步*地备份到另一个region的S3桶
  - 自动*snapshots*备份，帮我恢复了我删除的表格
  - 对于不可用的dirve和nodes，他会*内部自动replace*，毕竟是全托管的
  - 过去它只能在一个AZ中使用，现在有*Multi-AZ for RA3 clusters*了
  - 水平和垂直扩展都可以
  - 当创建新的Cluster后，旧的集群会用作read，会并行转移数据，使用CNAME切换到新的集群endpoint

- **Distribution Styles**：
  - Auto：自动分配
  - Even：所有row均匀分配，这没啥效果吧
  - Key：根据col进行分区，这个最常用
  - All：整个表复制到所有nodes，这也太大了

- **Importing/Exporting Data**：
  - *Copy*命令：
    - 并行，高效，从S3，EMR，DynamoDB，remote host导入数据
    - S3的导入需要manifest file和IAM role设置，在导入时候，如果数据是加密的，数据会被解密
    - 注意这个Copy是针对外部数据的，内部数据需要`insert into`或者`create table as`命令
    - 支持数据压缩以提速
    - 如果数据表格很narrow（就是col很少row很多）的情况，使用一次Copy会比较好，不然元数据会很大
    - *Cross-region*加密snapshot复制：要在目标region创建新的KMSkey用于加密，以及相应的copy grant权限，在本region中允许该权限进行copy行为
  - *Unload*命令，将数据导出到S3
  - *Enhanced VPC Routing*：最好设置好VPC，不然数据走的公网，就会很慢
  - *Auto-Copy from S3*：检测到S3的数据有变化就会自动导入
  - *Auraro 到 Redshift*的 Zero-ETL 自动 integration 数据复制功能
  - *Redshift Streaming Ingestion*，是从Kinesis Data Streams或者MSK的自动数据摄取
  - *DBLink*：同步和复制PostgreSQL（or RDS）数据库数据的功能，`create extension dblink;`

- *服务集成*：S3肯定，EMR，EC2，DynamoDB，DataPipeline，DataMigrationService，Glue
- *WLM（Workload Management）*：优化和管理 Amazon Redshift 集群中的查询性能。通过 WLM，用户可以配置查询队列，指定每个队列的内存和并发查询数，以确保资源的合理分配，比如有的查询时间太长，就会让优先级高的，和更短的查询优先执行
  - 默认的queue并发优先级别是5，superuser的优先级别则为1，最多设置8个queue，level可以多达50级
  - SQA（*Short Query Acceleration*）短查询可以有自己的专有查询space，避免被长的查询占用资源
    - select / CTAS
    - 内部使用ML预测查询花费的时间，你也可以自己定义，多少秒才是short
  - *Queue 设置方式*：
    - Priority
    - Concurrency Scaling Mode
    - User groups
    - Query groups
    - Query monitoring rules
- *并发扩展Concurrency Scaling*：支持多用户多查询的高并发扩展，自动增加cluster容量，通过WLM控制查询，发送查询到cluster的queue中
- **VACUUM**：recovers spaces from deleted rows

- **Redshift Resize**：
  - Elastic Resize：是通过增加和减少node数量实现的，cluster可能会有一些downtime，但是你的查询不会失败，会尽量保持连接
  - Classic Resize：会更改node的type之类的，cluster可能会在几小时到几天内变成read-only
  - Snapshot，restore，resize：如果不想让cluster不可用，可以用这种方式来创建新的cluster

- **Materialized Views**：物化视图
  - 进行了预计算和*结果预保存*的view，所以是一种需要不断和原表同步的view
  - 速度快是因为不断刷新结果，预先保存
  - 对于需要预先生成dashboard等使用情况很有用，比如下游是QuickSight
  - 创建`create materialized view`，自动刷新set `AUTO REFRESH` option on creation
  - MV的也可以基于其他MV创建

- **Redshift Data Sharing**：
  - 安全地将live的数据以*只读*的方式分享给其他cluster
  - 有助于workload隔离，降低主cluster的负荷
  - 有助于环境隔离，比如生产开发测试环境分开
  - 在数据交换的时候可以licensing data，这意味着可以进行数据买卖了
  - 细粒度控制，可以通过DB，schemas，tables，views和UDFs等来分享
  - 通过*Producer / Consumer*构架进行分享，这个真的好流行，两方数据都必须加密
  - 必须使用RA3nodes，Cross-region数据分享会产生transfer charges
  - 数据分享方式类型：Standard / AWS Data Exchange / AWS Lake Formation - managed

- **Redshift Lambda UDF**：
  - 可以在SQL中使用自定义的Lambda Function
  - 需要Lambda有相应的权限比如`AWSLambdaRole`，或者使用语句进行权限赋予`grant usage on language exfunc`for permissions
  - 使用SQL注册UDF：`create external function udf_name(int,int) returns int func_name`
  - 使用的时候比如可以在where中用UDF进行计算
  - *原理*：是Redshift将想要执行的内容通过*json格式*发送给Lambda，计算后的结果通过API返回给Redshift

- **Redshift Federated Queries**：
  - 联合查询意味着可以直接和其他RDS，Aurora等数据库进行join等操作，进行数据处理
  - 这种联合查询是单方面的：Redshift -> RDS/Aurora
  - 意味着可以使用其他数据库的live数据，并且甚至可以不需要ETL的Pipeline了，对其他数据库是*只读*权限
  - 将计算负荷分担到了其他的数据库，其他数据库要多花点钱
  - *原理*：通过建立连接，这不废话吗，其实Spectrum也是同样的原理来连接S3数据
    - 连接其他数据库，需要他们在同一个subnet或者通过VPC Peering进行连接，相互是可以看见的数据源
    - 权限：IAM Role -> Secret Manager
    - 语句：`create external schema ... URI endpoint IAM_ROLE ... SECRET_ARN ...`，spectrum也是这样的语句来建立S3的数据表
    - External Schema的详细信息存储在view中：`SVV_EXTERNAL_SCHEMAS`


## Migration & Transfer

### Application Discovery Service

- 完全的应用迁移，两种模式，Agentless 和 Agent-based，用于收集你本地的应用的各种参数
- 会被发现到S3中，所以可以用Athena进行查询

### Application Migration Services

- 原为Server Migration Services（SMS）
- rehost的方式，将本地服务持续地复制到云，使用AWS Replication Agent

### DataSync

- **数据迁移**transfer（S3，EFS，FSx之间），**数据同步**（S3，EFS，FSx），数据备份和归档，混合云构架
- 数据同步，可以是双向的，因为哪个修改都要进行同步咯
- 数据传输，可以*在AWS storage services*的服务之间进行*data*和*metadata*的复制（S3，EFS，FSx）
- 可以维持文件原有的*permissions*和*metadata*
- 下载并安装 DataSync Agent 到你的本地文件服务器
- 使用文件共享协议NFS，SMB
- 构架：一个*agent task*可能会用10GB带宽，当带宽不足，可以用*Snowcone*，它预安装了agent
  - 本地DataSync Agent - DX - PublicVIF - DataSync
  - 本地DataSync Agent - DX - PrivateVIF - PrivateLink - Interface VPC Endpoint - DataSync

### Snow Family

- 两种功能：Data Migration和Edge Computing
- Snow家族是离线数据传输设备，解决大量数据传输过慢（**网络传输带宽不足的问题**）的问题，AWS会发送你设备，你把数据装进去再发回给AWS
- Snowcone有离线发回数据的方式，还有使用DataSync发回数据的方式
- 需要安装snowball client/AWS OpsHub在你的服务器上
- **Data Migration**：
  - Snowcone（可以物理传输，也可以用DataSync传输回AWS）
  - Snowball Edge（Storage Optimized）（TB or PB）
  - Snowmobile（10PB-100PB）（安全，控温，GPS定位）
- **Edge Computing**：edge location基本是一个*离线的网络不行的地方，进行数据处理，机器学习等，甚至最后可以发回AWS*
  - Snowcone
  - Snowball Edge（Compute Optimized）
  - 可以跑EC2或者AWS Lambdafunctions（使用AWS IoT Greengrass）
- 可以使用**OpsHub**来管理snow家族，是一个可安装的软件（以前需要用CLI）
- 本地传输加速：Amazon S3 Adapter for Snowball

### DMS

- Database Migration Service
- 构架很简单：SourceDB - 安装了DMS的EC2 - targetDB
- SCT：DMS中自带Schema Conversion Tool，转换数据库结构用的，数据库相同就不用了谢谢
- 数据转移过程中SourceDB是可以使用的
- Source和Target可以是众多的本地或者云的数据库和服务，很多，但是*下面三个只能是target*：
  - Amazon Redshift，Kinesis Data Stream，OpenSearch

- Works over VPC Peering, VPN, DX
- 支持以下load方式：
  - Full Load
  - Full Load + CDC（**Change data Capture**）
  - CDC
- Continuous Data Replication实时或近实时地将数据从源系统复制到目标系统的过程目的，是保持两个或多个数据存储的同步。
- CDC在此过程中持续监控源数据库的变更，实时捕获这些变更，将变更以最小延迟传输到目标系统
- 支持Multi-AZ持续发布，使数据冗余和同步在不同的AZ

**Snowball + DMS 数据迁移过程**
- 本地用SCT将你数据进行转换并放进snowball设备，发给AWS
- AWS将数据load进S3
- DMS将S3的数据载入你的数据库

### AWS Transfer Family

- 用于S3和EFS的文件in/out服务，重点是**就是想使用FTP协议**的情况
- 可以集成现有的认证系统比如AD，AWS Cognito
- AWS Transfer for FTP/FTPS/SFTP
- 构架方式：Client - Route53 - Transfer（IAM Role） - S3/EFS
- Endpoint类型：
  - 完全被AWS托管的public类型，没法设置任何控制
  - VPC Endpoint类型的内部访问类型，可以在自己的VPC内部有私有IP和完全的访问控制（SG）
  - VPC Endpoint类型的internet-facing访问类型，可以在VPC内部设置，并通过ElasticIP设置公有IP，同时可以通过SG进行访问控制


## Compute

### EC2

（记点新的东西吧）
- AutoScaling模式，**EMR**就是依靠自动扩张，其他的还有DynamoDB，AutoScaling Group等
- 在EMR中，EC2是它的军队：MasterNode -> ComputeNodes(contain data) + TasksNodes(not contain data)

**AWS Graviton**：自己的处理器processors家族
- 支持的服务：MSK，RDS，MemoryDB，ElastiCache，OpenSearch，EMR，Lambda，Fargate

### Lambda

- 无服务器管理的网络构架方式，跑代码就行了，**感觉单一的小任务，无状态的任务，非动态网站，用Lambda足够了，虽然package管理很麻烦的，但是可以IaC**
- 构架：
  - *Serverless Website*：Client - API Gateway - Lambda - 集成Cognito - 后端使用DynamoDB
  - *Older history app*：ServerLog - KDS - Lambda - DynamoDB - ClientAPP
  - *Transaction rate alarm*：TransactionLog - KDS - KDA - KDS - Lambda - SNS
  - 作为S3和*ElasticSearch/OpenSearch*之间的粘合剂和桥梁
  - 事件数据处理流：S3 - Lambda - *DataPipeline*
  - *Data Copy to RedShift*：S3 - Lambda（DynamoDB管理数据load状态） - RedShift
- 处理Kinesis数据流的时候，太大的batch可能会timeout15min，另外payload限制为6MB，遇到error，会不断重试，有时会无法处理错误
- **File System Mounting**：
  - 可以挂载到EFS文件系统，通过*EFS Access Points*
  - 需要Lambda设置在相应的VPC中，而不是AWS的云中
  - EFS有ConnectionLimit，所以要注意挂载的Lambda数量等
- Use Case:
  - 实时文件处理
  - 实时流数据处理
  - ETL
  - Cron的替代
  - 处理Events事件
- Lambda Triggers：S3/SES/SNS/SQS/KDF/KDS（*poll the stream*）/DynamoDB/Config/CW/CloudFormation/APIGateway/CloudFront/Cognito/CodeCommit/IoT Button/Lex

### SAM

- AWS SAM（Serverless Application Model）是一个用于**构建和部署无服务器应用的框架**。它是一个开源框架，专门设计用于简化*AWS Lambda、API Gateway、DynamoDB、Step Functions*等无服务器资源的定义和管理。
- 使用简化的模板语法（基于AWS CloudFormation）来定义无服务器应用。SAM模板是*对CloudFormation模板的扩展*，提供了特定于无服务器应用的简化语法。
- SAM与*AWS CodePipeline、CodeBuild*等持续集成和持续交付（CI/CD）工具紧密集成，支持*自动化构建和部署*流程。
- 语法：官方Github中有所需要的一切代码示例，kami！
  - `sam init`初始化一个项目
  - `Type: 'AWS::Serverless::Function'`
  - `sam package`或者用`aws cloudformation package`将代码打包到云端
  - `sam deploy`或者用`aws cloudformation deploy`发布代码
  - `sam sync --watch`，`sam sync --code --(options)`快速同步本地变更，是**SAM Accelerate**功能，它使用服务API将变更*直接作用于*比如Lambda等服务，它之所以快速且同步，是因为它*Bypass CloudFormation*，直接自己干了
- 使用sam可以轻松添加*APIGateway到Lambda的event语法部分*，从而创建一个新的RestAPI端点
- 它的package，deploy功能，可以CI/CD持续集成，快速发布
- 使用Cloud9或者IDE插件就可以在本地开发

### AWS Batch

- **Run batch job as Docker Images**
- 两种方式一种是**AWS Fargate**，一种是Dynamic Provisioning for instance（EC2&Spot可设置price上限）
- fully serverless
- 处理高并发任务
- 可以用EventBridge设置schedule
- 可以用Step functions进行服务编排
- 构架方式：
  - 可以是我们案件那样的S3上传文件后Event驱动
    - 通过Lambda驱动APIcall启动Batch
    - 或通过EventBridge进行Batch trigger
  - 驱动后Batch会从ECR拉取镜像进行文件处理
  - 输出结果到S3或者DynamoDB等地方
- 这个构架很像我测试的GCP的*Cloud Run*的CICD方式，而且run的镜像部署是自动集成的
- AWS Batch和Lambda进行比较，就很像Cloud Run和Cloud Functions进行的比较

- **比较Glue和Batch**：
  - Glue是专用于ETL的，Apache Spark代码运行，Scala/Python服务，专注于ETL，拥有Data Catalog提高数据的可用性
  - Batch可以用于所有的计算相关的job，不只是ETL，ETL工作可以交给Glue，其他的可以交给Batch

- *比较App runner和Batch：*
  - AWS App Runner 如果你需要快速、简便地部署和运行 web 应用和 API 服务，而不想管理底层基础设施。像AppEngine
  - AWS Batch 则更适合那些需要运行大规模批处理作业、数据分析和高性能计算工作负载的用户，它提供了强大的资源管理和调度能力。

- *Multi Node Mode*
  - 高扩展能力，适合HPC高性能计算机
  - 适合高耦合的workloads
  - 只是为了一个job而创建了许多的nodes来执行它
  - 对Spot Instance不起作用
  - 如果EC2的lanch mode是placement group cluster就能工作的更好

## Container

* 容器的优点在于*无需依存于OS*，可以在任何机器上执行可预测行为的任何代码和服务。它是**微服务构架**的基础。
* DockerImage存放于DockerHub（公有），或者AWS的ECR，有公有仓库/私有仓库（自建）
* VM是操作系统级别的隔离，不共享资源，Docker是进程级别的隔离，共享底层设施资源
* build - push/pull - run

### ECS

- ECS use case：
  - Run MicroServices
  - Run Batch processing/Scheduled Tasks
  - Migrate Apps to Cloud
- ECS concepts:
  - ECS Cluster
  - ECS Services
  - Task Definitions(json file )
  - ECS Task
  - ECS IAM Roles
- **ECS 集成 ALB/NLB**：
  - ECS作为服务expose的时候使用LB
  - 使用Dynamic Port Mapping，分配流量：可以在一个EC2实例上发布多个container
  - ALB是标配，NLB的使用场景：需要高throughput，高性能，或者需要PrivateLink配对创建端点

- **数据存储**：使用EFS文件系统，ECS container可以进行 EFS 数据和文件节点的Mount，结合Fargate就是完全的serverless
  - 适合Multi-AZ的持久性存储分享
  - 注意，*S3不可以mount*为它的文件系统

- ECS安全：集成SSM ParameterStore和Secret Manager
- ECS Task Networking：None，host（使用底层host的interface），bridge（虚拟网络），awsvpc（使用自己的ENI和私有IP）

- **Launch方式1:使用EC2**:
- 也就是说Cluster的集群是通过EC2组成的，只要在EC2中安装*ECS agent*，就可以将他们集成到ECS的Cluster中去
- ECS Agent使用EC2 Instance Profile定义IAM Role，用于访问ECS，ECR，CloudWatch，SecretManager，SSM Parameter等
- 不同的服务需要不同的*ECS Task Role*，task role是在*task definition*中定义的
- **Launch方式2:使用Fargate**：
- 无需管理服务器，只需要定义*task definitions*

### EKS

- 开源框架
- ELB - AutoScaling - nodes - Pods
- Nodes托管模式：*Managed Node Group*和*Self-Managed Nodes*和*with Fargate*（use EFS）：完全托管
- 支持On-demand/Spot Instance
- CloudWatch监控：*CloudWatch Container Insights*
- DataVolumes：使用*CSI（Container Storage Interface）driver*，连接到EBS，EFS，FSx
- Kubernetes中的*Taints和Tolerations*是一种机制，用于控制哪些Pods可以调度到哪些Nodes上。这种机制通过向节点（Node）施加“污点”（Taint）来排斥Pods，然后Pods通过“容忍”（Toleration）这种污点来选择性地允许调度。

### Amazon Fargate work with ECS/EKS

- Fargate是一个Serverless的服务，无需构架基础设置，直接添加container

### ECR
- 支持image漏洞扫描（on push），版本控制和lifecycle
  - 基础漏洞扫描由ECR进行，并触发EventBridge发布通知
  - 高级Enhanced扫描通过Inspector进行，由Inspector触发EventBridge发布通知
- 通过CodeBuild可以register和push image到ECR
- 可以cross-region 或者 cross-account 地 replicate image
- 私有和公有，公有的叫ECR Public Gallery，后端是S3
- 通过IAM管理访问权限

### ECS/EKS Anywhere

- ECS部署在包括你的数据中心的任何地方
- 安装*ECS Container Agent和SSM Agent*
- 使用EXTERNAL的launch type
- 必须要有稳定的AWS Region连接

- EKS则需要安装在数据中心，使用EKS Anywhere Installer
- 使用*EKS Connector*从AWS连接到本地的EKS Cluster
- 以上是有连接的设置
- 无连接的设置需要安装EKS Distro，以及依赖开源工具管理Clusters


## Analytics

### AWS Glue

- 用途：*数据发现和管理（S3/RDS/Redshift/其他数据库）*，以及*ETL（事件驱动，时间驱动和on-demand）*
- 不适合Multiple ETL engines，它基于Spark，如果需要其他引擎，用Data Pipeling，EMR比较好，但是基本Spark无所不能了现在

- **Glue Crawler / Data Catalog**：
- 构架：S3 -> Glue -> (Redshift Spectrum/Athena/EMR) -> QuickSight
- 扫描S3数据，将数据schema化，可以并行执行，生成Data Catalog，只管理数据结构用于后续查询，数据还是在S3中
- 数据在S3中的*文件夹结构*，将决定Crawler如何生成数据*分区partition*方式，比如yyyy/mm/dd/hh等
- **Modify Data Catalog**：使用ETL脚本，重跑Crawler，更新schema和partitionKey，开启功能和option：*enableUpdateCatalog*和*partitionKeys*options，只支持S3，各种文件格式支持，但是parquet有自己独特代码格式，另外不支持嵌套的schema变更
- **Glue + Hive**：
* Hive（EMR）可以让你用SQL化等语言进行数据查询
* Glue Data Catalog则为它提供了一个元数据
* 也可以自主import一个Hive的元数据仓库到Glue中

- **Glue ETL**：以second计费
- 有自动代码生成/Scala/Python/PayForUse/在*Spark*平台上运行
- 加密：Server-Side at rest，SSL in transit
- 性能提升上，可以增加*DPU*（data processing units）来为底层Spark Jobs提速
  - 通过Job Metrics来帮助用户理解需要多少DPU
- Errors会传送到CloudWatch，后续可以发布SNS消息
- Glue Scheduler：进行jobs的schedule管理
- Glue Triggers用于自动化事件驱动的job运行
- *Bookmarks*：执行中防止重新处理re-run已经处理过的数据
- 集成*CW Events*：后续基于job的成功失败，执行LambdaFunction或者SNS通知，可以invoke EC2，发送events到Kinesis，或者驱动Step Function
- *Glue Development Endpoints*：用notebook作为开发环境开发ETLjobs，通过VPC和SG进行控制，以分钟计费

- **现在支持 Streaming 数据处理**：
- 支持serverless流数据处理ETL：Kafka/Kinesis -> transform in-flight -> S3/other data stores
- 依赖*Apache Spark Structured Streaming*库

- **DynamicFrame**：
  - 是*DynamicRecords*的集合，自描述，并且还有自己schema
  - 很像是*SparkDataFrame*，但是有更多的ETL功能
  - 有Scala和Python的API

- **Transformations**：
  - 内置处理：DropFields / DropNullFields / Filter / Join / Map
  - FindMatches ML：使用机器学习查找数据集中的匹配数据或重复数据
  - 文件格式转换，比如转换成Parquet就很好
  - 所有ApacheSpark可以做的转换都可以做到比如Kmeans机器学习

- **Glue resolveChoice**：
- 用于解决DynamicDataFrame中的数据歧义问题，比如同名的数据但是type不一样
- 数据合并或转换过程中，某个字段有多个数据类型时，resolveChoice 用于选择一个确定的数据类型，以确保数据的一致性
- 常用策略：
  - *cast*：将字段的数据类型转换为指定类型。
  - *project*：选择其中一个类型并放弃其他。
  - *make_cols*：将不同类型的数据分成不同的列。
  - *nullify*：将无法解析的数据转换为 null。
  - *match_catalog*：使用AWS Glue数据目录中的表定义来解决类型冲突。
  - *delete_col*：删除带有冲突的列。

- **Glue Studio**：
  - 可视化界面，用于创建ETL工作流
  - Job编辑（设置源，目的地和转换内容）
  - 监视Job执行的dashboard

- **Glue Data Quality**：
  * 实质上是一种数据的*规格合集*（rule set），可以自己定义也可以用官方推介的
  * 使用*DQDL*语言（Data Quality Definition Language）
  * 其结果可以用于决定是否让job失败成功，或者也可以发送到Cloud Watch

- **Glue DataBrew**：
  * 可视化*数据预处理工具*，导入数据集，在界面上解析数据，转换数据
  * 通过创建*Recipes*来进行数据转换工作
  * 可以定义Data Quality Rules
  * 可以通过custom SQL从RedShift和Snowflake中创建数据集
  * 集成IAM，KMS，SSL安全，集成CW和CloudTrail
  * 转换好的数据可以用于创建一个Job，并且可以cron执行，还可以直接在界面上下载转换好的数据结果
  * *Studio*是用于定义*工作流*，这个是用于*可视化数据转换*的非常详细的操作的
  * 处理**PII数据**：替换substitution，打乱shuffling，加密encrypt，删除delete，隐藏部分信息Mask，哈希化Hash

- **Glue Workflows**：
  * 编排服务，编排目标主要是Glue中的job，可以通过控制台，blueprint，或者API创建
  * Trigger可以是Schedule，On-demand，或者EventBridge的events事件

- 对比**Glue ETL和GCP的Dataproc**：
  - 服务类型：AWS Glue更侧重于ETL任务和自动化数据管道，而GCP Dataproc则侧重于通用的大数据处理和分析。
  - 技术栈：Glue支持PySpark等，而Dataproc则完全支持Apache Spark、Hadoop及其生态系统。
  - 集成性：两者均与各自平台的其他服务深度集成，Dataproc可以更好地与GCP的数据分析工具结合，如BigQuery和Dataflow。
  - 灵活性：Dataproc提供更灵活的集群管理和配置选项，适合复杂的大数据分析场景。
  - Google Cloud Dataproc 是GCP中使用Apache Spark的主要服务，对应AWS Glue的某些功能，但更专注于全面的大数据处理和分析能力。
  - AWS Glue 则是一种全面的ETL解决方案，适用于构建和管理数据管道，特别是处理ETL任务时具有便利性。

### Lake Formation

- **旨在让创建安全数据湖变得简单！**
- 自动导入数据和监控数据流，可设置分区，数据源可以是S3或者其他数据库等
- 统一的数据集中管理权限，加密和keys管理，高度*数据访问控制管理*
- 提供了ETL（提取、转换、加载）功能，帮助转换原始数据到分析就绪的格式，支持使用AWS Glue进行数据转换
- *数据治理*：审计和监控，支持对数据访问的监控和审计，帮助用户跟踪数据使用情况和权限变更
- 集成S3，下游使用Athena，RedShift，EMR进行高级分析
- Cost：本身没有费用，但是底层的Glue，S3，EMR，Athena，Redshift需要付费
- 支持Cross-account数据访问
- 支持*Governed Tables*
  * 是一种S3表的新形态
  * 这种表支持ACID事务处理
  * 可以和Kinesis流式数据集成，可以用Athena进行分析
  * 可以在row行和cell的级别上可以进行安全访问控制
  * 使用*Automatic Compaction*技术进行存储优化：自动合并存储系统或数据库中的小文件或小数据块，以减少碎片化，提高读写性能和存储效率。

- **安全和数据权限**：数据湖中，感觉安全真的很重要，也就是*数据治理*方式
  * 集成IAM users/roles，SAML或者external AWS accounts
  * 可以在databases，tables或者columns上用Policy tags
  * **Data Filters**：列，行，单元格cell级别的*安全管理*
    - 在进行`grant select`permission的时候用的
    - `All columns` + `row` filter = row-level security
    - `All rows` + specfic `columns` = column-level security
    - specific `columns` + specific `rows` = cell-level security
    - 可以如上通过控制台console创建，也可以通过`CreateDataCellsFilter`API


- 使用*步骤*：
  * 数据导入：定义数据源（例如S3、RDS），并使用Lake Formation导入数据到数据湖。
  * 数据转换：使用AWS Glue进行数据转换，准备分析就绪的数据。
  * 数据编目和元数据管理：自动生成数据目录，编目数据集并为其添加元数据。
  * 权限管理：设置细粒度的权限策略，确保用户和服务仅能访问他们被授权的数据。
  * 数据共享和分析：将数据湖中的数据与其他AWS服务集成，如Athena进行分析，或通过QuickSight进行可视化。
  * 治理和审计：使用Lake Formation的审计功能监控数据访问，确保数据使用符合治理政策。

- 在GCP中，与AWS Lake Formation 对应的服务是 **BigLake**：统一的数据湖分析平台，旨在简化大规模数据分析和管理。它集成了Google Cloud Storage和BigQuery，并扩展支持多种存储格式。

### Athena

- S3御用SQL查询工具，无需load数据，直接查S3
- 可用于分析所有存储在S3中的其他服务数据：CloudTrail，CloudFront，VPC，ELB等服务的*logs*
- 底层是*Prosto*：开源的分布式SQL查询引擎，专门用于高性能地查询大规模数据集
- 支持各种格式和数据类型：
  * 文件格式支持CSV，TSV，Json等人类可读格式
  * 文件格式支持*列式分布的ORC，Parquet*和*行式分布的Avro*格式
  * 注意：**使用列式文件可以提高性能，查询大文件比很多小文件的性能更高，使用分区键也可以提高性能**
  * 使用分区命令提升性能：`MSCK REPAIR TABLE`，是一个用于 Hive 和其他兼容 SQL 查询引擎（如 Amazon Athena）的命令。它用于修复分区表，更新 Hive 元存储中有关表分区的信息
  * `ALTER TABLE ADD PARTITION` 是手动指定新分区的方法，而 `MSCK REPAIR TABLE` 是自动发现和注册新分区的方法
  * 数据类型支持结构化，非结构化，半结构化数据
- Use Case：查询web logs，或者在数据load到Redshift之前进行预查询
- 集成：Jupyter，Zeppelin
- 集成：QuickSight
- 集成：ODBC/JDBC等的其他可视化工具
- 集成**Glue**的DataCatalog，当然其他各种query工具都可以集成，通过**Athena查询掌握DataCatalog**获取数据，并可以通过QuickSight可视化
- 通过IAM-based policy来限制database和tables级别的安全限制（database和Glue data catalog）

- **通过CTAS转换格式存储**：
  * 通过命令将数据重新存储为新的数据格式存储在S3中
  * `Create tabel new_table with (format = 'Parquet', write_compression = 'Snappy') as select ~`
  * `Create table new_orc_table with (extarnal_location = 's3 path~', format = 'ORC') as select ~`

- **Athena Workgroups**：
  - 是一种集成*IAM，CW，SNS*的工具，用于管理*users/teams/apps/workloads*的权限，以及他们能使用何种query等
  - 这个工具拥有自己的：Query记录，Data Limits，IAM Policies，和Encryption设置

- **Cost**：
  * 用多少花多少，1TB花5刀，*查询失败不花钱哈哈*，对于*DDL（Create/Alter/Drop）不花钱*
  * 查询ORC和Parquet这样的*列式文件*会省钱30%～90%
  * 其他的服务S3和Glue需要另行付费

- **Security**：
  * 访问控制使用IAM，ACLs，和S3 bucket policies
  * 相关的权限：AmazonAthenaFullAccess / AWSQuicksightAthenaAccess
  * S3的数据加密就用S3自己的就好了，跨账户的访问也是依靠S3的*bucket policy*
  * S3和Athena之间的数据传输加密靠*TLS传输层加密*

- **Ahena ACID tanssaction**：对原子事务处理的支持
  - Powered By Apache Iceberg：只需在创建表的时候加上`table_type` = `ICEBERG`即可，用户可以在row层级之行安全变更操作
  - 可以适配任何支持iceberg的服务比如EMR和Spark
  - 移除了custom record locking功能
  - 支持时间旅行功能`select statement`
  - Lake Formation的*Governed tables*也是另一种实现ACID的功能
  - 性能是通过*定期压实Periodic compaction*实现的，这翻译很绝

- **Amazon Athena for Apache Spark**:
  * 可以在Athena的控制台中跑Jupyter notebook，是KMS自动加密的
  * serverless，是可扩展的分析引擎（另一个是Athena SQL）
  * 依靠Firecracker快速扩张资源
  * 可以通过编程式的API和CLI进行访问
  * DPU：基于DPU付费，内部可以自动调节DPU的调度和执行size
  * 总之就是有这么个功能，除非是非常喜欢Athena，不知道用的人多不多

### EMR（Elastic MapReduce）

- 在EC2上跑的**Hadoop框架**，正因为是在EC2上跑，所以用户的*控制权更多*，可以选择安装Spark在集群上，自己掌控更多行为
- 内部工具包括*Spark，HBase，Presto，Flink，Hive*等
- 拥有*EMR Notebooks*可用
- EC2在集群中称为*节点Node*
  * Master node：leader node，就是一台EC2，感觉单点障碍，追踪任务，监控健康
  * Core node：HDFS数据节点，并且可以跑任务，至少要有一个core node
  * Task node：不存储数据，只跑任务，可有可无，用Spot instance很棒，省钱，并且不会丢失数据
- 两种集群：*Transient（临时）和Long-Running（长期）*
  - 前者跑完任务就自动删除了，后者可以买长期的，适合需要一直执行的任务
- 框架Frameworks和应用Applications是在cluster launch的时候就决定好的，**如何跑任务**：
  * 一种*直接接入master node*来踢，跑任务
  * 一种在*console定义任务steps*然后在console中invoke任务
  * 可以在*S3或者HDFS*中跑数据，输出到S3或其他地方，S3支持*强一致性*
  * 本地数据*local file system data*，以buffer和cache的形式存在，会随着node的关闭而消失
  * 存储还可以attach*EBS*，当集群关闭，会被删除，如果你手动删除cluster上的EBS，集群会以为EBS卷失败了他会替换一个新的！
- 集成各种AWS服务，包括IAM，VPC，CloudTrail等，DataPipeline可以用来schedule和start集群

- **EMR Managed Scaling**：支持*instance groups*和*instance fleets*的自动伸缩，增加从core node开始，减少从task node开始
- **EMR Serverless**：
  - 设置参数设置job然后就可以跑任务了，不需要管理底层server
  - 设置spark script，Hive query等定义job
  - All within one region（跨多个AZ）
  - 即使如此还是需要一定的知识，知道自己需要多少workers和设置
  - 初始化容量（*pre-Initialized capacity*）的时候为jobs多预留10%的capacity，防止spark的overhead
  - 全程加密，安全性OK
- **EMR on EKS**：支持在k8s上跑Spark作业，全托管服务，和k8s上的其他apps进行资源分享

- **数据加密**：
  - 在S3和node也就是EC2本地都会被加密
  - 在node中的disk加密分为：
    * EBS加密，KMS加密可以用于Root，但是LUKS加密无法用于Root
    * EC2 instance store的加密可以用NVMe加密和LUKS加密的方式

- NVMe加密：通常指的是硬件级别的加密，由NVMe固态硬盘自带的加密引擎执行，能够提供透明和高效的数据保护。
- LUKS加密：是一个软件加密方案，通过Linux内核的加密模块（dm-crypt）提供灵活的分区加密功能，通常用于加密整个磁盘或分区。

### Kinesis

- 托管的，实时流数据处理，3AZ自动同步复制
- 数据可以是logs，IoT数据，客户点击数据
- Kinesis Stream是低延迟的，可扩展的，流数据摄取
- Kinesis Analytics是实时数据分析平台，可用SQL
- Kinesis Firehose将数据流载入S3，Redshift，ElasticSearch和Splunk
- 者三个服务可以相连的
- 安全：KMS，IAM，VPC endpoints，HTTPS

**Kinesis Stream**

- 关键组件，*有序的Shard*，生产者，和消费者，有PubSub类似的功能
- *数据存留*为1天到一年，而KDF没有数据存留机制，只能传输
- 实时处理和可扩展吞吐能力
- KDS是*real-time*的，这是它和KDF的一个区别
- KDS可以通过自己编写代码来控制生产和消费逻辑
- 一旦数据被插入，就无法被删除，它很像是放大版的SQS，但是SQS的message是可以被删除的
- 关于*Shard*：
  - 里面的records是有序的，是有**partitionKey**的，同分区的数据会进入同shard
  - 批处理或单个信息处理
  - 两种capacity模式：On-demand 或者 Provisioned
  - **Shards的数量变更**是可以evolve，也就是演变的，可以重新分割*split shard*（hot shard就是流量较大的shard），或者*merge shards*，增加和减少数量
    - 注意，这种变更可能会导致顺序错误！比如先取得了child shard的数据3和4，然后才取得parent shard的数据1和2，导致顺序错误
    - 所以，要注意代码逻辑，先读完所有parent shard的数据，然后再读取child shard的数据

- 消息生产者：**Producer**
  - SDK是一种简单的生产者
  - **Kinesis Producer Library（KPL）**是 Amazon Kinesis 的一个关键组件，用于帮助开发者高效地将数据发送到 Kinesis 数据流中。它提供了优化的数据发送和处理机制，支持大规模、实时数据处理和分析，适用于需要快速处理和分析大量实时数据的应用场景。
    * 有同期和非同期（同步和异步）处理API可以用，*异步API*的性能更高
    * 可以进行*batching*处理，对数据进行*收集collect*和*集合aggregate*
  - Kinesis Agent：直接将服务器的log等送到Kinesis
  - 生产者每秒每shard，能发1000messages或者*1MB*消息
  - **SDK的API**的*PutRecord*是发送一条记录，*PutRecords*是复数体，发送很多记录
  - **ProvisionedThroughputExceeded**意味着发送的量超过了上限
    - 可以通过*backoff*进行retires重试
    - 可以*扩张shards的数量*
    - 确保你的*分区键*具有良好的分区功能而不是不平衡数据
  - **RecordMaxBufferedTime**参数设置较长会导致大的延迟，如果无法容忍这种延迟，就需要直接用SDK
  - **Producer处理重复数据**：生产者会发送重复数据的原因，一般是因为网络原因的timeout，消费者无法ack，导致生产者重复发送信息
    - *去重的方法是*：embed unique record ID到数据中去，来去重

- 消息消费者：**Consumer**
  - SDK，感觉内部都是通过api代码走的
  - Lambda，通过Event source mapping -> S3/DynamoDB/RedShift/OpenSearch
  - **KCL（Kinesis Client Library）**是亚马逊 Kinesis 的客户端库，用于开发消费者应用程序（Consumer Applications），这些应用程序从 Kinesis 数据流中读取和处理数据，可以分布式使用，并将checkout写入DynamoDB进行检查点管理
  - 两种消费者模式，*Classic*和**Enhanced Fan-Out**模式：Classic模式中所有消费者总共可以每shard读取*2MB*信息，以及最多5个APIcall，在Fan-out模式中，每个消费者每个shard可以读取2MB模式，并且没有APIcall需求，因为是*push model*
  - **Comsumer会重复读取数据两次的原因**：
    * 一个worker工作节点突然被终止
    * worker instances被增加或者移除
    * shards 被合并或者分割
    * 一个application被deployed
    - *解决方案*来说，确保app处理数据的幂等性，或者确保下游可以处理unique ID，确保更新比如merge处理


**Kinesis Data Firehose**

- 不能存储数据，就是一个管子
- 数据源中可以包括Kinesis Data Stream
- Record批大小最小可以有1MB
- 在KDF，可以用Lambda进行Data transformation，比如格式转换，数据变换，和压缩
- 数据写入目标：
  - *S3*
  - *Redshift*（是一种copy方法，需要通过从S3复制到redshift）
  - *OpenSearch*
  - 第三方：MongoDB，Datadog，*Splunk*等（Splunk收集机器生成的数据，未来IoT好像会很有用）
  - HTTP/S Endpoint
- 它是一种近乎实时的*near real time*，因为它是batch写入的，根据buffer的时间或者size来决定
  - buffer interval，可以从0到900秒
  - buffer size，可以设置几个MB
  - 一旦到达buffer设置rule，就会进行批处理，flushed（冲走）
- 如果需要实时构架，需要Kinesis Data Stream + Lambda的构架
- 可以将数据备份送往S3
- 注意：*Spark Streaming和KCL不会从KDF读取数据*

**Kinesis Data Analytics**

- 可以对KDS和KDF等，实时数据进行分析
- 还可以结合其他的静态reference数据进行SQL分析
- 分析结果，可以继续作为流数据输出给KDS或者KDF，输出到S3或者Redshift等
- Use Case：
  - Streaming ETL
  - 持续指标生成，比如一个比赛中的leaderboard
  - 响应式数据分析
- 可以用Lambda作为数据pre-processing
- 数据处理可以用SQL或者*Flink*（under the hood）
- schema discovery
- RANDOM_CUT_FOREST：机器学习模型，SQLFunction，用于检测数值异常

### Amazon Managed Apache Flink

- Flink是KDA的底层
- 实时数据分析、监控系统、事件驱动应用和数据管道等场景
- 支持复杂的流式和批处理任务，如数据过滤、聚合、窗口操作和机器学习
- 构架：Kinesis Data Streams/Managed streaming for Apache Kafka -> Managed Service for Apache Flink -> S3/Kinesis Data Streams/Kinesis Data Firehose

**Kinesis Video Stream**

- 它的流输出*不能输出*到S3中去，底层数据是在S3中，但是我们无法直接访问它
- 它的输出可以接*Rekognition*进行识别任务，后续的识别后数据可以跟Kinesis其他服务

### 流数据工程构架

- KDS数据源 - KDA数据分析 - 生成结果送往KDF - S3 - Redshift，或者ElasticSearch
- KDF直接摄取数据 - S3
- *DynamoDB Stream* - Lambda构架十分贵，替代方案就是KDS - Lambda - KDF - S3

### Amazon MSK

- Managed Streaming for Apache Kafka
- Kafka 是一个分布式流处理平台，用于构建实时数据管道和流应用，以高吞吐量和低延迟可靠地处理数据流。
- Multi-AZ高可用性
- 数据存储在EBS中
- Kafka 的关键组件包括主题（Topic）用于数据流的分类，生产者（Producer）发布消息，消费者（Consumer）读取消息，代理（Broker）存储消息，和 ZooKeeper 管理集群元数据。
- 多AZ部署高可用性
- 可以创建删除cluster，MSK会帮你管理nodes
- MSK的下游消费者：
  - Kinesis Data Analytics for Apache Flink
  - Glue Streaming ETL jobs powered by Apache Spark Streaming
  - Lambda
  - Application running on EC2，ECS，EKS
- 网络安全上，通过Kafka client和SG控制
- 数据安全上，通过KMS，TLS in-flight
- 认证认可上，Kafka ACLs加上其他的认证功能比如MutualTLS，或者IAM Access Controls全包认证认可
- *MSK Connect*数据连接
- *MSK Serverless*

- **对比Kinesis Data Stream的不同**：
  - KDS的message大小上限是1MB，MSK上限可以设置为10MB
  - DataStreams with shards - Kafka Topics with partitions
  - shard可以分割和合并，Topic的partition只能增加
  - 两者in-flight都可以加密，Kafka也支持不加密的Plaintext
  - 在认证上KDS只有IAM Policy，Kafka有好几种组合方式

### OpenSearch

- 是一个ElasticSearch和Kibana的*fork*
- 不适合事务处理OLTP，主要为了检索和分析
- 查询的对象都是*documents*，在index的基础上搜索，document->shard
- *Index State Management*
- *Cross-cluster Replication*：leader - followers
- *Use Cases*：
  - Log分析
  - 实时应用监控
  - 安全分析
  - 全文本搜索
  - 点击流分析
  - 索引
- *Logstash*：
  - 是CW Logs的替代品
  - 使用Logstash Agent管理
- *OpenSearch Dashboard（以前叫Kibana）*：
  - 提供实时仪表盘服务
  - 是CW仪表盘的替代方案
  - 安全地访问它需要EC2上一个前端代理服务器，DX连接到VPC，或者VPN连接
- *构架*
  - 可以收集和存放DynamoDB的数据流结果
  - 可以实时收集Kinesis Data Firehose的数据结果
- 资源的集合为*Domain*
- 可以*snapshots*到S3中
- *安全*：
  - Resource-based policies
  - Identify-based policies
  - IP-based policies
  - Request signing
  - VPC
  - Cognito
- *Storage*：
  - Hot Storage：EBS
  - UltraWarm/Warm Storage：S3 + Caching
  - Cold Storage：S3
- *稳定性stability的影响因素*：
  - 影响稳定性的第一原因经常是disk磁盘空间不够了，用完了
  - 有时候需要限制每个node的shards数量
  - shard/cluster不平衡，会影响，增加JVM的内存压力 -> 删除old或者不用的index
- *OpenSearch Serverless*：
  - On-demand autoscaling
  - 两种collections类型：timeseries / search
  - 通过KMS加密
  - 通过OCUs计算capacity（Opensearch compute unites）

### QuickSight

- 更像是一个BI网络应用，可视化，分页报告，分析仪表盘
- 数据源可以是文件，数据库，其他服务比如Athena，Redshift，还有Opensearch等
- **SPICE**：并行，in-memory，列存储数据，高速计算，machine code generation，载入数据超过30分钟会超时
- *安全*：多要素认证，VPC连接，私有VPC通过ENI连接，DX连接，可在row和col等level上设置访问安全权限
- QuickSight + RedShift，需要两个服务在同一个region
  - 两个服务在不同region的解决方案是通过设置Redshift的SG，允许QuickSight的IP range的访问
  - *跨账户*连接两个服务：
    * 可以设置Transit Gateway连接前提是需要是同一个组织和region
    * 或者设置不同region的Transit Gateway的Peering
    * 可以通过PrivateLink进行私有连接
    * 可以使用VPC Sharing来连接他们

- 用户安全管理：
  - 使用IAM或者用email登录
  - 集成AD connector（只有Enterprise Edition可以）

- 嵌入式仪表盘embedded dashboard用于嵌入应用中
  - 集成AD，cognito，SSO
  - JavaScript SDK/QuickSight API
  - 通过设置白名单domain，控制谁可以访问

- ML功能嵌入
  - 引以为豪的*Random cut forest*模型进行异常数值检测，forecasting预测
  - 为你的仪表盘增加 story of the data


### 知识补充

**Apache Hive**是一个构建在Hadoop之上的数据仓库系统，用于在Hadoop分布式文件系统（HDFS）上进行数据的查询和分析。它提供了一种类似于SQL的语言，称为HiveQL（Hive Query Language），使用户能够使用SQL语法来查询存储在Hadoop中的大规模数据集。Hive将SQL查询转换为MapReduce作业，以便在Hadoop集群上并行处理数据。Hive适用于批处理、ETL（提取、转换、加载）操作和数据分析。它支持用户自定义函数（UDFs），并能与其他大数据工具如Pig和Spark集成。通过抽象复杂的MapReduce操作，Hive大大降低了大数据分析的门槛，方便数据工程师和分析师高效处理和分析海量数据。

**Custom record locking** 是一种手动实现的机制，用于在数据库中防止多个用户或进程同时访问和修改同一条记录，以避免数据不一致或冲突。尽管这种方法提供了灵活性，可以根据特定业务逻辑实现锁定策略，但其缺点在于可能引入复杂性和错误，增加开发和维护成本。此外，不当的锁定策略可能导致死锁、性能下降、资源浪费，以及更难以处理并发问题，特别是在分布式系统中。使用内置数据库锁定机制通常更可靠和高效。

**Periodic compaction** 是一种定期执行的存储优化技术，用于合并和整理数据文件，以减少碎片化、提高存储效率和提升读写性能。在数据库和大数据系统（如HBase、Cassandra、Kafka）中，数据随时间增多，可能会产生许多小文件或分散的存储块。通过周期性地合并这些文件，系统可以减少I/O开销，降低存储成本，并提高查询效率。这种方法需要在性能和资源利用之间进行平衡，避免频繁合并对系统造成负担。

**Firecracker** 是一种开源虚拟化技术，由 AWS 开发，专为运行无服务器计算和容器化应用而设计。它通过微虚拟机（MicroVMs）提供安全隔离和高效资源利用，启动速度快，资源开销低，非常适合于 FaaS（Function as a Service）和 CaaS（Container as a Service）环境。Firecracker 是用 Rust 语言编写的，具备内存保护和并发安全特性。它被广泛应用于 AWS Lambda 和 AWS Fargate 等服务中，以提供轻量级、高性能的计算实例。

## Application Integration

### SQS

- 消费者poll消息后被处理然后被queue删除，也就是说一个消息只会被一个consumer消费
- 无服务器，和IAM认证集成
- 能处理突然很大的扩展不需要预测provision
- 解藕服务
- message的最大size是*256kb*
- 可以从EC2，Lambda读取数据
- 可以作为DynamoDB的writebuffer
- SQS FIFO是先进先出的，原本的 SQS 是没有顺序的，我的天
- 安全：KMS，HTTPS传输，IAM Policy，SQS Access Policy

- **SQS Extended Client**：
  - 为了发送超过256kb的数据
  - 实质是发送实际数据到S3，而message中只发送metadata的信息作为消息发送，然后consumer从S3中提取数据

- **DLQ**：当消息在visibility timeout的时间内多次都没被处理就会被送进dead letter queue，它适合之后的debug
  - Redrive to source：当你的代码问题修复了，可以将这些message重新放回原本的queue进行处理
  - 因为你可能会重复处理message，最好确保你的处理具有幂等性（Idempotency）比如upsert处理

- Lambda 的 *Event Source Mapping* 是一个配置，将 Lambda 函数与事件源（如 SQS 队列、DynamoDB 流、Kinesis 流等）关联起来。它使 Lambda 函数能够自动响应这些事件源的事件，无需手动触发。
  - 支持多种事件源，如 SQS 队列、DynamoDB Streams、Kinesis Streams、Amazon MQ、MSK 等。
  - 批量处理：Lambda 可以批量读取和处理事件，以提高效率和性能。例如，可以从 SQS 队列或 Kinesis 流中批量读取记录。
  - 并行处理：对于 Kinesis 和 DynamoDB Streams，Lambda 可以并行处理多个分片中的记录，提高吞吐量。
  - 重试机制：内置重试机制，确保在处理失败时自动重试，以提高事件处理的可靠性。
  - 自定义批量大小：可以配置每次读取的记录数量（批量大小），优化处理性能。

- *Kinesis Data Stream vs SQS*
  - KSD是为了流处理，信息最大可以存留1年，信息删除基于时间和设置，有序，对象大小1MB上限，信息会被消费一次以上
  - KDF在目的地最多可以有128MB，KDF本身不留存数据
  - SQS是为了解藕应用，信息被消费后就会被删除，最长存留14天，对象大小256kb，信息被消费一次以上，但是FIFO会正好被消费一次

### SNS

GCP的PubSub对标的是SQS和SNS两个服务。

- No ordering，No retention数据留存
- SNS主要是发布订阅服务PubSub
- *Event Producers*主要有：
  - CWAlarm，Budgets，Lambda，ASG，S3Bucket Events，DynamoDB，CloudFormation state change，RDS Events and so on
- 10万个topic上限，1250万个Subscriptions上限
- *SNS的Publish方法*：
  - Topic Publish（使用SDK）：创建topic和subscriptions，然后发布到topic
  - Direct Publish（为mobile apps SDK）
    - 创建platform application，创建platform endpoint
    - publish到endpoint，这更像是push操作
- *SNS的publish的目的地*：
  - SQS，Lambda，Kinesis Data Firehose，HTTP（S）endpoints，Emails（是一种端点）
- *Security*
  - HTTPSAPI通信加密
  - KMS存储加密
  - IAM Policy访问限制
  - SNS Access Policy：支持跨账户访问（就像S3的bucket access policy）

- **SQS + SNS fan out 构架**
  - 一个 SNS topic，多个 SQS subscriber
  - 比如一个S3 Event可以通过该构架发布给多个后续服务
  - SNS FIFO - SQS FIFO：FIFO的SNS只能以FIFO的SQS作为订阅者
  - **Message Filtering**
    - 如果subscription不过滤信息就会收到所有的message
    - 通过条件过滤可以收到各自想要的信息，Json Policy格式

- **Kinesis Data Firehose**
  - SNS Topic - KDF - S3/other services

- 服务器端发生错误则会自动应用重传策略
- 只有Http/s支持Custom delivery policies
- 当重传也失败了则会丢弃消息，除非设置了SNS Dead Letter Queues功能

### Step Functions

- 使用 JSON 格式的*状态机state machine*定义，开发人员可以轻松地编排和管理跨多个 AWS 服务的任务
  * task state：任务单元
  * choice state：条件单元
  * wait：delay状态
  * parallel：增加一些branch并行处理分支
  * map：也是并行的，对dateset里的每一个元素进行map处理
  * pass，success，fail状态单元
- 编排Lambda，Batch，ECS，DynamoDB，EMR，SQS，SNS等200多服务
- *可视化workflow*的流程，有点像Airflow
- trigger：Console，SDK，CLI，Lambda，API Gateway，EventBridge，CodePipeline，StepFunctions
- *不能集成Mechanical Turk*服务（在线人工劳动力），这里能集成它的服务是SWF服务
- Standard 和 Express Workflow两种类型：
  - 前者贵后者便宜
  - 前者执行时间可以长达1年，后者最大5分钟
  - 启动rate前者是每秒2000，后者是每秒10万
  - *前者确保执行一次，后者为最少执行一次*
- Express Workflow：包括*同步和异步驱动*，同步会确保得到API返回执行结果，后者只确保启动，同步的更适合编排服务，错误处理，并行处理，重试策略等
- Error Handling：重试或者增加alert功能触发EventBridge-SNS

### AppFlow

- 全托管，数据安全传输，从SaaS到AWS，的一种便捷的API解决方案
- Sources：Salesforce/SAP/Zendesk/Slack等
- Destination：S3/Redshift/Snowflake/Salesforce
- 基于时间，基于事件，On-demand
- 可以公网传输，也可以私网PrivateLink传输
- 支持数据验证validation和数据过滤filter

### Amazon EventBridge

- 时间cron驱动或者事件event驱动。
- event实际上是一种json格式数据。
  - 可以filter events
  - 源：EC2 Instance，CodeBuild（fail build），S3 Event，Trusted Advisor（new finding），CloudTrai（any API call）
- 可以发送到集成的组织event bus。
- 可以重现replay过去的event。
- event集合：event schema register
- 可以集成到第三方Partner或者自定义Custom的event bus。
- 如果跨账户使用，需要设置resource-based-policy。

### MWAA Airflow

- Batch-oriented workflow tool
- UseCase：复杂work flow，ML数据预处理，ETL流程
- 代码文件是可以ZIP的
- Scheduler和Workers使用的是AWS Fargate Container
- 构架上，需要连接Service VPC中的*Metadata database*和*Web server*的两个服务端点VPCE

## ⬆️数据工程的各种组件⬇️全面统筹和ML加持
## Security & Identity & Compliance

- 参考安全专家内容，只关注重点服务
- 最小权限原则*Least Privilege*：IAM Access Analyzer
- IAM：全球Global服务 / MFA / IAM Roles
- KMS：加密资源换region，需要新的key，也就是key不能在两个region共用，跨账户使用by key policy
- *Macie*：PII，with ML：S3 -> Macie -> EventBridge -> Topic
- Secret Manager：自动更新 / Multi-region Secret
- WAF：7layer应用保护，ACL / Global Accelerator的固定IP + WAF + ALB
- Shield：DDos，保护对象*ALB，CloudFront等*
- 在JDBC上使用TLS安全传输

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

- CloudWatch：
  * Metrics（namespaces，attributes），Metrics Stream（near-real-time）- firehose - S3 - Athena
  * Logs groups / streams / Logs Insights（query engine not real time）
  * CW Logs Subscription is real-time -> Kinesis -> log aggragation (通过policy设置可以跨账户)
  * Unified Agent

## Machine Learning

## Developer Tools
