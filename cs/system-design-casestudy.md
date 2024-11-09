- 关注整个构架，或者关注某个功能，理解scope很重要。
- **功能性设计 functional**和**非功能性设计 non-functional**。
- 功能性是实际的具体的功能。
- 非功能性一般是：Scalability可扩展性，Throuthput，StorageCapacity，Performance，Availibility
- *Back of the envelope calculation*：很多非功能性标准，无法立刻得到一个十分精确的计算结果，“信封背后的计算”是一种方便快捷的方法，用于在不需要高精度的情况下，快速得到一个大致的估算结果。这种方法强调的是效率和实用性，常用于需要快速决策的场合。
- 分布式系统很重要。
- 四步法：确定问题范围，总体设计，细节设计，总结
- **API Design**实际上是一种对系统中的个别功能的设计，包括数据结构，调用方法，返回结果的结构，以及对请求的处理方式。在学习Google Map的时候，体会到，很多地方不会自己去造一个地图，那么使用Google Map的API就是最好的方式，那么对Google Map的设计就是一种对API的设计。

角度：

- cost/time/skillStackLimit
- 考虑多方stakeholder，比如用户，客户，和开发者等
- 考虑时间线索，过去的日志，未来的监视
- 功能性是实际的具体的功能：比如Client -> Load Balancer -> API Gateway -> Backend Services (Microservices) -> Database
  - client：是功能的出发点
  - client-logic：load-balancer，proxy，data-transfer，api-gateway，或者DNS load-balancer
  - logic：一般是web servers中，作为services存在，service之间的通信（RPC>HTTP）
  - logic-data：一般通过driver和数据库通信
  - job处理环节：可使用ochistration系统，以及message-queue解耦，缓冲数据，异步处理，分布式调用，日志处理，流数据处理
  - data：数据设计一般设计数据库，缓存，CDN，CAP讨论，sharding，partitioning，master-slaves，replication，transaction
- 非功能性：Security，Scalability，Availibility，Performance（Throuthput，StorageCapacity）
  - 考虑每一个组件的性能
- High-Level 系统构架设计


# ------- 回答框架 -------
## --- Requirements/收集信息 ---

### 明确需求: func,non-func,cost,time,stakeholder
   - what are the requirements?
   - **澄清问题**：首先和面试官确认需求，确保对系统的目标和功能有清晰理解。例如：这个系统的用户规模是多少？系统的核心功能有哪些？
   - **非功能性需求**：了解性能要求（如响应时间、吞吐量）、可靠性（SLA、故障恢复）、可扩展性、存储需求、安全要求等。

## --- High-Level/假设和理论 ---
### 1. 系统组件划分/components
   - high level design, what are the components we need
   - **高层次架构**：给出系统的总体架构图，定义主要的组件（例如：API 层、数据库、缓存、负载均衡等）。
   - **模块划分**：详细说明每个模块的职责和如何交互。这里可以引入微服务、单体架构或分布式架构等概念。

### 2. 数据库设计/objects
   - talk about the database design
   - **数据模型**：讨论要存储的数据类型以及选择何种数据库（SQL vs NoSQL）。解释如何设计表结构或文档格式。
   - **索引与查询优化**：考虑数据的查询模式，如何优化查询性能（如索引、分片）。
   - **数据一致性**：讨论是否需要强一致性（例如，使用事务），或者可以接受最终一致性。

## --- Design Action/执行计划 ---
## --- Verify system/测试系统 ---
### 1. scalability: 扩展性与负载均衡
　 - scalability
   - **水平与垂直扩展**：说明系统如何扩展以应对用户增长，是通过增加更多服务器（水平扩展）还是通过增加硬件能力（垂直扩展）。
   - **负载均衡**：讨论负载均衡器的使用，确保请求能够均匀分发，避免系统过载。

### 2. performance: 缓存与优化
   - cache!
   - **缓存策略**：决定哪些数据可以缓存（如常用的查询结果、静态内容），以及使用何种缓存机制（如 Redis、Memcached）。
   - **缓存失效机制**：讨论缓存的失效策略（LRU、TTL）和缓存一致性问题。

### 3. reliability: 故障处理与容错
   - **高可用性设计**：讨论如何确保系统的高可用性，避免单点故障（如引入副本、自动故障转移）。
   - **备份与恢复**：考虑数据库和服务的备份策略，以及如何快速恢复服务。

### 4. availability: 可用性，降低延迟，提高用户体验
   - **CAP讨论**：数据一致性和可用性的权衡。

### 5. security: 安全性设计
   - security: data(store and transfer), person(auth), system(firewall)
   - **身份验证与授权**：选择适当的身份验证机制（如 OAuth、JWT），确保用户的权限管理。
   - **数据加密**：讨论数据在传输和存储中的加密策略，避免数据泄露。

## --- Improve and documentation/提升和记录 ---
### 1. log and alarm: 性能监控与日志
   - log and alarm
   - **监控与报警**：说明如何通过监控系统（如 Prometheus、Grafana）监控服务性能，以及如何设置报警。
   - **日志记录与分析**：考虑如何记录关键日志，便于问题排查和性能分析。

### 2. devops and improve 总结与扩展，永远可以进化的系统
   - **总结系统优缺点**：快速总结所设计系统的优点和潜在的风险点。
   - **未来扩展**：讨论系统将如何根据业务需求变化进行扩展和优化。

# ------- 学习案例 -------

## Dropbox

- 存储：对象存储，用户元数据存储，文件分片（优化传输和存储效率）
- 传输：负载均衡，数据压缩和加密
- 同步：服务端push，或者客户端pull

- client -> LB -> front end ->
  - get application server -> cache -> DB(metadata, object storage)
  - push/post app server -> queue compress the file -> update the metadate and the object storage

## Rate Limiter

- 限速的目的是，*防止users或者bots发送过量的请求*，有些出于安全（比如密码尝试次数），有些出于cost和availability考虑，或UX用户体验

- **fuctional**（表达它是什么东西）上来说可以设计一个*backend API*，针对API的各个功能（比如upload的各个instance）设计一个*共享的限速器*，封装为一个组件，存储所有实例的执行情况，这个很像模式设计中的，单例模式里的共享实例，然后可以设计比如*429*这种适用于http的错误码
- *Client端的比如浏览器里的JS中的限速OK？NO*，可能这会很simple，但是很多user可以直接bypass浏览器，而是用curl等直接hit你的API，然后就crush了，所以一定要有一个这样的limiter组件

- **Non-functional**（表达它的性能）主要关系到*throughput*和*latency*，这两个永远是最重要的考量点，我们当然不能用延迟换取限速，目的是限速器存在但是对用户来说似乎不存在的样子，考虑的要素：
  - limiter中设置的rules的数量，它决定了需要过滤多少规则，应该不会很多
  - Users和IPs的数量，一个IP大小是132byte，1一个billion的用户IP就是132GB？可以放进一个memory？
  - Availability：可用性和安全性的trade-off的考虑，call在fail后是open还是close，*这里的考虑因素是这个limiter会成为失败点，是否要让这个点成为整个API请求路上的一个绊脚石*

**components:**

- Reverse Proxy：它本身应该是一个反向代理，来决定是否要发送请求到后端
- Rule server：存放要进行过滤的rules
  - 规则服务器可以存在于后端，那么用于交互的就需要一个*cache服务器*，同时需要一个*worker*在规则变化的时候进行事件驱动，然后重写cache里的规则
- Memory key-value store like redis：需要读写功能，在一定时间内存储用户数据！this is fit
  - schema：key可以是userID，value可以是counts，通过redis的expire功能进行count重置0，api对象，timeUnit（用于滑动窗口的算法控制等），总之这里的数据是为了之后的算法功能实现
- APIs to forward，最终就是要进行路由的API了

**algorithm:**

- 固定窗口Fixed Window：挺好，但是不灵活
- 滑动窗口Sliding Window：灵活，精确，过去的一段时间的速率计算，缺点就是需要存储每个请求的timestamp

## TinyURL

functional:
- read url
- write url(generator)

components:
- client
- web server -> URL generator
  * 为了避免哈希冲突，甚至可以输出大部分的组合存储在一个数据库中，如果key被使用了则标记used
  - 多用户的并发请求要求ACID处理的原子性（atomic）和隔离性（isolation），来放置同时使用一个url
  - 事务一致性和cache速度之间是一种trade-off
- DB: NoSQL
- cache: in memory to get generated url(LRU算法)
- clean up server：非同步处理过期链接的服务器，对不再使用的url进行used标签解锁，并释放对长url的存储

- LB和数据库replica都是惯用伎俩

## Twitter

functional：

- follow others
- create tweets
- view feed：这个可能会需要机器学习算法之类的, read heavy system

components：

- client -> Front-end
- App servers（*LB*）- *cache*（最流行的tweets，使用*LRU*算法）加速必须
- cache DB -> DB：*Graph DB* instead of NoSQLDB 存储follow数据
- 数据库*Sharding*技术，使用uid来分区
- *Object Storage* 存储的是tweets数据包括文字图片音像等
- *CDN*：用文件分发系统（read-only）
- *PubSub* 处理新创建的tweet -> *Cluster/Worker*处理 -> 创建client可以立刻查阅的*Feed Cache*，所以多个cache有多种功能也是常态
  - 这些都是为了**降低延迟**


## Discord/Slack/Teams
*Funtional*
- background: groups, channels, chats
- arch: users -(web socket is better than http)-> server --> databases(MongoDB)
  - server --> cache

- *messages db*:
  - id, uid, memtionId, serverId, channelId, sent_at（shard at this）
  - last_read_at 会标识最后一次阅读信息的时间点，至于图标上的小红点（多少信息的count）可以存储在一个key-value store中
- *User Activity db*:
  - id, uid, serverId, channelId, last_read_at
  - (so user can return to where they left the channel)

*Non-Functional*
- Low latency

## Youtube

- *functional*: upload, watch, and search, recommendation, comments, analytics and so on
- focus on upload and watch video

- *non-functional*:
  - reliability: scale
  - availibility
  - consistary
  - secure: bots, rate limiter
  - low latency

- *arch*:
  - users - CDN(popular videos) fetch files from storage (mobiles, pcs)
  - users - load balancer - App Servers
  - users - Object Stores / metadata (upload(title, uid, video)) in NoSQL *MongoDB*
                          - compression video files - Message Queue - encoding - Object Storage
                          - video is chunked!!

## Google Drive

*Requirements*
- upload
- download
- 50M users, 15GB storage
- availibility/reliablity: replication multi-region (most important)

*high level design*
- file system (HDFS) / object store (GCS) --> object store is good
  - 但是Object的反面是你无法edit文件，只能重新创建一个新的版本
- arch:
  - users --> App servers --> cache --> kv store
                          --> object store
                          --> ZooKeeper handle heartbeat of load balancer
                          --> Garbage Collection: delete block store

- reduce cost:
  - block-level storage 块存储，降低数据传输量
  - deduplication 去重技术

## Google Map

**这个主题的线索，使用数据即可，分为四种数据：routing/map数据，traffic数据，结合map数据进行最短路径的计算，地理位置location数据记录用户自己的位置，静态image数据将图片传送给用户的手机**
*Requirements:*
- device: phone
*Functional:*
- routing

*Web server:*(logical parts)
- load balancer
- real-time calculate: websocket
- routing service:
  - Algorithm like bfs or some Geo one: *Spatial Indexing*
  - use 2 data for calculate: *map data* and *traffic data*

*Data:*
- map data: GraphDB
- location data: Cassadra (NoSQL and high write requirements)
- traffic data: Queue message -> BigTable
- image static data: chunked, Object storage, use CDN

- distributed system, shard by geo id
- use cache before DB
- schema:
  locaiton schema: userId, locationInfo

*Non-functional:*
- accuracy
- availability
- scalability
- reliablity
- latency tolerated

## Key-value store
*Functional:*
- get
- put
- delete

*Non-Functional:*
- durable
- scalability
- reliability
- consider CAP theory

*details:*
- sharding:
- partitioning：consistency hashing circle，devide data based on date(and so)
- replication
- indexing data: Cassadra writes very fast / BigTable / DynamoDB
- node failure: zookeeper
- concurrency write: Cassadra

## Distributed Message Queue
**任何一个云端的或者OSS应用的构架都是一个有趣的系统设计的范例。所以不如学习一下kafka。**
- pubsub system
- publisher
- subscriber

# 系统要素设计要点
### DNS
- 维护一个records表
- 层级结构，递归recursive和遍历的方式，遍历iterative的方式能减轻root server的压力
- cache：TTL越短可用性越高但是root服务器的压力更大
- 分布式系统：避免了单点故障，低延迟，高可用性
- UDP协议数据传输：速度快，性能高

### LoadBalancer
- 功能：
  - 健康检查nodes
  - TLS终止：减轻服务器压力
  - 流量预测分析
  - 服务注册和发现
  - 安全DDos拒绝，WAF设置
- 部署构架：
  - 集群以提高稳定性
  - 全局/全球LB和本地中心LB
  - layer4的LB可以维护一个TCP/UDP状态，layer7的LB需要cookie实现状态追踪
- DNS负载均衡是有局限性的
- 算法：
  - 轮询，加权轮询，最少连接，IPhash，URLhash等

### Distrubuted database
- SQL/NoSQL：终点区分在于数据关系，形式
- SQL数据库特点：
  - ACID事务处理
  - 通过正规化减少冗余
- NoSQL
  - 设计简单
  - 水平扩展容易
  - 根据操作和功能的性质分为不同的类别：
    - 文档数据库（如MongoDB/Firestore）
    - 键值存储（如Redis/DynamoDB）
    - 列族数据库（如Cassandra/HBase，他们的设计灵感来自于GoogleBigTable，实时读写，高并发场景）
    - 图数据库（如Neo4j）等
    - Object存储：GCS/S3

- 数据库复制和分区：
  - 目的：高可用性，灾害恢复，高性能读取，可扩展和高弹性
  - 同步和异步复制：master回复客户端的速度和数据一致性的影响，CAP分析
  - 分区：相比于垂直（分列），水平分区partition更好
    - 水平分区要有好的策略，防止热点：key-range，一致性哈希

### Key-Value Store（Redis，Cassandra）
- 实质是一种分布式哈希表
- 功能性API：get，update，put
- 一致性哈希：虽然可以在服务器变更的时候最小化数据移动，但是仍然可能成为热点，也就是说请求可能是不均衡的
- 并发写入，冲突处理：write batch会一并从memory写入sstable，ss代表sorted，可以进行快速二分查找，写入时候防止数据丢失的功能为transation log类似的事务处理，sstable是不可变的，所以在update的时候会创建新的sstable，这不仅有版本控制的功能，还提高了写入能力
- 非功能性：
  - 可扩展，因为一致性哈希也不能解决热点问题：replication, 水平partition（sharding）
  - 可用性，容错性：检测node failure，比如可以使用zookeeper进行服务发现进行中心管理，或者Gossip协议，每个节点检测其他节点的heartbeat，维护一个生存列表
  - 始终权衡可用性，一致性，成本效益，性能之间的关系

### CDN
### Unique ID Generator
- UUID：服务器ID，时间戳，节点，hash算法等生成
- Unix时间戳：毫秒为单位
- database
- Twitter Snowflake

### Client-Side Monitoring
- 指标选取和获得方式
  - 拉取或者推送
- 日志存储：Object存储，或者时间序列数据库
- 警报：基于阈值或者事件的行动
- HTTP500服务器故障原因排查：
  - DNS 名称解析失败
  - 从客户端到服务提供商的任何路由故障
  - 任何第三方基础设施（例如中间件和 CDN）的故障
  - **关于客户端可达性问题，有时候服务端并不会发生错误，比如BGP泄漏导致网站不可达**，所以我们可能需要客户端可达性检测系统，不只是在服务端设置检测！比如从web browser检测
### Server-Side Monitoring
- 服务器监控：CPU，memory，bandwith，disk
- 硬件系统监控：memory，disk，process
- 数据中心基础设施监控：switch，loadbalancer，CDN
- 网络监控：latency

- High Level：
  - data collector services -> push data or pull data
    - distrubuted message queues
    - service discovery: kubenetes, istio
  - database(timeseriess) store data
  - blob store stores metrics -> trigger alarms
  - query data service -> dashboard

### Distrubuted Cache
- RAM存储，所以存取快速
- LRU算法选择存留数据
- 写入数据的方式：批次写入数据库，或者每次直接写入数据库，或者只有在需要的时候从数据库读取
- 分布式缓存服务器的数据存储，遵循一致性哈希算法
