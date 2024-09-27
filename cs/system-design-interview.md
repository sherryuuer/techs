- 关注整个构架，或者关注某个功能，理解scope很重要。
- **功能性设计 functional**和**非功能性设计 non-functional**。
- 功能性是实际的具体的功能。
- 非功能性一般是：Scalability可扩展性，Throuthput，StorageCapacity，Performance，Availibility
- *Back of the envelope calculation*：很多非功能性标准，无法立刻得到一个十分精确的计算结果，“信封背后的计算”是一种方便快捷的方法，用于在不需要高精度的情况下，快速得到一个大致的估算结果。这种方法强调的是效率和实用性，常用于需要快速决策的场合。
- 分布式系统很重要。

角度：

- 功能性设计，非功能性设计
- 功能性是实际的具体的功能。
- 非功能性一般是：Scalability可扩展性，Throuthput，StorageCapacity，Performance，Availibility
- 系统构架

## Rate Limiter


## TinyURL

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

功能有哪些：

- follow others
- create tweets
- view feed：这个可能会需要机器学习算法之类的

组件：

- client
- App servers（*LB*）- *cache*（最流行的tweets，使用*LRU*算法）加速必须
- DB：*Graph DB* instead of NoSQLDB 存储follow数据
- 数据库*Sharding*技术，使用uid来分区
- *Object Storage* 存储的是tweets数据包括文字图片音像等
- *CDN*：用文件分发系统（read-only）
- *PubSub* 处理新创建的tweet -> *Cluster/Worker*处理 -> 创建client可以立刻查阅的*Feed Cache*，所以多个cache有多种功能也是常态
  - 这些都是为了**降低延迟**
