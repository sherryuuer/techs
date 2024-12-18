- Top-down 是自定向下当还没有清楚数据结构的时候
- Bottom-up 是自底向上是知道了数据结构的时候

- Top-down 从需求出发，选定functional的各种部分
- *Requirements*
  - high level requirements
  - user interviews
  - data collections
  - deep understanding
- *Data*
  - ER modeling
  - data structures
  - attributes
  - relations
- *Scalability*
  - vertical
  - horizontal
  - CAP trade-off discussion
- *Sharding*
  - row level
  - algorithm
  - 分片是多个数据库实例，复杂度高，备份、恢复、查询聚合等都需要额外的管理措施。Sharding还可能需要分片键的精心选择，以确保均匀分布数据和避免热点分片
- *Partitioning*
  - divide data based on date col, cluster is based on multi-cols
  - 分区是在同一个数据库实例上实现的，提高查询效率
- *replication/backup*
  - sync update
- *security*
  - SQL injection
