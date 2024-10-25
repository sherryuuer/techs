## 关于云，构架，系统设计的上层思考

- 所有服务都是API
- 所有的GUI设置都是IaC的映射
- 权限：**人**的权限，**资源**的权限，**实例**的权限，甚至是object的权限，API之间的互动需要足够的Permission
  - permission（权限）是对 AWS 资源执行操作的能力，而 policy（策略）是定义和管理这些权限的规则集合。策略可以授予或拒绝特定的权限，并可以与特定的实体（如用户、角色或资源）相关联
- 服务整合，统合，可视化，管理层级，TAG属性策略
- 疏结合，解藕，这个过程中可以进行filter和自动化安全应对，是很强大的功能，实质就是乐高构建
- 事件驱动，在各个领域，安全，workload，ETL等都是泛用的
- 批处理-->流处理 的进化
- 自动化技术（比如IaC），版本管理技术（比如S3的tier），docker技术（比如CI/CD，CloudRun的Build等），无处不在
- **网络是一切的基础**
- 依然是中心化的网络，比如集成用户，账户，日志，流量管理等
- 组织可以自定义规则和策略，比如config等策略，网络访问合规与否的策略，用于自动化分析和检测
- **安全方面**：
  - 安全监测现在很多使用机器学习技术
  - 主体数据：包括静态保护和传输中保护，加密技术等
    - 传输中数据加密：TLS/SSL=HTTPS
    - 存储中数据加密：Server-Side服务器进行加密和解密/Client-Side不信任服务器的情况客户端加密，又叫信封加密
  - 主体基础设施：安全基本都是网络安全方面的
  - 主体人和资源：认证认可
  - 未来时间线：危险检测和实践对应
  - 过去时间线：安全日志和监控分析
  - 统筹管理：组织，账户结构，设置，计费
- 服务层级：
  - Region
  - ZoneAZ
  - VPC-VPN
  - subnet
  - server/resource
- 认证和认可：
  - STS无处不在
  - SSO很强大
- **关注Monitering和Alarting**！
- **Communication**：和非技术人员的交流非常重要
- DevSecOps，自动部署相当重要


## 云构架和系统设计方案关键组件

- Cloud Native
  - Availability and Scalability
  - Ship feature quickly by
  - Microservices 疏结合，API连接
  - Containers <- Orchestrator (k8s): detect and repair, load balance, control runs
  - DevOps: CI/CD: Plan -> Code -> Build -> Test -> Release -> Deploy -> Operate -> Monitor
  - Cloud Native Open Standards

## 云原生

### Service Mesh

微服务构架的挑战：
- business logic
- communication logic
- security logic
- retry logic
- metric and tracing logic

服务网格，就是将上面的除了business logic之外的逻辑封装为proxy，通过sidecar服务模式，分布在各个服务节点上，作为一个sidecar存在，并通过kubenetes这样的control plane，在每次启动服务时候，将非业务需求的部分，装载到每个pod中，妙啊

**Istio**就是这种服务网格的实现。Istiod是Control Plane，开源的Envoy Proxy就是这种分布式的代理。
- 管理流量控制问题
- 服务之间的通信问题
- 分流和重试策略
- 服务发现
- 安全
- Metric和tracing功能


### Kubenetes
### Security
### DevOps




PS：任何的学习都离不开第一原理，底层，联想思考。
