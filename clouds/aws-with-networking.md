构架俯瞰：

- 单个Region/VPC：
  - 一个VPC，一个网关，一个ApplicationLoadBalancer，三层构架。
  - AZ两个/各自配置公共子网（NAT），私有子网（网络应用），私有子网（数据库）
  - Region中的其他AWS服务：VPC Endpoint（Gateway/Interface）
  - 连接到其他VPC网络应用：应用通过NetworkLoadBalancer暴露，LB连接PrivateLink，然后接入本VPC的Endpoint（上一个项目中的实现原理也是PrivateLink，它的目的是，VPC的对象不是另一个大范围的VPC网络，而是单个应用）
  - IP转换为DomainName：使用Route53，设置DNS服务器
  - 全球边缘内容分发服务：CloudFront会接入LoadBalancer，降低延迟，保证安全
- 多个VPC
  - VPC Peering
- VPC连接On-premise
  - Site2SiteVPN（经过Internet）：VGW-IPSecVPN连接本地
  - Client2VPC（通过客户端访问）：连接到VPC的ClientVPNEndpoint
  - DirectConnect（最安全选项）：物理
- 混合环境（VPC，VPN，Direct Connect）
  - Transit Gateway

## VPC

### 非云的虚拟网络拓扑

- 非云的构架拓扑结构：internet - router（Internet Gateway，从这里进入局域网=VPC）- 内部router - hub/switch交换机（从这里进入子网LAN（local area network）=subnet）
- switch交换机是LAN的入口，帮助将packets传送到正确的host主机。
- 而云的VPC和非云的网络拓扑其实在本质上是一样的。

### VPC Scope layers

- Global：
  - AWS Account：账户级别应该是最高的，你的账户可以跨很多Region，是全球的
  - Route53，Billing是全球服务
  - IAM是全球服务，跟着账户，根据权限，可以允许访问各个区域的服务
- Region：
  - VPC是Region级别的，VPC只能构建在一个Region中，然后跨好几个AZ
  - 一个Region中可以设置好几个VPC，有上限，可以申请上限增加
  - ELB是VPC级别的，可以在VPC级别上进行服务器负载均衡
  - S3，DynamoDB是Region级别的，你不能把他们放在VPC中，他们由AWS的Region网络管理
- AZ：Subnet和EC2，RDS等是AZ级别的

### VPC Building Blocks

- VPC Region choice
  - Default VPC是为了方便起见为用户创建的Region级别的VPC，若手动删除，也可以右上角随时恢复，没什么特别的地方
- VPC CIDR
  - IP也是一种协议，8x4bit数字，8bit代表0～255
  - CIDR是传统的网络类型ABC的替代品，有更高的分配和路由效率，灵活的子网划分，避免了浪费
  - Mask后算出的IP范围是从IP的起始位开始的 2^(32-mask) 个IP数量
  - 0-4和255不可用，是AWS在用：0是网址，1是VPC Router，2映射到Amazon-provided DNS，3是给future预留，255是NetworkBroadcastAddress，因为VPC不支持内部广播功能（计算网址数量的时候，要想到这5个不能用）
  - IPv4一般是/16或者/28，IPv6一般是是固定/56或者/64
  - IPv6地址全是Public的并且具有全球唯一性。不可以自己设定子网range，并且不提供Amazon Provided DNS hostname。
  - IPv6不支持Site2siteVPN，CustomGateway，NatDevice，和VPC Endpoint。只有IPv4支持。
  - EC2实例的IPv6地址在正常重启的情况下是保持不变的。只有在终止旧实例并启动新实例、手动关联新地址或发生故障迁移等情况下，IPv6地址才可能发生变化。AWS的这种设计提高了IPv6地址的可预测性和持久性。
  - Dual-stack mode是指具有IPv4和IPv6的地址。
- Route Tables：可以在VPC级别，也可以在Subnet级别
  - 创建VPC后，默认的RouteTable，默认VPC内的所有IP，服务器相互可以通信，比如Destination:10.10.0.0/16,Target:local
  - 如果将Route Table设置在Subnet级别，就可以控制子网的网络访问，这个时候，子网不会再看默认的路由表，而是使用自己的表
  - 如果有自定义的子网级别的路由表，优先级高于主路由表的路由
  - 如果要控制两个子网之间的通信还可以引入防火墙，路由表和防火墙是两个概念
  - 控制server级别的通信则需要，Security Group
- Subnets，CIDR，server
  - Subnet分为Public和Private
  - Public Subnet中服务：Web server，Load Balancer，NAT
  - Private Subnet中的服务：DB，App Server，没有Public IP，使用NAT进行网络通信
- IGW：接入外网
  - 前提是你的服务器PrivateIP对应有PublicIP（通过AWS的公共IP池分配）
  - 连接网络的条件，一个是IGW的设置，一个是0.0.0.0/0登陆到路由表
- Elastic IP
  - 通常我们自己的IP地址在路由器重启等情况会被自动分配IPv4地址，EC2自动分配（Amazon Pool within Region）的IP也是这样，当服务器重启，IP会变动。
  - 只有当你的ElasticIP和运行中的服务器绑定的时候不需要付费，其他时候，没绑定或者绑定了没有启动服务器，都是要付费的，因为这是浪费IP地址资源的行为。
  - 可以被attach到server，website，LB上。
- Security Group：EC2 level
  - 源可以是Port，可以是IP地址。还可以是另一个Security Group，当你需要接收很多EC2为源的时候，将他们和同一个SG绑定，就可以只设置这个SG为源了，当再添加新的EC2到源SG的时候，就不需要多余设置了。
  - Statefull
  - troubleshoting：如果APP（private subnet）显示timeout，可能是SG问题；如果APP显示Connection Refused，那就是APP错误或者还没running。
  - Default：inbound all blocked，outbound all authrized，可后续更改
  - 只能设置allow规则，也就是只有白名单
  - 创建EC2的时候需要设置SG
- NACL：Subnet level
  - Stateless
  - 通过设置端口范围(1024-65535)来允许这类端口的流量通过。同时可以结合其他条件(IP、协议等)来控制流量
  - 可以设置allow和deny规则，并且按照编号递增评估
  - 创建EC2的时候不需要设置NACL，因为和Subnet绑定就行了
- DNS：Route53 Resolver
