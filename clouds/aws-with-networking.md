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
- VPC CIDR
  - IP也是一种协议，8x4bit数字，8bit代表0～255
  - CIDR是传统的网络类型ABC的替代品，有更高的分配和路由效率，灵活的子网划分，避免了浪费
  - Mask后算出的IP范围是从IP的起始位开始的 2^(32-mask) 个IP数量
  - 0-4和255不可用，是AWS在用：0是网址，1是VPC Router，2映射到Amazon-provided DNS，3是给future预留，255是NetworkBroadcastAddress，因为VPC不支持内部广播功能（计算网址数量的时候，要想到这5个不能用）
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
  - 前提是你的服务器PrivateIP对应有PublicIP
  - 连接网络的条件，一个是IGW的设置，一个是0.0.0.0/0登陆到路由表
- Security Group：EC2 level
- NACL：Subnet level
- DNS：Route53 Resolver
