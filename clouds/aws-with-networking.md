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

- 当要创建构架的时候画一个构架图总是好的
- VPC Region choice
  - Default VPC是为了方便起见为用户创建的Region级别的VPC，若手动删除，也可以右上角随时恢复，没什么特别的地方
- VPC CIDR
  - IP也是一种协议，8x4bit数字，8bit代表0～255
  - CIDR是传统的网络类型ABC的替代品，有更高的分配和路由效率，灵活的子网划分，避免了浪费
  - Mask后算出的IP范围是从IP的起始位开始的 2^(32-mask) 个IP数量
  - 0-4和255不可用，是AWS在用：0是网址，1是VPC Router，2映射到Amazon-provided DNS，3是给future预留，255是NetworkBroadcastAddress，因为VPC不支持内部广播功能（计算网址数量的时候，要想到这5个不能用）
  - IPv4一般是/16（max）或者/28（min），IPv6一般是是固定/56或者/64
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
  - 可以设置allow和deny规则，黑白名单，并且按照编号递增评估
  - 创建EC2的时候不需要设置NACL，因为和Subnet绑定就行了
- NAT：
  - NATGateway
    - Managed by AWS，5GB带宽可以scalingUpto100GB
    - 高可用性：AWS的NAT高可用性功能通过在指定的可用区内使用冗余实例来实现高可用性。如果发生故障，AWS会自动重新分配 NAT 网关资源。只需在创建时启用该功能。
    - 设置在PublicSubnet，在PrivateSubnet的路由表中路由到该nat
    - AZ level，使用EIP
    - 支持协议TCP，UDP，ICMP
    - 1024-65535 port用于outbound connection
    - NAT本身就设置在公有子网，只要公有子网的路由可以接IGW，NAT就可以接，并且NAT被分配了EIP，天生就是向外连接的
  - NATInstance
    - 放在PublicSubnet，有效化公有IP，option：设置EIP
    - 需要用AWS的NAT AMIs
    - Disable Source/Destination Check：NAT实例的作用是将发自私有子网的流量进行源网络地址转换(SNAT)，使用自身的公有IP地址访问互联网，同时将互联网响应流量进行目标网络地址转换(DNAT)，发回发起请求的私有实例。源/目标检查是EC2实例的默认安全机制，它要求实例只能发送/接收使用自身IP地址作为源/目标的流量。但是对于充当NAT角色的实例，它需要转换流量的源IP和目标IP，这与源/目标检查机制相矛盾，所以要关闭。
    - PrivateSubnet的路由要路由到NatInstance的EIP，或者这个instance的ID本身（没有EIP的情况）
    - 因为它是一个EC2所以它有自己的好处，可能比较便宜，并且可以有自己的SG，可以设置portforward，或者bastion
- DNS：Route53 Resolver

### Advanced Topics（CIDR，ENI，BYOIP）

- 拓展VPC地址空间，增加第二个CIDRs（上限5个）
  - 不能和现存（包括peering）的地址空间CIDRs有重叠
  - 如果你的主CIDR是RFC1918的一个range，那你的第二个CIDR不可以是另一个RFC1918的不同range，*因为不同的range不能构成一个子网*。
    - RFC 1918是一份互联网标准文件，定义了专门保留用于私有网络使用的IP地址范围。这些私有IP地址范围包括:
      1. 10.0.0.0 - 10.255.255.255 (10.0.0.0/8前缀)
      2. 100.64.0.0 - 10.127.255.255  (100.64.0.0/10前缀)作为RFC 6598新增的共享地址空间，扩充了可用作本地网络和地址共享的IPv4地址池，是对RFC 1918的一个补充。
      3. 172.16.0.0 - 172.31.255.255 (172.16.0.0/12前缀) 
      4. 192.168.0.0 - 192.168.255.255 (192.168.0.0/16前缀)
      这些IP地址范围被*专门预留，用于构建私有的本地网络*，如家庭或企业内部网络。私有IP地址在互联网上是不可路由的，因此不会与公网上的IP地址冲突。
    - 扩展的时候必须是同一种range，比如主是10.0.0.0/16，增加第二个是10.1.0.0/16就可以
  - 如果你VPC的RouteTable中有某个CIDR（比如10.2.0.0/16）是destination，那么也不能设置和那个CIDR相同或者比他大的第二CIDR，可以设置比他小的CIDR（比如10.2.0.0/25）。

- **ENI弹性网卡**（*重新认识到了它的强大，哆啦A梦的传送门*）
  - 虚拟网卡的逻辑组件，基本相当于IP本身，一个主机可以有多个网卡（如ifconfig命令的结果所示）
  - 和EC2一样是AZlevel的
  - 可以包括以下属性：
    - 一个primary私有IPv4，一个主IPv6（6不需要私有）
    - 每个私有IPv4有一个ElasticIP
    - 一个或多个secondary pravite IPv4
    - 一个公有IPv4
    - 一个或多个IPv6
    - 一个或多个SG（安全组是和网卡绑定的）
    - 一个MAC地址（涉及使用的软件的license的时候经常是和MAC绑定的，所以将网卡带走就可以继续用软件）
    - 一个source/destination check flog（源/目标检查也是和网卡绑定的）
  - 一个EC2可以绑定多个ENI，但是Primary的ENI不能detach，可以将第二个之后的ENI进行detach，然后atach到其他的EC2上
  - 一个EC2可以拥有的IPv4地址上限是由，EC2的instance type，和ENI数量上限决定的
  - 不支持NIC Teaming（是一种将多个网络接口卡(NIC)绑定在一起作为一个虚拟接口使用的技术。主要目的是提高网络吞吐量和冗余性。），因为EC2的带宽上限是在创建时候决定好的
  - 如果对一个EC2附加来自同一个subnet的*两个ENI*可能会引起*非对称路由(Asymmetric Routing)*等网络问题，所以最好是一个EC2一个ENI，附加多个IP地址。路由表可能会基于不同的路由策略(如成本最优)选择不同的出站ENI，入站流量则倾向于沿原路径返回，与出站路径不同，形成非对称。
  - Use Cases：
    - *Requester Managed ENI*：是指在 AWS 云环境中，由客户(requester)自行管理和配置弹性网络接口(Elastic Network Interface, ENI)的一种模式。它让AWS的很多服务和网卡分离，实现用户对资源的（通过SG）管理。例如：
      - RDS创建仔AWS管理的VPC中，但是它的ENI创建在客户的VPC中，被用户控制。（lambda等其他服务也是）
      - EKS的Control-Plane master nodes创建在AWS管理的VPC中，但是它的ENI创建在客户的VPC中，就可以和worker nodes（是由客户在自己的 VPC 中创建的 EC2 实例）进行通信。
      - AWS Workspace和Appstream2.0的底层host是创建在AWS管理的VPC中，但是ENI创建在客户的VPC中，实现和客户的资源通信。
    - 创建Management network/Dual-homed instance（是指在AWS的虚拟私有云(VPC)环境中,一个EC2实例同时连接到两个不同的子网的配置，使得一个EC2可以同时访问两个子网的资源，或者实现不同的功能）
    - High Availability solution：只用一个ENI作为IP端口，当后面的EC2不可用，不删除ENI，而将该ENI atach到新的hot-standby的EC2上，实现高可用性，而不需要变更DNS等网络配置。如果对网络延迟不是很在意的小的APP可以进行这样的设计。
    - *Secondary IPs for PODs in EKS*
- BYOIP（Bring your own IP）
