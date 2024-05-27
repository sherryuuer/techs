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
  - 0-4和255不可用，是AWS在用：0是网址，1是VPC Router，2映射到Amazon-provided DNS（Route53 DNS Resolver），3是给future预留，255是NetworkBroadcastAddress，因为VPC不支持内部广播功能（计算网址数量的时候，要想到这5个不能用）
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
  - 迁移已有的公共IPv4或者IPv6路由到AWS（目的是为了继续使用自己的IP，比如该IP已经被客户使用，或者有自己的reputation，所以不想改变，或者可以*将AWS作为一个hot standby*之类的原因）
  - YourIP必须是在RIR注册过的。区域互联网注册管理机构 (RIR) 是负责特定地区内IP地址和自治系统号码（ASN）分配与管理的组织，确保网络资源的公平分配和互联网的稳定运行。支持三个：ARIN是美国，APNIC是亚太，RIPE是中东和欧洲。
  - YourIP必须历史干净，不然AWS有权拒绝你的IP迁移。
  - YourIPv4-range最多可以到/24，不能超过24（不能是25，可以是23）
  - YourIPv6-range最多可以到/48，但是如果你不想公开或者是在内部DirectConnect则最多可以到/56
  - 每个区域最多迁移5个IP range（包括4和6）
  - 授权：使用资源公共密钥基础设施（RPKI）创建一个路由原始授权（ROA：RouteOriginAuthorization）以授权AWS的自治系统号码ASN（16509和14618）来广播你的IP地址。
    - 这是为了确保你的IP地址只能通过指定的ASN进行广播，从而防止IP地址劫持和其他安全问题。
    - 通过创建和发布这个ROA，你告诉全球的路由器，只有AWS的指定ASN被授权广播你的IP地址前缀，从而提高网络安全性。
    - 简化的过程示例：1-登录RPKI管理平台：进入你所属RIR的管理界面。2-创建新ROA：填写你的IP前缀和AWS的ASN。3-确认和发布：确认信息无误并发布ROA。自治系统号码（ASN）通过边界网关协议（BGP）来广播其路由信息。BGP是一种用于在不同自治系统（AS）之间交换路由信息的标准协议。

## VPC DNS & DHCP

### VPC DNS Server（Route53 Resolver）

- **这部分内容专注于从VPC内部进行DNS解决。Route53的部分有更大的Scope。**
- 功能是：解决VPC内部资源（EC2）的DNS问题。
- 创建VPC就会被自动创建一个内部的DNS server，IP地址是VPCbaseIP+2，就是上面提到的AWS预用5个IP之一。或者IP地址也可以是一个虚拟IP：169.254.169.253(这个只能从VPC内部连)
- 通过以下方式解决request：这三者是*有顺序的*，按照顺序，自上而下进行DNS解决
  - *Route53 Private Hosted Zone*
  - *VPC internal DNS* 
  - Forwards other requests to *Public DNS*（including Route53 Public Hosted Zone，提到公共的就是指公共的互联网上的DNS服务）

- Route53 Private Hosted Zone（Route53的一项功能）*目的就是自定义自己VPC内的私有域名：域名：私有ip地址*
  - Private Hosted Zone 允许你在 Amazon VPC 中创建自定义的 DNS 命名空间，从而可以在私有网络内解析 DNS 名称。这些域名不会在公共互联网 DNS 服务器上可见，因此它们只在你指定的 VPC 内部有效。
  - 首先需要创建 Private Hosted Zone：
    - 在 AWS 管理控制台中，进入 Route 53 服务，选择“Hosted Zones”，然后创建一个新的“Private Hosted Zone”。
    - 为你的私有域名指定一个 DNS 名称（例如 example.internal）。
    - 选择一个或多个 VPC 与这个 Hosted Zone 关联。只有关联的 VPC 中的资源才能解析这个私有域名。
  - 然后配置资源记录：
    - 在 Hosted Zone 中，创建资源记录集（例如 A 记录、CNAME 记录），这些记录将指向你 VPC 内的资源（如 EC2 实例的私有 IP 地址）。
    - 例如一个 A 记录 webserver.example.internal 指向某个 EC2 实例的私有 IP 地址 10.0.0.5。
  - 进行 DNS 解析：
    - 当 VPC 内的资源（如 EC2 实例）发出对 webserver.example.internal 的 DNS 查询时，Route 53 使用 Private Hosted Zone 中的配置解析这个域名，并返回相应的 IP 地址。
    - 这保证了域名解析请求不会离开 AWS 网络，增强了安全性和性能。
  - 权限控制：
    - 通过关联特定的 VPC，确保只有这些 VPC 内的资源才能访问和解析 Private Hosted Zone 中的域名。你也可以使用 IAM 策略控制对 Hosted Zone 配置的访问权限。

- VPC（internal）DNS
  - 负责解析在VPC中的资源（如EC2实例、RDS实例等）的域名，确保它们能够通过内网进行通信。
  - 当你创建一个EC2的时候勾选为他分配DNS网址，会自动分配一个类似`ip-<ip address>.region.compute.internal`这样的地址，就可以使用内部的VPC DNS进行解决。
  - 但是相比这个不如用自己定义的域名，也就是在 Private Hosted Zone 中定义自己的域名。

- *Public DNS*（including Route53 Public Hosted Zone）
  - 在外网中递归地查询域名。
  - 这部分在Route53中更细致。
  - 这个部分的resolution和上面的不同之处在于，它*需要通过一个网关和互联网连接*，而上面两个不需要，是内部的resolution。

- Route53 Resolver Endpoint
  - 用于在VPC和本地on-premise网络之间进行DNS解析
  - 将DNS端点*ENI*配置在VPC中的subnet中，通过ENI进行request转发，实现解析
  - ENI包括Inbound和Outbound：Inbound是DNS在AWSVPC中，本地request到ENI，然后被转发到Route53Resolver解析。Outbound是DNS在本地服务器中，当VPC中的服务器请求解析时，如果内部无法解决，会根据条件通过ENI转发request到本地数据中心的DNS进行解析。

### DHCP Option sets

- **DHCP**（Dynamic Host Configuration Protocol）是一种网络管理协议，用于自动分配IP地址和其他网络配置参数给客户端设备，以便它们可以在网络上通信。DHCP简化了网络管理，不需要手动为每个设备配置网络设置。
- 工作原理如下：
  - 发现（Discover）：新设备（客户端）连接到网络时，会广播一个DHCP Discover消息，寻找可用的DHCP服务器。
  - 提供（Offer）：DHCP服务器收到Discover消息后，会返回一个DHCP Offer消息，提供一个可用的IP地址和其他网络配置信息（如子网掩码、网关地址、DNS服务器）。
  - 请求（Request）：客户端收到Offer后，选择一个DHCP服务器，并广播一个DHCP Request消息，表明它接受所提供的IP地址。
  - 确认（Acknowledge）：选定的DHCP服务器收到Request消息后，确认并发送一个DHCP Acknowledgment（ACK）消息，正式分配IP地址给客户端，同时提供其他必要的网络配置信息。
  - 租约lease管理：分配的IP地址有一个租约期限。在租约到期前，客户端需要向DHCP服务器请求续租，以继续使用该IP地址。

- 上一个部分的内部DNS解决可以达成，那么*EC2是如何知道要去2号IP找到这个VPC DNS的*，就是通过DHCP的设置。
- **DHCP Option Sets**：
  - 包括Domain Name，DNS设置，NTP服务器，和NetBIOS node type。
  - NTP（Network Time Protocol）是一种网络协议，用于同步计算机系统之间的时钟，以确保它们的时间一致。
  - NetBIOS（Network Basic Input/Output System）是一种用于局域网（LAN）中的通信协议，提供网络基本输入/输出服务。它允许应用程序在网络上的计算机之间进行通信，包括名称解析（将计算机名称转换为IP地址）、会话管理（建立和管理连接）、和数据传输。NetBIOS常用于早期的Windows网络环境，但在现代网络中，许多功能已被其他协议（如DNS和TCP/IP）所取代。
- 在VPC中创建的时候会生成一个默认的DHCP Option sets：
  - Domain-name={region}.compute.internal
  - name-servers=AmazonProvidedDNS
  - 它会为新创建的EC2创建内部hostname：<ip-ip-address>.<region>.compute.internal（美国东部US-east-1是<ip-ip-address>.ec2.internal），和`/etc/resolv.conf`，记载nameserver服务器的地址，和域名后缀<region>.compute.internal，这意味着你可以直接ping <ip-ip-address>也可以成功。
  - 它会为有publicIP的EC2创建外部hostname：ec2-<ip-address>.<region>.amazonaws.com（美国东部US-east-1是ec2-<ip-address>.compute-1.amazonaws.com）
  - 如果要给一个在public subnet中的EC2分配public ip，需要Enable DNS hostname。
  - 从内部的EC2进行ping Private EC2的时候，会返回私有ip，从外部进行ping的时候会返回公有ip。
- 无法对已有的DHCP进行更改，只能*创建一个新的DHCP*，然后attach到VPC中，以此*创建自己的内部域名（使用新创建的Private Hosted Zone）*，或者*不使用AmazonProvidedDNS，而是指向自己定义的DNS服务器*。设置后，需要等待内部租约lease更新，或者使用command（sudo dhclient -r eth0）进行手动更新。（一个VPC只能绑定一个DHCP option sets）
  - 自己的DNS服务器，需要UDP53端口进行通信。
- VPC DNS Attributes：
  - enableDnsSupport（DNS Resolution setting）是 AWS 中用于配置 VPC 的一个功能，它允许 VPC 中的实例使用默认的 DNS 解析服务来解析公共 DNS 域名，从而能够与互联网上的资源进行通信。默认启用，但是关闭了也可以自己设置Custom DNS Server。
  - enableDnsHostname（DNS Hostname setting）是 AWS VPC 中的一个配置选项，它允许 VPC 内的实例分配具有公开可解析主机名的 DNS 记录，从而使其他网络中的资源可以通过主机名来访问 VPC 中的实例。新VPC默认是禁用的。前置条件是 enableDnsSupport 为 True。
  - 使用Private Hosted Zone的前提是这两种属性都设置为True。

## Network Performance & Optimization

### Network Performance基础

**基础概念**

- *Bandwidth*带宽是指，在数据通信过程中，在单位时间从网络中的某一点能够通过的最大数据量。bit为单位。（byte是字节是8bit）
- *Latency*（延迟）是指数据从源发送到目的地所需的时间延迟，通常以毫秒(ms)为单位测量。它反映了网络传输的速度和响应时间，延迟越低，网络性能越好。
- hop是指数据包在传输过程中经过的一个*中间设备*（如路由器或网关）。每经过一个设备就算作一个hop，hop数越多，通常意味着数据包的路径越长，可能增加延迟。
- *Jitter*是指在网络中，数据包传输延迟的变化或波动，通常是指连续数据包之间到达时间的变动。高jitter会导致实时应用（如视频会议和在线游戏）的质量下降，因为数据包到达时间不一致会影响*流畅性*。
- *Throughput*是指在网络中单位时间内成功传输的数据量，通常以比特bit每秒（bps）、千比特每秒（kbps）或兆比特每秒（Mbps）等单位表示。它衡量网络传输效率和容量，throughput越高，网络能够处理的数据量就越大。受到带宽，延迟等的影响。
- PPS（Packets Per Second）是指网络设备每秒钟*处理的数据包数量*。它是衡量网络设备（如路由器、交换机）性能的一个重要指标，PPS越高，*设备处理数据包的能力*越强。
- *MTU*（Maximum Transmission Unit）是指网络中单个数据包的最大字节大小。MTU值*决定了数据包的最大尺寸*，设置合适的MTU可以优化网络性能，避免数据包分片，提高传输效率。
  - "Don't Fragment"是IP数据包中的一个标志位，指示网络设备不要对该数据包进行分片。如果数据包超过路径中的MTU值且设置了“Don't Fragment”标志（DF=1代表不允许分片），数据包将被丢弃并返回一个ICMP（*MTU path discovery*中，该协议必须被允许）错误消息，而不是被分片传输。
  - 受到错误消息后，client就会传输几个较小的数据包，重新传输。
  - 在EC2的设置，依赖于instance type是否支持
  - 在*ENI level*定义
  - check path MTU: tracepath amazon.com
  - check the MTU on your interface: ip link show eth0
  - set MTU value on Linux: sudo ip link set dev etho mtu 9001
- *Jumbo Frame*是指超过标准MTU（通常为1500字节）的以太网帧，通常大小在9000字节左右，几乎是MTU的六倍。使用Jumbo Frame可以*减少数据包数量*，从而降低CPU负载和提高网络效率，但需要所有网络设备都支持Jumbo Frame才能发挥其优势。
  - AWS的*内部*默认支持JumboFrame
    - 注意：如果通过IGW或Peering出去了就不支持了，降低为1500，所以*这里的使用要非常小心*，最好在可以使用的scope使用
    - 在*EC2集群放置组*（EC2 Cluster Placement Group）内使用JumboFrame，可以最大化throughput。（EC2集群放置组是一种用于容纳具有特定亲和性或亲缘性的EC2实例的逻辑分组。也许他们放置在同一个物理主机上。它可以提供更低的网络延迟和更高的网络吞吐量，适用于需要在集群中的实例之间进行低延迟和高吞吐量通信的应用程序。
    - *VPC Endpoint*：支持MTU 8500 bytes
    - *Internet Gateway*：出界进入互联网MTU变为1500 bytes
    - *Intra Region VPC peering*：区域内部，MTU 9001 bytes
    - *inter Region VPC peering*：区域之间的peering，MTU 1500 bytes
  - On-premise的情况
    - 使用VGW的VPN：MTU 1500 bytes
    - 通过Transit Gateway的VPN：Site2siteVPN：MTU 1500 bytes
    - 使用*AWS Direct Connect*，那么VPC之间和VPC与on-premise之间也支持JumboFrame
    - 通过Transit Gateway的Direct Connect：MTU 8500 bytes（VPC -> TransitGateway -> DX -> on-premise）

### EC2 Network performance Optimization

**这部分的加速，都是在传输距离，网卡专用，绕过系统，绕过中介上做功夫。**

- **Cluster Placement Group**
  - 逻辑组，同一个AZ内
  - 满足分布式应用，低延迟需求，比如HPC（高性能计算机，使用大规模的并行计算系统，以高速处理复杂的数学模型和大规模的数据集。）
- **EBS Optimized Instance**
  - EBS是一种网络存储drive，通过网络和instance通信，所以是有延迟的
  - EBS和instance之间通过ENI通信，那么和其他通信内容共享就会有较低的带宽
  - Optimized的版本，是IO接口是dedicated专有接口，以此优化EBS的通信性能，不需要和其他traffic进行竞争

- **Enhanced Networking**
  - Enhanced Networking是一种技术，旨在提高云计算环境（如AWS EC2实例）中的网络性能。通过使用特定的网络接口（如Elastic Network Adapter, ENA）和网络虚拟化技术，Enhanced Networking提供更高的带宽、更低的延迟和更低的抖动（jitter），适用于对网络性能要求较高的应用程序，如高性能计算（HPC）、大数据分析和实时应用。
  - 意味着PPS可能超过1M（每秒要传输100万数据包），目的是降低instance之间的传输延迟
    - 我的理解就是*EC2的通信可以直接绕过虚拟层，而是通过物理主机的网卡进行通信（通过ixgbevf或ENA），以降低延迟。*
    - SR-IOV（Single Root I/O Virtualization）是一种技术，用于提高虚拟化环境中网络和存储设备的性能和效率。它允许物理网络适配器（如以太网卡）在多个虚拟机之间直接共享，而不是通过主机操作系统进行中介。
    - PCI（Peripheral Component Interconnect 周边组件互联）是一种计算机总线标准，用于连接各种外部设备（如网卡、显卡、存储控制器等）到计算机的主板。PCI设备可以是物理设备，也可以是虚拟设备，SR-IOV则是针对虚拟化环境中PCI设备的一种技术扩展。PCI允许你的**VM和物理NIC网卡直接通信**，这里的VM就是EC2，也就是让EC2，和它所在的物理服务器的NIC进行通信。以此实现*低延迟，高传输*。
  - 支持Enhanced Networking的可用的Instance type：
    - *VirtualFunction（VF）uses ixgbevf driver*（up to 10GGbps）`ethtool -i eth0`输出driver:ixgbevf
    - *Elastic Network Adapter（ENA）*（up to 100Gbps）`ethtool -i eht0`输出driver:ena
    - ixgbevf驱动程序是Intel开发的一种用于虚拟功能设备的网络驱动程序，特别针对Intel 10 Gigabit Ethernet硬件。它支持SR-IOV技术，允许虚拟机直接访问物理网络适配器的虚拟功能（Virtual Function），从而提高网络性能和效率，减少延迟和CPU开销。
    - Elastic Network Adapter（ENA）是AWS提供的一种高性能网络接口，用于EC2实例。ENA支持高达100 Gbps的网络吞吐量，提供低延迟和高数据包处理能力。
  - 除此之外，也必须是支持Enhanced Networking的操作系统AMI

- **DPDK**
  - DPDK，全称为Data Plane Development Kit，它主要用于*加速网络数据包处理*，通常应用于网络设备和服务器上。DPDK通过*绕过bypass操作系统的内核网络栈*，直接在用户态进行数据包处理，从而大大提高数据包处理的速度和效率。所以它的目的是加速**操作系统内的包处理**。（SR-IOV是*加速instance和hypervisor之间的*包处理，这个还要经过一下操作系统再去网卡呢，DPDK直接操作系统都直接绕过了），算法很cool。

- **EFA**
  - 是ENA的一种类型，只能用于Linux系统，才能发挥它的功能，如果你放在Windows上，它就只是一个ENA了。
  - EFA，全称为Elastic Fabric Adapter，是亚马逊AWS提供的一种高性能网络接口。EFA旨在加速HPC（高性能计算）和机器学习应用中的网络通信。使得HPC内部使用MPI（Message Passing Interface，是一种用于编写*并行程序*的标准通信协议和编程模型），绕过bypass OS内核，直接和EFA进行包通信。总之，EFA通过支持RDMA（Remote Direct Memory Access）和OS Bypass，允许应用程序直接访问*网络硬件*（这里就是EFA），从而减少了通信延迟和CPU开销。

### Bandwidth limits（inside & outside VPC）

- 这部分的基础概念：**Network Flow 5 - tuple**
  - 在网络流量分析和网络流识别中，"5-tuple" 是指用于唯一标识网络流的一组五个参数。这五个参数是：
    * 源IP地址（Source IP Address）：数据包的发送方IP地址。
    * 目的IP地址（Destination IP Address）：数据包的接收方IP地址。
    * 源端口号（Source Port Number）：发送方应用程序的端口号。
    * 目的端口号（Destination Port Number）：接收方应用程序的端口号。
    * 协议（Protocol）：传输层协议，如TCP（传输控制协议）或UDP（用户数据报协议）。
  - 这五个参数组合起来可以唯一标识一个网络流，区分不同的通信会话。
  - 每个flow都有带宽限制，所以增加带宽，可以通过*Multiple-Flows*实现。这部分的内容，基本都是通过多个Flows的叠加来提升带宽。

- VPC的带宽限制包括：Internet Gateway，NAT Gateway，VPC Peering
  - No VPC specific limits
  - No limit for any Internet Gateway
  - No limit for VPC Peering
  - NAT Gateway每个带宽都是45Gbps，可以使用多个NAT Gateway来增加带宽到45Gbps以上。但是要注意不要跨到别的AZ去，那会另外付数据传输费。

- EC2的带宽限制
  - 受到instance本身的instance family，vCPU，传输目的地的影响
  - 在*region内传输*可以利用最大带宽
  - 通过Internet Gateway或者DX*传输到其他region*，最大可以利用*50%带宽*，并且需要是*Current Generation instance with 32vCPUs*，*32vCPUs以下*的实例只能限制在*5Gbps内*。
  - 通过VF提升单Flows带宽到5Gbps，集合带宽到10Gbps
  - 通过ENA提升
    - 单Flows带宽到*5Gbps*（outside placement group），*10Gbps*（inside placement group）
    - 集合Flows带宽到*100Gbps*，通过Multiple-Flows（在同VPC内 / VPC-peering / 通过VPC-endpoint到同区域的S3）
  - 一个特例AWS的 P4d instance，集群超级计算机可以有400Gbps的带宽

- VPN Connection & Direct Connect & Transit Gateway
  - VPG（*Virtual Private Gateway*）到On-premise之间的带宽限制是*1.25Gbps*。由于单个VPC不支持ECMP（Equal-Cost Multi-Path）功能，所以上限就是1.25Gbps了。
  - 通往*同一个VPG的VPN聚合*的带宽上限也是*1.25Gbps*
  - *DX*的带宽通过Port Speed限制
  - DX通往VPG的带宽也取决于Port的speed限制
  - 总之**涉及到DX就是取决于它自己的Port Speed，而关于VPC自身的VPN连接则上限为1.25Gbps。**
  - *TransitGateway*则支持ECMP，所以可以多路径提升聚合带宽。单个VPN带宽1.25Gbps上限，聚合带宽上限为*50Gbps*。

（**上面的数字会变化，会进化，但是底层原理是不变的，理解为什么很重要。**）

### Network I/O Credits

在AWS EC2中，Network I/O Credit是一种用于衡量和管理实例网络性能的机制，类似于EC2的CPU Credit系统。它主要用于突发型实例（如T3和T4g实例），帮助这些实例在需要时能够临时获得更高的网络带宽。以下是有关Network I/O Credit的详细解释：

1. **突发型实例**：
   - AWS的突发型实例（T3、T4g，R4等）设计用于处理偶尔的高负载工作，但通常在较低的基线性能下运行。
   - 这些实例在低负载时期会积累网络I/O Credit，当需要更高的网络带宽时，可以使用这些积累的Credit来提升网络性能。
2. **网络I/O Credit机制**：
   - 每个突发型实例在较低负载时会以一定速率积累Network I/O Credit。
   - 当实例需要更高的网络带宽（例如数据传输量突然增加）时，可以使用积累的Network I/O Credit来实现临时的网络性能提升。
3. **基线性能和突发性能**：
   - 突发型实例有一个基线baseline网络带宽限制，在这个限制内实例能够持续传输数据而不会消耗Network I/O Credit。
   - 如果实例需要超过基线带宽，系统会检查是否有足够的Network I/O Credit。如果有，实例可以短时间内使用更高的带宽。
4. **适用场景**：
   - 突发型实例适用于大多数时间网络需求较低，但偶尔需要较高网络带宽的工作负载。
   - 例如，开发和测试环境、小型网站和博客、突发性流量的应用等。
5. **监控和管理**：
   - AWS提供了工具和指标来监控Network I/O Credit的使用情况，如CloudWatch中的指标，可以帮助用户了解实例的网络性能和Credit使用情况。
通过使用Network I/O Credit，AWS突发型实例能够在需要时提供高网络性能，同时在大多数时间保持低成本。这种机制确保了灵活性和成本效益，适合处理不均衡的网络流量需求。
