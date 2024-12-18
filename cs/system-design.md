## System Design Concepts

---
### 1-Background

#### 计算机构架

主要组成部分包括，磁盘，内存，中央处理器，缓存。

**磁盘disk**包括HDD（硬盘）和SSD（固态）。后者更流行，它的成本要更高一些。

**内存RAM**是相对于磁盘价格更贵，储存空间更小，但是速度非常快的存储。计算机一旦关闭，内部的内容就会丢失。

**中央处理器**又叫CPU是电脑的核心计算部分，内置高速缓存。

**缓存cache**是一个重要的概念，应用于计算机体系结构以外的许多领域。例如，网络浏览器使用缓存来跟踪经常访问的网页，以更快地加载它们。这些存储的数据可能包括 HTML、CSS、JavaScript 和图像等。如果数据从页面缓存时起仍然有效，则加载速度会更快。但在这种情况下，浏览器使用磁盘作为缓存，因为发出互联网请求比从磁盘读取慢得多。

另外提一下摩尔定律，是一项观察结果，计算机内部CPU的晶体管数量每18个月就会翻一番，价格也会相应降低一倍。

#### 应用程序构架

**应用程序架构概述**：生产级应用程序的架构由多个组件组成，它们共同创建了一个健壮的系统。这些组件包括开发人员、服务器、用户、前端代码、存储系统等，每个组件都在不同的章节中详细介绍。

**开发人员视角**：开发人员编写代码，将其部署到服务器上。服务器处理来自其他计算机的请求，并需要持久存储来存储应用程序数据。存储系统可能是服务器内置的，也可能是外部的，通过网络连接。

**用户视角**：用户通过网络浏览器向服务器发出请求，服务器响应这些请求。如果服务器无法处理所有请求，就需要对其进行扩展，可以通过垂直扩展（升级服务器硬件）或水平扩展（增加服务器数量）来实现。

**负载均衡LB**：为了分担服务器的负载，可以使用负载均衡器将请求均匀地分配给多个服务器。

**日志和指标logging&metrics**：服务器具有日志服务，记录所有活动。为了更可靠地记录日志，通常会将其写入另一个外部服务器。此外，需要指标服务来收集服务器环境中的各种数据，以便开发人员可以了解服务器的行为并识别潜在的瓶颈。

**警报Alert**：开发人员可以设置警报，以便在某些指标未达到目标时收到推送通知，这样他们就不需要不断地检查指标是否正常。警报的设置有助于及时发现问题并采取行动。

#### 系统设计要求

设计大型和有效的分布式系统所需满足的要求：

1. **系统设计基础**：系统设计面试中，重点在于设计思路和分析，而不是关于每个组件的细节。设计一个有效的分布式系统需要考虑到数据的移动、存储和转换，以及系统的可用性、可靠性、容错性和冗余性等方面。

2. **思考方式**：设计大型企业系统时，可以将其简化为三个关键点：数据的移动（moving data）、存储（storing data）和转换（transforming data）。在分布式系统中，数据移动通常比本地数据移动更具挑战性。

3. **存储数据**：在设计大型分布式系统时，必然需要存储数据。需要考虑如何存储数据，比如使用数据库、Blob存储、文件系统或分布式文件系统等。选择合适的存储方式取决于使用情况。

4. **转换数据**：除了移动和存储数据外，系统还需要转换数据。转换数据的方式有很多，例如监控服务可以将服务器日志转换为成功和失败请求的百分比。无论转换过程多么复杂或简单，都需要考虑如何高效地转换数据。

5. **可用性（Availibility）**：可用性是一个有效系统的核心。它指的是系统在一定时间内处于可用状态的百分比。了解可用性的重要性，以及如何计算和提高可用性是设计系统的关键。

6. **可靠性、容错性和冗余性（Reliability, Fault Tolerance, and Redundancy）**：可靠性是指系统在一段时间内执行其预期功能而无故障或错误的能力。容错性指的是系统发生故障时，如何检测并自行修复问题。冗余性是指系统中备份组件的存在，以防主要组件发生故障。

7. **吞吐量（Throughput）和延迟（Latency）**：吞吐量指的是系统在一段时间内处理的数据量或操作数量。延迟指的是客户端发送请求到服务器响应请求之间的延迟时间。设计系统时需要考虑如何提高吞吐量和降低延迟。

总的来说，设计一个有效的系统需要考虑到数据的移动、存储和转换，以及系统的可用性、可靠性、容错性、冗余性、吞吐量和延迟等因素。

### 2-Networking

#### 网络基础知识

1. **网络概念**：网络是由多个设备连接在一起，通过一定的协议进行通信的集合。每个设备都有唯一的IP地址，用于在网络中进行标识和通信。

2. **IP地址**：IP地址是一个设备在网络中的唯一标识符，可以分为IPv4和IPv6两种类型。IPv4地址是32位的，通常表示为四个十进制数，而IPv6地址是128位的，通常表示为八组十六进制数。

3. **协议（protocols）**：数据在网络中传输时遵循特定的协议，如IP协议和TCP协议。IP协议负责将数据包从源地址传输到目标地址，而TCP协议则确保数据包按顺序到达目标地址。

4. **网络层次结构（network layers）**：计算机网络的协议被组织成不同的层次结构，以便逐层处理数据。常见的层次包括网络层、传输层和应用层。

5. **公网和私网（public & pravite network）**：公网IP地址是全球唯一的，可通过互联网进行访问，而私网IP地址只能在局域网内部访问。

6. **动态和静态IP地址（static & dynamic ip address）**：动态IP地址在设备连接到网络时由服务器动态分配，而静态IP地址则需要手动配置，通常用于服务器等需要固定地址的设备。

7. **端口（ports）**：端口是用于区分不同应用或服务的数字标识符，允许在同一设备上同时运行多个应用或服务。常见的端口号包括HTTP的80端口。

#### TCP&UDP

TCP（传输控制协议）和UDP（用户数据报协议）的用途和特点：

1. **TCP的用途**：TCP确保数据可靠传输，通过建立双向连接（即3次握手）来实现数据的可靠传输。TCP会对数据包进行排序和重传，以确保数据的可靠性。由于其可靠性，TCP常用于需要确保数据完整性和顺序性的应用，如文件传输、网页访问等。然而，TCP的可靠性带来了额外的开销和速度的降低。

2. **UDP的用途**：UDP虽然不如TCP可靠，但传输速度更快。UDP不会尝试重新发送丢失的数据包或对它们进行重新排序。UDP通常用于对数据传输速度要求较高、对可靠性要求较低的场景，如在线游戏、视频流等。在这些场景中，速度和效率比可靠性和错误校正更为重要。

总的来说，TCP用于需要可靠传输的场景，而UDP则适用于对速度要求较高、对可靠性要求较低的场景。

#### DNS

1. **域名系统（DNS）**：DNS类似于互联网的电话簿，将易于阅读的网站名称转换为数字IP地址。它通过分层的命名系统将域名映射到IP地址，从而帮助计算机确定请求的路由。

2. **ICANN和域名注册商**：ICANN负责管理域名系统的整体协调、安全性和运作。域名注册商类似于购物中心的租赁提供商，负责提供域名注册服务，并通过ICANN获得认证。他们帮助用户搜索可用的域名并完成注册过程。

3. **DNS记录**：DNS记录存储与域或子域相关的信息。其中，A记录是最常见的DNS记录类型，将域名与IPv4地址关联起来。DNS记录的作用是确保请求能够准确地路由到相应的服务器上。

4. **URL（统一资源定位符）的解剖**：URL包含多个部分，包括协议、域名、子域名、顶级域、路径和端口。通过解析URL，可以确定访问资源所需的协议、服务器位置和资源路径。

5. **端口**：端口号用于区分在同一设备上运行的多个应用程序或服务。通常情况下，HTTP协议使用默认端口80，而HTTPS协议使用默认端口443。端口号在URL中可以指定，但通常情况下不需要明确指定。

### 3-APIs

**客户端-服务器（Client-Server）模型**：

- **客户端**：在客户端-服务器模型中，客户端是指访问服务器提供的服务的应用程序或系统。它可以是从网页浏览器到电子邮件软件再到移动应用程序等各种形式，它通过发送请求获取特定信息或服务来发起通信。
- **服务器**：服务器是一台计算机、设备或软件，向网络上的客户端或其他服务器提供资源、数据、服务或功能。它等待来自客户端的请求，并通过满足这些请求来响应，例如提供网页、处理电子邮件或提供数据库访问。
- **可互换的角色**：客户端和服务器的角色可以根据上下文互换。一台计算机可以同时充当客户端和服务器的角色，例如当网站从第三方请求数据时。点对点（P2P）网络也展示了这种角色的可互换性。
- **主叫和被叫**：客户端通常被称为“主叫方”，因为它发起请求，而服务器则被视为“被叫方”，它提供服务或处理请求。

了解客户端-服务器模型在网络和系统设计中至关重要，因为它构成了跨网络进行通信和资源共享的基础。

**RPCs (Remote Procedure Call)**：

远程过程调用（RPC）是一种允许程序在不同机器上执行函数的技术，为分布式系统中的任务管理提供了简单高效的解决方案，使得程序能够跨多台计算机进行操作。在客户端-服务器模型中，客户端和服务器通过网络保持连接。

举个实际场景来说：假设我们在YouTube的搜索栏中输入“机器学习”，然后浏览器显示了所有有关机器学习的视频的列表。负责这个列表操作的代码并不驻留在我们的浏览器中。这是因为浏览器并不是视频存储的地方；相反，这段代码位于YouTube的服务器上。这个操作可能会使用一个名为listVideos('机器学习')的函数。即使看起来代码是在客户端执行，实际上它在后台是在调用YouTube的服务器来检索相关信息。

#### HTTP（超文本传输协议）

建立在IP和TCP之上的协议。作为开发人员，我们对此有一定的控制权。回想一下，基本上，IP负责将数据包从源传输到目的地，但不关心数据包到达的顺序。而TCP是建立在IP之上的，它确保数据包按照正确的顺序传递。HTTP则位于TCP之上。它是一种请求request/响应response协议。它是一组关于数据应该如何格式化和在网络上传输的规则，以及服务器和浏览器应该如何响应不同命令的协议。每次通过浏览器发起调用时，都会使用HTTP。

通过浏览器开发者工具可以很好的理解HTTP。在网页浏览器中输入网页地址，然后进入开发者工具，打开网络标签页会显示关键的细节，如请求的名称、状态码、请求类型、响应的大小以及完成请求所需的时间。

HTTP请求的解剖：

- **请求方法**：指示对提供的资源执行的期望操作。例如，我们可以使用GET（检索资源）、POST（提交要由资源处理的数据）、PUT（更新）和DELETE（删除）等方法。
- **请求URL/URI**：指定请求应该发送到哪里的URL。如所见，它还包含路径/results，并使用?search_query=something指定参数。这反映了我们在搜索框中输入的内容。
- **头部Headers**：头部提供了关于请求的附加信息。这些可以是响应头部或请求头部。
   - HTTP请求头部：提供了有关请求的必要信息，指示方法、方案，但可能最重要的是客户端可以接受的数据类型。如下图所示，我们的浏览器接受text/HTML。
   - HTTP响应头部：提供了关于响应或服务器的附加信息。它们还设置了cookie（Set-Cookie），指定了缓存行为（Cache-Control）和内容类型（Content-Type）。
- **主体Body**：并非每个HTTP请求都包含主体。例如，GET请求没有主体，因为我们没有发送任何数据；相反，我们是在检索数据。对于像POST或PUT这样的请求，主体包含我们想要发送到服务器的数据（有效负载）。

HTTP状态码是服务器通知客户端其请求状态的一种方式。状态码分为几个类别，帮助开发人员了解其请求的状态。以下是网络层次：

- 信息响应（100 - 199）：用于确认已接收到客户端的请求并正在处理。
  - 100 Continue：表示到目前为止一切正常，客户端应继续请求，或者如果请求已经完成，则忽略它。
  - 101 Switching Protocols：表示服务器正在切换到客户端从升级请求头中接收到的不同协议。

- 成功响应（200 - 299）：200级别的任何代码都表示成功响应。
  - 200 OK：表示请求已成功。这表示请求已被处理，并且服务器正在返回请求的数据。
  - 201 Created：表示请求已成功，创建了一个新资源。通常在POST或PUT请求后发送，确认资源已成功创建或更新。

- 重定向消息（300 - 399）：当请求的资源已被分配新的永久URL或在不同的URL上临时可用时使用。
  - 300 Multiple Choices：表示请求具有多个可能的响应。客户端应选择其中一个。
  - 301 Moved Permanently：表示请求的资源已永久移动到新位置，服务器正在将客户端重定向到此新位置。

- 客户端错误响应（400 - 499）：当服务器遇到无效或无法理解的客户端请求时使用。
  - 400 Bad Request：表示服务器遇到无效或无法理解的客户端请求。通常在请求中传递了不正确的参数时出现，导致了坏请求。
  - 401 Unauthorized：当客户端尝试未经适当身份验证或授权即访问受保护的资源时返回。例如，如果尝试删除您没有权限删除的视频，则服务器将以401 Unauthorized状态码响应。

- 服务器错误响应（500 - 599）：500-599范围内的服务器错误响应表示服务器在处理客户端请求时发生了错误。这些响应代码通常指示服务器内部出现了问题或失败，无法满足请求。

**SSL/TLS与HTTPS**：

- SSL/TLS：SSL（安全套接层）和TLS（传输层安全Transport Layer Security）是用于在网络上进行安全通信的协议。TLS通过加密数据来确保通信的安全性。虽然我们只是在高层次上讨论它，因为对系统设计来说并不是非常重要，但它在保护网络通信方面起着关键作用。

- HTTPS：HTTPS是HTTP与TLS的结合，用于安全地传输数据。HTTP本身容易受到中间人攻击的影响，而HTTPS通过SSL/TLS保护数据的安全性和完整性，防止了这种攻击。

- SSL与TLS：虽然SSL经常与TLS互换使用，但值得注意的是SSL在技术上是一个过时的术语，已被TLS取代。TLS是SSL的继任者，更安全、更强大。

- 最重要的协议：总的来说，对开发人员来说最有用的协议是HTTPS。HTTPS确保了数据的安全传输，对于保护用户隐私和数据完整性至关重要。

#### Web Socket

**应用层协议(Application level protocols)**包括 HTTP、WebSocket、FTP、SMTP、SSH 和 WebRTC等，它们中除了 WebRTC 使用 UDP 外，其余均使用 TCP。

虽然上面说到HTTP到泛用性和优势，但是对于实时聊天应用等需要实时通信的场景，HTTP 的单向请求-响应模式效率较低，需要频繁的轮询来保持实时性，存在性能开销大的问题。

所以这里出现了**WebSocket**：WebSocket 是一种支持双向通信的协议，可以实现客户端和服务器之间的实时数据传输，适用于实时聊天、直播应用等场景。相较于 HTTP，WebSocket 更高效、实时性更强，能够解决实时通信的需求。

**建立 WebSocket 连接**的过程，包括客户端发送 WebSocket 握手请求、服务器响应握手请求等步骤，WebSocket 的数据传输方式是双向的、实时的数据传输。一旦建立了 WebSocket 连接，客户端和服务器之间可以直接通过该连接进行双向通信，而无需像 HTTP 那样发起多个请求-响应往返。这意味着客户端和服务器可以随时发送数据给对方，并且接收方可以立即收到数据，从而实现实时的、即时的通信。WebSocket 的数据传输是基于 TCP 连接的，但它在连接的基础上提供了更高级别的通信抽象，以实现双向通信。

关于端口：WebSocket 默认使用端口 80，类似于 HTTP；WebSocket 安全连接使用端口 443，类似于 HTTPS。

#### API范式

API是用于客户端与服务器之间通过网络执行操作的方式。它们由一组规则和协议组成，用于构建和交互软件应用程序。API定义了程序（通常是软件库或操作系统）应该使用的方法和数据格式，以与其他软件进行通信。

三种不同的API范式，分别是REST、GraphQL和gRPC。每种范式都具有其独特的特点，并在特定场景中表现出色。

除了REST、GraphQL和gRPC之外，还有其他模式，如SOAP（简单对象访问协议）和WebHooks，但是上面一开始提到的三种是最流行的。分别看一下。

**REST**的关键概念：

- REST是符合REpresentational State Transfer架构风格设计原则和标准的API。
- REST利用简单的HTTP协议进行机器间通信，特别是客户端和服务器之间的通信。
- REST API需要客户端-服务器架构，其中客户端和服务器是通过网络进行通信的独立实体，这种分离支持了客户端和服务器的独立开发和更新。
- REST API是无状态的，每个客户端请求都必须包含处理请求所需的所有信息，服务器不应保留任何关于前一次客户端请求的信息，这有助于水平扩展。
- 通过示例说明了在REST API中状态的概念，以及在处理资源获取请求时，客户端如何将必要的数据包含在请求中，而不依赖服务器记住先前的请求状态。还没有提到的Load-Balancer构架，会有多个服务器存在，且可能随时丢弃服务器，因为无状态的服务器，也就是restapi更适合这种构架的实现。
- REST API不仅接受JSON格式的数据，而且响应的数据也封装在相同的格式中。JSON的名称源自其与JavaScript对象的结构相似，具有键-值对并支持各种层次的嵌套。JSON作为传输信息的首选数据格式。JSON中的键和值都被双引号括起来，表示它们的类型为字符串。
- **overfetching**是RESTAPI的主要问题，当我们请求内容的时候，会返回不必要的内容，或者内容不足，这是它的主要问题。这回造成网络资源浪费，或者达不到预期的应用效果。

**GraphQL**：

GraphQL作为一种替代API范例出现，旨在解决RESTAPI的局限性，特别是在缓解与过度获取和获取不足相关的问题方面。通过 GraphQL，客户端能够在单个请求中精确指定所需数据，无论数据在服务器上的结构如何。按照客户端的规范，服务器负责收集所有必要的数据并相应地对其进行格式化。

在 GraphQL 中，有两种主要类型的操作：queries和mutations。前者用于检索数据，而后者用于修改服务器上的数据。值得注意的是，GraphQL 通过单个端点（通常是 HTTP POST 端点）进行操作，所有查询都发送到该端点。GraphQL 的功能可以使用 SpaceX API 进行有效演示。

**gRPC**：

gRPC代表远程过程调用，来自谷歌。它是执行 RPC 的框架：一种允许程序在另一个地址空间（通常在共享网络上的另一台计算机上）执行过程的方法。gRPC 与 Web 套接字和 HTTP/2 一样，提供双向通信，或者通过单个 TCP 连接和服务器推送多路复用多个消息。

通常，gRPC 用于服务器与服务器之间的通信。在性能方面，它比 REST API 快得多，因为它使用协议缓冲区而不是 JSON 发送数据。协议缓冲区（protocal buffer）是一种与语言无关、与平台无关的可扩展机制，用于序列化结构化数据。

gRPC还提供流式（streaming）传输，即我们可以将数据从客户端推送到服务器，也可以从服务器推送到客户端。这可能会让人想起 WebSocket。然而，gRPC 并不是WebSocket的替代品，而是 REST API 的替代品。

#### API design

1. **API设计**：API设计的过程，包括API的表面区域或契约，以及考虑因素、限制、版本控制和API所涵盖的各种功能。

2. **API常见操作**：CRUD操作（创建、读取、更新、删除）。

3. **API文档**：通过定义API文档，为开发人员提供清晰的接口以与API交互的重要性，以确保一致的行为和无缝集成到其应用程序中。包括端点，操作，参数，数据格式等。

4. **向后兼容性**：考虑API向后兼容性的重要性，以确保对旧版本应用程序的平稳过渡，同时避免破坏现有功能。

5. **分页和GET请求**：通过在URL中包含限制和偏移参数来有效地实现分页，GET请求具有幂等性特性。

6. **API版本**：API版本控制很重要，通过更新版本来引入重大更改，处理安全漏洞时可能涉及的API版本更新。

虽然一般来说作为开发人员很少从零设计API但是掌握设计的基本原则，和有效地阅读文档，有助于开发人员更好地使用API。

### 4 - Caching

#### caching basic

CPU缓存是计算机速度最快的存储器，能够比RAM或磁盘更快地执行读写操作。缓存是数据的复制过程，可以将数据从RAM或磁盘复制到CPU的缓存中，使得读写速度更快。然而，缓存容量有限，通常只有几千字节或几兆字节，因此操作系统需要仔细选择要存储在缓存中的数据。除了单机缓存，分布式系统中也广泛使用缓存。

**从各个角度看缓存的运作方式：**

首先，客户端视角：得到请求后首先浏览器会检查内存缓存，如果不存在，则会去检查磁盘缓存是否存在，内有过去的历史信息资源，如果磁盘中也不存在，则会进行网络资源请求。

一个指标叫做缓存命中率，找到缓存是命中，没找到就是未命中。缓存命中率的计算方法是，命中缓存 /（命中缓存 + 未命中缓存）。

其次，服务器视角：服务器的disk比如数据库，而缓存比如Redis数据库，是一种缓存数据库。

服务器在缓存数据时的复杂性和不同缓存模式的实际运作方式：

1. **写回缓存（Write-back cache）**：在写回缓存模式下，数据首先被写入缓存，然后只有在缓存满时才会被写入磁盘。这意味着在数据更新时只需更新缓存，可以延迟写入磁盘，从而减少对磁盘的频繁访问。但是这种情况下，如果缓存丢失，则会造成数据损失，所以要看你的容忍度。

2. **写穿透缓存（Write-through cache）**：写穿透缓存模式下，数据被同时写入缓存和磁盘，无论数据是否被访问。这样做可以确保缓存和磁盘中的数据一致性，避免数据不一致的问题，但可能增加内存总线的负载。

3. **写通过缓存（Write-around cache）**：写通过缓存模式下，新数据直接写入磁盘，而不是写入缓存。只有当数据被访问时，才会被添加到缓存中。这样做可以节省缓存空间，但会导致首次访问时的较高延迟。

4. **缓存清理策略**：缓存清理策略决定了在缓存满时应该清除哪些数据以腾出空间。其中，常见的策略包括FIFO（先进先出）、LRU（最近最少使用）和LFU（最不经常使用）。

5. **LRU（最近最少使用）策略**：LRU缓存策略依据数据的最近使用情况来清理缓存，如果数据长时间未被访问，则被认为是最有可能被清除的。这种策略适用于需保持数据热度的场景，如Twitter中的热门推文。

6. **LFU（最不经常使用）策略**：LFU缓存策略根据数据被访问的频率来决定清理缓存，如果数据使用频率低，则可能被清除。然而，在某些情况下，LFU可能会出现问题，例如对于在过去长时间内频繁访问的旧数据，可能永远不会被清除。

综上所述，服务器在选择缓存策略时需要考虑数据的访问频率、内存和磁盘的负载以及数据的一致性需求。选择合适的缓存策略可以提高系统性能和响应速度。但是最常用的还是LRU策略。

#### CDNS

内容分发网络，是一组分布在世界各地，边缘的缓存服务器。以便将内容更快速的传递给用户。主要是静态文件HTML，CSS，JavaScript等。

现代边缘服务器使得真正提供动态内容成为可能。这可以通过无服务器JavaScript函数来实现，这些函数接收不同的输入，如设备类型、时间、用户位置和缓存动态内容。

同时CDNS可以实现类似loadbalancer的功能，当一个CDNS服务器出现故障的时候，会重定向到下一个最近的服务器。

**推送CDN**：

推送CDN特别适用于内容是静态的且不经常更改的网站。在推送CDN中，一旦新数据添加到原始服务器，它就会立即推送到所有缓存服务器。

值得注意的是，为了使推送CDN更有效，内容需要被广泛请求。由于推送CDN将内容分发到所有缓存服务器，如果内容没有被用户请求，则它可能效率低下。

适合推送CDN的一个典型用例是托管全球视频内容的网站。推送CDN的优势在于用户无需等待内容从原始服务器传输到CDN。

每个用户请求都会导致缓存命中，因为数据已经被复制过来。虽然内容所有者可以完全自由地确定哪些内容可以到达CDN，但他们也要负责不断更新和维护服务器上的文件。

**拉取CDN**：

如果客户端或用户要访问的数据不存在于CDN上，则会发生类似缓存的过程。首先检查缓存服务器，如果数据不存在，则缓存服务器将从原始服务器检索并缓存它以供未来请求使用。

与推送CDN相比，拉取CDN通过根据需要从原始服务器检索内容来操作，更符合我们之前讨论的缓存的原始概念。

像Twitter这样的网站是拉取CDN的理想使用者，因为它每秒生成的内容量巨大。对于Twitter团队来说，主动将所有内容推送到CDN将是一个重大负担。相反，当用户从服务器请求内容时，CDN将自动从服务器检索并将其存储在缓存中。

此外，值得注意的是，来自世界不同地区的用户可能请求不同的内容。例如，某些Twitter个人资料可能在一个地区比另一个地区更受欢迎。因此，位于不同地区的不同CDN可能具有针对各自受众定制的不同内容。

这通常与具有地理特定内容的较大平台更相关。但这仍然是一个有趣的工程概念值得考虑。
因此，过早地将所有内容推送到CDN可能会导致资源消耗过多，而拉取CDN方法允许根据用户请求动态检索和缓存内容。

### 5 - Proxies（代理）and Load Balancing

也就是代理和负载均衡。看这个单词别和协议搞错了。

转发代理服务器相当于一个网络中介，不仅可以隐藏我们的身份，也就是IP，还可以缓存内容提高效率。同时还可以控制网络流量。

对客户端进行的代理是**正向代理**。

反向代理对目标服务器而不是客户端进行匿名化。它也可以被视为入站代理。

**反向代理**位于互联网和服务器之间，其运行方式与正向代理不同。它不是接收来自客户端的请求并转发它们，而是接收请求并将它们转发到适当的服务器。反向代理的主要优点是它能够通过管理传入请求和在多个服务器之间分配负载来保护服务器。此外，它还提供针对分布式拒绝服务 (DDoS) 攻击的保护，从而保护网站。

正向代理充当客户端和互联网之间的中介，为客户端提供额外的安全和隐私层，而反向代理则为服务器提供类似的目的。反向代理的一个主要示例是内容分发网络 (CDN)。反向代理 (CDN) 处理请求，而不是客户端直接访问源服务器。如果请求的内容可用，反向代理会将其直接提供给客户端，从而有效减少延迟并减轻源服务器的负载。

#### 负载均衡器

反向代理的另一个广泛认可的示例是负载均衡器。与其他反向代理类似，负载均衡器驻留在客户端和服务器之间，充当它们通信的中介。

负载平衡可以通过多种方式实现。例如，每天访问量很大的热门网站不太可能使用单个服务器管理所有传入流量。在这种情况下，负载均衡器通过在已水平扩展的多个服务器之间智能地分配流量来发挥至关重要的作用。这确保了传入请求的高效处理和资源的最佳利用。

负载均衡器使用各种策略来有效地分发传入的网络请求。它们旨在优化资源使用、防止服务器过载并提高整体系统响应能力和性能。集中常见的负载均衡策略：

1. **轮询（Round Robin）**：按照事先设定的顺序依次将请求分配给每台服务器，循环往复。这种方法简单高效，但无法根据服务器的实际负载情况进行动态调整。

2. **最少连接（Least Connections）**：将新的请求分配给当前连接数最少的服务器，以保持服务器负载相对均衡。这种方法能够有效地避免服务器负载不均衡的情况。

3. **最短响应时间（Least Response Time）**：将新的请求分配给响应时间最短的服务器，通常通过监测服务器的响应时间来实现。这种方法能够使系统整体的响应时间最小化。

4. **IP哈希（IP Hash）**：根据客户端的IP地址将请求分配给特定的服务器，这样可以保证同一客户端的请求始终被分配到同一台服务器上，适用于需要保持会话状态的场景。

5. **加权轮询（Weighted Round Robin）**：在轮询的基础上引入权重，按照服务器的权重分配请求，权重越高的服务器被选中的概率越大，适用于不同服务器性能不均衡的情况。通过使用WRR技术，请求的分配与每个服务器的资源成正比，确保更有效地利用可用资源。“计算能力”涉及服务器的CPU处理能力、RAM、存储和网络带宽。

6. **加权最小连接数（Weighted Least Connections）**：在最少连接的基础上引入权重，按照服务器的权重选择连接数最少的服务器，以实现更灵活的负载均衡。

7. **基于域名（DNS-based Load Balancing）**：根据请求的域名将流量路由到不同的服务器集群，通常通过DNS解析实现。这种方法适用于需要根据域名进行流量分发的情况。

8. **内容敏感负载均衡（Content-aware Load Balancing）**：根据请求的内容特征（如URL、报文头部等）将流量分配到不同的服务器，以实现更精细的流量控制和处理。

9. **自适应负载均衡（Adaptive Load Balancing）**：根据服务器的实时负载情况和性能指标动态调整负载均衡策略，以实现更好的系统性能和资源利用率。

10. **四层和七层负载均衡**：在四层（TCP/UDP层）负载均衡中，负载均衡设备根据传输层的信息（如源IP地址、目标IP地址、源端口、目标端口等）来做出负载均衡决策，而不关心应用层的协议和数据内容。在七层（HTTP层）负载均衡中，负载均衡设备不仅可以根据传输层信息进行负载均衡，还能深入到应用层，根据HTTP请求的内容、URL等信息来做出负载均衡决策。七层负载均衡能够实现更加细粒度的流量控制和路由。

#### 一致性哈希

1. **哈希负载均衡**：哈希用于在负载均衡场景下将请求分配给服务器。每个请求都与一个唯一标识符（如IP地址、用户ID、请求ID）相关联，哈希函数用于将此标识符映射到一个服务器上。但是普通的哈希负载技术有些一些问题，因为是使用模运算，如果有一台服务器出现故障，就要对所有的请求进行新的模运算，这时候服务器数量变化了，可能导致原本的映射发生改变，从而导致不一致性，所以出现了下面的解决方案。

2. **一致性哈希原理**：一致性哈希确保即使在系统中添加或删除服务器时，请求与服务器的映射仍然保持稳定。它通过使用基于**环**的结构和哈希函数实现均匀的负载分布。

3. **处理服务器故障**：在一致性哈希中，如果服务器故障，它处理的请求将重新分配给环上顺时针方向的下一个可用服务器，以保持请求分配的一致性。

4. **模运算**：一致性哈希中使用模运算，以确保结果哈希值落在环上可用位置的范围内。这可以防止溢出问题，并促进请求的均匀分布。

5. **一致性哈希的应用**：一致性哈希适用于诸如内容传送网络（CDN）和数据库等场景，其中将特定用户路由到相同的缓存服务器或数据库服务器对于效率和一致性至关重要。

6. **与其他负载均衡技术的比较**：尽管一致性哈希提供了诸如一致路由和高效缓存利用等优点，但并不否定其他技术如轮询的有效性，后者在不涉及缓存的情况下可以很好地工作。

总的来说，一致性哈希确保分布式系统中的稳定和高效的负载分布，使其成为各种应用中的一项宝贵技术。

### 6 - Storage

#### SQL

**SQL数据库的底层数据结构是B+tree。**

B树是一种自平衡的搜索树，它被广泛应用于数据库索引、文件系统等场景中，用于支持高效的查找、插入和删除操作。

1. **平衡性**：B树是一种平衡树，它确保了树的高度相对较低，从而保证了查找、插入和删除操作的时间复杂度都是对数级别的。

2. **多路搜索**：B树的节点可以有多个子节点，这使得B树能够在每个节点上存储更多的键值对，从而减少了树的高度，提高了搜索效率。

3. **有序性**：在B树中，节点的键值按顺序存储，这使得范围查询等操作更加高效。

4. **磁盘IO优化**：B树的节点通常设计成与磁盘块大小相适配的大小，这样可以减少磁盘IO操作次数，提高数据访问效率。

与B树相比，B+树是一种改进的树型数据结构，它在B树的基础上做了一些优化：

1. **叶子节点存储数据**：在B+树中，所有的数据都存储在叶子节点上，而非内部节点。这样做使得B+树的内部节点能够容纳更多的索引项，从而减少了树的高度，提高了查询效率。

2. **链表连接叶子节点**：在B+树中，所有的叶子节点之间都通过链表连接起来，这样使得范围查询等操作更加高效。

3. **提高了顺序访问性能**：由于叶子节点之间通过链表连接，B+树支持高效的顺序遍历操作，因此适合用于范围查询等需要顺序访问的场景。

总的来说，B+树是一种优秀的数据结构，被广泛应用于数据库索引等场景中，它能够提供高效的查找、插入和删除操作，同时支持范围查询等功能。

**ACID** 是数据库事务的四个特性的首字母缩写，分别是原子性（Atomicity）、一致性（Consistency）、隔离性（Isolation）和持久性（Durability）。这些特性确保了数据库事务的可靠性、稳定性和一致性。

以下是对每个 ACID 特性的简要介绍：

1. **原子性（Atomicity）**：原子性指的是事务是不可分割的单位，事务中的所有操作要么全部执行成功，要么全部失败回滚。如果事务中的任何一个操作失败，那么整个事务都应该回滚到初始状态，不会留下部分完成的操作。

2. **一致性（Consistency）**：一致性确保了事务的执行不会破坏数据库的一致性约束。换句话说，如果数据库在事务开始之前处于一致的状态，那么事务结束后，数据库也应该保持一致的状态。这意味着事务执行的结果必须符合数据库的预定义规则和约束。

3. **隔离性（Isolation）**：隔离性指的是一个事务的执行应该与其他事务的执行相互隔离，使得每个事务感觉不到其他事务的存在，就好像它是唯一执行的事务一样。这可以防止并发事务之间的相互干扰和数据不一致性问题。在并发环境中，隔离性通常通过锁机制和多版本并发控制（MVCC）等技术来实现。

4. **持久性（Durability）**：持久性确保事务一旦提交成功，其修改的数据库状态将永久保存在数据库中，并且不会丢失，即使系统崩溃或发生故障也是如此。通常，持久性通过将事务的操作记录到日志中，以便在系统崩溃后可以恢复数据库到事务提交前的状态来实现。

这些 ACID 特性确保了数据库事务的可靠性和一致性，使得数据库系统能够在并发环境中有效地管理事务，并提供数据的一致性和持久性保证。

#### NoSQL：大数据时代的数据形态

NoSQL代表“Not only sql”，而不是“not“，我自己也已开始理解错误，它有些情况不能使用传统的sql工具进行查询。关键特性是：大规模数据，高速工作负载，可以水平扩展。几种常见的NoSQL数据库以及它们的特点和应用场景。以下是每种类型的主要知识点总结：

1. **键值数据库**：
   - 使用简单的键值方法存储数据。
   - 数据以键值对的形式存储，键作为唯一标识符。
   - 无模式，不需要预定义模式即可存储数据。
   - 不同的键和值可以具有完全不同的结构。
   - 适用于快速数据检索，例如 Redis 是其中的一个示例。

2. **文档数据库**：
   - 数据以文档形式存储，类似于 JSON。
   - 提供灵活性，可以独立添加或删除文档中的字段。
   - 数据存储格式与开发人员在应用程序代码中使用的文档模型格式相同。
   - MongoDB 是一个常见的文档数据库示例。

3. **宽列数据库**：
   - 将数据存储在列中而不是行中，可以高效存储和检索大型数据集。
   - 适用于需要高写入吞吐量和优化特定数据子集读取和聚合的场景。
   - 例如 Apache Cassandra 和 Google 的 BigTable 是宽列数据库的示例。

4. **图数据库**：
   - 使用类似图的结构存储数据，每个节点引用一个实体，节点通过边或关系相互连接。
   - 适用于表示实体之间各种类型的关系、友谊和关联。
   - 适用于复杂关系和互连性的数据场景，例如社交网络。
   - Facebook 是图数据库适用的典型例子。


同时SQL 数据库在处理规模和*分布式架构*方面存在问题，因为数据通常存储在单个服务器上，无法轻松拆分和分布到多个节点上。而NoSQL数据库考虑了分布式架构，并允许数据存储在多个节点上。（*删除外键约束*使得数据分布和拆分更为容易。）

**ACID vs. BASE**：
   - SQL 数据库通常使用 ACID（原子性、一致性、隔离性、持久性）来确保严格一致性。
   - NoSQL 数据库更倾向于使用 BaSE（基本可用、软状态、最终一致性）来描述其特性，注重最终一致性。

**关于最终一致性**：
   - 最终一致性是指系统在一段时间内可能出现不一致状态，但最终会达到一致状态。
   - 领导者/跟随者架构是提供最终一致性的一种机制，其中主节点leader负责更新数据，其余节点followers最终会同步更新。

总的来说，NoSQL 数据库通过弱化一致性要求和支持分布式架构来解决传统 SQL 数据库的限制，为大规模数据处理提供了更灵活和可扩展的解决方案。

对标传统的事务处理数据库，NoSQL是当今大数据时代的标配。

#### 数据库的复制和分片（Replication and Sharding）

1. **复制和分片**：
   - 复制和分片是用于提高分布式系统可用性和吞吐量的常用技术。
   - 复制是一种横向扩展。纵向扩展就是不断增加数据。但是数据不断增加，就需要分片技术。NoSQL很适合切片。形成分布式系统。

2. **复制**：
   - 用于处理单个数据库无法应对的请求量。
   - 包括领导者leader（主数据库）和追随者follower（从数据库）。（or master and slave）
     - client不能对slave进行write操作。
   - 有同步和异步两种方式：
     - **同步（Synchronous）复制**：保证数据一致性，但引入延迟。当主死机，最后更新的从可以立刻复制到主复活主。
     - **异步（Asynchronous）复制**：减少延迟，但可能导致数据不一致。
   - 可以实现主-主（多主）复制（Master-Master (Multi-Master) Replication），用于跨不同区域提供数据。

3. **分片**：
   - 当单个复制不足以处理高流量时使用。
   - 将数据库划分为较小的分片，每个分片托管在单独的服务器上。
   - 分片键决定数据如何分区，可以是基于范围(id, and so on)或基于属性(sex, and so on)。
   - 挑战包括确保相关数据最终位于同一分片中，以及维护 ACID 属性的困难。

4. **分片的挑战**：
   - 确保相关数据位于同一分片中可能复杂。
   - 维护 ACID 属性对关系数据库提出挑战。
   - NoSQL 数据库更适合分片，因为它们在设计时考虑了水平扩展。

5. **最终一致性**：
   - 在分布式系统中，数据副本最终将达到一致状态。
   - NoSQL 数据库提供最终一致性。

6. **基于哈希的分片**：
   - 使用一致哈希可最小化数据移动和重新平衡，以实现跨分片分发数据的方式。

#### CAP理论

数据库的CAP理论是分布式系统设计中的一个基本原理，它指出在一个分布式系统中，Consistency（一致性）、Availability（可用性）、Partition tolerance（分区容忍性）这三个特性不可能同时满足，只能满足其中的两个。这个理论由计算机科学家Eric Brewer在2000年提出。

下面我会对CAP理论的三个要素进行详细解释：

1. **Consistency（一致性）**：指的是在分布式系统中，所有节点的数据在同一时间上的一致性。即当一个节点对数据进行更新后，所有的节点都能够读取到最新的数据，并且数据的副本都能够达到一致的状态。在一致性要求下，系统对于每次读取操作都能够返回最新的数据值。一致性强调的是数据的实时更新和同步，确保每个节点都可以看到同样的数据状态。

2. **Availability（可用性）**：指的是系统能够保证在有请求时能够返回有效的响应，即系统保持可用状态。即使部分节点或者通信出现故障，系统依然能够继续响应客户端的请求，保持服务的可用性。在可用性要求下，系统对于每次请求都能够在有限的时间内给出响应，不一定是最新的数据。

3. **Partition tolerance（分区容忍性）**：指的是系统能够在网络分区或节点故障的情况下继续运行。网络分区是指分布式系统中的节点之间由于网络故障或者延迟等原因而无法通信的情况。分区容忍性要求系统能够在发生分区时保持部分节点之间的通信，同时继续提供服务。

CAP理论的重点在于指出在分布式系统中，无法同时满足一致性、可用性和分区容忍性这三个特性。这是因为在出现网络分区的情况下，如果要求保持一致性，那么系统可能需要阻塞请求以等待网络分区恢复，从而影响了可用性。反之，如果追求高可用性，系统可能会选择在出现网络分区时放弃一致性，允许部分节点返回旧的数据副本。C和A不兼容。

在实际系统设计中，根据具体的需求和场景，可以根据CAP理论的原则来选择满足一致性、可用性和分区容忍性中的两个，而牺牲另外一个。例如，对于大多数互联网应用来说，更倾向于保证可用性和分区容忍性，而对一致性的要求相对较低，因此常常采用AP（可用性和分区容忍性）的设计方案。而对于金融交易系统等对一致性要求较高的场景，则可能更倾向于保证一致性和分区容忍性，而对可用性要求稍低，采用CP（一致性和分区容忍性）的设计方案。

ACID原则和CAP是什么关系？

分布式系统 vs 单机数据库：CAP理论主要适用于分布式系统，强调在分布式环境下无法同时满足一致性、可用性和分区容忍性这三个特性。而ACID理论则更多地关注在单机数据库事务处理中保证数据的一致性、原子性、隔离性和持久性。

#### 对象存储（Object Storage）

数据库 VS 对象存储 - 有什么区别？

在数据库中，我们过滤和搜索数据的方式非常重要。然而，对于对象存储，文件夹的概念并不存在。数据库和对象存储都可以存储数据，但主要区别在于数据的**结构、可访问性和可扩展性**。在文件系统中，数据以类似于树的层次结构进行组织，其中文件存储在可以嵌套在其他文件夹中的文件夹中。

相比之下，对象存储将每条数据视为一个对象，包括实际数据、元数据和唯一标识符。与文件系统不同，对象存储**没有层次结构**。对象存储在平面地址空间中，由于不存在分层复杂性，因此与文件存储系统相比，可更轻松地进行扩展。对象存储从 BLOB（二进制大型对象）存储发展而来，通常用于存储**图像、视频和数据库备份**等项目。突出的例子包括 **AWS S3 和 Google Cloud Storage**。

从设计的角度来看，值得注意的是，通常不建议将图像或视频存储在数据库中。在数据库中查询特定图像或视频的情况很少见，将此类数据存储在数据库中会降低性能、增加存储要求并导致对数据库进行频繁的读写操作。

传统的 RDBMS 并未针对处理大文件进行优化，但对象存储的出现是应对这一挑战的解决方案。它专为高效处理**非结构化数据**而设计，非常适合存储大型文件。使用基于对象的存储的一个显着优势是其**可扩展性**，允许轻松扩展平面架构，而不会遇到与文件存储相关的限制。

从对象存储检索数据时，通常不执行像传统数据库的SQL那样从对象存储本身的直接读取。相反，直接向对象存储**发出网络 HTTP 请求**以获取数据。在系统设计访谈中，对象存储经常用于存储图像和视频，例如通过 Amazon Simple Storage Service (Amazon S3)。

### 7 - 大数据（Big data）

#### 消息队列 Message queues

**消息队列**：消息队列为应用服务器无法同时处理的大量请求提供了解决方案。它们将生产者（事件）和消费者（服务器）解耦（decoupling），并作为缓冲区来处理数据激增。

**示例（支付处理）**：支付处理展示了消息队列如何使系统受益，例如处理大型销售期间的高峰负载以及解耦服务，如订单下单和支付处理。

**推送/拉取模型（push/pull）**：消息队列与应用服务器使用拉取或推送模型进行交互。拉取模型要求应用程序监视队列以获取新消息，而推送模型将消息推送到服务器。两种模型都有关于服务器负载和延迟的优点和考虑因素。

**发布/订阅模型（pub/sub）**：发布者/订阅者模型解耦了发布者和订阅者，使它们不需要知道彼此的存在。发布者将消息发布到特定队列或主题，而一个或多个订阅者订阅这些主题。消息经由消息代理确保发布到主题的所有消息都成功传递给该主题的所有订阅者。订阅者以独立的方式并以自己的速度处理消息。

这些概念提供了一种灵活、可扩展且高效的方法来处理大量的数据和请求，同时确保系统的可靠性和稳定性。

#### MapReduce

MapReduce 涉及大数据处理，能够处理大量数据、对其执行计算并生成结果。MapReduce 是一种编程模型，结合了专门为处理和生成大型数据集而设计的特定实现。该模型对于跨 TB 甚至 PB 的海量数据集的分布式计算特别有利。

中间涉及两个概念，批处理batch processing和流处理stream processing。

**MapReduce功能及实现**
在 Apache Hadoop 等 MapReduce 框架中，系统通常由一个“主”节点（master）和多个“工作”节点（worker or slave）（也称为“从”节点）组成。以下是字数统计场景在这样的系统中的表现：

- 主节点（master）：该节点的任务是管理 MapReduce 作业在工作节点之间的分配。它密切关注每个任务的状态，并在发生任何故障时重新分配任务。
- 工作节点（worker）：这些节点是实际数据处理发生的场所。主节点为每个工作节点分配一部分数据和MapReduce 程序的副本。
- Map阶段：每个工作节点对其分配的数据部分执行Map操作。在我们的场景中，这需要将每个单词映射到一个键值对，其中键是单词，值是单词的频率。
- 洗牌Shuffle和排序sort阶段：在映射阶段之后，工作节点重新组织键值对，以便将与同一键关联的所有值分组在一起。此过程称为洗牌和排序阶段。因此，例如，给定单词“The”，如果worker 1 处理了 3 次出现，worker 2 处理了 7 次 worker 3 处理了 100 次，那么在此阶段它们将被分组在一起。
- 归约reduce阶段：对每组值执行归约操作，生成每个字的最终计数。然后将该结果写入某种形式的存储或数据库。

这是一种先进的分而治之的思想，在函数式编程中有map方法和reduce方法，内部是一样的原理，但是应用在系统中就会发挥巨大的并行运行效率效应。

现在主要的是用Spark实现。

stream处理现在很重要的是SparkFlink服务，在Amazon的kinesis中占据Analysis服务的位置。

## ZTM Thinking model

### Components

![picture](system-design.png)

- **Full Picture**：
- 网络是最重要的，因为他是传递服务的唯一手段
- Components / Simple Architecture:
  - Client (DNS / CDN) - (Load Balancer) WebServer（Application Logic / Database）

- **Web Server**先发送所有的页面渲染文件，HTML/CSS/JS等，然后通过Restapi不断发送json文件来动态改变内容
  - 安全：**WAF**很重要
  - 主要的host **Business Logic**的地方，这里的逻辑代码基本上等于，针对clients的

- **Load Balancer**：降低负荷策略
  - **资源调度**关系到服务器**性能**的是什么：CPU算力，Memory（RAM），Storage，Network带宽
  - **Scaling**：Vertical（垂直） & Horizontal（水平增加服务器数量）
  - **Latency**：一切都是为了解决延迟问题，假设所有的client的延迟都是一样的，这是round robin策略的原因
  - **Session Persistence**：如何确保新的server可以接手原本的user的*session*信息，相关策略：
    - **Reverse Proxy**：（反向代理）代理分配资源，这里是为服务器代理，**Weighted Round Robin**：每台服务器的资源是不同的，这可以决定权重分配
  - **Redundant LB**：因为如果只有一台LB它本身就是一个single point failure，所以可以有多个LB
  - **Server Clustering**：服务器集群加上Healthy Check，比如GCE的集群，或者k8s的集群，他们作为node存在（k8s中是Pods作为服务）
    - 这个node集群的强大之处在于，和一般的LB控制的server集群不同，他们*有一个同样的entry point*，是*作为一个整体存在的*，他们之间是*sync互相之间的信息的*，当一台node失效，其他的nodes之间会进行信息同步
    - 这个nodes集群中可能有一部分是passive的，他们不做事但是他们和其他active同步，当需要的时候他们会接手服务，他们已经保有session信息
    - 这种情况的LB构架就是：LB -> server clustering(many nodes)

- **Database**：数据库不只是存储和取回数据的地方，它和web server之间的数据交互速度是容易成为瓶颈的
  - 所以**Caching**服务相当重要
  - 相对于reliability，*avilability和speed更加重要*，因为时间成本很高

- **Job Server**：为处理数据库上游数据的逻辑
  - 相对于Web server中的business logic，这里处理的是*application logic*，也就是将上游的*Data sources*，进行job处理，然后送入web server需要使用的数据输入**Database**，这就是job的作用
  - 我所处理的数据仓库的job系统，虽然是不接入web server的，但是它可以
  - **Job Queue**：整个job系统通过queue来接收job，通过*优先级*和*先入先出*来控制job的顺序
    - 分布式系统的Job有自己的调度系统，比如我的airflow的Job schedular
    - 又比如用户的处理请求，比如文件处理，或者邮件认证的mail服务，就是event trigger类型
    - 定时处理的cron类型任务，比如数据处理
    - 消息队列系统比如像 Kafka 来处理微服务之间的异步消息通信，实现事件驱动架构。比如，当订单系统创建订单时，它可以发布一个事件，库存服务会订阅该事件并相应地更新库存
    - 第三方API任务，比如与外部系统的交互产生的任务，比如从支付网关、社交媒体、物流服务中接收到的任务

- **Services**：它是对一个整体服务的分解，一个大的服务包括了哪些小的features，这些就是services，而每一个service，都可以看成是一个复杂的项目系统，比如用户**内容请求服务**，**认证服务**都是首先单独存在的
  - *每一个service*都是一个*domain*，他们都有*独立的任务*，并且每个任务都可以做的非常好
  - 认证服务中的**Access Token**是一个非常重要的东西，通过它，web server的逻辑中，就知道了是哪个用户，那么在对其他*services*的交互中，就能够知道该用哪个用户的信息进行请求任务了，也就能知道用户的**session**，甚至用户的**权限是什么**，这就是**认证（authetication）和认可（authorization）**，它促成了服务之间的交互，**微服务构架，疏结合**
  - 结构：**Client -> LB -> Web Server -> Services**
  - ![picture](system-design-services.png)

- **Data hose**&**Data warehouse**：从web server拿到的数据，存入数据库，仅此无他
- **Cloud Storage**：存储所有形式的各种数据，数据湖，或者任何web server的数据文件
  - **CDN**则是用于deliver文件数据的组件，它从Storage中得到文件，CDN 具备缓存的特性，可以被视为 Web Server 缓存的一种高级形式，但它的功能远超出缓存，主要目标是通过分布式架构加速内容传输、内容动态优化，负载均衡，提升用户体验，并提供附加的安全性功能，诸如 DDoS 防护、Web 应用防火墙（WAF）等，它是一个更加**Global**的服务

### Important Things

- 实际的经验是无可取代的
- 系统设计的目标要明确
- **Engineer and build systems that meet the needs of the business in a coherent, efficient and organized way.**(以连贯、高效和有组织的方式设计和构建满足业务需求的系统。)
- **Design the architecture, interfaces, and data.**
- Core Principles: **Availability** and **Reliability**
- 可用性和可靠性都关系到**时间**，1个9到9个9，可用性和cost是一种trade-off
- 网络模型TCP/IP
- **Proxy**：客户端代理是Forward Proxy，代理服务的是反向代理Reverse Proxy，可以是**LB**，也可以是NGNIX这种服务器
  - 反向代理的作用比如caching
- **数据库**和数据，是一个和网络同样重要的模块
- **CAP**理论：对于一个分布式数据存储系统，不能同时完美地满足以下三个特性，一致性（Consistency），可用性（Availability），分区容错性（Partition Tolerance），因为是分布式系统，P总是需要的，如果CA那就要放弃分布式系统只用一个数据库了
  - 在设计分布式系统时，必须根据具体的业务需求来选择系统的优先级
  - 如果业务对数据一致性要求非常高，可以选择**CP**系统，但需要考虑到可用性在网络故障时可能受到影响
  - 而如果系统的可用性是优先级，**AP**系统会更合适
  - 这个理论对理解分布式数据库（如NoSQL数据库）的设计选择非常有帮助，帮助架构师在系统设计中做出适当的权衡
- **ACID**：atomic原子性，consistent一致性，isolated隔离性保证多个并发事务处理不会相互影响，durability持久性保证数据永久存在不会丢失，主要针对**relational-db**
- **BASE**：基本可用性（Basic Availability）、软状态（Soft state）和最终一致性（Eventual consistency），在一定程度上牺牲强一致性，来换取系统的可扩展性和高可用性，主要针对**NoSql**数据库
