## 网络基础

### OSI Model

- layer1：物理层负责通过物理媒体（如电缆、光纤、无线电波等）传输原始比特流。它定义了物理设备的规格和接口，如电压、电缆类型、传输速率、信号传输方式等。
  - 协议和标准：Ethernet（以太网）物理层标准，USB，Bluetooth，IEEE 802.11（Wi-Fi）
- layer2：*链路层*负责将物理层传输的比特流组织*成帧*，并提供可靠的*节点到节点*的数据传输。它负责帧的传输控制、错误检测和纠正、流量控制等。
  - 协议和标准：Ethernet（以太网）数据链路层协议，PPP（点对点协议），HDLC（高级数据链路控制），*MAC*（媒体访问控制）地址
  - 子层sublayers：LLC（*NIC网卡*）逻辑链路控制层，MAC媒体访问控制层
- layer3：*网络层*负责在不同网络之间路由*route*和传输数据包*packets*（数据包可以跨越多个网络）。它管理网络间的路径选择、逻辑地址IP分配、网络间的转发和流量控制等。
  - 协议和标准：*IP（互联网协议），ICMP（互联网控制消息协议），ARP（地址解析协议），OSPF（开放最短路径优先），BGP（边界网关协议）*
- layer4：*传输层*提供可靠的数据传输服务，并负责*数据的分段segmentation、重组和流量控制*。它确保数据包在端到端传输中的完整性和顺序。
  - 协议和标准：*TCP（传输控制协议），UDP（用户数据报协议）*，SCTP（流控制传输协议）
- layer5：会话层负责建立、管理和终止应用程序之间的会话（连接）。它管理会话的建立和同步，并提供对话控制和数据交换。
  - 协议和标准：NetBIOS（网络基本输入输出系统），PPTP（点对点隧道协议），RPC（远程过程调用）
- layer6：表示层负责数据格式化、加密和解密、数据压缩和解压缩。它确保发送方和接收方之间的数据格式一致，使得数据在传输过程中可以被正确解析和展示。
  - 协议和标准：SSL/TLS（安全套接层/传输层安全协议），JPEG（图像压缩格式），ASCII（文本编码标准），MPEG（视频编码标准）
- layer7：*应用层*直接为应用程序提供网络服务和接口。它处理用户接口相关的事务，并负责应用程序间的通信和数据交换。
  - 协议和标准：*HTTP/HTTPS（超文本传输协议/安全超文本传输协议），FTP（文件传输协议），SMTP（简单邮件传输协议），DNS（域名系统）*

### 封装（Encapsulation）和解封装（Decapsulation）

在网络通信中，封装（Encapsulation）和解封装（Decapsulation）是关键的过程，它们描述了数据在网络协议栈中从发送到接收的转换过程。

- 封装是将数据从高层协议添加到低层协议中的过程。
- 在发送数据时，数据从应用层开始，一层层向下传递，每一层都会添加特定的协议头或尾（称为协议数据单元，PDU）。
- 最底层的数据单元最终会变成网络可以传输的帧或数据包。

- 解封装是接收数据时从低层协议到高层协议的过程。
- 数据从物理层开始，一层层向上传递，每一层都会去掉自己的协议头或尾，并将数据传递给上一层。
- 最终，高层应用程序收到去掉了所有协议头的原始数据。


数据封装和解封装过程（宛如人生）

假设我们有一个 Web 客户端向服务器发送 HTTP 请求的过程，以下是每一层的封装示例：

1. **应用层**：
   - 数据：`"GET /index.html HTTP/1.1\r\nHost: www.example.com\r\n\r\n"`
   - 无协议头。

2. **表示层**：
   - 假设表示层没有对数据进行处理。
   - 数据保持不变。

3. **会话层**：
   - 假设会话层也没有对数据进行处理。
   - 数据保持不变。

4. **传输层**：
   - 数据：应用层数据 + 传输层头（例如 TCP 头）。
   - 封装后的段（segment）：
     ```
     TCP Header + Application Data
     ```

5. **网络层**：
   - 数据：传输层段 + 网络层头（例如 IP 头）。
   - 封装后的数据包（packet）：
     ```
     IP Header + TCP Header + Application Data
     ```

6. **数据链路层**：
   - 数据：网络层数据包 + 数据链路层头和尾（例如 Ethernet 头和尾）。
   - 封装后的帧（frame）：
     ```
     Ethernet Header + IP Header + TCP Header + Application Data（payload）+ Ethernet Trailer
     ```

7. **物理层**：
   - 数据：数据链路层的帧被转换成比特流。
   - 发送比特流通过物理介质传输。

当服务器收到这个 HTTP 请求时，解封装过程如下：

1. **物理层**：
   - 接收到比特流。
   - 将比特流转换成帧，并传递给数据链路层。

2. **数据链路层**：
   - 接收帧。
   - 去掉数据链路层头和尾，得到网络层的数据包，并传递给网络层。
     ```
     IP Header + TCP Header + Application Data
     ```

3. **网络层**：
   - 接收数据包。
   - 去掉网络层头，得到传输层的段，并传递给传输层。
     ```
     TCP Header + Application Data
     ```

4. **传输层**：
   - 接收段。
   - 去掉传输层头，得到应用层数据，并传递给会话层。
     ```
     Application Data
     ```

5. **会话层**：
   - 假设会话层没有对数据进行处理。
   - 数据保持不变。

6. **表示层**：
   - 假设表示层没有对数据进行处理。
   - 数据保持不变。

7. **应用层**：
   - 接收应用数据。
   - 解析 HTTP 请求并进行响应。

- **模块化和分层设计**：封装和解封装使得每一层可以专注于特定的功能，从而实现了网络协议的模块化和分层设计。这有助于协议的开发、维护和升级。
- **互操作性**：不同厂商和系统之间可以通过标准化的封装和解封装协议进行互操作。
- **故障诊断**：分层的封装和解封装过程有助于网络故障的隔离和诊断，因为可以在特定的层次上查找问题。
- **数据安全和控制**：通过封装过程，数据可以在不同的层次上被加密、压缩和处理，从而提高了数据的安全性和传输效率。

**这个过程太有意思了，宛如一个人的完整一生。从淳朴到厚重最后再回归淳朴。返璞归真。带了很多了头衔，帽子，内容，负载了TTL生命的长度，过程中会经过各种人生旅程，但是最后到达终点的时候，只是为了回归本我，中间一路的嵌套，只是为了自我保护罢了，都很不容易呐。**

TCP的header中的*Port*指示了设备上的哪个应用。
*MTU*是payload最大传输单元。