## Routing problem(network layer)

ICMP（Internet Control Message Protocol，互联网控制消息协议）是网络协议栈中用于发送控制消息的协议，常用于网络诊断和错误报告。它是IP协议的一部分，通常用于网络设备之间的通信，以报告连接问题和网络状态。

### ICMP的主要功能
1. **错误报告**：当路由器或主机遇到问题（例如网络无法到达、主机不可达、端口不可达等），它会发送ICMP消息通知源主机问题的类型。
2. **网络诊断**：`ping`和`traceroute`等工具都是基于ICMP的，用于测试主机的可达性和追踪数据包的路由路径。

### 常见的ICMP消息类型
1. **Echo Request / Echo Reply**：用于`ping`命令。`ping`工具发送ICMP Echo Request消息，目标主机收到后会返回一个Echo Reply，确认其网络可达性。
2. **Destination Unreachable（目的地不可达）**：如果数据包无法到达目的地，路由器会返回一个目的地不可达的ICMP消息。
3. **Time Exceeded（超时）**：当数据包在网络中停留的时间超过其`TTL`（Time to Live）时，路由器会丢弃该数据包并发送超时消息。`traceroute`工具通过控制`TTL`来追踪数据包的路由路径。
4. **Redirect（重定向）**：当网络拓扑发生变化时，ICMP会引导主机选择更优的路径发送数据包。

### 使用场景
- **`ping`命令**：测试主机可达性
- **`traceroute`命令**：检测数据包的路由*路径*
- **网络问题排查**：监测并报告传输中的网络错误和阻塞情况

## port connection(transport layer)
`netcat`（简称 `nc`）是一个网络工具，常被称为“网络瑞士军刀”。它可以用于创建、读写 TCP 和 UDP 连接，具有数据传输、端口扫描、网络调试等多种功能。它适用于网络管理、故障排查、甚至轻量级的服务器和客户端开发。

### `netcat` 的主要功能

1. **端口扫描**：快速检测目标主机的开放端口。
   ```bash
   nc -zv <target_ip> <port_range>
   ```
   - **`-z`**：只扫描端口，不发送数据。
   - **`-v`**：输出详细信息。

2. **数据传输**：可以作为一个轻量的服务器和客户端在两台机器之间传输文件或数据。
   - **作为服务器**：
     ```bash
     nc -l -p <port> > received_file
     ```
     - **`-l`**：监听模式，使`netcat`充当服务器。
     - **`-p`**：指定端口。
   - **作为客户端**：
     ```bash
     nc <target_ip> <port> < file_to_send
     ```

3. **远程Shell访问**：创建一个简单的反向Shell，适用于远程管理和测试。
   - **在受控端监听**：
     ```bash
     nc -l -p <port> -e /bin/bash
     ```
   - **在控制端连接**：
     ```bash
     nc <target_ip> <port>
     ```

4. **聊天**：可以利用 `netcat` 在两台机器之间建立简单的聊天会话。

5. **网络调试**：可以用于测试TCP/UDP连接、发送或接收数据包。

### 常用示例
1. **连接到远程主机的特定端口**：
   ```bash
   nc <target_ip> <port>
   ```
2. **发送消息并断开连接**：
   ```bash
   echo "Hello, World!" | nc <target_ip> <port>
   ```

### 注意事项
`netcat` 是一个强大的工具，但在某些网络环境中可能被限制使用，因为它也常被黑客工具用于恶意活动，如端口扫描和远程访问。

## Name Resolution
`nslookup`（Name Server Lookup）是一个用于查询 DNS（Domain Name System）记录的命令行工具。它可以帮助用户获取域名和 IP 地址之间的映射信息，以及其他与 DNS 相关的记录。`nslookup` 是网络故障排查和调试的常用工具。

### `nslookup` 的主要功能

1. **查询域名的 IP 地址**：通过提供域名，`nslookup` 可以返回相应的 IP 地址。
   ```bash
   nslookup example.com
   ```

2. **反向查找 IP 地址**：可以根据 IP 地址查询对应的域名。
   ```bash
   nslookup <ip_address>
   ```

3. **查看 DNS 记录类型**：可以查询特定类型的 DNS 记录（如 A、AAAA、CNAME、MX、NS 等）。
   ```bash
   nslookup -type=MX example.com
   ```

4. **指定 DNS 服务器**：可以通过指定不同的 DNS 服务器进行查询，以验证 DNS 解析的问题。
   ```bash
   nslookup example.com <dns_server_ip>
   ```

5. **交互模式**：`nslookup` 支持交互模式，可以通过输入 `nslookup` 进入该模式，允许用户多次执行查询而不需要重复输入命令。
   ```bash
   nslookup
   ```

### 注意事项
- `nslookup` 在一些系统上可能已被更现代的工具（如 `dig`）取代，后者提供更详细的信息和更灵活的查询选项。
- 在某些情况下，DNS 记录可能会缓存，因此查询结果可能与实际记录不同。使用 `nslookup` 有助于检查 DNS 解析的实时状态。

## security problem
- DNS hacking

## Analysis reasons
1. 选择最简单的能解决问题的办法，也许是Google
2. 从数据生命循环考虑
   - 一路上从硬件，软件，收集信息（控制变量法）
   - 弄清楚scope是在client还是路由还是云服务
   - 可视化每一个可能发生问题的layer
3. 再现问题然后解决问题

- 数据一路上网络未达的可能原因：
  - DNS域名解析过期
  - 客户端问题400code：netstat，traceroute
    - 客户端的主机效能问题需要针对server的troubleshooting
    - 浏览器问题
    - 周围的人是否有同样问题 -> 以此判断问题是哪个*scope*的，internal还是external还是cloud问题
  - 在到达App之前，可能在网路上遇到路由问题
  - App端的端口错误问题
  - App端的防火墙阻挡
  - App的某个node没有工作，用云网络工具诊断或者直接访问相关的node host
  - VM带宽不足，需要更高带宽的AMI
  - Code效率低，设计问题
  - 数据库Performance低下
  - 突然user增多，而缺少scalability对应方式
  - 安全问题：DDos攻击，钓鱼网站
