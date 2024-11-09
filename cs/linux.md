## file system
- /usr/share/doc 中有各种系统说明文件
- 文件系统file system也是一种数据结构，tree！
- lsblk：命令会列出所有disk
- fdisk：文件系统分区
- mkfs：创建文件系统
- Linux中一切都是文件
- 分区 -> 格式化 -> 挂载 -> 配置自动挂载（可选） -> 访问和使用
- RAID技术简直就是分布式系统的先驱
- 进程管理使用的schedular，对任务进行管理，有Queue进行任务FIFO管理
- find查找文件名，grep查找文件内容，比如日志
- awk命令：是一种文本处理工具，适合处理结构化文本（如表格数据、日志文件等），以逐行处理的方式对数据进行筛选、格式化、统计等操作，pattern match，来自一种编程语言
- sed命令：流编辑器，适用于文本的搜索、替换、插入、删除等操作，常用于批量修改文件内容，比如用正则表达替换
- stdin: standard input
- stdout: standard output > output >> add
- stderr: standard error 2> errorfile.txt > dev/null
- | pipeline is good!-> xargs,tee

## kernel
- 目的是系统初始化，进程管理，内存和硬件管理
- Device driver用于驱动硬件
- uname命令
- layers：Devices -> FileSystem -> Memory -> Process -> SystemCallInterface
- 这些组件以Kernel Modules的形式存在

## boot process
- Boot loader: a small program in ROM, load kernel from storage device （GRUB2）
- 硬件 -> 软件 -> 网络
- 操作系统 -> /boot -> init进程 -> 运行级别 -> /etc/init.d -> 用户登陆 -> login shell

## system components
- Localization Options本地化选项：语言，日期，键盘等
- GUI用户图形界面，和远程桌面协议（Secure shell也是）
- SSH Port Forwarding可以安全连接到远程端口
- **systemd**是所有进程的父进程！初始化所有init进程，系统中的process都是services，他们以deamon（守护进程）的形式存在，
- **systemctl**管理这些services的启动等活动

## Network
- hostname也可以作为主机或者node的另一个识别方式
- 设备：switch（data link level），router（internet level）
- 数据的不同叫法：物理层为bit，链路层为frame，网络层为packet
- DHCP动态主机配置协议是个好东西，方便，UDP协议
- ABC私有网络：10./172.16/192.168
- subnet是一种逻辑网络分区
- LDAP是集中式用户权限控制协议，身份和访问控制
- Syslog集中式日志服务
- Monit是一个监控组件
- 可以设置为LoadBalancer服务
- 网络设置：
  - `hostnamectl set-hostname`
  - `hostname`
  - `nmcli`
  - `ifconfig` check ip address MAC address etc
  - `ip adder show` show all ip address
- DNS的nameserver存储在 /etc/resolv.conf 中

## package and software
- RPM: redhat package, CentOS, redhat, `yum`, `dnf`
- dpkg: debian package, kaliLinux, debian, `apt`
- Linux repo
- wget, curl, tar, gzip, zip

## security
- CIA：机密性，完整性，可用性
- Authentication：pin，password，passphrase
- Token：一次性验证码
- LDAP：TCP-IP based directory service protocol
- lowest permissions
- shred：wipe a storage device
- best practices：
  - enable SSL/TLS -> protocol: OpenSSL
  - configure SSH, ssh-keygen
  - change service defaults: SSH/HTTPS
- PAM：用户认证框架，auth，account，session，password
- VPN: IPSec protocol, 分为transport-mode和tunnel-mode，后者多用于site2siteVPN
- SELinux/AppArmor

## Task Automation
- bash scripting：crontab，at（一次性）

## trouble shooting

### 系统状态相关
- **查看系统负载**
  - `top`：实时显示系统的进程和资源使用情况。
  - `htop`：类似于`top`，但界面更友好（需要单独安装）。
  - `uptime`：查看系统运行时间和平均负载。
  - `proc/cpuinfo`：查看CPU信息和逻辑cores。

- **查看内存使用情况**
  - `free -h`：以人类可读格式显示内存使用情况。
  - `vmstat`：查看内存、CPU，process，i/o等资源的详细信息。vm是指virtual memory。
  - `proc/memoinfo`：有各种fields指示了memo各种信息。

- **查看磁盘使用情况**
  - `df -h`：以人类可读格式显示文件系统磁盘使用情况。
  - `du -sh <目录>`：查看指定目录的大小。
  - `iostat`：显示CPU和磁盘I/O统计信息（需要安装`sysstat`）。
  - `iotop`：实时监控各进程的磁盘I/O使用情况。


### 网络相关
- **查看网络连接和端口状态**
  - `netstat -tuln`：查看当前的网络连接和监听的端口。
  - `ss -tuln`：类似`netstat`，但速度更快。
  - `lsof -i`：显示使用特定端口的进程。

- **诊断网络连接**
  - `ping <主机名/IP>`：检查主机是否在线。
  - `traceroute <主机名/IP>`：追踪数据包到目标主机的路径。
  - `curl -I <URL>`：查看URL的HTTP响应头，验证服务是否可访问。

- **查看网络接口状态**
  - `ifconfig` 或 `ip addr show`：查看网络接口信息。
  - `ethtool <网络接口>`：查看或修改网络接口的配置（需要root权限）。
  - `ip link show`：显示网络链路状态。

- **检查防火墙**
  - `firewalld`：查看防火墙状态（CentOS/RHEL）。
  - `ufw status`：查看防火墙状态（Ubuntu）。
  - `iptables -L`：查看防火墙规则。

### 进程和服务相关
- **查看进程状态**
  - `ps aux`：显示所有进程的详细信息。
  - `ps -ef`：另一个格式显示所有进程信息。
  - `pstree`：以树状结构显示进程关系。

- **查找进程**
  - `pgrep <进程名>`：根据名称查找进程ID。
  - `pidof <进程名>`：返回进程ID。
  - `lsof -p <PID>`：查看某个进程打开的文件。
  - `nice / renice`：修改进程的优先级。
  - `kill`：关闭进程。

- **管理服务**
  - `systemctl status <服务名>`：查看服务状态（适用于systemd）。
  - `systemctl restart <服务名>`：重启服务。
  - `service <服务名> status`：查看服务状态（适用于SysV）。

### 日志文件相关
- **查看系统日志**
  - `journalctl`：查看systemd日志。
  - `tail -f /var/log/syslog` 或 `tail -f /var/log/messages`：实时查看系统日志。
  - `dmesg`：查看系统引导和内核消息日志。

- **查看特定服务日志**
  - `tail -f /var/log/<服务名>.log`：实时查看某服务的日志文件。

### 硬件信息
- **查看硬件信息**
  - `lscpu`：显示CPU信息。
  - `lsblk`：显示块设备信息（磁盘和分区）。
  - `lshw`：显示详细的硬件信息（需要root权限）。

- **查看硬盘健康状态**
  - `smartctl -a /dev/sdX`：查看硬盘SMART信息（需要安装`smartmontools`）。

### 性能分析工具
- **资源使用分析**
  - `sar`：收集、报告和保存系统活动信息（需要安装`sysstat`）。
  - `sysctl`：runtime收集CPU的参数。
  - `strace <command>`：追踪一个命令的系统调用，用于调试。
  - `perf top`：查看系统实时性能（CPU使用等）。

### 常用故障排查组合
- **CPU内存使用高时**
  - `top`：查看具体哪个进程占用CPU高。
  - `ps -eo pid,ppid,cmd,%mem,%cpu --sort=-%cpu | head`：按CPU使用率排序的进程列表。
  - `free -h`：确认内存总量和使用量。
  - `ps aux --sort=-%mem | head`：按内存使用率排序的进程列表。
  - /proc/cpuinfo -> uptime -> sar -> /proc/memoinfo -> free -> vmstat

- **磁盘I/O性能问题**
  - `iotop`：查看I/O使用量高的进程。
  - `iostat -x`：查看详细的I/O统计。

- **网络连接问题**
  - `nmcli` network manager cli
  - `ifconfig`, `ip adder` -> NIC check
  - `nslookup`, `dig`, `host` -> DNS name
  - `whois` -> get domain name info
  - `netstat` check latency
  - `ping` 检查网络连通性。
  - `traceroute` report network path between source and destination for every `hop`/jump point，很多时候原因可能是路由表配置错误
  - `netstat`、`ss`(socket state)：gether info about TCP connections on the system
  - `nmap` scan network
  - `wireshark`, `tcpdump` packet sniffer
  - `netcat`, `nc` scan network, connection checking
  - `iperf`, `iftop` check the maximum throughput of an interface! some time it is NIC problem
