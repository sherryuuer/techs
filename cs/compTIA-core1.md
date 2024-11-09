### 思考方式
- 硬件和软件
- 多方终端：IoT，手机，电脑，云服务器等，*一切皆是电脑*
- 网络
- 虚拟技术和云计算
- TroubleShoting
- input，process，output，storage，想一下这些其实也是所有云component的组成部分

### 硬件！
- 端口，cable，插口，API同一种概念，软件和硬件是可以迁移思考的
- 数据传输电缆USB
- 视频传输HDMI其实也有很多类型，还有DisplayPort，Thunderbolt
- 存储传输，主要就是数据传输吧，Thunderbolt，SATA

- 主板（Motherboard）：input，output，processing，storage，有不同规格的主板，可以插不同的卡
- CPU中64位才够吧，ARM（苹果的就是）可以延长电池寿命哦！并且释放热量更少，除了Windows似乎都是这种芯片

### Networks

- **layers**
- Link layer: local
- Internet layer: where to send
- Transport layer: how (TCP/UDP) to send
- Application layer: what to send

- **Ports and Protocols**
- port is a communication endpoint: inbound and outbound
- SFTP/SSH, all 22 port
- DHCP: 67, 68 port，is UDP connection
- HTTP: 80, basic / HTTPS: 443 port
- POP3: receive mail, 110 port
- *TCP*：ssh，http，https
- *Remote access servers*: Telnet23(不安全，发送的是纯文本), ssh22, RDP(远程桌面)

### TroubleShoting
- **六步故障排除法TroubleShoting**：
  1. *Identify the problem*：查看问题，查看最近针对相关组件发生的变化，进行总结
  2. *Establish a theory of probable cause*：列出显而易见的原因列表然后排查，包括阅读日志，谷歌其他人的经验，以及进行内部外部，软件硬件各个方面的问题原因排查
  3. *Test the theroy to determine the cause*：如何测试原因，控制变量法就很不错
  4. *Establish a plan of action to resolve the problem and implement the solution*：比如更换有问题的组件，重新锁定问题根源，寻找更好的解决方案，设定方式，或者best practice，持续观察结果
  5. *Verify full system functionality*：查看整个系统是否有影响，如果在这个阶段就找到将问题完全解决的方法，长久对应则最好
  6. *Document the findings，actions and outcome*



### knowledge
- 驱动Driver是一种将command转换为针对硬件的操作command的工具，相当于命令翻译。
- b=bits, B=bytes，他们相差8倍，所以Gb和GB不是一个概念，小心被骗
