## 使用Python实现的套接字编程


### 套接字 Sockets

在之前的笔记中讲过**套接字 Sockets**套接字与程序密切相关，是程序和计算机网络的接口。在网络编程中，程序可以使用套接字来建立连接、发送和接收数据以及错误对应，并处理与网络通信相关的各种任务。开发者可以使用编程语言提供的套接字库（如Python中的socket模块）来创建套接字，并在程序中实现网络通信功能。它是一种软件接口，和硬件无关。

想象一下你在和朋友通话的情景。你和朋友之间需要一种方式来进行交流，就像网络中的计算机之间需要一种方式来进行通信。而套接字就像是你们之间的电话线和电话机一样，它提供了一种通信的接口和机制，使得你们可以相互交流。

在这个比喻中，电话线就是套接字的连接，它连接了两个通信的端点（你和你的朋友、两台计算机）。而电话机就是套接字本身，它负责发送和接收消息（声音或数据）。

当你想要和朋友通话时，你拿起电话机，拨打朋友的电话号码，就像创建了一个套接字并指定了目标地址和端口一样。然后你们可以开始进行通话，你说话，朋友听到，并且朋友也可以回答你的问题或者提出自己的观点。这就像是通过套接字发送和接收数据。

当通话结束后，你挂断电话，就像关闭了套接字连接一样。这样，你们之间的通信就结束了。

在Socket编程中，通常有两种类型的Socket：流式Socket（TCP）和数据报Socket（UDP）。

1. **流式Socket（TCP）**：流式Socket提供了面向连接的、可靠的数据流传输服务。它基于TCP协议，在通信之前需要建立连接，然后可以通过连接进行双向的数据传输。TCP套接字提供了流式数据传输的服务，确保数据的可靠传输和顺序传递。

2. **数据报Socket（UDP）**：数据报Socket提供了无连接的、不可靠的数据传输服务。它基于UDP协议，通信过程中不需要建立连接，而是直接将数据报发送给目标地址。UDP套接字提供了数据报形式的数据传输服务，适用于实时性要求较高、对数据可靠性要求较低的场景。

总之，Socket是一种用于网络通信的编程接口，提供了在不同计算机之间进行数据传输和通信的机制。通过Socket编程，可以轻松实现各种网络应用程序，包括服务器、客户端和网络通信工具。

### Python的Socket编程

一般来说，使用Python的库进行套接字编程，主要有如下步骤：

1. 创建Socket对象：使用`socket.socket()`函数创建一个Socket对象，指定通信使用的协议和地址族（如IPv4或IPv6）。

2. 绑定到地址和端口（服务器端）：如果是服务器程序，需要将Socket绑定到一个地址和端口上，以便客户端可以连接到该地址和端口。

3. 监听连接（服务器端）：开始监听连接请求，等待客户端连接。

4. 接受连接（服务器端）：接受来自客户端的连接请求，并返回一个新的Socket对象和客户端的地址信息。

5. 建立连接（客户端）：客户端使用`connect()`方法与服务器建立连接，指定服务器的地址和端口。

6. 发送和接收数据：一旦建立了连接，服务器和客户端都可以使用`send()`和`recv()`方法来发送和接收数据。

7. 关闭连接：通信结束后，需要调用`close()`方法关闭Socket连接。

总之，Socket编程是一种用于网络通信的编程技术，通过Python的`socket`模块可以方便地实现各种网络应用程序，包括服务器、客户端和网络通信工具。

总之就是想玩一下，顺便记录一下笔记。

### Python套接字编程 step 1: 创建实例和绑定套接字

实现内容：

- 客户端将向服务器发送一行文本。
- 服务器将接收数据并将每个字符转换为大写。
- 服务器将把大写字符发送给客户端。
- 客户端将接收它们并将其显示在其屏幕上。

首先导入库，并创建一个对象：

```python
import socket

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
print(s)

# <socket.socket fd=3, family=AddressFamily.AF_INET, type=SocketKind.SOCK_DGRAM, proto=0, laddr=('0.0.0.0', 0)>
```

输出的内容是： socket.socket(family, type, proto, fileno)

- family 属性用于分配可以和套接字进行通信的地址类型。AF_INET 表示 IPv4，是最常用的地址类型。AF_INET6 则是 IPv6，AF_UNIX 是 Unix 域套接字。
- type 是指连接类型，SOCK_DGRAM 指定应用程序要使用用户数据报协议(UDP)，UDP不太可靠，但不需要建立初始连接。现在的简单尝试就是在 UDP 中构建这些服务器和客户端程序对。SOCK_STREAM指定应用程序要使用传输控制协议(TCP)。虽然 TCP 需要一些初始设置，但它比 UDP 更可靠。

然后绑定套接字（binding the sockets）：

当一个程序绑定到地址时，它告诉操作系统将来自指定地址的网络数据包传递给该程序处理。在网络编程中，程序通过套接字（Socket）来与网络通信。套接字可以通过绑定到一个特定的网络地址和端口来监听网络上的数据流，这样就可以接收来自其他计算机的数据。

绑定到地址意味着指定了套接字应该监听哪个网络接口（网卡）和端口号。网络接口可以是本地回环接口（例如IPv4的 127.0.0.1 或者 IPv6的 ::1），也可以是计算机上的其他网络接口（例如局域网接口或者互联网接口）。

当程序绑定到一个地址时，它会告诉操作系统只有来自该地址的数据包才会被传递给这个程序处理。这对于服务器程序来说特别重要，因为它们需要监听特定的地址和端口，以便客户端可以连接到它们并与之通信。

```python
import socket

# Setting up a socket
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) 
port = 3000
hostname = '127.0.0.1'
s.bind((hostname, port)) # Binding the socket to a port and IP address
print('Listening at {}'.format(s.getsockname())) # Printing the IP address and port of socket

# Listening at ('127.0.0.1', 3000)
```
bind 函数中参数数量根据 family 不同而不同。端口选3000，是为了避开0-1024的预留端口。

hostname的设置：

如果在本地计算机上进行操作，请将其设置为 IPV4 的本地主机 127.0.0.1 地址。127.0.0.1 是 IPv4 地址中的本地回环地址，表示本机。当程序绑定到这个地址时，它会监听来自本机的所有数据包。因此，当服务器程序绑定到 '127.0.0.1' 时，它将只能接受来自本机的连接，而不会接受来自其他计算机的连接。

另外，还可以将其设置为''代表INADDR_ANY，它表示指定程序打算接收发送到指定端口的数据包，该端口可以是该计算机上配置的任何 IP 地址。或者可以将其设置为分配给自己的计算机的任何一个特定 IP 地址。

### Python套接字编程 step 2: 编写一个UDP服务器

首先定义这个小服务器的功能：接收客户端的文字，打印，将文字转换为大写字母，发送回客户端。

以下的代码部分，监听客户端的消息，使用`while True:`进行无限监听。定义了UDP datagram 的最大size，执行的时候，代码会停止，直到收到消息。最后将消息和客户的地址存储在定义好的变量中。

```python
MAX_SIZE_BYTES = 65535 # Mazimum size of a UDP datagram
while True:
    data, clientAddress = s.recvfrom(MAX_SIZE_BYTES) # Receive at most 65535 bytes at once
```

下面几步处理从比特流 byte stream 收到的信息：

- 将信息解码为ASCII字符。
- 然后进行大写字母处理。
- 打印处理好的数据。
- 再对处理好的数据进行编码。
- 将数据送回给客户端。

```python
while True:
    data, clientAddress = s.recvfrom(MAX_SIZE_BYTES)

    message = data.decode('ascii')
    upperCaseMessage = message.upper()
    print('The client at {} says {!r}'.format(clientAddress, message))
    data = upperCaseMessage.encode('ascii')
    s.sendto(data, clientAddress)  
```

Just code:

```python
import socket

MAX_SIZE_BYTES = 65535
# Setting up a socket
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) 
port = 3000
hostname = '127.0.0.1'
s.bind((hostname, port))
print('Listening at {}'.format(s.getsockname()))

while True:
    data, clientAddress = s.recvfrom(MAX_SIZE_BYTES)
    message = data.decode('ascii')
    upperCaseMessage = message.upper()
    print('The client at {} says {!r}'.format(clientAddress, message))
    data = upperCaseMessage.encode('ascii')
    s.sendto(data, clientAddress) 
```

### Python套接字编程 step 3: 编写一个UDP客户端程序

创建一个客户端套接字实例，并且不需要绑定端口，系统会帮我们搞定，这时候上场的就是临时端口，是一种动态的端口。并且可以打印现在绑定到的动态端口的端口号（如下显示的('0.0.0.0', 0)是因为我们还没开始使用，使用后可以再检查）。

临时端口的作用是确保在客户端与服务器进行通信时，不同客户端之间的连接可以同时存在，而不会产生冲突。由于每个连接都有一个唯一的临时端口号，因此可以确保不同的连接之间不会产生混淆。

```python
import socket

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
print('The OS assigned the address {} to me'.format(s.getsockname()))

# The OS assigned the address ('0.0.0.0', 0) to me
```

接下来使用input方法输入我们要发送的信息并编码和发送到，我们之前创建的监听中的服务端。

这时候就可以打印一下临时端口看看了。

```python
import socket

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
message = input('Input lowercase sentence:' )
data = message.encode('ascii')
s.sendto(data, ('127.0.0.1', 3000))
print('The OS assigned the address {} to me'.format(s.getsockname()))
```

定义接收从服务端发回来的信息：（这和之前的服务端的定义类似）

```python
MAX_SIZE_BYTES = 65535 # Mazimum size of a UDP datagram
data, address = s.recvfrom(MAX_SIZE_BYTES) 
```

最后对收到的信息进行解码和打印即可。

```python
text = data.decode('ascii')
print('The server {} replied with {!r}'.format(address, text))
```

Just code:

```python
import socket

MAX_SIZE_BYTES = 65535 # Mazimum size of a UDP datagram

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
message = input('Input lowercase sentence:' )
data = message.encode('ascii')
s.sendto(data, ('127.0.0.1', 3000))
print('The OS assigned the address {} to me'.format(s.getsockname()))
data, address = s.recvfrom(MAX_SIZE_BYTES) 
text = data.decode('ascii')
print('The server {} replied with {!r}'.format(address, text))
```

### Python套接字编程 step 3: 写一个脚本同时可以用来运行服务端或者客户端

使用上面的代码，编写文件`udp.py`。可以执行以下程序用于测试。

- python udp.py client
- python udp.py server

脚本如下：

```python
import argparse, socket

MAX_SIZE_BYTES = 65535 # Mazimum size of a UDP datagram

def server(port):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    hostname = '127.0.0.1'
    s.bind((hostname, port))
    print('Listening at {}'.format(s.getsockname()))
    while True:
        data, clientAddress = s.recvfrom(MAX_SIZE_BYTES)
        message = data.decode('ascii')
        upperCaseMessage = message.upper()
        print('The client at {} says {!r}'.format(clientAddress, message))
        data = upperCaseMessage.encode('ascii')
        s.sendto(data, clientAddress)

def client(port):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    message = input('Input lowercase sentence:' )
    data = message.encode('ascii')
    s.sendto(data, ('127.0.0.1', port))
    print('The OS assigned the address {} to me'.format(s.getsockname()))
    data, address = s.recvfrom(MAX_SIZE_BYTES) 
    text = data.decode('ascii')
    print('The server {} replied with {!r}'.format(address, text))

if __name__ == '__main__':
    funcs = {'client': client, 'server': server}
    parser = argparse.ArgumentParser(description='UDP client and server')
    parser.add_argument('functions', choices=funcs, help='client or server')
    parser.add_argument('-p', metavar='PORT', type=int, default=3000,
                        help='UDP port (default 3000)')
    args = parser.parse_args()
    function = funcs[args.functions]
    function(args.p)
```

服务台1:执行服务器端的代码，监听等待。

```bash
root@user:/# echo Starting Server .... && python3 /usercode/udp.py server
Starting Server ....
Listening at ('127.0.0.1', 3000)
The client at ('127.0.0.1', 46734) says 'sally'
```

服务台2:执行客户端代码，发送信息，然后就立刻返回了消息。

```bash
root@user:/# python3 /usercode/udp.py client
Input lowercase sentence:sally
The OS assigned the address ('0.0.0.0', 46734) to me
The server ('127.0.0.1', 3000) replied with 'SALLY'
```

### Python套接字编程 step 4: 改进客户端随意接受消息的部分

在客户端的代码中`data, address = s.recvfrom(MAX_SIZE_BYTES) `使得客户端可以随意接受来自各个计算机的消息，这是不好的，这一步是为了改进这个问题。

第一种是使用 connect 方法，这种方法限定客户端只和一个ip端口连接。代码如下：

```python
import socket

MAX_SIZE_BYTES = 65535 # Mazimum size of a UDP datagram

def client(port):
    host = '127.0.0.1'
    port = 3000
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect((host, port))
    message = input('Input lowercase sentence:' )
    data = message.encode('ascii')
    s.send(data)
    print('The OS assigned the address {} to me'.format(s.getsockname()))
    data = s.recv(MAX_SIZE_BYTES) 
    text = data.decode('ascii')
    print('The server replied with {!r}'.format(text))
```
它强制连接到服务器的端口和地址。同时关注一下代码发现，通过`connect()`的连接方式，原本的`recvfrom`和`sendto`方法，可以简化为`recv`和`send`。

但是在现实中我们的客户端总是和多个端口进行联系，所以这里是一种传统的一对多的对应方法：

```python
import socket

MAX_SIZE_BYTES = 65535 # Mazimum size of a UDP datagram

def client(port):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    hosts = []
    while True:
        host = input('Input host address:' )
        hosts.append((host,port))
        message = input('Input message to send to server:' )
        data = message.encode('ascii')
        s.sendto(data, (host, port))
        print('The OS assigned the address {} to me'.format(s.getsockname()))
        data, address = s.recvfrom(MAX_SIZE_BYTES)
        text = data.decode('ascii')
        if(address in hosts):
            print('The server {} replied with {!r}'.format(address, text))
            hosts.remove(address)
        else:
            print('message {!r} from unexpected host {}!'.format(text, address))
```

这里创建了一个 hosts 列表，当给一个服务器发送消息的时候，将该服务器信息加入列表，当收到消息后，再移除该服务器。

如果收到的消息不在这个列表中，则表明是不明来源。

### Python套接字编程 step 5: 写一个简单的chat app

使用上面学过的内容，编写一个简单的chat app。同样是脚本的形式。最终使用终端进行对话测试。

OK下面就是我修改出来的非常基础的本地对话小程序了，虽然基础但是在上面可以构架很多东西，有兴趣可以玩玩看。

```python
import argparse
import socket

MAX_SIZE_BYTES = 65535  # Mazimum size of a UDP datagram


def server(port):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    hostname = '127.0.0.1'
    s.bind((hostname, port))
    print('Listening at {}'.format(s.getsockname()))
    while True:
        data, clientAddress = s.recvfrom(MAX_SIZE_BYTES)
        message = data.decode('ascii')
        print(f'The client at {clientAddress} says {message}')
        replay = input('replay:')
        data = replay.encode('ascii')
        s.sendto(data, clientAddress)


def client(port):
    host = '127.0.0.1'
    port = 3000
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect((host, port))
    client_name = input('What is your name? ')
    print('You can start conversition now!')
    while True:
        message = input(f'{client_name}: ')
        data = message.encode('ascii')
        s.send(data)
        data = s.recv(MAX_SIZE_BYTES)
        text = data.decode('ascii')
        print(f'The server replied with {text}')


if __name__ == '__main__':
    funcs = {'client': client, 'server': server}
    parser = argparse.ArgumentParser(description='UDP client and server')
    parser.add_argument('functions', choices=funcs, help='client or server')
    parser.add_argument('-p', metavar='PORT', type=int, default=3000,
                        help='UDP port (default 3000)')
    args = parser.parse_args()
    function = funcs[args.functions]
    function(args.p)
```

### Python套接字编程之TCP

刚刚为止都是UDP的程序，那么TCP如何，直接上代码然后再说要点：

```python
import argparse, socket

def recvall(sock, length):
    data = b''
    while len(data) < length:
        more = sock.recv(length - len(data))
        if not more:
            raise EOFError('was expecting %d bytes but only received'
                           ' %d bytes before the socket closed'
                           % (length, len(data)))
        data += more
    return data

def server(port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(('127.0.0.1', port))
    sock.listen(1)
    print('Listening at', sock.getsockname())
    while True:
        print('Waiting for a new connection')
        sc, sockname = sock.accept()
        print('Connection from', sockname)
        print('  Socket name:', sc.getsockname())
        print('  Socket peer:', sc.getpeername())
        message = recvall(sc, 16)
        print('  message from client:', repr(message))
        sc.sendall(b'Goodbye, client!')
        sc.close()
        print('  Closing socket')

def client(port):
    host = '127.0.0.1'
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((host, port))
    print('Client has been assigned the socket: ', sock.getsockname())
    sock.sendall(b'Greetings, server')
    reply = recvall(sock, 16)
    print('Server: ', repr(reply))
    sock.close()

if __name__ == '__main__':
    choices = {'client': client, 'server': server}
    parser = argparse.ArgumentParser(description='Send and receive over TCP')
    parser.add_argument('role', choices=choices, help='which role to play')
    parser.add_argument('-p', metavar='PORT', type=int, default=3000, help='TCP port (default 1060)')
    args = parser.parse_args()
    function = choices[args.role]
    function(args.p)
```

需要注意的要点：

- TCP connect()调用引发完整的 TCP 三向握手。三向握手可能会失败，connect()调用也可能会失败。
- TCP 架构中为每个新连接创建一个新套接字。
- TCP 服务器上的一个套接字专用于持续侦听新的传入连接。
- 当连接成功时，该监听套接字会专门为该连接创建一个新套接字。当连接终止时，关联的套接字将被删除。
- 每个套接字以及每个连接都由唯一的 4 元组标识：(local_ip, local_port, remote_ip, remote_port)。检查所有传入的 TCP 数据包，以确定它们的源地址和目标地址是否属于任何此类当前连接的套接字。
- 与 UDP 不同，只要发送方和接收方通过路径连接并且双方都处于活动状态，TCP 数据段就会被传送。
- 发送 TCP 实体可能会将 TCP 段分割成数据包，因此接收 TCP 实体必须重新组装它们。这在我们的小程序中不太可能发生，但在现实世界中却经常发生。因此，我们需要注意一次调用后缓冲区中是否有剩余数据需要发送或接收。

关于代码的具体实现就不再赘述了。

以上就是这次玩代码的全过程～！
