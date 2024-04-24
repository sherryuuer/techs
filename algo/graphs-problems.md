## 图算法的相关问题

### 图的解释

Graph（图）是图论中的一个重要概念，它是由一组节点（或顶点）和连接这些节点的边组成的数据结构。从不同视角解释图可以理解为：

1. **数学视角**：
   - 在数学中，图是由一组顶点（节点）和一组连接这些顶点的边（边）组成的抽象数据结构。顶点表示图中的实体或对象，边表示顶点之间的关系或连接。
   - 图可以是有向的（有向图）或无向的（无向图），即边可以有方向或者没有方向。
   - 图可以是带权重的（加权图），即边上带有权重或者值。

2. **计算机科学视角**：
   - 在计算机科学中，图是一种常见的数据结构，用于表示网络、路径、关系等抽象概念。
   - 图可以用邻接矩阵或邻接表等方式来表示和存储。邻接矩阵是一个二维数组，用于表示图中节点之间的连接关系；邻接表是一种链表的数组，用于表示每个节点的相邻节点列表。
   - 图可以用于解决各种问题，如路径搜索、网络分析、社交网络分析等。

3. **实际应用视角**：
   - 在现实世界中，图被广泛应用于各种领域，如社交网络分析、交通网络规划、电路设计、组织结构分析等。
   - 社交网络可以被建模为图，其中用户是节点，用户之间的关系（如好友关系）是边。通过分析图的结构和属性，可以揭示社交网络中的模式、趋势和影响力。
   - 交通网络也可以被建模为图，其中交通节点是顶点，道路或路径是边。通过分析交通图，可以优化路线规划、减少拥堵和提高效率。

图是一种灵活的数据结构，可以用于表示和解决各种问题，在数学、计算机科学和实际应用中都有重要的地位和作用。

### 问题1:Network Delay Time

力扣743题，是一道中等难度问题。提示给出一组网络节点1到n，同时给出一组边的权重，并且`times[i] = [ui, vi, wi]`，列表中的三个元素，分别是起点，终点，和权重。给出的k是起始点。通过以下给出的input，返回最长路径的权重。

这里求最长路径，是因为题设假定从k点发出了一个信号，希望所有n个node节点都收到这个信号，如果不能让所有的节点都收到信号，那么返回-1。比如Input: times = [[2,1,1],[2,3,1],[3,4,1]], n = 4, k = 2，得到Output: 2。

这道题要用到的是Dijkstra算法求最短路径。

具体步骤：

- 构建邻接表，以起点为key，终点和权重的元祖为value。
- 初始化存放结果的最短路径哈希表和一个最小堆：结果表存放的是，到达某节点需要的最短路径，最小堆已存放出发点的权重和节点组成的元祖。
- 如果最小堆不为空，则从最小堆中弹出最短路径加入结果。同时对于该节点，将它的所有，不在结果列表中的邻接节点加入最小堆以备下次使用。
- 检查最终结果表，如果程度为n则最终返回结果表中的最大路径权重。反之返回-1，因为这说明有些节点无法到达。

代码：
```python
from heapq import heappop, heappush
def network_delay_time(times, n, k):
    adj = {}
    for i in range(1, n + 1):
        adj[i] = []
    for u, v, w in times:
        adj[u].append((v, w))

    shortest = {}
    minheap = [[0, k]]
    while minheap:
        w1, n1 = heappop(minheap)
        if n1 in shortest:
            continue
        shortest[n1] = w1

        for n2, w2 in adj[n1]:
            if n2 not in shortest:
                heappush(minheap, [w1 + w2, n2])
    return max(shortest.values()) if len(shortest) == n else -1
```

题解答案，大同小异：

```python
from queue import PriorityQueue
from collections import defaultdict

def network_delay_time(times, n, k):
    adjacency = defaultdict(list)
    for src, dst, t in times:
        adjacency[src].append((dst, t)) 

    pq = PriorityQueue()
    pq.put((0, k)) 
    visited = set() 
    delays = 0   
      
    while not pq.empty():
        time, node = pq.get()
      
        if node in visited:
            continue
          
        visited.add(node)  
        delays = max(delays, time) 
        neighbours = adjacency[node]

        for neighbour in neighbours:
            neighbour_node, neighbour_time = neighbour
            if neighbour_node not in visited:
                new_time =  time + neighbour_time
                pq.put((new_time, neighbour_node))
  
    if len(visited) == n:
        return delays

    return -1
```

学习笔记：时间复杂度为边的数量和进行最小堆插入的时间的乘积：O(elogn)。空间复杂度为O(e + n)。

### 问题2:Paths in Maze That Lead to Same Room

一个迷宫由n个房间组成，部分房间有走廊相连。给定一个二维整数数组corridors，`corridors[i]=[room1, room2]`。它表示第i条走廊，连接着room1和room2。

设计者想要弄清楚这个迷宫的混乱度，confusion_score。混乱度的定义是，迷宫中长度为3的环的数量。比如1→2→3→1是长度为3的环，但是1→2→3→4和1→2→3→2→1就不是。

两个不同的环定义为，只要有一个或者以上不同的房间出现在两个环中。

题解方法：构造邻接表，然后遍历每两个房间，看该房间的相邻房间是否是彼此的相邻房间。因为每个长度为3的环会被遍历3次，所以最后除以3。

代码如下：

```python
from collections import defaultdict
from itertools import combinations


def number_of_paths(n, corridors):
    g = defaultdict(set)
    for a, b in corridors:
        g[a].add(b)
        g[b].add(a)

    res = 0
    for i in range(1, n + 1):
        for m, n in combinations(g[i], 2):
            if m in g[n]:
                res += 1
    return res // 3
```
题解代码：

```python
from collections import defaultdict

def number_of_paths(n, corridors):
    neighbours = defaultdict(set)
    cycles = 0

    for room1, room2 in corridors:
        neighbours[room1].add(room2)
        neighbours[room2].add(room1)
        cycles += len(neighbours[room1].intersection(neighbours[room2]))

    return cycles
```
这个题解很巧妙，两个房间的相交房间的数量，就是他们成环的数量。`cycles += len(neighbours[room1].intersection(neighbours[room2]))`计算了两个房间的相邻房间集合的交集，并将其长度加到混乱度计数器中。这里的交集长度即为两个房间之间的长度为3的环的数量。

学习笔记：时间复杂度为O(mxn)，空间复杂度为O(n^2)。

### 问题3:Bus Routes

力扣815题，hard难度。

给定一个routes数组，其中第 i 辆巴士永远沿着 route[i] 路线行驶。例如 routes[0] = [1, 5, 7]，那么巴士0号的行驶路线永远是1-5-7。同时给定src你的起始站，和终点站dest，找出从起始点到终点，乘坐最少数量的巴士数量。

例如input： routes = [[1,2,7],[3,6,7]], source = 1, target = 6，得到output： 2，因为最好的策略是乘坐第一趟巴士到 7 号巴士站，然后乘坐第二趟巴士到6号巴士站。

解题思路：

- 创建一个邻接表，车站-bus列表。
- 初始化一个queue，包含出发车站和bus数量。
- 遍历queue直到它空了，或者已经到达终点。
- 每次遍历都访问出队的车站，并且对相连的车站进行入队。
- 每次遍历中如果有新的bus经过了该车站，则对count加一。
- 最后返回bus的count。

代码如下：

```python
from collections import deque


def minimum_buses(bus_routes, src, dest):
    stb = {}
    for i in range(len(bus_routes)):
        for station in bus_routes[i]:
            if station not in stb:
                stb[station] = []
            if i not in stb[station]:
                stb[station].append(i)
    visit = set()
    queue = deque([[src, 0],])
    while queue:
        print(queue)
        source, count = queue.popleft()
        if source == dest:
            return count
        if source in stb:
            for bus in stb[source]:
                if bus not in visit:
                    for station in bus_routes[bus]:
                        queue.append([station, count + 1])
                    visit.add(bus)
    return -1


routes = [[1, 2, 7], [3, 6, 7]]
source = 1
target = 6
res = minimum_buses(bus_routes=routes, src=source, dest=target)
print(res)
```

学习笔记：题解参考和我的代码除了变量，基本一样不再赘述了。时间复杂度和空间复杂度应该是O(rxs)也就是巴士和车站的乘积，因为我们进行的是双层遍历。同时使用了一个额外的变量visit用于存储已经乘坐过的bus。
