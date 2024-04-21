## 图算法的相关问题

### 图的解释


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
