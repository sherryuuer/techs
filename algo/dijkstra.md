## 最短路径算法：迪杰斯特拉Dijkstra算法

---

### 概念

学习过dfs深度优先搜索后，要知道dfs的局限是，它的图中是没有权重的，dijkstra算法就是对标没有权重的dfs，他是一个有权重的图算法，但是它自己也有局限性，那就是不能有负权重。

Dijkstra算法是一种用于在加权图中找到单源最短路径的贪婪算法。其终极目标是找到起始点到*所有*节点的最短路径。

---
它由荷兰计算机科学家Edsger Dijkstra在1956年提出，被广泛应用于**网络路由**和其他领域。该算法通过逐步扩展一个集合来维护已知的最短路径，直到到达目标顶点。

以下是Dijkstra算法的详细步骤：

1. **初始化：** 创建两个集合，一个用于存放已经找到最短路径的顶点（称为已知集合），一个用于存放待处理的顶点（称为未知集合）。同时，初始化距离数组，用于记录从源点到各顶点的最短路径长度。将源点的距离设为0，其他顶点的距离设为无穷大。

2. **选择源点：** 将源点加入已知集合，并将源点到自身的距离设为0。

3. **更新距离：** 对于与源点直接相邻的顶点，更新它们到源点的距离。如果通过当前源点到达这些顶点的路径比它们当前的最短路径更短，就更新它们的距离。

4. **选择最短路径：** 从未知集合中选择距离最短的顶点，将其加入已知集合。这一步确保每次选择的都是当前已知集合外距离最短的顶点。

5. **更新距离：** 重复步骤3，对于新加入已知集合的顶点，更新它们通过新路径到源点的距离。

6. **重复步骤4和5：** 重复选择最短路径和更新距离的步骤，直到所有顶点都加入已知集合为止。

7. **得到最短路径：** 当所有顶点都加入已知集合后，最短路径的信息已经被计算出来。通过距离数组，可以获取从源点到每个顶点的最短路径长度。
---
Dijkstra算法的关键在于每一步选择当前未知集合中距离最短的顶点，确保每次加入已知集合的顶点都是当前最短路径的一部分。算法的时间复杂度通常为O(V^2)或O(E + V log V)，其中V是顶点数，E是边数。后者可以通过使用**最小堆**来优化到O(E + V log V)。

### 算法实现

最关键的部分是设置了一个最小堆，每次都保证从堆中弹出的是最短路径。

```python

import heapq

# Given a connected graph represented by a list of edges, where
# edge[0] = src, edge[1] = dst, and edge[2] = weight,
# find the shortest path from src to every other node in the 
# graph. There are n nodes in the graph.
# O(E * logV), O(E * logE) is also correct.
def shortestPath(edges, n, src):
    adj = {}
    for i in range(1, n + 1):
        adj[i] = []

    for s, d, w in edges:
        adj[s].append((d, w))
    
    shortest = {}
    minheap = [[0, src]]
    while minheap:
        w1, n1 = heapq.heappop(minheap)
        if n1 in shortest:
            continue
        shortest[n1] = w1

        for n2, w2 in adj[n1]:
            # 或者可以是shortest，这里的条件是为了不让程序陷入无限循环。
            if n2 not in minheap:
                heapq.heappush(minheap, [w1 + w2, n2])
    return shortest
```

**注意点**：有向图的情况adj只需要一个方向，无向图adj需要双向添加。并且权边一定是正数。

### leetcodes

- 网络延迟时间[leetcode743 题目描述](https://leetcode.com/problems/network-delay-time/description/)

提示给我出一组网络节点1到n，同时给出一组边的权重，并且`times[i] = [ui, vi, wi]`，列表中的三个元素，分别是起点，终点，和权重。给出的k是起始点。通过以下给出的input，返回最长路径的权重。

这里求最长路径，是因为题设假定从k点发出了一个信号，希望所有的node节点都收到这个信号，如果不能让所有的节点都收到信号，那么返回-1。

input：

```
Input: times = [[2,1,1],[2,3,1],[3,4,1]], n = 4, k = 2
Output: 2
```

这是一个典型的最短路径问题，而且输入还和上面该问题的一样所以直接使用原本的代码，并对输出进行小的更改即可。

```python
class Solution(object):
    def networkDelayTime(self, times, n, k):
        """
        :type times: List[List[int]]
        :type n: int
        :type k: int
        :rtype: int
        """
        from heapq import heappop, heappush
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

- 在上升的水中游泳[leetcode778 题目描述](https://leetcode.com/problems/swim-in-rising-water/description/)

- 最大概率路径[leetcode1514 题目描述](https://leetcode.com/problems/path-with-maximum-probability/description/)
