## 图算法：Prim&Kruskal-最小生成树算法

---
### 概念引导

Prim算法是一种用于求解最小生成树（Minimum Spanning Tree）的贪心算法。最小生成树是一个无环的连通子图，它包含了图中的所有顶点，但只包含足够的边，以确保图是连通的，且没有形成环路。

Prim算法的基本思想是从一个起始顶点开始，逐步选择与当前生成树相邻的边中权重最小的边，将其加入到生成树中，然后将新加入的顶点标记为已访问。这个过程一直持续，直到生成树包含了图中的所有顶点。

以下是Prim算法的一般步骤：

1. **选择起始顶点：** 从图中选择一个起始顶点作为生成树的根节点。

2. **初始化：** 将所有顶点标记为未访问，将与起始顶点相邻的边的权重和顶点加入到候选集合中。

3. **循环：** 在候选集合中选择权重最小的边（即最小权重的边），将连接的顶点标记为已访问，将这条边加入到生成树中。同时更新候选集合，将新加入的顶点的所有相邻边加入候选集合。

4. **重复：** 重复步骤3，直到生成树包含了图中的所有顶点。

Prim算法的时间复杂度取决于实现方式，可以通过使用最小堆（priority queue）来加速每次找到最小权重边的过程，使得算法的复杂度可以达到 O(E log V)，其中 E 是边的数量，V 是顶点的数量。

总体而言，Prim算法是一种简单而有效的求解最小生成树问题的算法，特别适用于稠密图。

代码示例：求得到的最小生成树的总权重的题解。

```python
import heapq

def minimumSpanningTree(edges, n):
    adj = {}
    for i in range(n):
        adj[i] = []
    for n1, n2, weight in edges:
        adj[n1].append([n2, weight])
        adj[n2].append([n1, weight])

    visit = set()
    res = 0 # total weight of the tree
    minheap = [[0, 0]]

    while minheap and len(visit) < n:
        weight, v = heapq.heappop(minheap)
        if v in visit:
            continue

        visit.add(v)
        res += weight
        for neighbor, weight in adj[v]:
            if neighbor not in visit:
                heapq.heappush(minheap, [weight, neighbor])
    return res if len(visit) == n else -1
```

Kruskal算法是一种用于解决最小生成树（Minimum Spanning Tree，MST）问题的贪婪算法。最小生成树是连接图中所有节点，并且总权重最小的树。

以下是Kruskal算法的基本步骤：

1. **初始化：** 将图中的所有边按照权重从小到大进行排序。
   
2. **创建空的最小生成树集合：** 开始时，最小生成树集合为空。

3. **逐步选择边：** 从排序后的边列表中依次选择边，如果选择的边不会形成环路（即加入这条边后不会使得最小生成树集合中的节点之间形成环），则将该边加入最小生成树集合。

4. **更新并查集：** 为了检测环路，通常使用并查集数据结构。在每次选择边后，需要更新并查集，将相关的节点合并。

5. **重复步骤3和4：** 重复以上步骤，直到最小生成树集合包含了所有节点。

Kruskal算法的主要思想是通过不断选择权重最小的边，逐步构建最小生成树，同时保证不形成环路。由于边是按权重排序的，所以贪婪地选择最小的边不会导致环路的形成。这使得Kruskal算法具有很好的性能，并且易于实现。

需要注意的是，Kruskal算法适用于无向图。如果是有向图，需要使用其他算法，如Prim算法。

```python
# 首先需要用到python的堆库进行更方便的取最小边操作
import heapq 
# 并查集的实现，第一可以检查是否可以组成一个无循环的树，并且需要不断更新
class UnionFind:
    def __init__(self, n):
        self.par = {}
        self.rank = {}

        for i in range(1, n + 1):
            self.par[i] = i
            self.rank[i] = 0
    
    # 辅助函数，找到节点的父节点
    def find(self, n):
        p = self.par[n]
        while p != self.par[p]:
            self.par[p] = self.par[self.par[p]]
            p = self.par[p]
        return p

    # 通过rank进行树的合并
    # 将返回False如果已经被结合了，这说明是循环，在最小生成树的实现上作为是否将边加入结果的，条件判断
    def union(self, n1, n2):
        p1, p2 = self.find(n1), self.find(n2)
        if p1 == p2:
            return False
        
        if self.rank[p1] > self.rank[p2]:
            self.par[p2] = p1
        elif self.rank[p1] < self.rank[p2]:
            self.par[p1] = p2
        else:
            self.par[p1] = p2
            self.rank[p2] += 1
        return True

# Given an list of edges of a connected undirected graph,
# with nodes numbered from 1 to n,
# return a list edges making up the minimum spanning tree.
def minimumSpanningTree(edges, n):
    # 初始化一个
    minHeap = []
    # 初始就将所有的节点和权重加入，最小堆，之后使用。
    for n1, n2, weight in edges:
        heapq.heappush(minHeap, [weight, n1, n2])
    # 初始化并查集
    unionFind = UnionFind(n)
    # 初始化结果的最小生成树为一个空的列表
    mst = []
    # 当结果列表，尚未拥有所有节点的情况下，不断循环添加
    while len(mst) < n - 1:
        # 从最小堆中取出最小的节点和相连权边
        weight, n1, n2 = heapq.heappop(minHeap)
        # 判断是否能将两个节点结合进并查集
        if not unionFind.union(n1, n2):
            # 如果结合后，会循环，则跳过该次循环
            continue
        # 如果不会变成循环则将其加入结果
        mst.append([n1, n2])
    # 返回结果
    return mst
```

### leetcode逐行解析

- 连接所有点的最小成本[leetcode1584 题目描述](https://leetcode.com/problems/min-cost-to-connect-all-points/description/)

- 在最小生成树中查找关键边和伪关键边[leetcode1489 题目链接](https://leetcode.com/problems/find-critical-and-pseudo-critical-edges-in-minimum-spanning-tree/description/)

Kruskal算法的应用。是一道hard难度的题。
