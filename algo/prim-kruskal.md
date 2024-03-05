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

求最小生成树的所有的边的解题：初始化不同的结果上面是res这里是最小生成树mst列表。while循环的条件也不同。

```python
import heapq

# Given a list of edges of a connected undirected graph,
# with nodes numbered from 1 to n,
# return a list edges making up the minimum spanning tree.
def minimumSpanningTree(edges, n):
    adj = {}
    for i in range(1, n + 1):
        adj[i] = []
    for n1, n2, weight in edges:
        adj[n1].append([n2, weight])
        adj[n2].append([n1, weight])

    # Initialize the heap by choosing a single node
    # (in this case 1) and pushing all its neighbors.
    minHeap = []
    for neighbor, weight in adj[1]:
        heapq.heappush(minHeap, [weight, 1, neighbor])

    mst = []
    visit = set()
    visit.add(1)
    while len(visit) < n:
        weight, n1, n2 = heapq.heappop(minHeap)
        if n2 in visit:
            continue

        mst.append([n1, n2])
        visit.add(n2)
        for neighbor, weight in adj[n2]:
            if neighbor not in visit:
                heapq.heappush(minHeap, [weight, n2, neighbor])
    return mst
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

### 二者区别

Prim算法（普里姆算法）和Kruskal算法（克鲁斯卡尔算法）都是用于解决最小生成树（Minimum Spanning Tree，MST）问题的经典算法，它们之间的主要区别在于其工作方式和实现细节。

1. **基本原理**：
   - Prim算法基于节点来构建最小生成树。它从一个初始节点开始，然后逐步添加与当前最小生成树相连的最小权重边所连接的节点，直到覆盖所有的节点。
   - Kruskal算法基于边来构建最小生成树。它将图中的所有边按照权重从小到大进行排序，然后依次考虑这些边，如果加入某条边不会形成环路，则将其加入最小生成树中，直到最小生成树中包含了所有的节点。

2. **工作方式**：
   - Prim算法通常以节点为中心展开搜索，通过维护一个优先队列或者最小堆来选择下一个要加入的节点，并找到连接当前最小生成树与新节点的最小权重边。
   - Kruskal算法则是在整个边集合上迭代，通过并查集等数据结构来判断是否会形成环路，并将合适的边加入最小生成树中。

3. **时间复杂度**：
   - 在密集图中，Prim算法的时间复杂度通常为O(V^2)，其中V是节点的数量。这是因为在优先队列或最小堆的维护上需要花费较多时间。
   - Kruskal算法的时间复杂度通常为O(E log E)，其中E是边的数量，因为需要对边集合进行排序。

4. **适用情况**：
   - 当图是稀疏图（边的数量相对较少）时，Kruskal算法通常更为高效，因为它的时间复杂度与边的数量相关。
   - 当图是稠密图（边的数量相对较多）时，Prim算法通常更为高效，因为它的时间复杂度与节点的数量相关。

总的来说，Prim算法更适用于稠密图，而Kruskal算法更适用于稀疏图。在实际应用中，可以根据具体情况选择合适的算法来解决最小生成树问题。

PS：什么是稠密图和稀疏图。

在图论中，稠密图和稀疏图是两种不同的图的类型，它们主要通过边的数量来区分。

1. **稠密图**：
   - 稠密图是指边的数量接近于节点的数量的图。换句话说，稠密图中的节点之间有较多的边相连。
   - 在稠密图中，边的数量通常接近于节点数量的平方级别。
   - 稠密图的特点是边之间连接比较紧密，图中大部分节点都直接或者间接地相连。

2. **稀疏图**：
   - 稀疏图是指边的数量远小于节点的数量的图。换句话说，稀疏图中的节点之间较少有边相连。
   - 在稀疏图中，边的数量通常接近于节点数量的线性级别或者更低。
   - 稀疏图的特点是节点之间连接相对较少，图中只有少量节点直接或者间接相连。

稠密图和稀疏图的区别主要体现在边的数量上。在实际应用中，对于不同类型的问题，可能会更倾向于使用适合该类型图的算法来解决，以达到更高的效率。


### leetcode逐行解析

- 连接所有点的最小成本[leetcode1584 题目描述](https://leetcode.com/problems/min-cost-to-connect-all-points/description/)

也是一个最小生成树的问题。给一个2d平面的点points的列表。输出最小的权重满足点之间的曼哈顿距离最短。

输入输出如下：

```
Input: points = [[0,0],[2,2],[3,10],[5,2],[7,0]]
Output: 20
```

题解：该问题和典型解法一样，只有在构造邻接表的时候，需要自己算出距离弄清关系。

```python
class Solution(object):
    def minCostConnectPoints(self, points):
        """
        :type points: List[List[int]]
        :rtype: int
        """
        import heapq
        n = len(points)
        adj = {i:[] for i in range(n)}  # i:[cost, node-j]
        for i in range(n):
            x1, y1 = points[i]
            for j in range(i + 1, n):
                x2, y2 = points[j]
                dist = abs(x1 - x2) + abs(y1 - y2)
                adj[i].append([dist, j])
                adj[j].append([dist, i])
            
        visit = set()
        res = 0
        minheap = [[0, 0]]  # cost, point
        
        while minheap and len(visit) < n:
            weight, v = heapq.heappop(minheap)
            if v in visit:
                continue

            visit.add(v)
            res += weight
            for weight, neighbor in adj[v]:
                if neighbor not in visit:
                    heapq.heappush(minheap, [weight, neighbor])
        return res
```

感想：在图中的算法，一定要多注意点和边等要素，总是容易搞混。

- 在最小生成树中查找关键边和伪关键边[leetcode1489 题目链接](https://leetcode.com/problems/find-critical-and-pseudo-critical-edges-in-minimum-spanning-tree/description/)

Kruskal算法的应用。是一道hard难度的题。

输入输出如下：

```
Input: n = 5, edges = [[0,1,1],[1,2,1],[2,3,2],[0,3,2],[0,4,3],[3,4,3],[1,4,6]]
Output: [[0,1],[2,3,4,5]]
```

首先需要并查集数据结构。然后对题解进行处理。处理过程中将edges列表加入原始序列index以便追踪，在遍历循环中，查找生成的树的最小权重，通过最小权重判断。

什么是关键边？就是少了这个边就无法生成最小树或者权重大于原本的最小权重。什么是伪关键边？是除了关键边之外，如果生成的最小树依然有同样的最小权重，那么就是伪关键边。

代码如下：

```python
# 并查集数据结构implement
class UnionFind:
    def __init__(self, n):
        self.par = [i for i in range(n)]
        self.rank = [1] * n

    def find(self, v1):
        while v1 != self.par[v1]:
            self.par[v1] = self.par[self.par[v1]]
            v1 = self.par[v1]
        return v1

    def union(self, v1, v2):
        p1, p2 = self.find(v1), self.find(v2)
        if p1 == p2:
            return False
        if self.rank[p1] > self.rank[p2]:
            self.par[p2] = p1
            self.rank[p1] += self.rank[p2]
        else:
            self.par[p1] = p2
            self.rank[p2] += self.rank[p1]
        return True

class Solution:
    def findCriticalAndPseudoCriticalEdges(self, n: int, edges: List[List[int]]) -> List[List[int]]:
        # Time: O(E^2) - UF operations are assumed to be approx O(1)
        for i, e in enumerate(edges):
            # 将原始index加入列表，以便追踪
            e.append(i) # [v1, v2, weight, original_index]

        # 相当于一个最小堆，以便从最小权重开始添加进树
        edges.sort(key=lambda e: e[2])

        # 初始化最小生成树的权重为0，并初始化一个并查集uf
        mst_weight = 0
        uf = UnionFind(n)
        # 通过遍历找到最小权重，用于之后的判断
        for v1, v2, w, i in edges:
            if uf.union(v1, v2):
                mst_weight += w
        # 初始化关键边和伪关键边列表
        critical, pseudo = [], []

        # 开始遍历判断，可以说是一种暴力循环
        for n1, n2, e_weight, i in edges:
            # 判断不带当前边的情况
            weight = 0
            uf = UnionFind(n)
            for v1, v2, w, j in edges:
                # 这里通过条件ij不想等，排出当前的边
                if i != j and uf.union(v1, v2):
                    weight += w
            # 如果根本生成不了一个树或者权重大于最小权重了那么说明该边是关键边
            if max(uf.rank) != n or weight > mst_weight:
                critical.append(i)
                # 跳过当前循环，判断下一个边
                continue
            
            # 判断带当前边的情况
            # 使用当前的边初始化并查集和权重
            uf = UnionFind(n)
            uf.union(n1, n2)
            weight = e_weight
            # 开始计算权重和合并树
            for v1, v2, w, j in edges:
                if uf.union(v1, v2):
                    weight += w
            # 如果得到的权重和最小权重一致那么就是伪关键边，不会和关键边重合，因为是先判断的关键边，在上面已经continue了
            if weight == mst_weight:
                pseudo.append(i)
        # 返回结果即可
        return [critical, pseudo]
```
之后就可以尝试将代码去掉自己来写。

注意：这只是所有解法中的一种，以理解为主，可以找到更多的练习和情况进行加深理解，和自身的能力泛化。
