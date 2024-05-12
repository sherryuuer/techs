## 树算法：并查集（union-find or disjoint-set-union）

### 什么是并查集

初始状态，每个树都是一个节点，他们是一群节点，也是一个森林。

两个树结构可以不是组合在一起的。相当于是一个树的森林，他们一开始可以是没有组合在一起的。每一棵树只存储们的父级节点，可以考虑，将他们的父节点看成他们自己。

树的结合，是通过比较他们的 rank 也就是 height 来实现的，**将更小的树结合到更大的树上**。

主要用于处理一些不相交集合的合并与查询问题。它提供了两个主要操作：查找（Find）和合并（Union）。这种数据结构常常被用于解决一些与集合划分相关的问题，例如连通性问题。还用于查找是否有循环。

以下是并查集的一些常见应用：

1. **图的连通性问题：** 并查集可用于判断图中的节点是否连通，即是否存在一条路径连接两个节点。在处理图的最小生成树算法中，如 Kruskal 算法，也会使用并查集来判断两个节点是否在同一连通分量中。

2. **网络连接问题：** 在网络设计和维护中，可以使用并查集来管理设备之间的连接关系。这对于网络中的设备故障排除和资源管理都是有用的。

3. **朋友关系问题：** 在社交网络中，可以使用并查集来管理朋友关系。当两个人成为朋友时，可以通过合并两个人所在的集合来表示他们之间的关系。

4. **区域合并问题：** 在图像处理、地理信息系统等领域，有时需要合并或划分区域。并查集可以用于有效地合并或查询区域。

5. **任务调度问题：** 在一些任务调度场景中，可能需要合并或拆分任务集合，以便有效地管理和调度任务。并查集可以用于处理这类问题。

6. **动态连接性问题：** 并查集可用于维护动态连接性，即在一系列的连接和断开操作中判断两个元素是否属于同一集合。

并查集的优势在于它具有较高的效率，特别是在处理大规模的数据集时。其基本操作的时间复杂度近似为常数，使其成为处理一些实际问题的有效工具。

并查集代码实现：

```python
class UnionFind:
    def __init__(self, n):
        self.par = {}
        self.rank = {}

        for i in range(1, n + 1):
            self.par[i] = i
            self.rank[i] = 0

    def find(self, n):
        # to find root of n
        p = self.par[n]
        while p != self.par[p]:
            self.par[p] = self.par[self.par[p]]
            p = self.par[p]
        return p

    def union(self, n1, n2):
        p1, p2 = self.find(n1), self.find(n2)
        if p1 == p2:
            return False

        if self.rank[p1] > self.rank[p2]:
            self.par[p2] = p1
        elif self.rank[p2] > self.rank[p1]:
            self.par[p1] = p2
        else:
            self.par[p1] = p2
            self.rank[p2] += 1

        return True
```

### leetcodes

- 冗余连接leetcode684[题目描述](https://leetcode.com/problems/redundant-connection/description/)

题意给出一个图，无向，通过节点和边连接，但是图中有一条多余的边，去掉这个边之后，图就会成为一个树，实质上，图中有一个循环，目的就是找到这个边。

输入输出如下：如果有多个解，返回最后一个出现的边。

```python
Input: edges = [[1,2],[1,3],[2,3]]
Output: [2,3]
```

解答方案就是使用并查集，解决方法就在union方法中，在该方法中，如果这个边的节点已经无法进行结合了也就是False，那么这就是解答。总之实质上就是找到会造成循环的那个边，或者说，就是找到已经连接好了的那个edge。

```python
class Solution(object):
    def findRedundantConnection(self, edges):
        par = [i for i in range(len(edges) + 1)]
        rank = [1] * (len(edges) + 1)

        def find(n):
            p = par[n]
            while p != par[p]:
                par[p] = par[par[p]]
                p = par[p]
            return p
        
        def union(n1, n2):
            p1, p2 = find(n1), find(n2)
            if p1 == p2:
                return False
            
            if rank[p1] > rank[p2]:
                par[p2] = p1
                rank[p1] += rank[p2]
            else:
                par[p1] = p2
                rank[p2] += rank[p1]
            return True
        
        for n1, n2 in edges:
            if not union(n1, n2):
                return [n1, n2]
```

- 账户合并[leetcode721 题目描述](https://leetcode.com/problems/accounts-merge/description/)

顾名思义就是，一个人的账户列表，可能一个人有好几个账户列表，要判别是同一个人，只要列表的第二个开始的邮件地址有重复的，就可以判别为，是同一个用户。

同一个人一定同名，但是同名的不一定是同一个用户。能判别的只有他们的邮件地址有相同。为每个人建立并查集。

- [leetcode128 题目描述](https://leetcode.com/problems/longest-consecutive-sequence/description/)

haven't done yet
