## 图算法：拓扑排序（Toplogical Sort）

---

### 概念解析

**拓扑**是什么：

拓扑（Topology）是数学的一个分支，研究空间中保持不变的性质，即那些在连续变形下保持不变的性质。拓扑学主要关注空间的形状和结构，而不关心具体的度量和距离。

在拓扑学中，一个空间可以由开集的概念来描述。开集是一个集合，对于集合中的每个点，都存在一个包含该点的开集。通过这种方式，可以描述空间中的开放部分和它们之间的关系，而不需要引入具体的度量或距离概念。

拓扑学的研究对象包括拓扑空间、拓扑映射、同胚等概念。拓扑学的应用领域很广泛，包括物理学、工程学、生物学等各种科学领域。在计算机科学中，拓扑学也被用于网络拓扑、数据结构等方面。

讲真，我完全没看懂这个概念。

**DAG**是什么：

DAG 是指有向无环图（Directed Acyclic Graph）。在图论中，图是由节点（或顶点）和边组成的一种数据结构，有向图表示节点之间的有向关系，无环图表示在图中不存在环路，即不存在从某个节点出发经过若干边最终回到该节点的情况。

DAG 具有以下特点：
1. **有向图（Directed Graph）：** 每条边都有一个方向，从一个节点指向另一个节点。
2. **无环图（Acyclic Graph）：** 不存在从某个节点出发经过若干边最终回到该节点的路径。

DAG 在计算机科学和算法中有广泛的应用，特别是在任务调度、依赖关系表示、编译器优化等领域。例如，软件编译过程中的源代码可以被表示成一个DAG，每个节点表示一个编译单元，边表示编译单元之间的依赖关系。在这种情况下，DAG 可以用于确定编译单元的执行顺序，以提高编译效率。

DAG 也常用于表示计算机网络中的依赖关系，比如任务调度中的任务依赖图。在这样的应用中，DAG 的拓扑排序（Topological Sorting）是一种常用的算法，可以用来确定节点的合理执行顺序。

什么是**拓扑排序**：

拓扑排序是一种对有向无环图（DAG）进行排序的算法，其目的是将图中的节点排成线性序列，使得图中的任意一条有向边都按照从前到后的顺序排列。这个排序反映了图中节点之间的依赖关系，通常用于解决任务调度、编译器优化等问题。

拓扑排序的算法主要包含以下步骤：

1. **选择入度为0的节点：** 从图中选择一个入度为0的节点，入度表示指向该节点的边的数量。这样的节点是图中没有依赖关系的节点，可以作为排序的起点。

2. **输出该节点并删除其所有出边：** 输出当前选择的节点，并将它的所有出边删除。这相当于移除当前节点及其相关的依赖关系。

3. **更新相邻节点的入度：** 更新当前节点的所有邻接节点的入度，即将与当前节点相邻的节点的入度减1。

4. **重复步骤1-3：** 重复执行上述步骤，直到所有节点都被输出。如果图中存在环路，那么图无法进行拓扑排序。

拓扑排序的输出结果是一个满足依赖关系的节点序列。如果图中存在环路，那么不可能完成拓扑排序，因为环路表示存在循环依赖，无法确定一个合理的节点顺序。

拓扑排序的时间复杂度通常为 O(V + E)，其中 V 是节点数，E 是边数。这是因为每个节点和每条边都要被访问一次。

```python

```

### leetcode 逐行解析

- 课程安排[leetcode207 题目描述](https://leetcode.com/problems/course-schedule/description/)

How does cycle work?


```python
class Solution(object):
    def canFinish(self, numCourses, prerequisites):
        """
        :type numCourses: int
        :type prerequisites: List[List[int]]
        :rtype: bool
        """
        preMap = {c: [] for c in range(numCourses)}
        # c : course; p: prerequisites
        for c, p in prerequisites:
            preMap[c].append(p)
        cycle = set()

        def dfs(c):
            if c in cycle:
                return False
            if preMap[c] == []:
                return True

            cycle.add(c)
            for pre in preMap[c]:
                if not dfs(pre):
                    return False
            
            # else, True:
            cycle.remove(c)
            preMap[c] = []
            return True
        
        for c in range(numCourses):
            if not dfs(c):
                return False
        return True
```

- 课程安排2[leetcode210 题目描述](https://leetcode.com/problems/course-schedule-ii/description/)

典型的拓扑排序题。

```python

```

- 课程安排3[leetcode1462 题目描述](https://leetcode.com/problems/course-schedule-iv/description/)

深度优先搜索。

```python

```
