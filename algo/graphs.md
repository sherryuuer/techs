## 图数据结构（graphs）

---
### 图的基础

图（Graph）是一种非常重要的数据结构，广泛应用于算法和计算机科学领域。图由节点（顶点）和边组成，表示对象之间的关系。以下是图结构的基本知识：

1. **节点（顶点）：** 图中的基本单元，表示对象或实体。节点可以有附加信息，称为"属性"。

2. **边：** 连接两个节点的线，表示节点之间的关系。边可以是有向的（箭头表示方向）或无向的（双向连接）。

3. **有向图和无向图：**
   - **有向图（Directed Graph）：** 边有方向，从一个节点指向另一个节点。
   - **无向图（Undirected Graph）：** 边没有方向，表示节点之间的双向关系。
   
   还有一种**DAG**是有向无环图，在我的数据ETL中作为一种数据处理的流程结构存在，当我学了图论之后才知道它的基础原理。

4. **权重（Weight）：** 边上可以关联一个权重，表示节点之间的距离、成本或其他度量。这种图称为带权图。

5. **度（Degree）：** 节点的度是与其相连的边的数量。在有向图中，分为入度和出度，分别表示指向该节点的边和从该节点发出的边的数量。

6. **路径（Path）：** 由边连接的节点序列称为路径。路径的长度是边的数量。

7. **环（Cycle）：** 如果图中存在一条路径，使得路径的起点和终点是同一个节点，则称之为环。

8. **连通图（Connected Graph）：** 无向图中，如果任意两个节点之间都存在路径，则称图是连通的。

9. **强连通图（Strongly Connected Graph）：** 有向图中，如果任意两个节点之间都存在双向路径，则称图是强连通的。

10. **稀疏图和稠密图：**
    - **稀疏图：** 边的数量相对较少。
    - **稠密图：** 边的数量相对较多。

11. **邻接矩阵和邻接表：**
    - **邻接矩阵：** 使用矩阵表示节点之间的关系，矩阵元素表示边的存在与否，适用于稠密图。
    - **邻接表：** 使用链表或数组表示节点之间的关系，适用于稀疏图。

12. **图的遍历：** 遍历是访问图中所有节点的过程，主要有深度优先搜索（DFS）和广度优先搜索（BFS）两种方法。

这些是图结构的基本概念，图还有许多高级概念和算法，如最短路径算法、最小生成树算法、拓扑排序等。不同的问题可能需要不同的图算法来解决。

### 矩阵Matrix的DFS和BFS

Matrix的DFS：计算从矩阵的左上角到右下角有多少条路可以走。

```python
# Matrix (2D网格)
grid = [[0, 0, 0, 0],
        [1, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 1, 0, 0]]

# 计算有多少条路 (回溯方法)
def dfs(grid, r, c, visit):
    ROWS, COLS = len(grid), len(grid[0])
    if (min(r, c) < 0 or # 边界内
        r == ROWS or c == COLS or # 边界内
        (r, c) in visit or grid[r][c] == 1): # 已经访问过或者遇到墙
        return 0
    if r == ROWS - 1 and c == COLS - 1: # 代表走出去了
        return 1

    visit.add((r, c)) # 加入访问过的列表

    count = 0
    # 从四个方向进行进一步搜索
    count += dfs(grid, r + 1, c, visit)
    count += dfs(grid, r - 1, c, visit)
    count += dfs(grid, r, c + 1, visit)
    count += dfs(grid, r, c - 1, visit)
    # 在回溯节点，取消对当前节点的访问记录，以便其他路径的使用
    visit.remove((r, c))
    return count
```

Matrix的BFS：寻找从左上到右下的最短路径。

```python
from collections import deque

# Matrix (2D Grid)
grid = [[0, 0, 0, 0],
        [1, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 1, 0, 0]]

def bfs(grid):
    ROWS, COLS = len(grid), len(grid[0])
    visit = set()
    queue = deque()
    queue.append((0, 0))
    visit.add((0, 0))

    length = 0
    while queue:
        for i in range(len(queue)):
            r, c = queue.popleft()
            if r == ROWS - 1 and c == COLS - 1:
                return length

            neighbors = [[0, 1], [0, -1], [1, 0], [-1, 0]]
            for dr, dc in neighbors:
                if (min(r + dr, c + dc) < 0 or
                    r + dr == ROWS or c + dc == COLS or
                    (r + dr, c + dc) in visit or grid[r + dr][c + dc] == 1):
                    continue
                queue.append((r + dr, c + dc))
                visit.add((r + dr, c + dc))
        length += 1
```

### 邻接表（Adjacency List）

邻接表的初始化：通过dict实现。

```python
# GraphNode used for adjacency list
class GraphNode:
    def __init__(self, val):
        self.val = val
        self.neighbors = []

# Or use a HashMap
adjList = { "A": [], "B": [] }

# Given directed edges, build an adjacency list
edges = [["A", "B"], ["B", "C"], ["B", "E"], ["C", "E"], ["E", "D"]]

adjList = {}

for src, dst in edges:
    if src not in adjList:
        adjList[src] = []
    if dst not in adjList:
        adjList[dst] = []
    adjList[src].append(dst)
```

通过邻接表实现的深度优先搜索。

```python
# Count paths (backtracking)
def dfs(node, target, adjList, visit):
    if node in visit:
        return 0
    if node == target:
        return 1
    
    count = 0
    visit.add(node)
    for neighbor in adjList[node]:
        count += dfs(neighbor, target, adjList, visit)
    visit.remove(node)

    return count
```

通过邻接表实现的广度优先搜索。

```python
from collections import deque
# Shortest path from node to target
def bfs(node, target, adjList):
    length = 0
    visit = set()
    visit.add(node)
    queue = deque()
    queue.append(node)

    while queue:
        for i in range(len(queue)):
            curr = queue.popleft()
            if curr == target:
                return length

            for neighbor in adjList[curr]:
                if neighbor not in visit:
                    visit.add(neighbor)
                    queue.append(neighbor)
        length += 1
    return length
```
