## 树算法：深度优先搜索和广度优先搜索

---
### 概念思考

两个简洁而强大的算法思想。通过递归和回溯在图或者树中进行搜索。

dfs使用了栈的结构，先进后出。bfs使用了队列结构，先进先出。在树和图中的实现方式不同，但是基本思想都是一样的。

广度优先搜索，就像是是root扔下一块石头，然后水波不断扩展的样子。

### 适用范围

广度优先搜索（BFS）和深度优先搜索（DFS）是两种基本的图搜索算法，它们在解决不同类型的问题时具有不同的适用范围。

**广度优先搜索（BFS）的适用范围：**

1. **最短路径问题：** BFS可以用于寻找无权图中两个节点之间的最短路径。由于BFS会逐层遍历图，因此当搜索到目标节点时，它一定是经过的最少边数的路径之一，因此可以用于解决最短路径问题。

2. **状态空间搜索：** 在状态空间搜索问题中，BFS通常用于找到从初始状态到目标状态的最短路径。例如，BFS可以应用于迷宫问题、拼图游戏等。

3. **拓扑排序：** 如果一个图是有向无环图（DAG），BFS可以用于对其进行拓扑排序，即确定图中节点的线性顺序，使得图中的任意一条边的起点在排序中都排在终点之前。

4. **连通性检测：** BFS可以用于检测图的连通性，即确定图中的各个节点是否可以通过边相互到达。

**深度优先搜索（DFS）的适用范围：**

1. **遍历和搜索：** DFS通常用于图的遍历和搜索，它会沿着图的深度方向尽可能地遍历图中的节点。DFS在搜索树或图中的路径时可能会沿着一个分支一直深入，直到到达叶子节点，然后回溯到前一个分支的未探索节点。

2. **拓扑排序：** DFS也可以用于对有向无环图进行拓扑排序，类似于BFS，但DFS生成的拓扑排序结果可能不同。

3. **连通性检测：** DFS可以用于检测图的连通性，它能够通过探索整个图并标记访问过的节点来确定图中的连通分量。

4. **回溯算法：** 在组合优化问题中，DFS通常与回溯算法一起使用，通过深度优先搜索树上的节点来探索解空间，并在搜索过程中进行剪枝以提高效率。

总的来说，广度优先搜索和深度优先搜索在图搜索和状态空间搜索问题中都有广泛的应用，选择合适的算法取决于具体问题的特点和要求。

### 树中的代码实现

深度优先搜索dfs

```python
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

def inorder(root):
    if not root:
        return    
    inorder(root.left)
    print(root.val)
    inorder(root.right)

def preorder(root):
    if not root:
        return    
    print(root.val)
    preorder(root.left)
    preorder(root.right)

def postorder(root):
    if not root:
        return    
    postorder(root.left)
    postorder(root.right)
    print(root.val)
```

广度优先搜索bfs

```python
from collections import deque

class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

def bfs(root):
    queue = deque()

    if root:
        queue.append(root)
    
    level = 0
    while len(queue) > 0:
        print("level: ", level)
        for i in range(len(queue)):
            curr = queue.popleft()
            print(curr.val)
            if curr.left:
                queue.append(curr.left)
            if curr.right:
                queue.append(curr.right)
        level += 1
```

### 在图中的代码实现

当你使用深度优先搜索（DFS）和广度优先搜索（BFS）时，你需要考虑如何表示图。

最常用的，是使用邻接列表表示的图。

```python
from collections import deque

class Graph:
    def __init__(self):
        self.graph = {}

    def add_edge(self, u, v):
        if u not in self.graph:
            self.graph[u] = []
        self.graph[u].append(v)

    def dfs(self, start):
        visited = set()

        def dfs_recursive(node):
            nonlocal visited
            if node not in visited:
                print(node, end=' ')
                visited.add(node)
                for neighbor in self.graph.get(node, []):
                    dfs_recursive(neighbor)

        dfs_recursive(start)

    def dfs(self, start):
        visited = set()
        stack = [start]

        while stack:
            node = stack.pop()
            if node not in visited:
                print(node, end=' ')
                visited.add(node)
                # 将邻居节点逆序压入栈中，保持深度优先搜索的顺序
                stack.extend(reversed(self.graph.get(node, [])))

    def bfs(self, start):
        visited = set()
        queue = deque([start])
        visited.add(start)

        while queue:
            node = queue.popleft()
            print(node, end=' ')
            
            for neighbor in self.graph.get(node, []):
                if neighbor not in visited:
                    queue.append(neighbor)
                    visited.add(neighbor)

# 示例用法
g = Graph()
g.add_edge(0, 1)
g.add_edge(0, 2)
g.add_edge(1, 2)
g.add_edge(2, 0)
g.add_edge(2, 3)
g.add_edge(3, 3)

print("DFS starting from node 2:")
g.dfs(2)
print("\nBFS starting from node 2:")
g.bfs(2)
```
例子中，`Graph` 类表示图，使用邻接列表来存储图的边。`add_edge` 方法用于添加边。`dfs` 和 `bfs` 方法分别执行深度优先搜索和广度优先搜索。

