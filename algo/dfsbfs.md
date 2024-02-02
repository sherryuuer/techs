## 树算法：深度优先搜索和广度优先搜索

---
### 概念思考

两个简洁而强大的算法思想。

通过递归和回溯在图或者树中进行搜索。

dfs使用了栈的结构，先进后出。bfs使用了队列结构，先进先出。在树和图中的实现方式不同，但是基本思想都是一样的。

广度优先搜索，就像是是root扔下一块石头，然后水波不断扩展的样子。

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

在这个例子中，`Graph` 类表示图，使用邻接列表来存储图的边。`add_edge` 方法用于添加边。`dfs` 和 `bfs` 方法分别执行深度优先搜索和广度优先搜索。

注意，这只是一个基本的示例，你可能需要根据实际问题进行适当的修改。
