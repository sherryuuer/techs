## 广优先搜索bfs相关问题

---
### 适用范围

当答案距离根节点近，并且可以一层一层进行探索的情况。

- 最短路径问题：BFS可以用于寻找无权图中两个节点之间的最短路径。由于BFS会逐层遍历图，因此当搜索到目标节点时，它一定是经过的最少边数的路径之一，因此可以用于解决最短路径问题。
- 状态空间搜索：在状态空间搜索问题中，BFS通常用于找到从初始状态到目标状态的最短路径。例如，BFS可以应用于迷宫问题、拼图游戏等。
- 拓扑排序：如果一个图是有向无环图（DAG），BFS可以用于对其进行拓扑排序，即确定图中节点的线性顺序，使得图中的任意一条边的起点在排序中都排在终点之前。
- 连通性检测：BFS可以用于检测图的连通性，即确定图中的各个节点是否可以通过边相互到达。
- HTML的网页结构DOM就是一个巨大的树结构。

### 问题1:Symmetric Tree

题意为对称树，也就是判断一个二叉树，以他的root节点为轴，左右是否对称的问题。是一道很经典的问题（是，但我是查了才知道经典。）。

步骤解析：使用队列queue解答，这是一种很好的解法比较直观，后面还会给出一种递归的解答方法。

- 将根节点的左右节点加入一个queue（这个queue使用Python自带的库deque构造）。
- 在每次遍历中，出队两个节点进行比较（这两个节点分别是left和right节点）。
  - 如果这两个节点都为空，则继续下一轮。
  - 如果这两个节点中有一个为空，则不对称，返回False。
  - 如果这两个节点的值不想等，则不对称，返回False。
- 入队操作，按照左节点的左，右节点的右，左节点的右，右节点的左的顺序将节点加入队列。
- 完成遍历后返回True。

```python
from collections import deque


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def isSymmetric(root):
    if not root:
        return True

    queue = deque([root.left, root.right])

    while queue:
        left_node = queue.popleft()
        right_node = queue.popleft()

        if not left_node and not right_node:
            continue
        if not left_node or not right_node:
            return False
        if left_node.val != right_node.val:
            return False
        
        # append queue
        queue.append(left_node.left)
        queue.append(right_node.right)
        queue.append(left_node.right)
        queue.append(right_node.left)

    return True
```

另外还有这种递归的方法，这种方法就是，优雅吧，但是对我这种新手也不是很容易写出来，即使上面那种我也很费劲就是了。很多算法的学习不是解迷，而是学习一种方法。仅供参考：

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def isSymmetric(root):
    def isMirror(node1, node2):
        if not node1 and not node2:
            return True
        if not node1 or not node2:
            return False
        return (node1.val == node2.val) and isMirror(node1.left, node2.right) and isMirror(node1.right, node2.left)

    if not root:
        return True
    return isMirror(root.left, root.right)
```

学习笔记：这道题的时间复杂度和空间复杂度都是O(n)，因为是从头到尾的遍历了所有的元素，并且使用了额外针对所有的元素的空间。
