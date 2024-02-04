## 回溯算法：树迷宫（Tree Maze）

---
### 什么是树迷宫

如概念自身表达，就是一个树形状的路径迷宫，从root出发，走到leaf节点的通路。

### 代码示例

这个示例是通过在一个树迷宫中寻找路径的问题，来理解这个概念。**如果遇到0节点，则意味着此路不通**，需要回溯到上一个节点，寻找新的路径，如果是一个有可能组成路径的节点，则会被append到path中去。

```python
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

def canReachLeaf(root):
    if not root or root.val == 0:
        return False
    
    if not root.left and not root.right:
        return True
    if canReachLeaf(root.left):
        return True
    if canReachLeaf(root.right):
        return True
    return False

def leafPath(root, path):
    if not root or root.val == 0:
        return False
    path.append(root.val)

    if not root.left and not root.right:
        return True
    if leafPath(root.left, path):
        return True
    if leafPath(root.right, path):
        return True
    path.pop()
    return False
```
