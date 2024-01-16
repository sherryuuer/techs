## 树算法：迭代深度优先搜素（Iterative DFS）

---

### 它是什么

深度优先搜索（DFS）是一种用于图和树等数据结构的遍历算法，它从起始节点开始，沿着一条路径尽可能深地探索，直到到达末端，然后回溯并继续探索其他路径。

传统的递归深度优先搜索在实现上使用了函数调用栈来保存中间状态，但在处理深度较大的图时，可能导致栈溢出。为了解决这个问题，可以使用迭代深度优先搜索，通过使用显式的数据结构（通常是栈）来维护状态信息，而不是依赖于系统调用栈。

迭代 dfs 实质上是自己搞了一个 stack 存放你访问过的节点，在遍历完应有的深度后，返回来重新从 stack 弹出你访问过的节点，相当于手动开栈。

代码实现：

```python
class TreeNode:
    def __init__(self, val, left, right):
        self.val = val
        self.left = left
        self.right = right

# O(n)
def inorder(root):
    stack = []
    cur = root

    while cur or stack:
        if cur:
            stack.append(cur)
            cur = cur.left
        else:
            cur = stack.pop()
            print(cur.val)
            cur = cur.right

def preorder(root):
    stack = []
    cur = root

    while cur or stack:
        if cur:
            print(cur.val)
            if cur.right:
                stack.append(cur.right)
            cur = cur.left
        else:
            cur = stack.pop()

def postorder(root):
    stack = [root]
    visit = [False]

    while stack:
        cur, visited = stack.pop(), visit.pop()
        if cur:
            if visited:
                print(cur.val)
            else:
                stack.append(cur)
                visit.append(True)
                stack.append(cur.right)
                visit.append(False)
                stack.append(cur.left)
                visit.append(False)
```

### leetcode 逐行解析

- 二叉树先序遍历[leetcode144 题目描述](https://leetcode.com/problems/binary-tree-preorder-traversal/description/)

- 二叉树后序遍历[leetcode145 题目描述](https://leetcode.com/problems/binary-tree-postorder-traversal/description/)

- 二叉树迭代搜索[leetcode173 题目描述](https://leetcode.com/problems/binary-search-tree-iterator/description/)
