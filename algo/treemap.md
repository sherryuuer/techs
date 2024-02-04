## TreeMap：基于红黑树的树映射数据结构

---
### 什么是红黑树

红黑树（Red-Black Tree）是一种自平衡的二叉搜索树，它在每个节点上添加了一个额外的属性，用于表示节点的颜色，可以是红色或黑色。红黑树通过一系列的规则来保持树的平衡，以确保在最坏情况下，树的高度保持对数级别，从而保证各种操作的时间复杂度为 O(log n)。

红黑树的规则包括：

1. **节点颜色：** 每个节点要么是红色，要么是黑色。
  
2. **根节点：** 根节点是黑色的。

3. **叶子节点（NIL节点）：** 每个叶子节点都是黑色的空节点（NIL节点）。

4. **相邻节点：** 如果一个节点是红色的，则它的子节点都是黑色的。这保证没有两个相邻的红色节点。

5. **路径黑色节点数：** 对于树中的任意一个节点，从该节点到达其所有后代叶子节点的任意路径上，经过的黑色节点数量是相同的。

由于这些规则，红黑树在执行插入、删除等操作时，能够保持树的平衡，防止树的高度过高，确保了树的查找、插入和删除等操作都能够在对数时间内完成。

红黑树广泛应用于计算机科学领域，例如在**实现集合、映射**等数据结构时，以及一些编程语言的内部实现中，如 Java 的 `TreeMap` 和 `TreeSet`。

### 什么是TreeMap

TreeMap是一个基于红黑树实现的有序映射（Sorted Map），它有一些特性，比如有序，平衡，以保证时间复杂度是对数时间。

在 Python 中，有一个类似于 Java 中 `TreeMap` 的数据结构，那就是 `collections` 模块中的 `OrderedDict`。虽然 `OrderedDict` 并不是基于红黑树，但它是一个有序字典，可以按照元素插入的顺序进行遍历。以下是一个简单的示例：

```python
from collections import OrderedDict

# 创建一个有序字典
ordered_dict = OrderedDict()

# 添加键值对
ordered_dict[3] = "Three"
ordered_dict[1] = "One"
ordered_dict[4] = "Four"
ordered_dict[2] = "Two"

# 遍历输出键值对
for key, value in ordered_dict.items():
    print(f"{key}: {value}")
```

输出结果将按照键的插入顺序显示：

```
3: Three
1: One
4: Four
2: Two
```

在 Python 中，如果只需要按照键的顺序进行遍历而无需实现其他特定的有序映射行为，使用 `OrderedDict` 就足够了。如果需要更丰富的有序映射功能，可以考虑使用第三方库，例如 `sortedcontainers` 或 `bintrees`。这些库提供了基于二叉搜索树的有序映射实现，类似于 Java 中的 `TreeMap`。

### Python实现TreeMap的代码

初始化二叉树的树节点。注意到该树节点不只是有一个值而是一个健值对。

```python
# Binary Search Tree Node
class TreeNode:
    def __init__(self, key: int, val: int):
        self.key = key
        self.val = val
        self.left = None
        self.right = None

```

创建一个TreeMap数据结构，包括插入，移除，寻找最大值，寻找最小值，遍历排序健值对操作。

插入操作是根据二分查找树进行操作的，创建一个新的节点，current节点是当前root节点，不断比较左右节点，小于向左，大于向右，等于则更新该节点。所以说insert是可能会有一个更新update操作的。

```python
# Implementation for Binary Search Tree Map
class TreeMap:
    def __init__(self):
        self.root = None

    def insert(self, key: int, val: int) -> None:
        newNode = TreeNode(key, val)
        if self.root == None:
            self.root = newNode
            return

        current = self.root
        while True:
            if key < current.key:
                if current.left == None:
                    current.left = newNode
                    return
                current = current.left
            elif key > current.key:
                if current.right == None:
                    current.right = newNode
                    return
                current = current.right
            else:
                current.val = val
                return
```

get操作很好理解，通过二分查找树不断寻找想要得到的健，然后取得它的值即可。

```python
class TreeMap:
    def __init__(self):
        self.root = None

    def get(self, key: int) -> int:
        current = self.root
        while current != None:
            if key < current.key:
                current = current.left
            elif key > current.key:
                current = current.right
            else:
                return current.val
        return -1
```

取得整个数据结构中的最小和最大值value。找最大和找最小其实用了两种写法，找最大直接合并了找最小方法的两个函数，个人还是比较喜欢一次写完一个函数的方法。

```python
class TreeMap:
    def __init__(self):
        self.root = None

    def getMin(self) -> int:
        current = self.findMin(self.root)
        return current.val if current else -1

    # Returns the node with the minimum key in the subtree
    def findMin(self, node: TreeNode) -> TreeNode:
        while node and node.left:
            node = node.left
        return node

    def getMax(self) -> int:
        current = self.root
        while current and current.right:
            current = current.right
        return current.val if current else -1
```

移除remove操作。和二叉树里面移除操作是一样的，只是在最后更新节点的时候，记得将val也更新而已。使用一个helper函数，进行递归操作。

```python
class TreeMap:
    def __init__(self):
        self.root = None

    def remove(self, key: int) -> None:
        self.root = self.removeHelper(self.root, key)

    # Returns the new root of the subtree after removing the key
    def removeHelper(self, curr: TreeNode, key: int) -> TreeNode:
        if curr == None:
            return None

        if key > curr.key:
            curr.right = self.removeHelper(curr.right, key)
        elif key < curr.key:
            curr.left = self.removeHelper(curr.left, key)
        else:
            if curr.left == None:
                # Replace curr with right child
                return curr.right
            elif curr.right == None:
                # Replace curr with left child
                return curr.left
            else:
                # Swap curr with inorder successor
                minNode = self.findMin(curr.right)
                curr.key = minNode.key
                curr.val = minNode.val
                curr.right = self.removeHelper(curr.right, minNode.key)
        return curr
```

遍历二叉树，从左到右的顺序，也就是从小到大，将每个节点推入结果数组result中，返回result。

```python
class TreeMap:
    def __init__(self):
        self.root = None

    def getInorderKeys(self) -> List[int]:
        result = []
        self.inorderTraversal(self.root, result)
        return result

    def inorderTraversal(self, root: TreeNode, result: List[int]) -> None:
        if root != None:
            self.inorderTraversal(root.left, result)
            result.append(root.key)
            self.inorderTraversal(root.right, result)
```
