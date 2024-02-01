## 树结构：一种基本的树，二叉树（Binary tree）

---
### 概念引导

嗯就是一棵倒着的树。上面是根下面是枝叶节点，每一个root根节点，都有两个叶节点，左边和右边，左边的数字都小于root，右边的数字都大于root。这里要说的二叉树，其实接的是二分查找那一篇，二分查找是有序数组的算法，这里的二叉树，保证左边的枝比右边的枝叶的值都小。你把这个树拉开用数组的样子呈现，也是一个有序的数组。

总的来说就是严格的二叉树，保证左边的节点都比root根小，右边的节点都比root大。

二叉树的python实现：可以看出在search上，其实和数组的二分查找很像。

在我看来树最美的地方就是他经常用**递归的方式**进行操作，遍历，迭代，探索发现，一种不断拓展的美。

```python
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None


class BST:
    def __init__(self, root) -> None:
        self.root = TreeNode()

    def search(self, root, target):
        if not root:
            return False

        if target < root.left.val:
            return self.search(root.left, target)
        elif target > root.right.val:
            return self.search(root.right, target)
        else:
            return True
```

节点的插入：

节点的插入保证永远插入的地方都是最下面的叶子节点。递归地往下查找，直到找到最下面的叶子，和它进行比较和链接。

```python
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

# 插入一个节点，返回其root。
def insert(root, val):
    if not root:
        return TreeNode(val)
    
    if val > root.val:
        root.right = insert(root.right, val)
    elif val < root.val:
        root.left = insert(root.left, val)
    return root
```

删除节点相对复杂一点，因为可能要删除有child的节点。

具体步骤分解：我将注释写入代码

```python
# 返回一个树的最小值，一般是左边，所以只需要不断的递归left的root直到最深处的left就是最小值
# 这个method在下面的remove方法中要用到
def minValueNode(root):
    curr = root
    while curr and curr.left:
        curr = curr.left
    return curr

# 删除一个节点，返回其root。
def remove(root, val):
    # 特殊情况：如果不存在root那么你什么也删除不了，所以返回None
    if not root:
        return None
    
    # 判断要删除的数值比根大
    if val > root.val:
        # 那么要删除的节点在该根的右边，递归地从右边删除该节点，更新右边分支（这就是需要写root.right重新赋值的原因）
        root.right = remove(root.right, val)
    # 判断要删除的数值比根小
    elif val < root.val:
        # 那么要删除的节点在该根的左边，递归地从左边删除该节点，更新左边分支（这就是需要写root.left重新赋值的原因）
        root.left = remove(root.left, val)
    else:  # else意味着找到了该节点
        # 如果这个节点没有左边的分支
        if not root.left:
            # 那么你要更新的节点就直接和，要删除的节点的右边的分支连上就可以了
            return root.right
        # 如果这个节点没有右边的分支
        elif not root.right:
            # 那么你要更新的那部分分支，就直接和要删除的节点的左边连上就可以了
            return root.left
        # 上面两个步骤相当于跳过了要删除的节点，想起链表中删除节点的方法，其实是一样的，就是跳过要删除的节点
        # 并且上面两个判断保证了不会丢失节点
        else: # else意味着该节点左边右边都有的情况
            # 找到右边分支中最小的节点（因为左边分支都小于右边分支的节点，所以只要找一个右边节点中最小的放在root上就可以保证树的正确）
            minNode = minValueNode(root.right)
            # 将root（也就是要删除的节点）的val更新为右边分支中最小节点的值
            root.val = minNode.val
            # 这时候树中有两个相等val的节点，一个是右边分支中的最小值节点一个是要删除的这个根节点，所以只需要将右边分支中的这个最小值删除即可。
            root.right = remove(root.right, minNode.val)
    # 返回更新后的root
    return root
```
