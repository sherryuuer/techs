## 深度优先搜索dfs相关问题

---
### 适用范围

- 遍历和搜索：DFS通常用于图的遍历和搜索，它会沿着图的深度方向尽可能地遍历图中的节点。DFS在搜索树或图中的路径时可能会沿着一个分支一直深入，直到到达叶子节点，然后回溯到前一个分支的未探索节点。
- 拓扑排序：DFS也可以用于对有向无环图进行拓扑排序，类似于BFS，但DFS生成的拓扑排序结果可能不同。
- 连通性检测：DFS可以用于检测图的连通性，它能够通过探索整个图并标记访问过的节点来确定图中的连通分量。
- 回溯算法：在组合优化问题中，DFS通常与回溯算法一起使用，通过深度优先搜索树上的节点来探索解空间，并在搜索过程中进行剪枝以提高效率。

### 问题1:Diameter of Binary Tree

这个问题要求计算二叉树的直径。二叉树的直径定义为树中任意两个节点之间的最长路径的长度。这条路径可能经过根节点，也可能不经过根节点。

实现步骤：

- 选择树中的一个节点作为根节点。
- 计算从该根节点出发，经过该节点的最长路径长度。这可以通过递归地计算左右子树的高度之和来实现。
- 在树的所有节点中，找到最长的路径，其长度即为二叉树的直径。
- 解决这个问题的一种常见方法是采用递归的方式进行深度优先搜索（DFS）。对于每个节点，我们可以计算其左右子树的高度，并将它们相加得到当前节点的直径。然后，- 我们递归地应用这个过程，直到找到整个树的直径。

代码实现：

```python
def diameter_of_binaryTree(root):
    maxDiameter = 0
    
    def dfs(node):
        nonlocal maxDiameter
        if node is None:
            return 0
        leftDepth = dfs(node.left)
        rightDepth = dfs(node.right)
        # 更新直径
        maxDiameter = max(maxDiameter, leftDepth + rightDepth)
        # 返回当前节点的深度
        return max(leftDepth, rightDepth) + 1
    
    dfs(root)
    return maxDiameter
```

学习笔记：如果是暴力破解法，则需要对每一个节点进行计算，它到树中其他节点的最远距离，然后追踪距离，这会让时间复杂度达到n的平方次。使用上面的这种方法，时间复杂度则为n，因为所有的节点只需要遍历一次。

### 问题2:Serialize and Deserialize Binary Tree

其实我一点都不懂序列化是什么意思，所以先查一下概念吧。

"Serialize and Deserialize Binary Tree" 是一种将二叉树序列化（转换为字符串表示）和反序列化（从字符串表示还原为二叉树）的算法。这种算法通常用于将二叉树存储到文件或者通过网络传输。

序列化（Serialize）：是将二叉树转换为字符串表示的过程。在序列化过程中，我们需要将二叉树的结构和节点值转换为一个字符串。通常使用先序遍历（Pre-order Traversal）的方式来实现序列化，即按照根节点-左子树-右子树的顺序遍历二叉树，并将节点的值转换为字符串，并使用特定的符号（例如逗号或空格）来分隔节点值。如果节点为 null，则用特定的标记例如 "#"表示。

反序列化（Deserialize）：是将字符串表示的二叉树转换回原始的二叉树结构的过程。在反序列化过程中，我们需要解析字符串，并将其转换为二叉树的结构和节点值。通常采用递归的方式进行反序列化。我们首先解析字符串中的第一个节点值，并创建一个节点。然后递归地调用反序列化函数来构建左子树和右子树，直到所有节点都被处理完毕。

比如考虑以下二叉树：
```
    1
   / \
  2   3
     / \
    4   5
```
序列化的结果可能是："1,2,#,#,3,4,#,#,5,#,#"。

这种算法通过将二叉树转换为字符串表示，实现了二叉树的持久化存储和传输。这种算法具有简单高效的特点，适用于需要将二叉树保存到文件或者通过网络传输的场景。

解题思路：

serialize：

- 从根节点开始进行先序遍历。
- 对于每个节点，如果节点为空，则将其值序列化为 "#"，表示空节点；否则，将节点值转换为字符串，并将其与左子树和右子树的序列化结果拼接起来。
返回序列化后的字符串。

deserialize 函数：

- 定义一个辅助函数 helper，它接收一个迭代器作为参数，用于处理当前节点及其子节点。
- 从字符串中逐个读取节点值，如果节点值为 "#"，则返回 None 表示空节点；否则，创建一个节点，并递归地构建其左子树和右子树。
- 将输入的字符串切分为一个列表，并创建一个迭代器。
- 调用 helper 函数，并返回反序列化后的二叉树根节点。

```python
# Definition of a binary tree node
#
# class TreeNode:
#     def __init__(self, data):
#         self.data = data
#         self.left = None
#         self.right = None

from ds_v1.BinaryTree.BinaryTree import TreeNode

def serialize(root):
    """将二叉树序列化为字符串表示"""
    if not root:
        return "#,"  # 空节点用 "#" 表示
    
    # 序列化当前节点值，以及左右子树
    left_serialized = serialize(root.left)
    right_serialized = serialize(root.right)
    
    return str(root.data) + "," + left_serialized + right_serialized

def deserialize(stream):
    """将字符串表示反序列化为二叉树"""
    def helper(stream):
        val = next(stream)  # 从迭代器中取出一个值
        if val == "#":
            return None
        node = TreeNode(int(val))
        node.left = helper(stream)
        node.right = helper(stream)
        return node
    
    stream = iter(stream.split(","))  # 将字符串分割成列表，并创建一个迭代器
    return helper(stream)
```

以上是自编和修改，以下是官方题解：他加入了一个M的maker，同时带有编号，也许带这个编号在某些场景下有好的作用，整体上方法大同小异，都是通过递归的方法，将左右树枝分别进行处理和添加。

```python
# Initializing our marker
MARKER = "M"
m = 1

def serialize_rec(node, stream):
    global m

    if node is None:
        stream.append(MARKER + str(m))
        m += 1
        return

    stream.append(node.data)

    serialize_rec(node.left, stream)
    serialize_rec(node.right, stream)

# Function to serialize tree into list of integers.
def serialize(root):
    stream = []
    serialize_rec(root, stream)
    return stream

def deserialize_helper(stream):
    val = stream.pop()

    if type(val) is str and val[0] == MARKER:
        return None

    node = TreeNode(val)

    node.left = deserialize_helper(stream)
    node.right = deserialize_helper(stream)

    return node

# Function to deserialize integer list into a binary tree.
def deserialize(stream):
    stream.reverse()
    node = deserialize_helper(stream)
    return node
```

学习笔记：同样的，树的算法加上递归的算法，体现的是一种递归的美，让人欲罢不能。这道题在时间复杂度上，因为遍历了所有的节点，所以是O(n)，空间复杂度分情况，因为是递归的从上到下经历了整个树的高度，所以应该是O(height)，如果是平衡的二叉树，就是对数时间，如果是不平衡的，最大可以是O(n)。

### 问题3:Binary Tree Maximum Path Sum

题目的意思是给定一个二叉树，要求找到一条路径，使得该路径上的节点值之和最大。这条路径可以从二叉树的任意节点出发，沿着父节点、左子节点和右子节点之间的连接线移动，并且每个节点最多只能被访问一次。要求找到一条最大路径和是指从某个节点出发，经过该节点的路径上所有节点值之和，是所有路径和中的最大值。

以下是代码的步骤方法总结：

- 初始化一个变量max_sum，他是一个全局的最大和，为负无穷。
- 对于一个子节点，它对路径的贡献值为它自己本身的值。如果不存在，则为0。
- 或者它对路径贡献的最大值为，它自己的值，加上左右节点中最大的那个值（负数忽略不计）（这里要递归的计算了）。
- 更新用于储存最大值的变量max_sum。最后返回这个值。

代码：

```python
# Definition of a binary tree node
# class TreeNode:
#     def __init__(self, data):
#         self.data = data
#         self.left = None
#         self.right = None

from ds_v1.BinaryTree.BinaryTree import TreeNode

def max_path_sum(root):
    max_sum = float('-inf')

    def helper(node):
        nonlocal max_sum

        if not node:
            return 0
        
        # 计算左右子树的最大路径和，不考虑负数
        left_sum = max(0, helper(node.left))
        right_sum = max(0, helper(node.right))
        
        # 更新全局的最大和
        max_sum = max(max_sum, node.data + left_sum + right_sum)
        
        # 返回以当前节点为根节点的子树的最大路径和
        return node.data + max(left_sum, right_sum)

    helper(root)
    return max_sum
```

学习笔记：每次我拿到题，感觉最难的地方，在于梳理题目的意思，每次在理解题意上要花的时间，远远大于之后的步骤。这样正说明了，解决问题的时候，对问题的理解至关重要。

一开始我没有理解helper函数中最后的return的部分，为什么加号后面是选取最大的那个，想清楚后理解了，这里返回的东西，要在下一轮中，在left_sum和right_sum中使用，他们要作为递归后的一个分叉来使用，因为是一个分支，所以当然是选择最大的那个。很多时候我都是困在一个地方。

时间复杂度是O(n)，空间复杂度是树的高度。
