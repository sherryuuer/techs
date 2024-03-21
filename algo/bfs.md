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

### 问题2:Vertical Order Traversal of a Binary Tree

这是一个关于二叉树的问题，题目要求按照垂直顺序遍历二叉树，并按照每个节点的位置从左到右进行排序。

具体来说，对于给定的二叉树，我们需要按照节点的水平位置，从左到右的顺序，将每个节点的值组织成一个二维列表。在列表中，每个子列表表示同一水平位置上的节点值，而不同水平位置的子列表则按照水平顺序排列。

例如，考虑以下二叉树：

```
    3
   / \
  9  20
     / \
    15  7
```

按照垂直顺序遍历这棵二叉树，可以得到以下结果：

```
[
  [9],
  [3, 15],
  [20],
  [7]
]
```

在这个结果中，第一个子列表包含水平位置 -1 上的节点 9，第二个子列表包含水平位置 0 上的节点 3 和 15，第三个子列表包含水平位置 1 上的节点 20，第四个子列表包含水平位置 2 上的节点 7。题目的要求是按照这种方式，从左到右遍历二叉树，并按照每个节点的位置将节点值进行组织。

解题思路：

- 从根节点开始一层一层遍历二叉树。
- 将节点和和节点对应的index一同推入队列queue。
- 如果节点有左边节点就将左节点的index设为根节点index - 1，如果有右边节点就将右节点的index设为根节点index + 1。
- 追踪最小和最大的节点以供之后for循环使用。
- 为每一个index从小到大建立一个列表结果。
- （我觉得这里如果使用最小堆，就可以免得排序直接从最小到大弹出了）

按照步骤来写代码就是这样：

```python
from collections import deque
# Definition for a binary tree node
# class TreeNode:
#     def __init__(self, data):
#         self.data = data
#         self.left = None
#         self.right = None

from ds_v1.BinaryTree.BinaryTree import TreeNode

def vertical_order(root):
    if not root:
        return []

    queue = deque([(root, 0)])
    minIndex = 0
    maxIndex = 0
    indexToValues = {}

    while queue:
        current, index = queue.popleft()
        
        minIndex = min(minIndex, index)
        maxIndex = max(maxIndex, index)

        if index not in indexToValues:
            indexToValues[index] = []
        indexToValues[index].append(current.data)

        if current.left:
            queue.append((current.left, index - 1))

        if current.right:
            queue.append((current.right, index + 1))

    result = []
    for i in range(minIndex, maxIndex + 1):
        if i in indexToValues:
            result.append(indexToValues[i])

    return result
```

题解给出的答案如下，大同小异，省略了几行可以合并的代码。

```python
from collections import defaultdict, deque

def vertical_order(root):
    if root is None:
        return []

    node_list = defaultdict(list)
    min_column = 0
    max_index = 0
    queue = deque([(root, 0)])

    while queue:
        node, column = queue.popleft()

        if node is not None:
            temp = node_list[column]
            temp.append(node.data)
            node_list[column] = temp

            min_column = min(min_column, column)
            max_index = max(max_index, column)

            queue.append((node.left, column - 1))
            queue.append((node.right, column + 1))

    return [node_list[x] for x in range(min_column, max_index + 1)]
```

学习笔记：题目的时间复杂度是O(n)，因为要遍历整个树的所有节点。空间复杂度也是O(n)，因为使用了额外的空间用来存储所有的节点。在做题中，最重要的是整理步骤，然后按照步骤梳理结构，也许最大的误区在于，我们总是希望从头到尾完美的写好一段代码，以至于从一开始就想找到所有需要初始化的变量，正确的数据结构，但其实只要想清楚了步骤，然后按照步骤逐步添加需要的部分，才是一开始应该修炼的部分，熟能生巧是下一阶段的事情。

### 问题3:Word Ladder

Word Ladder（单词梯子）问题是一类经典的单词游戏和编程题目，其中给定了两个单词（起始单词和目标单词），以及一个单词列表，要求从起始单词变换到目标单词，每次只能改变一个字母，并且变换后的单词必须在给定的单词列表中。

例如，给定单词列表 ["hot", "dot", "dog", "lot", "log", "cog"]，起始单词是 "hit"，目标单词是 "cog"，则一种最短的变换路径可以是： "hit" -> "hot" -> "dot" -> "dog" -> "cog"

Word Ladder 问题可以用图论的思想来解决，将单词列表中的每个单词看作图中的节点，如果两个单词之间只有一个字母不同，则在它们之间连一条边。然后可以使用广度优先搜索算法来找到从起始单词到目标单词的最短路径。

以下是解决 Word Ladder 问题的一般步骤：

- 构建图：遍历单词列表，对于每个单词，找到与其只有一个字母不同的其他单词，将其相连。
  - 将word列表初始化为一个set，初始化一个队列将src起始单词推入，初始化一个counter存储队列长度。
- 使用 BFS 算法：从起始单词开始，按层次进行广度优先搜索，找到从起始单词到目标单词的最短路径。
  - 从queue中弹出单词，然后在单词set中找到和该单词相差一个字母的单词，加入queue。
- 回溯路径：一旦找到目标单词，可以通过回溯路径来重建从起始单词到目标单词的变换路径。在这道题中是返回追踪的counter也就是变换次数。

Word Ladder 问题在编程面试中经常出现，也是一种常见的算法练习题目。它既考察了图论和搜索算法的知识，又考察了对数据结构和算法的设计和实现能力。

这里是我的代码：

```python
from collections import deque
def is_one_char_diff(word1, word2):
    if len(word1) != len(word2):
        return False
    
    diff_count = 0
    # 比较两个单词的每个字符
    for i in range(len(word1)):
        if word1[i] != word2[i]:
            diff_count += 1
            # 如果不同字符数量超过1，则不满足条件，直接返回 False
            if diff_count > 1:
                return False
    return True

def word_ladder(src, dest, words):
    if dest not in words:
        return 0
    
    word_list = set(words)
    queue = deque([src])
    counter = 1

    while len(queue) > 0:
        for _ in range(len(queue)):
            cur = queue.popleft()
        
            if cur == dest:
                return counter

            for word in word_list.copy():
                if is_one_char_diff(cur, word):
                    queue.append(word)
                    word_list.remove(word)

        counter += 1
    return 0
```

题解代码如下：他这里没有另外加入单词比较的函数，而是直接用二十六个字母构建了一个相差一个字母的单词temp然后进行26轮乘以单词长度次的比较。

```python
def word_ladder(src, dest, words):
    myset = set(words)

    if dest not in myset:
        return 0

    q = []
    q.append(src)
    length = 0

    while q:
        length += 1
        size = len(q)

        for _ in range(size):
            curr = q.pop(0)

            for i in range(len(curr)):
                alpha = "abcdefghijklmnopqrstuvwxyz"
       
                for c in alpha:
                    temp = list(curr)
                    temp[i] = c
                    temp = "".join(temp)
 
                    if temp == dest:
                        return length + 1

                    if temp in myset:
                        q.append(temp)
                        myset.remove(temp)
    return 0
```

不过我还是喜欢自己的那种写法，下面将该写法优化一下：优化了单词比较函数。另外使用了visited，来存储是否访问过单词。以及简化了遍历queue时候的循环。

```python
from collections import deque

def is_one_char_diff(word1, word2):
    # 计算两个单词之间的字符差异数量
    diff_count = sum(c1 != c2 for c1, c2 in zip(word1, word2))
    return diff_count == 1

def word_ladder(src, dest, words):
    if dest not in words:
        return 0
    
    visited = set()
    queue = deque([(src, 1)])

    while queue:
        cur, level = queue.popleft()
        
        if cur == dest:
            return level

        for word in words:
            if word not in visited and is_one_char_diff(cur, word):
                visited.add(word)
                queue.append((word, level + 1))
    
    return 0
```

学习笔记：将单词列表转换为集合的时间复杂度为 O(n)，其中 n 是单词列表的长度。在 BFS 搜索中，遍历了每个单词，并检查其与其他单词的差异。对于每个单词，我们需要检查它与所有其他单词的差异，因此总的时间复杂度为 O(n * m)，其中 n 是单词列表的长度，m 是单词的平均长度。时间复杂度为 O(n * m)。

将单词列表转换为集合需要额外的空间来存储单词。集合的大小取决于单词列表的长度，因此空间复杂度为 O(n)，其中 n 是单词列表的长度。而在 BFS 搜索中，使用了一个队列来存储待处理的单词，以及一个集合来存储已经访问过的单词。这两个数据结构的空间复杂度取决于它们的大小，最坏情况下，它们的大小可能与单词列表的长度相同，因此空间复杂度为 O(n)。因此，整个算法的空间复杂度为 O(n)。

说实话这三道题对我来说并不容易，现在还是不太会立刻能写出来的状态，有空再进行分别的练习。
