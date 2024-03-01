## 树结构算法：段树（Segment Tree）

---

### 概念

Segment Tree（段树）是一种数据结构，主要用于处理一维区间或区间上的操作。它在解决一些范围查询问题时非常有用，例如在数组中找到某一区间的最小值、最大值、总和等。

Segment Tree 的核心思想是分治。将数组划分成一些小的区间，然后为每个区间建立一颗线段树。每个节点代表一个区间，根节点表示整个数组，而叶子节点表示数组中的单个元素。通过在节点中存储有关区间的信息（例如最小值、最大值、总和等），可以在树中高效地执行范围查询。

Segment Tree 的常见操作包括建立、更新和查询。建立过程通常是自底向上的，递归地将每个区间划分为两半，然后将父节点的信息计算为两个子节点的信息的合并。更新操作用于修改数组中的元素值，而查询操作则用于获取某一范围内的信息。

这种数据结构通常用于解决需要频繁进行范围查询的问题，例如在数值模拟、统计学等领域。在算法竞赛和编程竞赛中，Segment Tree 是一个常见而强大的工具，可以帮助解决一些复杂的问题。

创建数据结构：

```python
class SegmentTree:
    def __init__(self, total, L, R):
        self.sum = total
        self.left = None
        self.right = None
        self.L = L
        self.R = R
        
    # O(n)
    @staticmethod
    def build(nums, L, R):
        if L == R:
            return SegmentTree(nums[L], L, R)

        M = (L + R) // 2
        root = SegmentTree(0, L, R)
        root.left = SegmentTree.build(nums, L, M)
        root.right = SegmentTree.build(nums, M + 1, R)
        root.sum = root.left.sum + root.right.sum
        return root

    # O(logn)
    def update(self, index, val):
        if self.L == self.R:
            self.sum = val
            return
        
        M = (self.L + self.R) // 2
        if index > M:
            self.right.update(index, val)
        else:
            self.left.update(index, val)
        self.sum = self.left.sum + self.right.sum
        
    # O(logn)
    def rangeQuery(self, L, R):
        if L == self.L and R == self.R:
            return self.sum
        
        M = (self.L + self.R) // 2
        if L > M:
            return self.right.rangeQuery(L, R)
        elif R <= M:
            return self.left.rangeQuery(L, R)
        else:
            return (self.left.rangeQuery(L, M) +
                    self.right.rangeQuery(M + 1, R))

```

### leetcodes

- 区间和leetcode307题：[题目链接](https://leetcode.com/problems/range-sum-query-mutable/description/)

是一道典型的段数数据结构构造的题。

输入输出也是典型的力扣的方式：

```python
Input
["NumArray", "sumRange", "update", "sumRange"]
[[[1, 3, 5]], [0, 2], [1, 2], [0, 2]]
Output
[null, 9, null, 8]
```

代码如下：说实话整个代码看起来相当长，其实就是上面的段树的改写。增加了一个Node的数据结构用于创建段树结构，该数据结构用于创建题目中所要求的数据结构。返回的只有一个root。因此需要写出辅助函数来创建对应的方法。

由于对OOP掌握的还不是很熟练，所以这种改写其实是一个很好的练习。

```python
class Node:
    def __init__(self, total, left, right):
        self.sum = total
        self.leftroot = None
        self.righroot = None
        self.left = left
        self.right = right


class NumArray(object):

    def __init__(self, nums):
        """
        :type nums: List[int]
        """
        def build(nums, left, right):
            if left == right:
                return Node(nums[left], left, right)
            mid = (left + right) // 2
            root = Node(0, left, right)
            root.leftroot = build(nums, left, mid)
            root.rightroot = build(nums, mid + 1, right)
            root.sum = root.leftroot.sum + root.rightroot.sum
            return root

        self.root = build(nums, 0, len(nums)-1)

    def update(self, index, val):
        """
        :type index: int
        :type val: int
        :rtype: None
        """
        def _update(root, index, val):
            if root.left == root.right:
                root.sum = val
                return 

            mid = (root.left + root.right) // 2
            if index > mid:
                _update(root.rightroot, index, val)
            else:
                _update(root.leftroot, index, val)
            root.sum = root.leftroot.sum + root.rightroot.sum
        return _update(self.root, index, val)

    def sumRange(self, left, right):
        """
        :type left: int
        :type right: int
        :rtype: int
        """
        def _sumRange(root, left, right):
            if left == root.left and right == root.right:
                return root.sum
            mid = (root.left + root.right) // 2
            if left > mid:
                return _sumRange(root.rightroot, left, right)
            elif right <= mid:
                return _sumRange(root.leftroot, left, right)
            else:
                return _sumRange(root.leftroot, left, mid) + _sumRange(root.rightroot, mid + 1, right)

        return _sumRange(self.root, left, right)
```

在力扣的题解tab还有前人，利用数组的优良性质写出了简单的解法，如下：使用内置的sum方法，和长度的控制构建的方法。

```python
class NumArray:
    nums = []
    s = 0
    l = 0

    def __init__(self, nums):
        self.nums = nums
        self.s = sum(nums)
        self.l = len(nums)

    def update(self, index: int, val: int):
        self.s -= self.nums[index]
        self.nums[index] = val
        self.s += self.nums[index]

    def sumRange(self, left: int, right: int):
        if right - left > self.l // 2:
            ans = sum(self.nums[:left]) + sum(self.nums[right + 1:])
            return self.s - ans
        else:
            return sum(self.nums[left: right + 1])
```
