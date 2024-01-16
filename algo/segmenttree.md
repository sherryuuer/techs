## 树结构算法：段树（Segment Tree）

---

### 它是什么

Segment Tree（段树）是一种数据结构，主要用于处理一维区间或区间上的操作。它在解决一些范围查询问题时非常有用，例如在数组中找到某一区间的最小值、最大值、总和等。

Segment Tree 的核心思想是将数组划分成一些小的区间，然后为每个区间建立一颗线段树。每个节点代表一个区间，根节点表示整个数组，而叶子节点表示数组中的单个元素。通过在节点中存储有关区间的信息（例如最小值、最大值、总和等），可以在树中高效地执行范围查询。

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
        root.left = SegmentTree.build(0, L, M)
        root.right = SegmentTree.build(0, M + 1, R)
        root.sum = root.left.sum + root.right.sum
        return root

    # O(logn)
    def updata(self, index, val):
        if self.L == self.R:
            self.sum = val

        M = (self.L + self.R) // 2
        if index > M:
            self.right.update(index, val)
        else:
            self.left.update(index, val)
        self.sum = self.left.sum + self.right.sum

    # O(logn)
    def rangeQuery(self, L, R):
        if self.L == L and self.R == R:
            return self.sum

        M = (self.L + self.R) // 2
        if L > M:
            return self.right.rangeQuery(self, L, R)
        elif R <= M:
            return self.left.rangeQuery(self, L, R)
        else:
            return (self.left.rangeQuery(self, L, M) + self.right.rangeQuery(self, M + 1, R))

```

### leetcode 逐行解析
