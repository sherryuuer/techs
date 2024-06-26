## 数组算法：前缀和（prefix sums）

---

### 前缀和

顾名思义就是 index 之前的元素的和。

示例一个求 index 区间的元素的 sum 的数据结构设计。

```python
class PrefixSum:
    def __init__(self, nums):
        # 定一个一个空的数组用来存储前缀和数组
        self.prefix = []
        # 初始化前缀和添加器
        total = 0
        # 遍历给定的数组
        for n in nums:
            # 每次都将当前的元素添加到total
            total += n
            # 在当前的位置也就是index上，添加当前的total到前缀和数组
            self.prefix.append(total)

    def rangeSum(self, left, right):
        # 该方法计算区间[left, right]之间元素的和

        # 取得区间右端位置的前缀和
        preRight = self.prefix[right]
        # 取得区间左边位置-1的前缀和，为了防止这个left是左边端点导致out of index错误，所以给一个判断，如果是0则直接返回0，因为最左边的元素的左边来所，前缀和是0
        preLeft = self.prefix[left - 1] if left > 0 else 0
        # 区间的和就是right端点的前缀和减去left端点的前缀和
        return preRight - preLeft
```

### leetcode

- 不可变数组范围求和[leetcode303 题目描述](https://leetcode.com/problems/range-sum-query-immutable/description/)

典型的前缀和问题。题解和上面的基本概述的代码是一样的，内容不变，但是力扣的内部执行带入比较特殊，但是这里不太需要在意。

input和output如下方式：

```
Input
["NumArray", "sumRange", "sumRange", "sumRange"]
[[[-2, 0, 3, -5, 2, -1]], [0, 2], [2, 5], [0, 5]]
Output
[null, 1, -1, -3]
```

有时候感觉读题也是一门学问。

```python
class NumArray(object):

    def __init__(self, nums):
        """
        :type nums: List[int]
        """
        self.prefix = []
        total = 0
        for n in nums:
            total += n
            self.prefix.append(total)

    def sumRange(self, left, right):
        """
        :type left: int
        :type right: int
        :rtype: int
        """
        rightSum = self.prefix[right]
        leftSum = self.prefix[left - 1] if left > 0 else 0 
        return rightSum - leftSum
```

说点别的，一开始python基础不好的时候，对类的创建很不熟，比如这里的init部分，total不需要带self是因为他不是一个属性，而是计算过程中的变量，在其他部分的method中，也不会用到这个变量，可是说是内部的变量，所以不需要self，但是prefix列表是前缀树的一个属性，而且是重要属性，所以是self的。

---

- 不可变二维矩阵范围求和[leetcode304 题目描述](https://leetcode.com/problems/range-sum-query-2d-immutable/description/)

前缀和增维变体问题。其实是矩阵求面积的问题，这种问题容易超出边界，添加辅助边界真的 YYDS！

- 查找透视索引[leetcode724 题目描述](https://leetcode.com/problems/find-pivot-index/description/)

读题读出的意思就是，求前缀和和后缀和相等的索引位置。如果没有则返回-1。

hint：考虑动态地计算出后缀和。

- 除自身之外的数组乘积[leetcode238 题目描述](https://leetcode.com/problems/product-of-array-except-self/description/)

题目要求不能使用除法运算。

- 子数组和为 k[leetcode560 题目描述](https://leetcode.com/problems/subarray-sum-equals-k/description/)

关键在于用字典节约计算时间，如何在同时计算前缀和的同时，计算出有多少结果。
