## 回溯算法：子集（subsets）

---

### 它是什么

回溯是一种试探性算法，每次都选择走不走下一步，很像一种后悔药。在子集问题上一般是给定一个有重复数字或者，没有重复数字的数组，然后找到所有符合条件的不重复的子集。

本质上是一种深度优先搜索算法。

```python

```

### leetcode 逐行解析

- 子集[leetcode78 题目描述](https://leetcode.com/problems/subsets/description/)

```python
class Solution(object):
    def subsets(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        res = []
        curset = []

        def dfs(idx):
            if idx >= len(nums):
                res.append(curset[:])
                return
            # curset include the idx num
            curset.append(nums[idx])
            dfs(idx + 1)
            # backtracking, not include the num
            curset.pop()
            dfs(idx + 1)
        
        dfs(0)
        return res
```

- 子集2[leetcode90 题目描述](https://leetcode.com/problems/subsets-ii/description/)

问题中的数组中有重复的数字，和无重复数组的关键不同在于，需要先对数组排序，然后在回溯的时候，只有在回溯前加入该数字，回溯后，迭代index的位置，直到不重复的数字index为止。

```python

```
