## 回溯算法：子集（subsets）

---

### 概念

回溯是一种试探性算法，每次都选择走不走下一步，很像一种后悔药。在子集问题上一般是给定一个有重复数字或者，没有重复数字的数组，然后找到所有符合条件的不重复的子集。

本质上是一种深度优先搜索算法。

题解示例是给一个没有重复数字的数组，找出所有的不重复（顺序不同也被视为同一个答案）的子集。

```python
# Time: O(n * 2^n), Space: O(n)
def subsetsWithoutDuplicates(nums):
    subsets, curset = [], []
    helper(0, nums, subsets, curset)
    return subsets

def helper(i, nums, subsets, curset):
    if i >= len(nums):
        subsets.append(curset[:])
        return

    # include current i - index value
    curset.append(nums[i])
    helper(i + 1, nums, subsets, curset)
    # clear up to reset the status
    curset.pop()
    # not include i value
    helper(i + 1, nums, subsets, curset)
```

如果给出的数组中有重复数字，如何进行求解：和上面的情况不同之处在于，首先要对数组排序，然后在进行回溯的时候，一种是一个一个迭代添加，另一个情况是跳过所有重复的元素。仅此不同而已。

```python
# Time: O(n * 2^n), Space: O(n)
def subsetsWithDuplicates(nums):
    nums.sort()
    subsets, curset = [], []
    helper(0, nums, subsets, curset)
    return subsets

def helper(i, nums, subsets, curset):
    if i >= len(nums):
        subsets.append(curset[:])
        return

    # include i index value
    curset.append(nums[i])
    helper(i + 1, nums, subsets, curset)
    # clear up to reset the status
    curset.pop()
    # not include i value with all the duplicates
    while i + 1 < len(nums) and nums[i + 1] == nums[i]:
        i += 1
    helper(i + 1, nums, subsets, curset)
```

### leetcodes

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
