## 回溯算法：排列（Combinations）

---

### 它是什么

顾名思义就是数学上的排列组合。

```python

```

### leetcode 逐行解析

- 排列[leetcode77 题目描述](https://leetcode.com/problems/combinations/description/)

在代码中自己碰到的最重要的感触就是，helper函数中前面两个return条件的顺序很重要。必须先将符合条件的当前comb添加到最终结果，然后再判断当前的i是不是超出范围了，不然就会丢失很多结果。

```python
class Solution(object):
    def combine(self, n, k):
        """
        :type n: int
        :type k: int
        :rtype: List[List[int]]
        """
        res = []
        curcomb = []

        def helper(i, curcomb, res, n, k):
            if len(curcomb) == k:
                res.append(curcomb[:])
                return
            if i > n:
                return

            curcomb.append(i)
            helper(i + 1, curcomb, res, n, k)
            curcomb.pop()

            helper(i + 1, curcomb, res, n, k)

        helper(1, curcomb, res, n, k)
        return res
```

方法二：

```python
```

- 排列求和[leetcode39 题目描述](https://leetcode.com/problems/combination-sum/description/)

每个数字都可以重复使用的数列，给一个sum目标，求出符合条件的排列组合。

这道题也是绕了好久，主要在于内部函数自己脑内没有区分清楚是i是index还是数字本身，所以有一个清晰的逻辑在算法中太重要了。

```python
class Solution(object):
    def combinationSum(self, candidates, target):
        """
        :type candidates: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        res = []
        total = 0
        curset = []
        def dfs(i, curset, total):
            if total == target:
                res.append(curset[:])
                return
            if i >= len(candidates) or total > target:
                return

            curset.append(candidates[i])
            dfs(i, curset, total + candidates[i])
            curset.pop()
            dfs(i + 1, curset, total)
        dfs(0, curset, total)
        return res
```

- 电话号码的字母组合[leetcode17 题目描述](https://leetcode.com/problems/letter-combinations-of-a-phone-number/description/)


