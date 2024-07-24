## 回溯算法：组合（Combinations）

---

### 概念

顾名思义就是数学上的排列组合。从n个数字1到n中抽取k个数字，返回所有的组合。因为是组合所以即使顺序不同也是同一个set。

```python
# Given n numbers (1 - n), return all possible combinations
# of size k. (n choose k math problem).
# Time: O(k * 2^n)
def combinations(n, k):
    combs, curComb = [], []
    helper(1, combs, curComb, n, k)
    return combs

def helper(i, combs, curComb, n, k):
    if len(curComb) == k:
        combs.append(curComb[:])
        return
    if i > n:
        return

    # include i
    curComb.append(i)
    helper(i + 1, combs, curComb, n, k)
    curComb.pop()
    # not include i
    helper(i + 1, combs, curComb, n, k)
```

但是这种解法存在问题就是，会出现很多空子集的输出，浪费了时间和内存。所以如果从i开始，下一步只提取i+1开始的元素，就会节省很多资源。

优化的解法，其实只有helper函数的后半部分不一样：

```python
# Time: O(k * C(n, k))
def combinations(n, k):
    combs, curComb = [], []
    helper(1, combs, curComb, n, k)
    return combs

def helper(i, combs, curComb, n, k):
    if len(curComb) == k:
        combs.append(curComb[:])
        return
    if i > n:
        return

    for j in range(i, n + 1):
        curComb.append(j)
        helper(j + 1, combs, curComb, n, k)
        curComb.pop()
```

这很像是n层**暴力循环loop**的解法，这里使用了回溯，其实暴力训练也可以用回溯的方式来写，所谓的回溯，其实就是**在使用完了一个元素后，将它弹出这个数组，然后进行下一个遍历和尝试**，希望之后会再有新的理解。暴力循环会非常暴力，比如下面这样：

```python
# 暴力破解法
def combinations(n, k):
    combs = []
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            if k == 2:
                combs.append([i, j])
            else:
                for l in range(j + 1, n + 1):
                    if k == 3:
                        combs.append([i, j, l])
                    else:
                        for m in range(l + 1, n + 1):
                            if k == 4:
                                combs.append([i, j, l, m])
                            # 继续增加嵌套循环来处理更高阶的组合数
                            # 以此类推...
    return combs

# 示例用法
n = 4
k = 2
print(combinations(n, k))
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
