## 动态规划：0-1背包问题（0-1 knapsack）

---

### 概念引导

背包问题最初在书上看到的是小偷偷东西，背包有限，面对既有条件，有限的背包容量，可以偷的商品数量，和对应的价值，怎么装包可以让价值最大化的问题。

它是动态规划问题的经典问题。通过一个简单的状态转移方程，可以迅速求解。从暴力解法，到内存解法，到动态规划数组，到优化的动态规划数组，是一个层层递进的算法优化过程。

暴力求解，其实相当于二叉树问题，每一个商品都决定要不要装包，在暴力循环中找到最大利润。即使是暴力解法，也显示出了递归的美，通过不断呼出helper函数，拿回skip当前物品，和include当前物品的两种情况下的利润，来进行比较，以取得最大利润的值。

从第0个物品开始递归。dfs函数的目的主要是为了呼出helper函数，没有也没关系。关注helper函数的实现内容。

```
0 
- skip 
  1 - skip
      2 - skip
      2 - include
    - include
      2 - skip
      2 - include
      ...
- include
  1 - skip
      2 - skip
      2 - include
    - incude
      2 - skip
      2 - include
      ...
```

通过以上的二叉树递归实现过程和如下的代码实现过程，可以看出暴力求解简明扼要，但是中间存在大量的重复计算。

```python
def dfs(profit, weight, capacity):
    return dfsHelper(0, profit, weight, capacity)

def dfsHelper(i, profit, weight, capacity):
    if i == len(profit):
        return 0
    
    # not include i
    skip = dfsHelper(i + 1, profit, weight, capacity)
    # include i
    # 更新背包容量
    newCapacity = capacity - weight[i]
    if newCapacity >= 0:
        include = profit[i] + dfsHelper(i + 1, profit, weight, newCapacity)
        maxprofit = max(skip, include)

    return maxprofit
```

内存法，其实是在暴力求解的基础上的优化，通过上面的结构图可知，暴力求解中会有很多的重复计算，所以将他们存入内存数组，可以节省时间和内存。
从代码中就可以看出，实质上就是将计算结果存入了cache数组，这为后面的动态规划打下基础。

```python
def memoization(profit, weight, capacity):
    N, M = len(profit), capacity
    cache = [[-1] * (M + 1) for _ in range(N)]
    return memoHelper(0, profit, weight, capacity, cache)

def memoHelper(i, profit, weight, capacity, cache):
    if i == len(profit):
        return 0
    if cache[i][capacity] != -1:
        return cache[i][capacity]
    
    # not include i
    skip = memoHelper(i + 1, profit, weight, capacity, cache)
    # include i
    newCapacity = capacity - weight[i]
    if newCapacity >= 0:
        include = profit[i] + memoHelper(i + 1, profit, weight, newCapacity, cache)
        cache[i][capacity] = max(skip, include)
    
    return cache[i][capacity]
```

动态规划，数组填充。

```python
def dp(profit, weight, capacity):
    N, M = len(profit), capacity
    dp = [[0] * (M + 1) for _ in range(N)]

    # fill the edge case
    for i in range(N):
        dp[i][0] = 0
    for c in range(M + 1):
        if c >= weight[0]:
            dp[0][c] = profit[0]

    for i in range(1, N):
        for c in range(1, M + 1):
            skip = dp[i - 1][c]
            # include
            include = 0
            if c - weight[i] >= 0:
                include = profit[i] + dp[i - 1][c - weight[i]]
            dp[i][c] = max(skip, include)

    return dp[N - 1][M] 
```

动态数组的优化方法：考虑到其实每次我们只使用了上一行的数组，所以可以再次缩小空间复杂度，只需要两行来解决问题。优化后的方法其实和原本的动态规划数组方法一样，只是将dp改为一行，每次初始化一个curRow当前数组，更新完当前数组后，将dp改写为当前数组，继续更新下一行而已。

```python
def dp(profit, weight, capacity):
    N, M = len(profit), capacity
    dp = [0] * (M + 1)

    # fill the edge case
    for c in range(M + 1):
        if c >= weight[0]:
            dp[c] = profit[0]

    for i in range(1, N):
        curRow = [0] * (M + 1)
        for c in range(1, M + 1):
            skip = dp[c]
            # include
            include = 0
            if c - weight[i] >= 0:
                include = profit[i] + dp[c - weight[i]]
            curRow[c] = max(skip, include)
        dp = curRow

    return dp[M]
```

### leetcode逐行解析

- 子数组和相等[leetcode416 题目描述](https://leetcode.com/problems/partition-equal-subset-sum/description/)

给定一个数组，判断如果将数组分为两个子数组，这两个子数组的sum是否能相等。返回布尔值。

输入输出示例：

```
Input: nums = [1,5,11,5]
Output: true
Explanation: The array can be partitioned as [1, 5, 5] and [11].
```

关键的部分是将所有可能的目标和（也就是nums和的一半）全都存储在一个hashset数据结构中。

```python
class Solution(object):
    def canPartition(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        if sum(nums) % 2:
            return False
        # 目标就是找到这个和的一半
        target = sum(nums) // 2
        # 初始化一个hashset数据结构，添加一种情况（什么数字都不选的情况和为0）作为初始
        dp = set()
        dp.add(0)
        # 循环不断找到所有可能的和，并进行判断
        for i in range(len(nums)):
            nextdp = set()
            for t in dp:
                if t == target:
                    return True
                nextdp.add(t)
                nextdp.add(t + nums[i])
            dp = nextdp
        return False
```

还有另一种动态规划网格的写法不是很懂所以手动推演一下：

```python
class Solution(object):
    def canPartition(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        if sum(nums) % 2:
            return False
        target = sum(nums) // 2
        dp = [False] * (target + 1)
        dp[0] = True

        for num in nums:
            for j in range(target, num - 1, -1):
                dp[j] = dp[j] or dp[j - num]

        return dp[target]
```

假设输入是[1, 2, 5, 2]，target就是5。dp数组就是[True, False, False, False, False, False]。一共有5个False。

第一次遍历从num是1，j从5到1倒着遍历。

```
dp[5] = dp[5] or dp[4] = False
dp[4] = dp[4] or dp[3] = False
dp[3] = dp[3] or dp[2] = False
dp[2] = dp[2] or dp[1] = False
dp[1] = dp[1] or dp[0] = True

dp = [True, True, False, False, False, False]
```

可以看到dp到index1更新为了True，说明可以构成sum为1的子数组。这是肯定的因为遍历num为1，这个时候只要取一个num也就是1就够了。

下面遍历num为2的情况。j从5到2倒着遍历。

```
dp[5] = dp[5] or dp[3] = False
dp[4] = dp[4] or dp[2] = False
dp[3] = dp[3] or dp[1] = True
dp[2] = dp[2] or dp[0] = True

dp = [True, True, True, True, False, False]
```

下面的情况也是以此类推。遍历num为5的情况。j从5到5倒着遍历。

```
dp[5] = dp[5] or dp[0] = True

dp = [True, True, True, False, False, True]
```

这里已经可以得到dp[target]也就是True的结果了。通过遍历数组中的每个元素 num，以及从 target 到 num 的每个可能的和 j（这样我们可以避免重复计算），更新 dp[j] 的值。具体地说，dp[j] 表示我们是否可以找到一些元素的子集，它们的和等于 j。我们尝试用当前元素 num 来更新 dp[j]，如果我们可以找到一些之前的元素构成和为 j - num 的子集，那么我们也可以将当前元素添加到这个子集中，得到和为 j 的子集。因此，dp[j] 的更新规则是 dp[j] = dp[j] or dp[j-num]。

这就是一种拆小为大的思想，整个dp数组实际上计算了所有可能的target内的数字的布尔结果。

总的来说第一种方法更好理解一些和更省资源。

**反思和总结**：总的来说这道题很棒，让我在过程中感受到了务必要用一种解决子问题的观点看动态规划问题，而不仅仅是网格。网格只是表象，将大问题拆分成小问题，才是主要的思想核心。
