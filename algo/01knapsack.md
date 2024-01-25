## 动态规划：0-1背包问题（0-1 knapsack）

---

### 概念引导

背包问题最初在书上看到的是小偷偷东西，背包有限，面对既有条件，有限的背包容量，可以偷的商品数量，和对应的价值，怎么装包可以让价值最大化的问题。

它是动态规划问题的经典问题。通过一个简单的状态转移方程，可以迅速求解。从暴力解法，到内存解法，到动态规划数组，到优化的动态规划数组，是一个层层递进的算法优化过程。

暴力求解，其实相当于二叉树问题，每一个商品都决定要不要装包，在暴力循环中找到最大利润。

```python
def dfs(profit, weight, capacity):
    return dfsHelper(0, profit, weight, capacity)

def dfsHelper(i, profit, weight, capacity):
    if i == len(profit):
        return 0
    
    # not include i
    skip = dfsHelper(i + 1, profit, weight, capacity)
    # include i
    newCapacity = capacity - weight[i]
    if newCapacity >= 0:
        include = profit[i] + dfsHelper(i + 1, profit, weight, newCapacity)
        maxprofit = max(skip, include)

    return maxprofit
```

内存法，其实是在暴力求解的基础上的优化，因为暴力求解中会有很多的重复计算，所以将他们存入内存数组，可以节省时间和内存。
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
