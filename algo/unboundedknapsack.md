## 动态规划：无限背包问题（Unbounded Knapsack）

---

### 概念引导

无限背包是0-1背包的拓展。加上一个，物品可以无限拿的条件。

它也是动态规划问题的经典问题。通过一个简单的状态转移方程，可以迅速求解。从暴力解法，到内存解法，到动态规划数组，到优化的动态规划数组，是一个层层递进的算法优化过程。

暴力求解。

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
        include = profit[i] + dfsHelper(i, profit, weight, newCapacity)
        maxprofit = max(skip, include)

    return maxprofit
```

这里和0-1问题的唯一不同就是倒数第三行，可以再次包含项目i。

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
        include = profit[i] + memoHelper(i, profit, weight, newCapacity, cache)
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
                include = profit[i] + dp[i][c - weight[i]]
            dp[i][c] = max(skip, include)

    return dp[N - 1][M] 
```

动态数组的优化方法：考虑到其实每次我们只使用了上一行的数组，所以可以再次缩小空间复杂度，只需要两行来解决问题。优化后的方法其实和原本的动态规划数组方法一样，只是将dp改为一行，每次初始化一个curRow当前数组，更新完当前数组后，将dp改写为当前数组，继续更新下一行而已。

```python
def optimizedDp(profit, weight, capacity):
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
                include = profit[i] + curRow[c - weight[i]]
            curRow[c] = max(skip, include)
        dp = curRow

    return dp[M]
```

这里唯一不同点是倒数第三行不用dp而是用当前行的数值。

### leetcode逐行解析

- 322题硬币找零（[题目链接](https://leetcode.com/problems/coin-change/description/)）

这道题也是一道典型的动态规划题目。给定一组硬币，和一个总金额，使用最少数量的硬币数量，组成这个目标金额，如果无法完成就返回-1。

动态规划的几个步骤：
- 1， 定义状态
- 2， 初始化
- 3， 确定状态转移方程
- 4， 遍历计算
- 5， 返回结果

针对这道题：

1. **定义状态：** 定义一个一维数组 `dp`，其中 `dp[i]` 表示组成金额 `i` 的最小硬币数量。

2. **初始化：** 将 `dp[0]` 初始化为 0，因为组成金额为 0 时，不需要任何硬币。

3. **状态转移方程：** 对于每个金额 `i`，我们考虑每一种硬币的情况，选择其中硬币数量最小的情况。状态转移方程为：`dp[i] = min(dp[i], dp[i - coin] + 1)`，其中 `coin` 为硬币的面值。加一是因为要加上这个硬币的数量1。

4. **遍历计算：** 从金额 1 开始，一直计算到目标金额 `amount`。

5. **返回结果：** 返回 `dp[amount]`，即组成目标金额的最小硬币数量。

Python 代码：

```python
def coinChange(coins, amount):
    # Initialize dp array with values representing infinity
    dp = [float('inf')] * (amount + 1)
    
    # Base case: 0 coins needed to make amount 0
    dp[0] = 0

    # Iterate through all amounts from 1 to target amount
    for i in range(1, amount + 1):
        # Consider each coin and update the dp array
        for coin in coins:
            if i - coin >= 0:
                dp[i] = min(dp[i], dp[i - coin] + 1)

    # If dp[amount] is still infinity, it means it's not possible to make the amount
    return dp[amount] if dp[amount] != float('inf') else -1

# Example usage:
coins = [1, 2, 5]
amount = 11
result = coinChange(coins, amount)
print(result)
```

这个动态规划解法的时间复杂度为 O(amount * len(coins))，其中 `amount` 是目标金额，`len(coins)` 是硬币的种类数量。
