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

### leetcodes

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

- 983题Minimum Cost For Tickets（[题目链接](https://leetcode.com/problems/minimum-cost-for-tickets/description/)）

题意理解你为下一年的旅行计划买火车票，旅行日期最大可以到365天，也就是一个`days`列表最大有365天，同时给出火车通票的列表`costs`代表1日，7日，30日的通票的价格。求解怎么买可以覆盖`days`列表的所有日期，还可以用最少的钱。

关键词是最少最大，或者多少种方法，就可能是提示，动态规划。

**动态规划并不代表需要将每个细度的结果都找到，也可以是一个hashset用来动态存储结果。**

示例input和output：

```
Input: days = [1,4,6,7,8,20], costs = [2,7,15]
Output: 11
Explanation: For example, here is one way to buy passes that lets you travel your travel plan:
On day 1, you bought a 1-day pass for costs[0] = $2, which covered day 1.
On day 3, you bought a 7-day pass for costs[1] = $7, which covered days 3, 4, ..., 9.
On day 20, you bought a 1-day pass for costs[0] = $2, which covered day 20.
In total, you spent $11 and covered all the days of your travel.
```

递归算法的代码：

```python
class Solution(object):
    def mincostTickets(self, days, costs):
        """
        :type days: List[int]
        :type costs: List[int]
        :rtype: int
        """
        dp = {}
        def dfs(i):
            if i == len(days):
                return 0
            if i in dp:
                return dp[i]

            dp[i] = float("inf")
            for d, c in zip([1, 7, 30], costs):
                # 计算出下一次开始旅行的天，需要不超出列表长度
                j = i
                while j < len(days) and days[j] < days[i] + d:
                    j += 1
                dp[i] = min(dp[i], c + dfs(j))
            return dp[i]

        return dfs(0)
```

动态规划解法是相似的：

```python
class Solution(object):
    def mincostTickets(self, days, costs):
        """
        :type days: List[int]
        :type costs: List[int]
        :rtype: int
        """
        dp = {}

        for i in range(len(days) - 1, -1, -1):
            dp[i] = float("inf")
            for d, c in zip([1, 7, 30], costs):
                j = i
                while j < len(days) and days[j] < days[i] + d:
                    j += 1
                dp[i] = min(dp[i], c + dp.get(j, 0))

        return dp[0]
```
---
这道问题是一个经典的动态规划问题，通常称为“旅行计划问题”。给定一组旅行日子 `days` 和对应的旅行费用 `costs`，我们需要确定最小的费用，以便在这些日子中旅行一次。

这段 Python 代码使用了动态规划来解决这个问题。下面我将逐步解释代码的工作原理：

1. `dp` 字典的初始化：在这个字典中，我们将存储每一天旅行的最小费用。开始时，我们将其初始化为一个空字典。

2. 逆向遍历旅行日子 `days`：为了方便动态规划的计算，我们从最后一天开始向前遍历旅行日子。

3. 在内部循环中，我们使用了三种不同类型的旅行方式（1 天、7 天和 30 天），并使用 `zip` 函数将其与相应的费用 `costs` 配对。对于每一种旅行方式，我们要计算从当前这一天开始的旅行费用。

4. 我们使用一个 `while` 循环来找到可以用于旅行的最后一天 `j`。我们找到的这一天需要满足两个条件：它必须在 `days` 中存在，且其日期小于当前这一天加上旅行天数 `d`。

5. 一旦找到了这一天 `j`，我们计算出从当前这一天到 `j` 这段时间的旅行费用 `c`，并将其加上从 `j` 处开始的动态规划值 `dp[j]`。

6. 最后，我们更新当前这一天的动态规划值 `dp[i]`，将其设置为找到的最小费用。如果没有找到适当的旅行方式，我们将当前这一天的费用设为无穷大（`inf`）。

7. 在完成所有的遍历后，我们返回第一天的动态规划值 `dp[0]`，这个值就是旅行计划的最小费用。

这段代码实现了一个动态规划解法，通过反向遍历旅行日子并使用动态规划表来存储已解决的子问题，避免了重复计算，从而有效地解决了这个问题。

