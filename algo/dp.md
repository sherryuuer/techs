## 动态规划（Dynamic Programming）

---
### 什么是动态规划

动态规划（Dynamic Programming，简称DP）是一种解决多阶段决策问题的数学优化方法，通常用于优化问题的求解。它将原问题分解为若干个子问题，并通过保存子问题的解来避免重复计算，从而提高效率。动态规划在解决许多最优化问题、路径规划问题等方面具有广泛的应用。

动态规划有两个核心特点：

1. **最优子结构（Optimal Substructure）：** 问题的最优解可以通过子问题的最优解来构建。换句话说，全局最优解可以通过局部最优解得到。

2. **重叠子问题（Overlapping Subproblems）：** 在解决问题的过程中，会反复遇到相同的子问题。动态规划通过存储子问题的解来避免重复计算，提高效率。

动态规划的一般解决步骤包括：

1. **定义状态：** 明确定义问题的状态，找到问题规模的变量。

2. **定义状态转移方程：** 找到问题状态之间的关系，建立状态转移方程，描述问题规模变化的规律。

3. **初始化：** 对问题规模最小的情况进行初始化，通常是边界条件。

4. **递推计算：** 从问题规模最小的情况开始，逐步递推计算出更大规模的问题的解。

5. **保存结果：** 在计算的过程中，保存子问题的解，避免重复计算。

6. **返回结果：** 根据最终问题规模的解得出原问题的解。

动态规划常见的应用场景包括：

- **最短路径问题：** 如Dijkstra算法和Floyd-Warshall算法。
- **最长公共子序列问题：** 用于比较两个序列的相似性。
- **背包问题：** 如0/1背包问题、多重背包问题。
- **编辑距离问题：** 用于计算两个字符串之间的相似度。
- **最优BST（二叉搜索树）：** 用于构建具有最小搜索代价的二叉搜索树。

总的来说，动态规划是一种通过将复杂问题分解成简单子问题并保存其解决方案来解决问题的方法，从而实现更高效的算法。

### 1D的动态规划问题：斐波那契数列

斐波那契数列的暴力破解方法如下：

```python
# Brute Force
def bruteForce(n):
    if n <= 1:
        return n
    return bruteForce(n - 1) + bruteForce(n - 2)
```

这个人们已经非常熟悉了，但是中间会重复发生大量计算重复的问题。所以通过内存优化的方法，可以将结果存储在cache里。

也就是进化版：

```python
# Memoization
def memoization(n, cache):
    if n <= 1:
        return n
    if n in cache:
        return cache[n]

    cache[n] = memoization(n - 1) + memoization(n - 2)
    return cache[n]
```

这是内存版的动态规划，也可以称为自顶向下的动态规划。

而大家常说的真正的动态规划是自底向上的，在斐波那契数列问题上，其实计算过的历史，并不需要，空间复杂度其实可以降为2个空间的数组。

每次计算完对前两个元素的sum计算，最左边的元素空间就可以被丢弃了。

代码如下：

```python
# Dynamic Programming
def dp(n):
    if n < 2:
        return n

    dp = [0, 1]
    i = 2
    while i <= n:
        tmp = dp[1]
        dp[1] = dp[0] + dp[1]
        dp[0] = tmp
        i += 1
    return dp[1]
```

注意到在计算过程中，需要的空间只需要index-0和index-1两个位置上的元素而已。

### 2D动态规划：路径count问题

给一个4x4的Matrix，从左上到右下的所有可能路径，中间没有障碍物，只能往右边和下面移动。

暴力破解的方法依然是递归：要计算所有的路径数量，自顶向下，也就是从（0，0）出发，向右和向下的路径的总和。从而递归地找到所有的路径数量。

```python
# Brute Force - Time: O(2 ^ (n + m)), Space: O(n + m)
def bruteForce(r, c, rows, cols):
    if r == rows or c == cols:
        return 0
    if r == rows - 1 and c == cols - 1:
        return 1
    
    return (bruteForce(r + 1, c, rows, cols) +  
            bruteForce(r, c + 1, rows, cols))

print(bruteForce(0, 0, 4, 4))
```

cache的方法是暴力破解的进化，将重复计算的部分存储在和Matrix一样大小的2D矩阵中。

```python
# Memoization - Time and Space: O(n * m)
def memoization(r, c, rows, cols, cache):
    if r == rows or c == cols:
        return 0
    if cache[r][c] > 0:
        return cache[r][c]
    if r == rows - 1 and c == cols - 1:
        return 1
    
    cache[r][c] = (memoization(r + 1, c, rows, cols, cache) +  
        memoization(r, c + 1, rows, cols, cache))
    return cache[r][c]

print(memoization(0, 0, 4, 4, [[0] * 4 for i in range(4)]))
```

真动态规划是更进一步的简化，只需要一行的空间复杂度，一次更新一行，最终找到第一行的结果。

```python
# Dynamic Programming - Time: O(n * m), Space: O(m), where m is num of cols
def dp(rows, cols):
    prevRow = [0] * cols

    for r in range(rows - 1, -1, -1):
        curRow = [0] * cols
        curRow[cols - 1] = 1
        for c in range(cols - 2, -1, -1):
            curRow[c] = curRow[c + 1] + prevRow[c]
        prevRow = curRow
    return prevRow[0] 
```
