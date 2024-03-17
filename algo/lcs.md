## 动态规划：最长公共子序列（LCS）

---

### 概念引导

最长公共子序列（Longest Common Subsequence，简称LCS）是指两个序列中都存在的最长的子序列。这个子序列**不要求连续**，只要在原序列中的相对顺序保持一致即可。这与最长公共子串（Longest Common Substring）不同，后者要求子序列必须是**连续**的。

例如，对于序列"ABCBDAB"和"BDCAB"，它们的最长公共子序列是"BCAB"。

LCS可以通过动态规划算法来求解。算法的核心思想是构建一个二维数组，用来记录两个序列之间的最长公共子序列的长度。数组中的每个元素`dp[i][j]`表示序列A的前i个元素与序列B的前j个元素之间的最长公共子序列的长度。

具体的动态规划过程如下：
1. 初始化一个二维数组`dp`，其维度为`(m+1) x (n+1)`，其中m和n分别是两个序列的长度。
2. 从左上角开始，逐行逐列地填充`dp`数组。如果序列A的第i个元素与序列B的第j个元素相等，则`dp[i][j] = dp[i-1][j-1] + 1`；否则，`dp[i][j] = max(dp[i-1][j], dp[i][j-1])`，即取左方和上方的最大值。
3. 最终`dp[m][n]`就是两个序列的最长公共子序列的长度。

得到`dp`数组后，可以通过回溯的方式找到具体的最长公共子序列。

LCS算法的应用非常广泛，其中一些例子包括：
1. **文本相似度比较：** 可以用LCS算法来比较两个文本的相似度，找出它们之间的最长公共子序列。
2. **版本控制系统：** 版本控制系统中常用LCS算法来比较两个版本之间的差异，以便合并修改或回滚。
3. **基因序列比对：** 在生物信息学中，LCS被用于比较基因序列，找出它们之间的共同特征。
4. **视频压缩：** 在视频编码中，可以使用LCS来找出连续视频帧之间的共同部分，从而实现更好的压缩效果。

**小感想**：

动态规划这种题，一开始你解了几个泛用情况，你以为你会了，不其实你还不会，今天感觉其实很多题你开始觉得会啦，然后改变了条件你又不会了，那你还是没会，哪里没会？？核心思想。比如动态规划你要去找子问题

Python的算法实现：

DFS：除了冗余的计算，这种方法如此清晰。因为通过dfsHelper的递归的部分，清晰地定义了小问题。

```python
# Time: O(2^(m + n), Space: O(m + n))
def dfs(s1, s2):
    return dfsHelper(s1, s2, 0, 0)

def dfsHelper(s1, s2, i1, i2):
    if i1 == len(s1) or i2 == len(s2):
        return 0

    if s1[i1] == s2[i2]:
        return 1 + dfsHelper(s1, s2, i1 + 1, i2 + 1)
    else:
        return max(dfsHelper(s1, s2, i1 + 1, i2),
                   dfsHelper(s1, s2, i1, i2 + 1))
```

一般来说这种问题有两个变量，改成Memoization问题也是两个变量，如果出现了三个变量的情况会使得情况变得非常复杂，并且是更难的问题，很少出现。将这两个变量放进一个cache中就是memoization的解法：

```python
# Time: O(n * m), Space: O(n + m)
def memoization(s1, s2):
    N, M = len(s1), len(s2)
    cache = [[-1] * M for _ in range(N)]
    return memoHelper(s1, s2, 0, 0, cache)

def memoHelper(s1, s2, i1, i2, cache):
    if i1 == len(s1) or i2 == len(s2):
        return 0
    if cache[i1][i2] != -1:
        return cache[i1][i2]

    if s1[i1] == s2[i2]:
        cache[i1][i2] = 1 + memoHelper(s1, s2, i1 + 1, i2 + 1, cache)
    else:
        cache[i1][i2] = max(memoHelper(s1, s2, i1 + 1, i2, cache),
                            memoHelper(s1, s2, i1, i2 + 1, cache))
    return cache[i1][i2]
```

接着就是进化到动态规划的解法：

```python
# Time: O(n * m), Space: O(n + m)
def dp(s1, s2):
    N, M = len(s1), len(s2)
    dp = [[0] * (M + 1) for _ in range(N + 1)]

    for i in range(N):
        for j in range(M):
            if dp[i] == dp[j]:
                dp[i + 1][j + 1] = 1 + dp[i][j]
            else:
                dp[i + 1][j + 1] = max(dp[i + 1][j], dp[i][j + 1])
    return dp[N][M]       
```

优化的动态规划算法，将计算空间压缩为两行。

```python
# Time: O(n * m), Space: O(m)
def optimizedDp(s1, s2):
    N, M = len(s1), len(s2)
    dp = [0] * (M + 1)

    for i in range(N):
        curRow = [0] * (M + 1)
        for j in range(M):
            if s1[i] == s2[j]:
                curRow[j + 1] = 1 + dp[j]
            else:
                curRow[j + 1] = max(dp[j + 1], curRow[j])
        dp = curRow
    return dp[M]       
```

### leetcode解析

- 115题不同的子序列（[题目链接](https://leetcode.com/problems/distinct-subsequences/description/)）
