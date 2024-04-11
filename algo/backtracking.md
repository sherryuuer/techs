## 回溯算法：Backtracking 相关问题

---
### 问题概述

回溯算法是一种解决问题的算法，通常用于在一个大的问题空间中搜索所有可能的解，直到找到满足特定条件的解为止。它通常用于解决组合优化、排列组合、棋盘游戏、图搜索等问题。

回溯算法的基本思想是通过递归地尝试所有可能的解，当发现当前解无法满足条件时，就回溯到上一个状态，尝试其他可能的解。其核心是"试错"，在搜索过程中，不断地选择一个候选解，并检查是否满足问题的条件，如果满足则继续搜索下去，如果不满足则进行回溯，撤销上一步的选择，尝试其他的选择。

通常，回溯算法包括三个重要部分：

- 选择Choose：根据问题的特性，选择一个候选解进行尝试。
- 约束Constraint：对选择的候选解进行约束条件的检查，判断是否满足问题的要求。
- 回溯Backtrack：如果当前选择的候选解不满足约束条件，就进行回溯，撤销上一步的选择，尝试其他的选择。

回溯算法的框架通常是递归的，每次递归调用代表一次选择过程。在递归调用中，需要在每一步做出选择，并进行约束检查。当找到符合条件的解时，算法停止，并返回结果；当遍历完所有可能的选择后，仍未找到满足条件的解时，算法也会停止。

虽然回溯算法通常是一种暴力搜索方法，但是在实际应用中，通过一些剪枝策略可以减少搜索空间，提高算法的效率。但是回溯方法比暴力破解更好，因为我们不必生成所有可能的解决方案并从中选择所需的解决方案。它为我们提供了在每次可能的递归调用时检查所需条件的选项。如果满足条件，我们将继续探索这条道路。如果不是，我们会退后一步，探索另一条路。通过这种方式，我们可以避免生成冗余的解决方案。

### 问题1:Flood Fill

这个问题也可以归类为[图结构：矩阵和邻接表的DFS和BFS](algo/graphs.md)问题。

Flood Fill（洪水填充）是一种经典的图像处理算法，用于将图像中的连通区域填充为指定的颜色。该算法通常用于**图像编辑软件中的填充工具**（interesting），也用于计算机图形学中的图像处理和区域分割任务。

在 Flood Fill 算法中，给定一个起始点（seed point）和目标颜色，该算法会将起始点所在的连通区域中所有的像素都填充为目标颜色。填充过程是通过搜索相邻像素的方式进行的，当相邻像素的颜色与起始点的颜色相同，并且未被填充过时，就将其填充为目标颜色，并将其加入到填充队列中继续搜索。

Flood Fill 算法可以用递归或非递归的方式实现。在递归实现中，从起始点开始向四个方向（上、下、左、右）搜索相邻像素，并对相邻像素进行颜色填充，然后递归调用自身来处理相邻像素的连通区域。而非递归实现通常采用队列数据结构来存储待处理的像素，通过循环遍历队列中的像素并进行颜色填充，直到队列为空为止。

这道题中是一个矩阵，给定的填充是一个target的数字。以下是力扣给的例子，源点是（1，1）的坐标，目标颜色的数字是 2，最终结果是和源点坐标邻接的位置，都转换为 2。

```
Input: image = [
    [1,1,1],
    [1,1,0],
    [1,0,1]
    ], sr = 1, sc = 1, color = 2
Output: [
    [2,2,2],
    [2,2,0],
    [2,0,1]]
```

解题思路：

- 首先判断当前点 (row, col) 是否越界或者颜色已经改变，若是则直接返回。
- 将当前点 (row, col) 的颜色改为新的颜色 new_color。
- 分别递归调用 flood_fill() 函数填充当前点的上、下、左、右四个方向的相邻区域。
- 返回填充完成后的矩阵。

代码尝试：

```python
def flood_fill(grid, sr, sc, target):
    source = grid[sr][sc]
    visit = set()
    def dfs(grid, r, c, source, target):
        
        if r < 0 or c < 0 \
        or r >= len(grid) or c >= len(grid[0]) \
        or grid[r][c] != source \
        or (r, c) in visit:
            return

        grid[r][c] = target
        visit.add((r, c))
        dfs(grid, r - 1, c, source, target)
        dfs(grid, r + 1, c, source, target)
        dfs(grid, r, c - 1, source, target)
        dfs(grid, r, c + 1, source, target)

    dfs(grid, sr, sc, source, target) 

    return grid  
```

参考答案题解：同样是深度优先搜索，遍历的时候将坐标提取出来了。

```python
def flood_fill(grid, sr, sc, target):
    if grid[sr][sc] == target:
        return grid
    else:
        old_target = grid[sr][sc]
        grid[sr][sc] = target
        dfs(grid, sr, sc, old_target, target)

        return grid


def dfs(grid, row, col, old_target, new_target):
    adjacent_cells = [[0, 1], [1, 0], [-1, 0], [0, -1]]

    grid_length = len(grid)
    total_cells = len(grid[0])

    for cell_value in adjacent_cells:
        i = row + cell_value[0]
        j = col + cell_value[1]

        if i < grid_length and i >= 0 and j < total_cells and j >= 0 and grid[i][j] == old_target:
            grid[i][j] = new_target
            dfs(grid, i, j, old_target, new_target)
```

补充，这道题还可以用 stack 进行迭代的方法，是一种优化的方法，尝试一下：

```python
def flood_fill(grid, sr, sc, target):
    source = grid[sr][sc]
    stack = [(sr, sc)]
    visit = set()

    while stack:
        r, c = stack.pop()

        if r < 0 or c < 0 \
        or r >= len(grid) or c >= len(grid[0]) \
        or grid[r][c] != source \
        or (r, c) in visit:
            continue

        grid[r][c] = target
        visit.add((r, c))
        stack.append((r - 1, c))
        stack.append((r + 1, c))
        stack.append((r, c - 1))
        stack.append((r, c + 1)) 

    return grid  
```
顺利通过，是一种优化的方法。还是很喜欢stack。

学习笔记：时间和空间复杂度都是O(m x n)。

### 问题2:Word Search

经典的搜索问题，在一个二维字符网格中查找给定的单词是否存在。单词可以在网格中按顺序横向或纵向连续出现，但不能跨越网格中的字符。

具体来说，给定一个二维字符网格和一个单词，需要判断是否能在网格中找到单词的完整路径。路径的方向可以是水平或垂直的，但不能是对角线方向。每个单元格中的字符只能使用一次。

例如，给定一个网格和单词 "ABCCED"：

```
[
  ['A','B','C','E'],
  ['S','F','C','S'],
  ['A','D','E','E']
]
```

可以看到单词 "ABCCED" 存在于网格中，路径为 (0,0) -> (0,1) -> (0,2) -> (1,2) -> (2,2) -> (2,1) -> (2,0)。

Word Search问题的解法通常使用深度优先搜索（DFS）算法来搜索可能的路径。我们从网格中的每个字符开始，尝试构建以该字符为起点的路径，递归地向四个方向探索，直到找到完整的单词或无法继续探索为止。这就是完整了解题思路了。回溯问题的解决思路很好说清楚，剩下的就是逻辑处理了。

代码尝试：本着先尝试再修改的原则。

```python
def word_search(grid, word):

    def dfs(grid, visit, r, c, idx, word):
        if r < 0 or c < 0 \
                or r >= len(grid) or c >= len(grid[0]) \
                or (r, c) in visit \
                or idx > len(word) - 1:
            return
        if grid[r][c] == word[idx] and idx == len(word) - 1:
            return True

        visit.add((r, c))

        res = dfs(grid, visit, r - 1, c, idx + 1, word) or \
            dfs(grid, visit, r + 1, c, idx + 1, word) or \
            dfs(grid, visit, r, c - 1, idx + 1, word) or \
            dfs(grid, visit, r, c + 1, idx + 1, word)
        if res:
            return True

        visit.remove((r, c))  # 回溯时要移除访问标记

        return False

    for row in range(len(grid)):
        for col in range(len(grid[0])):
            if grid[row][col] == word[0]:
                print(row, col)

                if not dfs(grid, set(), row, col, 0, word):
                    continue
                else:
                    return True
    return False


grid = [
    ["N", "W", "L", "I", "M"],
    ["V", "I", "L", "Q", "O"],
    ["O", "L", "A", "T", "O"],
    ["R", "T", "A", "I", "N"],
    ["O", "I", "T", "N", "C"]
]
word = "LATIN"

res = word_search(grid, word)
print(res)
```

然后进行代码优化：

- 让代码更快地判断返回，可以在一开始就判断是否到了最后一个字符，那么就可以直接返回 True，节省了不必要的空间和时间。
- 四个方向的遍历使用direction列表进行。
- 回溯不要忘记remove元素的步骤。
- 主程序的判断部分也没必要那么复杂，直接符合条件返回True即可。

```python
def word_search(grid, word):
    def dfs(grid, visit, r, c, idx, word):
        if idx == len(word) - 1:
            return True

        visit.add((r, c))

        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < len(grid) and 0 <= nc < len(grid[0]) and (nr, nc) not in visit and grid[nr][nc] == word[idx + 1]:
                if dfs(grid, visit, nr, nc, idx + 1, word):
                    return True

        visit.remove((r, c))

        return False

    for row in range(len(grid)):
        for col in range(len(grid[0])):
            if grid[row][col] == word[0]:
                if len(word) <= 1 or dfs(grid, set(), row, col, 0, word):
                    return True

    return False
```

参考答案给的代码：结构和上面的一样。

```python
# Function to search a specific word in the grid
def word_search(grid, word):
    n = len(grid)
    m = len(grid[0])
    for row in range(n):
        for col in range(m):
            if depth_first_search(row, col, word, 0, grid):
                return True
    return False

# Apply backtracking on every element to search the required word
def depth_first_search(row, col, word, index, grid):
    if len(word) == index:
        return True

    if row < 0 or row >= len(grid) or col < 0 or col >= len(grid[0]) \
            or grid[row][col] != word[index]:
        return False

    temp = grid[row][col]
    grid[row][col] = '*'

    for rowOffset, colOffset in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
        if depth_first_search(row + rowOffset, col + colOffset, word, index + 1, grid):
            return True

    grid[row][col] = temp
    return False
```

学习笔记：

这道题的时间复杂度取决于两个方面：

1. 遍历整个二维网格：这一步的时间复杂度为 O(m*n)，其中 m 和 n 分别是网格的行数和列数。

2. 在每个可能的起始位置上进行深度优先搜索：最坏情况下，我们需要在网格的每个位置上都进行深度优先搜索。在每个搜索过程中，我们最多需要搜索目标单词的长度个位置。因此，深度优先搜索的时间复杂度为 O(m*n*l)，其中 l 是目标单词的长度。

综合考虑以上两个方面，整体的时间复杂度为 O(m*n*l)。

至于空间复杂度，主要取决于深度优先搜索的递归调用栈以及记录已访问位置的集合。在最坏情况下，递归调用栈的深度为目标单词的长度，因此空间复杂度为 O(l)。另外，记录已访问位置的集合也会消耗额外的空间，其大小最多为网格的大小，即 O(m*n)。因此，整体的空间复杂度为 O(m*n + l)。

### 问题3:N-Queens

N皇后问题是一个经典的组合问题，旨在找到在N×N的棋盘上放置N个皇后，使得它们互相不受攻击。在国际象棋中，皇后可以沿着水平线、垂直线和对角线移动，因此在解决N皇后问题时，需要确保任何两个皇后都不在同一行、同一列或同一对角线上。

这个问题是一个NP难问题，在一般情况下没有有效的多项式时间算法来解决。通常采用回溯算法来解决N皇后问题。回溯算法是一种深度优先搜索算法，它尝试每一种可能的解决方案，并在遇到无效解时进行回溯，尝试其他的路径。

N皇后问题的解决方案通常包括以下步骤：
1. 定义棋盘：创建一个N×N的棋盘，其中每个格子表示一个位置，初始状态下所有位置都为空。
2. 回溯搜索：从第一行开始，在每一行选择一个位置放置皇后，并检查是否满足皇后不相互攻击的条件。如果满足条件，则继续递归地放置下一行的皇后；如果不满足条件，则回溯到上一行，尝试其他位置。
3. 终止条件：当所有皇后都成功放置在棋盘上时，得到一个有效解；当所有可能的位置都尝试完毕时，回溯到上一行，继续尝试其他解决方案，直到找到所有解或者没有解为止。

N皇后问题的解决方案数量随着N的增加呈指数级增长，因此对于较大的N，求解可能会非常耗时。对于较小的N，可以通过回溯算法在合理的时间内找到所有解决方案。

代码尝试：

```python
def solve_n_queens(n):

    def is_safe(row, col, queens):
        # queues is the list for all the quenes that has been added
        # check if put a new queue at (row, col) is safe
        for r, c in queens:
            if row == r or col == c or abs(row - r) == abs(col - c):
                return False
        return True

    def backtracking(row, queens):
        # check from rowth
        if row == n:
            res.append(queens[:])
            return
        for col in range(n):
            if is_safe(row, col, queens):
                queens.append((row, col))
                backtracking(row + 1, queens)
                queens.pop()

    res = []
    backtracking(0, [])

    return len(res)
```

给出的题解参考和我上述的方法类似，使用回溯试错重来。

```python
def is_valid_move(proposed_row, proposed_col, solution):
    for i in range(0, proposed_row):
        old_row = i
        old_col = solution[i]
        diagonal_offset = proposed_row - old_row
        if (old_col == proposed_col or
            old_col == proposed_col - diagonal_offset or
                old_col == proposed_col + diagonal_offset):
            return False
            
    return True

def solve_n_queens_rec(n, solution, row, results):
    if row == n:
        results.append(solution[:])
        return

    for i in range(0, n):
        valid = is_valid_move(row, i, solution)
        if valid:
            solution[row] = i
            solve_n_queens_rec(n, solution, row + 1, results)

# Function to solve N-Queens problem
def solve_n_queens(n):
    results = []
    solution = [-1] * n
    solve_n_queens_rec(n, solution, 0, results)
    return len(results)
```

学习笔记：

- 时间复杂度：
  - 在回溯函数中，每个皇后都要尝试放置在每一行的每一个列位置上，因此有 n 行和 n 列，总共有 n^2 种尝试的可能性。
  - 对于每个放置皇后的位置，都要检查之前已经放置的皇后位置，因此在 `is_safe` 函数中，需要遍历已经放置的皇后位置，时间复杂度为 O(n)。
  - 因此，整个算法的时间复杂度为 O(n^3)。

- 空间复杂度：
  - 空间复杂度主要取决于保存解的列表，以及递归过程中的栈空间。
  - 在递归过程中，每次递归调用都会创建一个新的列表 `queens`，其中保存了当前已经放置的皇后的位置。这些列表的长度最大为 n，因此递归过程中的栈空间最大为 O(n)。
  - 在保存解的列表中，最多会保存 n! 个解，每个解包含 n 个位置，因此解的总空间复杂度为 O(n^2 x n!)。
  - 因此，整个算法的空间复杂度为 $O(n^2 x n! + n)。

总的来说，这个算法的时间复杂度是 O(n^3)，空间复杂度是 O(n^2 x n! + n)。虽然时间复杂度相对较高，但是由于 N 皇后问题的规模通常较小（通常 n 值不会很大），因此这个算法在实践中是可行的。
