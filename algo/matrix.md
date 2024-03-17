## 数据结构：矩阵Matrix

---
### 适用范围

在数据结构中我很少看到单独讲矩阵的，突然看到了拿出来讲的很新鲜，但矩阵确实是一个很重要的工具：对于矩阵的操作主要包括旋转，缩放，遍历等。而且已经经历过很多矩阵的应用场景。

- 数字图像处理，对图像进行缩放加工，在深度学习卷积神经网络也是通过对图像进行矩阵的各种处理。
- 作为图的一种结构方式用来解决问题。是一种邻接表。
- 在动态规划中，作为网格工具储存已经计算过的结果。
- 解方程和线性代数的数学工具，一些网格类游戏的解决工具。

### 问题1:Set Matrix Zeros

题目描述很简单，一个矩阵中任何位置发现了零，则将该元素所在的行和列，都变成零，有一种病毒传播的感觉。暂且称为零的传播哈哈。

一种简单的做法是重新构建一个矩阵，对原矩阵进行遍历，如果发现了零，则将新的矩阵的相对的行列都设置为零，最终再将所有元素拿回来。可以想象如果矩阵很大，就会有很高的时间和空间复杂度。

实现步骤：

- 初始化几个变量，行长度，列长度，frow和fcol为布尔值，通过遍历第一行列标注是否有零存在。
- 从第二行第二列的元素开始遍历，当找到一个零元素，就对它所在的行和列的第一行第一列的对应位置的元素，设置为零。（这里的核心思想我想就是，将第一行和第一列作为一种路标，在最后使用，但是为了知道原始的第一行第一列是否有零，所以设置了frow和fcol变量）
- 检查第一列和第一行的元素，将为零的行和列都转换为零。
- 检查frow和fcol变量，如果有为True，将对应的第一行或者第一列转换为零。

代码实现：

```python
def set_matrix_zeros(mat):
    rows, cols = len(mat), len(mat[0])
    frow, fcol = False, False
    for i in range(rows):
        if mat[i][0] == 0:
            fcol = True
            break
    for j in range(cols):
        if mat[0][j] == 0:
            frow = True
            break

    for r in range(1, rows):
        for c in range(1, cols):
            if mat[r][c] == 0:
                mat[0][c], mat[r][0] = 0, 0

    for i in range(1, rows):
        if mat[i][0] == 0:
            for j in range(1, cols):
                mat[i][j] = 0
    for j in range(1, cols):
        if mat[0][j] == 0:
            for i in range(1, rows):
                mat[i][j] = 0

    if frow:
        for j in range(cols):
            mat[0][j] = 0
    if fcol:
        for i in range(rows):
            mat[i][0] = 0

    return mat
```

学习笔记：

中规中矩地按照步骤自己写了解答，和答案没有区别就不贴答案了，时间复杂度为O(mxn)，空间复杂度为O(1)，因为没有使用额外的空间。

### 问题2:Rotate Image

将一个正方形矩阵向右九十度旋转的问题。

以下是解决这个问题的步骤总结：

1. 运行一个循环，其中行数范围从0到n/2。
2. 在这个循环内部，运行一个嵌套循环，其中列数范围从行数到n - 行数 - 1。这些循环遍历矩阵中的四个单元格组。在这个嵌套循环中，我们执行三次交换：
   - 将左上角单元格的值与右上角单元格的值交换。
   - 将左上角单元格的值与右下角单元格的值交换。
   - 将左上角单元格的值与左下角单元格的值交换。
3. 当前的四个单元格组已经旋转了90度。我们现在移动到外部循环的下一次迭代，以旋转下一组。
4. 重复上述过程，直到整个矩阵都被旋转。

通过这些步骤，我们可以完成矩阵的顺时针旋转90度。

说实话这个步骤我很迷惑，比如对角线的右手边的两个元素要走一个Z字形，我理解了题解的方法是从可视化表示才理解的。

我更喜欢自己这个思路：

- 首先对矩阵进行转置操作。
- 然后对每一行进行左右逆转。

仅此而已。而且这两种方法的时间复杂度都是O(n^2)，空间复杂度都是O(1)。

首先是我觉得自己的这种比较好理解的方法：转置和左右逆。

```python
def rotate_image(matrix):
    n = len(matrix)
    for i in range(n):
        for j in range(i, n):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
    
    for i in range(n):
        matrix[i] = matrix[i][::-1]
    return matrix
```

以下是官方的三次旋转法的题解：

```python
def rotate_image(matrix):
    n = len(matrix)
    for row in range(n // 2):
        for col in range(row, n - row - 1):
            matrix[row][col], matrix[col][n - 1 - row] = matrix[col][n - 1 - row], matrix[row][col]
            matrix[row][col], matrix[n - 1 - row][n - 1 - col] = matrix[n - 1 - row][n - 1 - col], matrix[row][col]
            matrix[row][col], matrix[n - 1 - col][row] = matrix[n - 1 - col][row], matrix[row][col] 
    return matrix
```

学习笔记：矩阵的算法还是挺快乐的。

### 问题3:Spiral Matrix

对一个矩阵进行螺旋遍历，从左上角开始顺时针。将遍历的数字存储在数组中。简单来说就是一个矩阵螺旋遍历的问题。

以下是代码的步骤方法总结：

- 初始化变量：获取输入矩阵的行数和列数，并初始化指示当前位置的行号和列号变量（起始位置在左上角元素的左边也就是（0，-1）），以及一个表示遍历方向的变量（direction），并初始化结果列表（result）为空。
- 循环遍历：在矩阵的行数和列数都大于0的情况下，执行循环。每次循环中，都先向右遍历一行，然后向下遍历一列，再向左遍历一行，最后向上遍历一列。这个过程可以让遍历轨迹呈螺旋状。
- 更新行列数和方向：每次遍历完一行或一列后，更新剩余的行数和列数，并改变遍历方向（direction *= -1），以便下一次循环中可以沿着正确的方向进行遍历。
- 返回结果：遍历完成后，返回结果列表。
这个算法以较为直观的方式实现了螺旋矩阵的顺时针遍历功能，而不需要复杂的数据结构或算法。

代码以及对照答案修改过后：

```python
def spiral_order(matrix):
    rows, cols = len(matrix), len(matrix[0])
    r, c = 0, -1
    direction = 1
    result = []

    while rows > 0 and cols > 0:
        for _ in range(cols):
            c += direction
            result.append(matrix[r][c])
        rows -= 1

        for _ in range(rows):
            r += direction
            result.append(matrix[r][c])
        cols -= 1

        direction *= -1
    
    return result
```

学习笔记：时间复杂度是遍历的长度也就是O(m x n)，也就是整个矩阵的大小。空间复杂度为O(1)，因为没有使用额外的空间。这是一道很不错的题。查了一下还有一种活用内置函数`zip()`来模拟顺时针遍历螺旋矩阵的过程。好聪明。具体步骤如下：

1. 每次取出矩阵的第一行，并将其添加到结果列表中。
2. 然后，将矩阵逆时针旋转90度，并重复步骤1，直到矩阵为空。

以下是这种方法的Python代码实现：

```python
def spiral_order(matrix):
    result = []
    while matrix:
        result += matrix.pop(0)
        matrix = list(zip(*matrix))[::-1]
    return result
```

这种方法简洁明了，利用了Python内置函数的强大功能，而不需要手动控制遍历方向。
