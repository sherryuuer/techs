## 堆和优先队列（Heap & Priority queue）

---
### 概念引导

堆（Heap）是一种特殊的树状数据结构，通常用于实现优先队列。堆分为最大堆和最小堆，它们有以下特性：

1. **最大堆（Max Heap）：** 在最大堆中，每个节点的值都大于或等于其子节点的值。根节点的值是整个堆中的最大值。

2. **最小堆（Min Heap）：** 在最小堆中，每个节点的值都小于或等于其子节点的值。根节点的值是整个堆中的最小值。

另外，二叉堆是一棵二叉树，是一棵完全二叉树，**除了最低层节点从左到右连续填充外，树的每一层都被完全填充**。也就是只有最后一行后面开始才能有空缺节点。

堆的一个关键性质是，对于任意节点 i，其父节点和子节点的索引之间存在一定的关系。对于索引 i 的节点：
（注意这里为了公示简化，内部设计从index1开始的，节点0设为dummy，和链表那时候一样）

- 父节点的索引是 `i // 2`
- 左子节点的索引是 `2i`
- 右子节点的索引是 `2i + 1`

堆的主要应用之一是实现优先队列。

**优先队列（Priority Queue）：** 优先队列是一种抽象数据类型，具有队列的特点，但是每个元素都有一个与之关联的优先级。优先级高的元素在队列中具有更高的优先级，因此在出队时会先被取出。

堆可以很好地实现优先队列的操作。在最小堆中，根节点始终是最小元素，而在最大堆中，根节点始终是最大元素。这使得在插入和删除操作时能够以 O(log n) 的时间复杂度保持堆的性质。

以下是堆和优先队列的基本操作：

1. **插入（Insert）：** 将一个元素插入到堆中。在插入后，需要保持堆的性质，通过上浮（在最小堆中）或下沉（在最大堆中）操作来实现。

2. **删除（Delete）：** 从堆中删除一个元素。在删除后，同样需要保持堆的性质，通过上浮或下沉操作来实现。

3. **获取最值（Get Min/Max）：** 获取堆中的最小值或最大值，这是优先队列的主要操作。

4. **合并（Merge）：** 将两个堆合并成一个堆。

这些操作使得堆和优先队列在许多场景中非常有用，例如调度算法、图算法（如Dijkstra最短路径算法）等。

### Python的最小堆

在Python中有一个最小堆实现的包非常好用。那就是

但是这里还是要用Python实现一下，感觉如果不同原理实现一下就很难理解。比如机器学习的算法也是。确实需要手动实现一下。

在这里的实现，堆其实是放在一个数组中实现的，通过索引的数学关系，进行各种操作。

```python
class MinHeap:
    def __init__(self):
        # 初始化堆为一个节点
        self.heap = [0]

    def push(self, val):
        # push新的节点到最后（以保证结构稳定）
        self.heap.append(val)
        i = len(self.heap) - 1

        # Percolate up：自底向上移动节点，直到满足父节点更大，不满足就和父节点互换
        # 时间复杂度是树的高度也就是logn
        while i > 1 and self.heap[i] < self.heap[i // 2]:
            self.heap[i], self.heap[i // 2] = self.heap[i // 2], self.heap[i]
            i = i // 2

    def pop(self):
        # 这里代表只有一个dummy节点没什么可以弹出的
        if len(self.heap) <= 1:
            return -1
        # 只有一个节点，直接弹出，新手注意，这里的pop是list的pop最后一个元素的那个内置方法
        if len(self.heap) == 2:
            return self.heap.pop()
        # 将最后要return的结果存储在res里
        res = self.heap[1]
        # 将heap数组的最后一个元素pop并放在，空缺的index-1位置（这样的做法是为了确保结构稳定）
        self.heap[1] = self.heap.pop()
        # 初始化index为1
        i = 1

        # 自顶向下，因为现在堆顶是一个从底下拿过来的数字，需要移动到正确的位置，满足顺序正确
        while i * 2 < len(self.heap):  # 当至少有一个左边节点的情况下进行循环
            # 如果有右边节点，并且右边节点小于左边节点，且右边节点小于父节点
            if i * 2 + 1 < len(self.heap) and self.heap[i * 2 + 1] < self.heap[i * 2] and self.heap[i * 2 + 1] < self.heap[i]:
                # 交换右边节点和父节点
                self.heap[i], self.heap[i * 2 +
                                        1] = self.heap[i * 2 + 1], self.heap[i]
                # 更新i的位置到右边节点的位置
                i = i * 2 + 1
            # 如果不满足上面的条件，则是在有左边节点的情况下，且左边节点小于父节点
            elif self.heap[i * 2] < self.heap[i]:
                # 那么交换父节点和左边节点
                self.heap[i], self.heap[i * 2] = self.heap[i * 2], self.heap[i]
                # 更新i的位置到左边节点的位置
                i = i * 2
            # 其他的情况则上述都不满足，则代表已经到了正确的位置
            else:
                break
        return res

    def top(self):
        # 取得最小元素但不弹出
        if len(self.heap) > 1:
            return self.heap[1]
        else:
            return -1

    def heapify(self, arr):
        # 为了简化数学计算，在开头加一个0
        arr = [0] + arr
        self.heap = arr
        # 只有一半的节点有子节点
        # 自顶向下比，自底向上更有效率，因为可以省掉一半的， 没有child的节点
        cur = len(self.heap) // 2

        # 自顶向下交换节点，使得顺序正确
        while cur > 0:
            i = cur
            # 这里的while交换和前面的pop里的是一样的
            while i * 2 < len(self.heap):
                if i * 2 + 1 < len(self.heap) and self.heap[i * 2 + 1] < self.heap[i * 2] and self.heap[i * 2 + 1] < self.heap[i]:
                    self.heap[i], self.heap[i * 2 +
                                            1] = self.heap[i * 2 + 1], self.heap[i]
                    i = i * 2 + 1
                elif self.heap[i * 2] < self.heap[i]:
                    self.heap[i], self.heap[i *
                                            2] = self.heap[i * 2], self.heap[i]
                    i = i * 2
                else:
                    break
            # 更新要进行换位操作的节点位置，直到开头的节点为止
            cur -= 1


# 由于自顶向下和自底向上的方法多次重用，所以这里将方法重写的的写法：
class MinHeap:
    def __init__(self):
        self.heap = [0]

    def _percolate_up(self, i):
        while i > 1 and self.heap[i] < self.heap[i // 2]:
            self.heap[i], self.heap[i // 2] = self.heap[i // 2], self.heap[i]
            i = i // 2

    def _percolate_down(self, i):
        while i * 2 < len(self.heap):  # at least have the left child
            # have the right , and right child is least than root and left
            if i * 2 + 1 < len(self.heap) and self.heap[i * 2 + 1] < self.heap[i * 2] and self.heap[i * 2 + 1] < self.heap[i]:
                # swap root with the right
                self.heap[i], self.heap[i * 2 +
                                        1] = self.heap[i * 2 + 1], self.heap[i]
                i = i * 2 + 1
            # left is least than the root
            elif self.heap[i * 2] < self.heap[i]:
                self.heap[i], self.heap[i * 2] = self.heap[i * 2], self.heap[i]
                i = i * 2
            # at the right position
            else:
                break

    def push(self, val):
        self.heap.append(val)
        i = len(self.heap) - 1

        # Percolate up
        self._percolate_up(i)

    def pop(self):
        if len(self.heap) <= 1:
            return -1
        if len(self.heap) == 2:
            return self.heap.pop()

        res = self.heap[1]
        # move the last val to the top
        self.heap[1] = self.heap.pop()
        i = 1

        # Percolate down
        self._percolate_down(i)
        return res

    def top(self):
        if len(self.heap) > 1:
            return self.heap[1]
        else:
            return -1

    def heapify(self, arr):
        # make the arr 's 0 th element to the end to make it a heap
        # arr.append(arr[0])  # this will cause index out of range
        arr = [0] + arr
        # now it is heap
        self.heap = arr
        # there are only half nodes have children
        # 从上往下推比，从下往上更有效率，因为可以省掉一半的， 没有child的nodes
        cur = len(self.heap) // 2

        # Percolate down
        while cur > 0:
            i = cur

            # Percolate down
            self._percolate_down(i)

            cur -= 1
```


### Python中的heapq包

在Python中，有一个名为`heapq`的内置模块，提供了对堆的支持。`heapq`模块实现了最小堆的功能，但是通过取反操作可以实现最大堆的效果。以下是一些`heapq`模块中常用的函数：

1. **heapify(iterable):** 将可迭代对象转换为堆。时间复杂度为 O(n)。

    ```python
    import heapq
    
    data = [3, 1, 4, 1, 5, 9, 2]
    heapq.heapify(data)
    print(data)  # 输出: [1, 1, 2, 3, 5, 9, 4]
    ```

2. **heappush(heap, elem):** 将元素压入堆。

    ```python
    import heapq
    
    heap = [1, 3, 5, 7, 9]
    heapq.heappush(heap, 4)
    print(heap)  # 输出: [1, 3, 4, 7, 9, 5]
    ```

3. **heappop(heap):** 弹出并返回堆中的最小元素。

    ```python
    import heapq
    
    heap = [1, 3, 4, 7, 9, 5]
    smallest = heapq.heappop(heap)
    print(smallest)  # 输出: 1
    print(heap)      # 输出: [3, 5, 4, 7, 9]
    ```

4. **heappushpop(heap, elem):** 先将元素压入堆，然后弹出并返回堆中的最小元素。

    ```python
    import heapq
    
    heap = [3, 5, 7, 9]
    result = heapq.heappushpop(heap, 4)
    print(result)  # 输出: 3
    print(heap)    # 输出: [4, 5, 7, 9]
    ```

5. **heapreplace(heap, elem):** 弹出并返回堆中的最小元素，然后将元素压入堆。

    ```python
    import heapq
    
    heap = [3, 5, 7, 9]
    result = heapq.heapreplace(heap, 4)
    print(result)  # 输出: 3
    print(heap)    # 输出: [4, 5, 7, 9]
    ```

这些函数使得在Python中使用堆变得非常方便，特别是在解决一些需要快速找到最小值或最大值的问题时。当我们需要最大堆的时候，只需要将数字乘以-1就可以实现，拿出来的时候再次乘以-1.最常用的事pop和push，两个足已。
