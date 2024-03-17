## 有序数组的算法：二分查找（Binary search）

---
### 概念引导

二分查找（Binary Search）是一种在有序数组或列表中快速定位目标值的搜索算法。它通过反复将有序数组分成两半，然后确定目标值可能在哪一半中，从而将搜索范围缩小一半。这个过程不断重复，直到找到目标值或确定目标值不在数组中。

二分查找的基本步骤如下：

1. 确定搜索范围的起始点（通常是整个数组）和终点。
2. 计算中间元素的索引。
3. 比较中间元素与目标值的大小。
   - 如果中间元素等于目标值，则找到了目标，算法结束。
   - 如果中间元素大于目标值，则目标值可能在左半部分，缩小搜索范围到左半部分。
   - 如果中间元素小于目标值，则目标值可能在右半部分，缩小搜索范围到右半部分。
4. 重复步骤2和步骤3，直到找到目标值或搜索范围为空。

二分查找的时间复杂度是 O(log n)，其中 n 是数组的元素个数。由于每次比较都能将搜索范围缩小一半，因此它在大型有序数据集中具有高效的性能。这使得二分查找在查找任务中比线性搜索更为高效。

### Python代码实现

按照上面的步骤逐步实现即可。

```python
def binary_search(arr, target):
    left, right = 0, len(arr) - 1

    while left <= right:
        mid = (left + right) // 2
        
        if target > mid:
            left = mid + 1
        elif target < mid:
            right = mid - 1
        else:
            return mid
    return -1
```

还有一种`search range`的实现更常用，也就是在条件判断的时候插入一个函数进行判断。

它是一种二分查找的模板，查找在一个范围内是否存在这个

```python
# Binary search on some range of values
def binarySearch(low, high):

    while low <= high:
        mid = (low + high) // 2

        if isCorrect(mid) > 0:
            high = mid - 1
        elif isCorrect(mid) < 0:
            low = mid + 1
        else:
            return mid
    return -1

# Return 1 if n is too big, -1 if too small, 0 if correct
def isCorrect(n):
    if n > 10:
        return 1
    elif n < 10:
        return -1
    else:
        return 0
```

如果用递归如何实现：

```python
def binary_search(nums, low, high, target):

    if (low > high):
        return -1

    # Finding the mid using integer division
    mid = low + (high - low) // 2

    # Target value is present at the middle of the array
    if nums[mid] == target:
        return mid

    # If the target value is greater than the middle, ignore the first half
    elif nums[mid] < target:
        return binary_search(nums, mid + 1, high, target)

    # If the target value is less than the middle, ignore the second half
    return binary_search(nums, low, mid - 1, target)
```

### 二分查找的变体问题1 First Bad Version

在一个有序数组（1到n个版本号组成的数组）中存在第一个bad version，同时有一个检测器返回版本号v是否是bad，问题要求通过二分查找的方法，找到第一个bad version的编号，和操作次数（这里当然就是logn次的计算）

```python
import main as api_call
def is_bad_version(v):
    return api_call.is_bad(v)
```

其实二分查找已经很熟悉了，这里因为计算mid的时候方法不一样，所以标记出来。

```python
def first_bad_version(n):
    first, last = 1, n
    counter = 0
    while first < last:
        # 注意这里的计算方法
        mid = first + (last - first) // 2
        if is_bad_version(mid):
            last = mid
        else:
            first = mid + 1
        counter += 1
    return first, counter
```

在这个代码中，mid如此计算是为了避免整数溢出问题，并且确保在二分搜索过程中得到正确的中间索引值。

当使用 `(first + last) // 2` 计算 mid 时，如果 `first` 和 `last` 都是非常大的整数，它们的和可能会导致整数溢出。为了避免这种情况，使用 `(last - first) // 2` 来计算 mid。这样做的好处是保证了mid在合理的范围内，避免了整数溢出的问题。

使用 `(last - first) // 2` 计算 mid，仍然可以确保 mid 是居中的索引值，因为 `(last - first) // 2` 表示了 first 和 last 之间的距离的一半，再加上 first 就得到了 mid。这样可以保证在每次迭代中，搜索区间被正确地减半，确保算法的正确性。

### 二分查找变体问题2 Search in Rotated Sorted Array

数组不完全是顺序排列的，可能有一部分被移动了。比如数组：[1, 3, 4, 5, 6, 9]被调整为：[5, 6, 9, 1, 3, 4]。对于这样的数组，同样要找到目标数字的下标的问题。

```python
def binary_search_rotated(nums, target):
    low, high = 0, len(nums) - 1

    while low <= high:
        # mid = (low + high) // 2
        mid = low + (high - low) // 2
        if nums[mid] == target:
            return mid
        
        # 这里的要包括等于的情况，下面也是，在这里犯错了
        if nums[low] <= nums[mid]:
            if nums[low] <= target < nums[mid]:
                high = mid - 1
            else:
                low = mid + 1
        elif nums[mid] <= nums[high]:
            if nums[mid] < target <= nums[high]:
                low = mid + 1
            else:
                high = mid - 1
    return -1
```

整个流程分解为以下步骤：

- 将数组一分为二。
- 检查左半边是否有序，如果是的：
  - 检查target是否在这半边，是的话缩小范围到左半边进行二分搜索即可。
  - 如果不是，则排除这半边。缩小范围。
- 上面的情况为否，则检查右半边是否有序，如果是的：
  - 检查target是否在这半边，是的话缩小范围到右半边进行二分搜索即可。
  - 如果不是，则排除这半边，缩小范围。
- 没找到值返回-1。

对我来说以上步骤都很明晰，但是处理边界问题的时候出现了问题，我的思考是：**要注意将所有可能的情况都包括进去**

还有一种是递归方法的代码：大同小异，步骤一样的：

```python
def binary_search_rotated(nums, target):
    return binary_search(nums, 0, len(nums) - 1, target)

def binary_search(nums, low, high, target):

    if low > high:
        return -1
    
    mid = low + (high - low) // 2
    if nums[mid] == target:
        return mid
        
    if nums[low] <= nums[mid]:
        if nums[low] <= target < nums[mid]:
            return binary_search(nums, low, mid - 1, target)
        return binary_search(nums, mid + 1, high, target)
    elif nums[mid] <= nums[high]:
        if nums[mid] < target <= nums[high]:
            return binary_search(nums, mid + 1, high, target)
        return binary_search(nums, low, mid - 1, target)
```

### 问了一下大模型二分查找的应用场景

二分查找广泛应用于各种领域，尤其是在需要高效查找有序数据集的情况下。以下是一些二分查找的应用场景：

1. **查找算法：** 二分查找是一种经典的查找算法，用于在有序数组或列表中查找特定元素。

2. **排序算法的一部分：** 二分查找常用于一些排序算法中的子步骤，例如快速排序。

3. **数据库检索：** 在数据库中，如果数据按某个列有序存储，可以使用二分查找来加速检索特定值的记录。

4. **文件系统搜索：** 在文件系统中，如果文件名按字母顺序排列，可以使用二分查找来快速定位文件。

5. **游戏开发：** 在游戏中，当需要在有序列表中查找某个元素时，二分查找可以提高查找效率。

6. **网络路由表查找：** 在网络路由中，二分查找可用于在路由表中快速定位目标地址。

7. **统计学和数据分析：** 在某些统计和数据分析任务中，二分查找用于在有序数据集中找到中位数等特定值。

8. **图形学：** 在计算机图形学中，例如在光线追踪算法中，可以使用二分查找来确定光线与场景中物体的交点。

9. **金融领域：** 在金融领域，二分查找可用于在按时间顺序排序的数据中查找特定的交易或事件。

总的来说，二分查找是一种通用且高效的算法，适用于许多需要在有序数据中查找特定值的场景。


