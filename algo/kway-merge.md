## 算法应用：K-way Merge 问题

---
### 问题解释和适用范围

K-way Merge 问题是一个经典的排序和合并算法问题，它的目标是将 K 个有序数组合并成一个有序数组。这个问题在诸如外部排序、数据库查询优化、日志合并等领域都有广泛的应用。

举例来说，假设有 K 个有序数组：

```
Array 1: [2, 5, 8, 12]
Array 2: [3, 6, 9, 11]
Array 3: [1, 4, 7, 10] 
```

K-way Merge 问题的目标是将这 K 个有序数组合并成一个有序数组：(听起来很简单)

[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

解决 K-way Merge 问题的一种常见方法是使用最小堆（Min Heap）数据结构。具体步骤如下：

- 将 K 个有序数组的首个元素放入一个最小堆中，同时记录元素所属的数组编号。
- 从最小堆中取出堆顶元素（即当前 K 个数组中最小的元素），将其放入输出数组中，并从所属数组中取出下一个元素放入最小堆中。
- 重复上述步骤，直到所有的元素都被取出并放入输出数组中为止。

这种方法的时间复杂度是 O(N log K)，其中 N 是所有输入数组中的元素总数。这是因为每次插入和删除操作都需要 O(log K) 的时间，总共需要进行 N 次插入和删除操作。

当然也不全是使用堆，还有双指针技术。

K-way Merge 问题是一个非常有用和常见的问题，对于处理大规模数据的场景具有重要意义。

### 问题1:Merge Sorted Array

有序数组合并问题，要求将两个已排序的数组合并为一个有序数组，本身题很简单，如果按照上面一个部分的步骤来解题就可以但是，这道题中规定不可以使用多余空间。

举例来说，假设有两个已排序的数组，他们长这样：

```
Array 1: [1, 3, 5, 7，0，0，0，0，0]
Array 2: [2, 4, 6, 8, 9]
```

结果应该是[1, 2, 3, 4, 5, 6, 7, 8, 9]。注意到第一个数组中零的长度是第二个数组的长度。将第二个数组合并到第一个数组，并且不使用额外空间，覆盖第一个数组中的零的位置。

解决思路：

- 初始化两个指针 p1, p2 分别指向两个数组的最后一个非零元素。
- 再初始化一个指针 p，指向第一个（长的，融合对象）数组的最后一个元素。
- 比较 p1 和 p2 的大小，将 p 的数值设置为两者中较大的那个。（因为是从后向前比较，所以选择最大的排在后面）
- 遍历直到第二个数组的数字都遍历结束融合。

题解尝试：问题很简单，但是会忽视一种 nums1 里的数字都遍历完了但是 nums2 还有剩余的情况，所以我在一次比较后，将nums2剩余的数字进行了移动。（关注第二次while循环。）

```python
def merge_sorted(nums1, m, nums2, n):
    pointer_1 = m - 1
    pointer_2 = n - 1
    pointer = m + n - 1

    while pointer_2 >= 0 and pointer_1 >= 0:
        if nums1[pointer_1] > nums2[pointer_2]:
            nums1[pointer] = nums1[pointer_1]
            pointer_1 -= 1
        else:
            nums1[pointer] = nums2[pointer_2]
            pointer_2 -= 1
        pointer -= 1

    while pointer_2 >= 0:
        nums1[pointer] = nums2[pointer_2]
        pointer_2 -= 1
        pointer -= 1
        
    return nums1


res = merge_sorted([1, 2, 3, 0, 0, 0], 3, [4, 5, 6], 3)
print(res)
```

题解答案：我喜欢他这个答案，在条件判断中包括了所有的内容，在第二个条件中判定如果 p1 不再大于零，就直接走else将 p2 的元素不断加入 p 的位置就可以了，学习了。

```python
def merge_sorted(nums1, m, nums2, n):
    p1 = m - 1  
    p2 = n - 1 
    for p in range(n + m - 1, -1, -1):
        if p2 < 0:
            break
        if p1 >= 0 and nums1[p1] > nums2[p2]:
            nums1[p] = nums1[p1]
            p1 -= 1
        else:
            nums1[p] = nums2[p2]
            p2 -= 1
    return nums1
```
学习笔记：这道题的时间复杂度为O(m+n)因为进行了一次遍历，空间复杂度为O(1)因为没有使用额外的数组等空间，只用了几个指针。

### 问题2:Kth Smallest Number in M Sorted Lists

这道题目要求找到 M 个已排序列表中第 K 小的数字。如果有重复则视为不同的要素，如果input为空则返回零，如果input的总元素不足k，则返回最后一个最大的元素。在leetcode中是一道median难度的题。

一种解决这个问题的方法是使用最小堆（Min Heap）。解题思路：

- 将每个列表的第一个元素（最小的元素）和它所属的列表索引加入堆中。
- 不断从堆中弹出最小的元素，并将对应列表的下一个元素加入堆中。
- 直到弹出了第 K 个最小的元素为止。
- 如果列表用完也不够 K 个元素，则返回最后一个弹出的元素即可。

代码尝试：思路很清晰，简单的通过了所有的示例。在第一次遍历L的时候记得判断是否为空。

```python
from heapq import *


def k_smallest_number(lists, k):
    minheap = []
    counter = 0
    # track list index, element index
    for li, L in enumerate(lists):
        if L:
            heappush(minheap, [L[0], li, 0])

    if not minheap:
        return 0

    while minheap:
        num, li, ei = heappop(minheap)
        counter += 1
        L = lists[li]
        if ei < len(L) - 1:
            heappush(minheap, [L[ei + 1], li, ei + 1])

        if counter == k:
            return num

    return num


res = k_smallest_number([[2, 6, 8], [3, 7, 10], [5, 8, 11]], 5)
print(res)
```

附上所给题解：除了写法稍有不同，其他没什么不一样的。思路一样。仅作参考，不做分析。

```python
def k_smallest_number(lists, k):
    list_length = len(lists)
    kth_smallest = []

    for index in range(list_length):
        if len(lists[index]) == 0:
            continue
        else:
            heappush(kth_smallest, (lists[index][0], index, 0))

    numbers_checked, smallest_number = 0, 0
    while kth_smallest:
        smallest_number, list_index, num_index = heappop(kth_smallest)
        numbers_checked += 1

        if numbers_checked == k:
            break

        if num_index + 1 < len(lists[list_index]):
            heappush(
                kth_smallest, (lists[list_index][num_index + 1], list_index, num_index + 1))

    return smallest_number
```

学习笔记：时间复杂度来说，第一次遍历将 m 个列表的第一个元素推入堆，是 O(mlogm)，第二次遍历弹出 k 个元素，同时要推入元素，是 O(klogm)，因此总体的时间复杂度是 O((m + k)logm)。空间复杂度，只使用了一个大小为 m 的最小堆，所以为 O(m)。
