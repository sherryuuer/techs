## 堆数据结构的 Top K Elements 问题

---
### Top K Elements 问题

Top K Elements 问题是指在一个数据集合中，找出具有最大（或最小）前 K 个元素的问题。这个问题在数据处理和算法设计中经常遇到，解决这个问题有助于找出数据集中的关键信息或者快速筛选出重要的元素。

通常，Top K Elements 问题可以分为两类：

1. **最大的 K 个元素**：在一个数据集合中，找出具有最大值的前 K 个元素。

2. **最小的 K 个元素**：在一个数据集合中，找出具有最小值的前 K 个元素。

解决 Top K Elements 问题的常用方法包括：

- **排序后取前 K 个元素**：将数据集合排序，然后取出前 K 个（最大或最小）元素。这种方法的时间复杂度通常为 O(n log n)，其中 n 是数据集合的大小。

- **堆（Heap）**：使用堆结构来高效地找出前 K 个元素。可以使用最大堆来找出最大的 K 个元素，也可以使用最小堆来找出最小的 K 个元素。这种方法的时间复杂度通常为 O(n log K)，其中 n 是数据集合的大小。

- **快速选择算法（QuickSelect）**：这是一种改进的快速排序算法，它可以在平均情况下以线性时间复杂度 O(n) 的时间复杂度内找出第 K 小（或第 K 大）的元素。因此，可以使用快速选择算法来解决 Top K Elements 问题。

- **计数排序**：对于特定范围内的整数数据集合，可以使用计数排序来快速找出前 K 个元素。计数排序的时间复杂度为 O(n + k)，其中 n 是数据集合的大小，k 是数据范围。

Top K Elements 问题在各种场景下都有应用，例如在数据分析、机器学习、搜索引擎优化等领域。通过快速找出最重要的元素，可以帮助人们更好地理解数据集合的特征和趋势，以及进行进一步的分析和决策。

以上结解说我很能纳得。但是在我学习的材料里，使用堆解决问题的时候它如此描述：

let’s look at how this pattern takes steps to solve the problem of finding the top k largest elements (using min-heap) or top k smallest elements (using max-heap).

对，在解决k个最大问题的时候，使用最小堆，在解决k个最小问题的时候，使用最大堆。

我觉得是来自己语言说法的漏洞，其实他们是两个问题，一个是解决前k个元素，一个是解决第k个元素。如果是前k个所有的元素的话，那么用最大堆一直弹出k个即可，如果是第k个那就不一样了，因为最大堆只能弹出最大的，无法弹出第k大的，只有用最小堆，限制堆的大小为k，将大的元素不断加进去，就可以取到这个堆的最小元素，也就是第k大的元素了。（英语中elements使用了复数，迷惑了我很久，直到开始做例题）

### 问题1:Kth Largest Element in a Stream

这个问题是指在一个数据流中，动态地找出第 K 大的元素。也就是说，随着数据流的不断输入，我们需要实时地找出第 K 大的元素。

解决 "Kth Largest Element in a Stream" 问题的一种常见方法是使用最小堆（Min Heap）。题中要求实现两个方法，一个是初始化数据结构`init`另一个是`add`方法，当加入一个新元素的时候动态的返回第 K 大的元素。

具体编码步骤如下：

- 首先创建一个大小为 K 的最小堆，堆中存放当前流中的前 K 大元素（使用之后要定义的add方法）。
- 当新元素进入数据流时，我们将其与堆顶元素（堆中最小的元素）进行比较。
- 如果新元素大于堆顶元素，则将堆顶元素替换为新元素，并重新调整堆，保持堆的性质。
- 如果新元素小于等于堆顶元素，则不进行任何操作。
- 在任何时刻，堆顶元素即为当前数据流中的第 K 大元素。

编码练习：

```python
import heapq

class KthLargest:
    # Constructor to initialize heap and add values in it
    def __init__(self, k, nums):
        self.size = k
        self.heap = []
        for n in nums:
            self.add(n)

    # Adds element in the heap and return the Kth largest
    def add(self, val):
        if len(self.heap) < self.size:
            heapq.heappush(self.heap, val)
        elif val > self.heap[0]:
            heapq.heappop(self.heap)
            heapq.heappush(self.heap, val)
        return self.heap[0] if len(self.heap) == self.size else -1
```
通过测试，和答案没什么区别。

学习笔记：这种方法的时间复杂度为 O(logK)，因为在调整堆的过程中，堆的高度不会超过 logK。空间复杂度就是使用的额外的堆的大小长度为K的堆所以是O(K)。这是一道有趣的题。

### 问题2:Reorganize String

题目要求重新排列给定字符串中的字符，使得任意两个相邻的字符都不相同。如果这样的重新排列是可能的，则返回任意一种可能的重新排列结果；如果不可能，则返回空字符串。

举个例子，假设输入字符串为 "aab"，则返回 "aba" 是一个有效的重新排列，因为相邻字符都不相同。而输入字符串 "aaab"，则无法找到有效的重新排列，因此返回空字符串 ""。

解决步骤：（似乎是贪心）

- 使用一个hashmap统计输入字符串中每个字符出现的频率。
- 根据以上的hashmap字符频率构建一个最大堆（或优先队列）。
- 遍历堆，从堆中取出频率最高的两个字符，将它们加入结果字符串，并将它们的频率减一，然后重新放回堆中。
- 重复步骤3。
- 最后的判断如果剩下的一个字符频率大于1，则说明无法满足要求，返回空字符串。

编码尝试：下面的代码看起来很长，是因为我写了三次。这三次都是很快乐的尝试和学习。

- 第一次我一次迭代一个元素，根据直觉写了整个过程，虽然很复杂很多重复代码，但是是一个正确的方案。
- 第二个函数我改用一次弹出两个元素的方法，让代码更加清晰简洁了一些。
- 第三个函数我将第二个函数中进行判断的`if res and res[-1] == char1[1]:`的部分去掉了，因为在leetcode中我看到其实不需要这一个步骤，因为根据最大堆的性质，不需要进行显式的比较也可以满足条件。比如第一次输出了a和b分别是最大和次大的元素，这两个都减去count一次，在第二次迭代的时候a必然在b的前面，不可能出现a和a相邻的情况。

```python
# importing libraries
from collections import Counter
import heapq


def reorganize_string(str):
    res = ''
    count = Counter(list(str))
    count_list = []
    for c, v in count.items():
        heapq.heappush(count_list, [-1 * v, c])

    while len(count_list) > 1:

        if not res:
            char = heapq.heappop(count_list)
            res += char[1]
            char[0] += 1
            if char[0] != 0:
                heapq.heappush(count_list, char)

        else:
            char1 = heapq.heappop(count_list)
            if res[-1] == char1[1]:
                char2 = heapq.heappop(count_list)
                res += char2[1]
                char2[0] += 1
                if char2[0] != 0:
                    heapq.heappush(count_list, char2)
                heapq.heappush(count_list, char1)
            else:
                res += char1[1]
                char1[0] += 1
                if char1[0] != 0:
                    heapq.heappush(count_list, char1)
        print(count_list)

    return res + count_list[0][1] if count_list[0][0] == -1 else ''


def reorganize_string2(str):
    res = ''
    count = Counter(str)
    count_list = [[-v, c] for c, v in count.items()]
    heapq.heapify(count_list)

    while len(count_list) >= 2:
        char1 = heapq.heappop(count_list)
        char2 = heapq.heappop(count_list)

        if res and res[-1] == char1[1]:
            res += char2[1]
            res += char1[1]
        else:
            res += char1[1]
            res += char2[1]

        char1[0] += 1
        char2[0] += 1

        if char1[0] != 0:
            heapq.heappush(count_list, char1)
        if char2[0] != 0:
            heapq.heappush(count_list, char2)

    # 这里的条件比较复杂
    if count_list and count_list[0][0] != -1:
        return ''
    if count_list:
        return res + count_list[0][1]
    return res


str = 'fofjjb'
res = reorganize_string2(str)
print(res)


def reorganize_string3(str):
    res = ''
    count = Counter(str)
    count_list = [[-v, c] for c, v in count.items()]
    heapq.heapify(count_list)

    while len(count_list) >= 2:
        char1 = heapq.heappop(count_list)
        char2 = heapq.heappop(count_list)

        res += char1[1]
        res += char2[1]

        char1[0] += 1
        char2[0] += 1

        if char1[0] != 0:
            heapq.heappush(count_list, char1)
        if char2[0] != 0:
            heapq.heappush(count_list, char2)

    if count_list and count_list[0][0] != -1:
        return ''
    if count_list:
        return res + count_list[0][1]
    return res
```

下面是一个题解给的答案：它使用一个previous存储前一次的迭代的字符，而且它是一个一个字符迭代的和我的方法不同。可以作为一个不同的视角，不管是一次迭代两个还是一个，核心思想不变都是使用最大堆进行的贪心算法。

关于每次迭代，它这里如何满足和前面的字符不相等：
  - 取出频率最高的字符，并将其放入结果字符串。
  - 更新堆中字符的频率，并重新构建最大堆。
  - 再次从堆中取出频率最高的字符，并将其放入结果字符串。
  - 检查上一次取出的字符与当前取出的字符是否相同，如果相同，则从堆中取出频率第二高的字符，并将其放入结果字符串。
  - 重复步骤2至4，直到堆中没有字符。

```python
# importing libraries
from collections import Counter
import heapq

def reorganize_string(str):

    char_counter = Counter(str)
    most_freq_chars = []

    for char, count in char_counter.items():
        most_freq_chars.append([-count, char])

    heapq.heapify(most_freq_chars)

    previous = None
    result = ""

    while len(most_freq_chars) > 0 or previous:

        if previous and len(most_freq_chars) == 0:
            return ""

        count, char = heapq.heappop(most_freq_chars)
        result = result + char
        count = count + 1

        if previous:
            heapq.heappush(most_freq_chars, previous)
            previous = None

        if count != 0:
            previous = [count, char]

    return result
```

学习笔记：这种方法保证了相邻字符不相同，并且尽可能地利用了高频率的字符。但需要注意的是，如果输入字符串中存在超过一半的相同字符，则无法重新排列使得相邻字符不相同，此时返回空字符串。

通过三次代码迭代，找到问题，找到思维盲点的过程非常有趣。

时间复杂度的分析：将字符推入堆的时间复杂度是O(logc)，c是字符的数量。因为字符数量有上限26，所以可以作为常数，需要遍历整个字符串所以是O(nlogc)，总的来说就是O(n)。空间复杂度上不管是hashmap还是堆都是有c上限的数据结构，也就是常数。所以空间复杂度为O(1)。

### 问题3:K Closest Points to Origin

题意很简单，给出平面上的一组点的坐标，找出前k个距离原点最近的点。

解题思路：

- 遍历将点推入最小堆。
- 计算点到原点的距离。
- 最后pop返回堆的k个元素即可。

这道题可以说比上面几个更清晰简单。

代码尝试：

```python
import heapq


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.dst = (x**2 + y**2)**0.5


def k_closest(points, k):
    minheap = []
    for p in points:
        point = Point(p[0], p[1])
        heapq.heappush(minheap, [point.dst, p])

    res = []
    for _ in range(k):
        dst, p = heapq.heappop(minheap)
        res.append(p)

    return res


points = [[-1, -3], [-4, -5], [-2, -2], [-2, -3]]
print(k_closest(points, 3))  
```
这个代码应该是正确的，在本地测试OK但是console告诉我有问题，我loop了input对象才发现里面都是点对象，`<point.Point object at 0x7fe0cabcdf40>`意料之外，他们是不可比较的，所以在类中定义了一个计算距离的method。

所以修改代码：

修改后的代码通过了第一批用例，但是在第二批中出现了列表中的点完全相等的情况。这样在push的时候会导致第一个元素比较距离发现完全相等，于是瞬移比较第二个元素，导致比较的元素因为是instance而报错的情况。于是我使用enumerate，将他们的index加入minheap，使用index作为第二位比较对象，就没问题了。

```python
import heapq


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    # implemented
    def cal_distance(self):
        return (self.x**2 + self.y**2)**0.5


def k_closest(points, k):
    minheap = []
    for i, point in enumerate(points):  # point is instance
        # use i as the second element to avoid the type error
        heapq.heappush(minheap, [point.cal_distance(), i, point])

    res = []
    for _ in range(k):
        _, _, point = heapq.heappop(minheap)
        res.append(point)

    return res
```

答案示例：它这个答案在处理我刚刚的instance比较的问题上，使用了类内部的魔术方法`__lt__`，当需要进行比较的时候就会调用这个魔术方法。另外format方法我不再会用了，大部分都是用fstring。

```python
class Point:
    # __init__ will be used to make a Point type object
    def __init__(self, x, y):
        self.x = x
        self.y = y

    # __lt__ is used for max-heap
    def __lt__(self, other):
        return self.distance_from_origin() > other.distance_from_origin()

    # __str__ is used to print the x and y values
    def __str__(self):
        return '[{self.x}, {self.y}]'.format(self=self)

    # distance_from_origin calculates the distance using x, y coordinates
    def distance_from_origin(self):
        # ignoring sqrt to calculate the distance
        return (self.x * self.x) + (self.y * self.y)

    __repr__ = __str__

# main.py

from point import Point
import heapq

def k_closest(points, k):
    points_max_heap = []

    for i in range(k):
        heapq.heappush(points_max_heap, points[i])

    for i in range(k, len(points)):
        if points[i].distance_from_origin() \
         < points_max_heap[0].distance_from_origin():
            heapq.heappop(points_max_heap)
            heapq.heappush(points_max_heap, points[i])

    return list(points_max_heap)
```

考虑了一下我还是用上面的方法修改我自己的代码。原本我以为反正时间复杂度相差不大，但是其实还是有差别的。

全推进堆然后抽出答案的话时间负载度是线性遍历的 n 乘以堆计算的 logn，但是使用上面的方法构筑堆的话，时间复杂度是nlogk。n如果很大就不太好。另外这个方法不是使用最小堆而是使用最大堆，每次都比较堆顶的最大元素来决定是否插入新的元素，反复推入和推出是用了pushpop方法，还是区别不小而且很有意思！

所以修改一下：

```python
import heapq


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    # implemented
    def cal_distance(self):
        return (self.x**2 + self.y**2)**0.5


def k_closest(points, k):
    maxheap = []
    for i in range(k):
        heapq.heappush(maxheap, [-1 * points[i].cal_distance(), i, points[i]])

    for i in range(k, len(points)):
        if points[i].cal_distance() < -1 * maxheap[0][0]:
            heapq.heappushpop(maxheap, [-1 * points[i].cal_distance(), i, points[i]])

    return [p[2] for p in maxheap]
```

OK

学习笔记：如上所述，时间复杂度是O(nlogk)，空间复杂度是O(k)也就是额外的堆大小。快乐的三道题。
