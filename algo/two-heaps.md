## 堆算法：两个堆（Two Heaps）

---

### 概念

简单的说就是一种数据结构，为了实现动态中位数的算法。一个数组将它调整为由两个堆组成的数据结构，一个是最大堆，一个是最小堆。也就是将一个数组一刀切两段。切断的地方就是两个部分的中位数所在的部分，如果完美切段，两边一样长那么中位数就是中间截断部分的平均，如果切开的不等长，有一边长一个元素，那么那个元素就是这个中位数。

通过动态插入，大小和长度调整，实现快速获得中位数的数据结构。

使用python的heapq库实现。因为heapq是最小堆，所有要在大小堆之间交换数字的时候，都把val乘以-1就可以了，很清晰。

使用两个堆可以实现什么问题？通过做leetcode502的IPO问题，看出，当想要最大化一个数组，同时最小化一个数组的量，取得权衡的时候，两个堆可以很好的发挥作用。同时，利用python的heapq库，可以很好的解决数据结构的问题，他是一个最小堆，但是-1出奇迹啊。

### 基本结构的代码实现（找中位为例）

```python
import heapq

class Median:
    def __init__(self):
        self.small, self.large = [], []

    def insert(self, num):
        # push to the max heap and switch if needed
        heapq.heappush(self.small, -1 * num)
        if (self.small and self.large and (-1 * self.small[0]) > self.large[0]):
            val = -1 * heapq.heappop(self.small)
            heapq.heappush(self.large, val)

        # handle uneven size
        if len(self.small) > len(self.large) + 1:
            val = -1 * heapq.heappop(self.small)
            heapq.heappush(self.large, val)
        if len(self.large) > len(self.small) + 1:
            val = -1 * heapq.heappop(self.large)
            heapq.heappush(self.small, val)
    
    def getMedian(self):
        if len(self.small) > len(self.large):
            return -1 * self.small[0]
        elif len(self.small) < len(self.large):
            return self.large[0]
        return ((-1 * self.small[0]) + self.large[0]) / 2
```

### 问题1: 
- [leetcode480 题目描述](https://leetcode.com/problems/sliding-window-median/description/)


### 问题2:Find median from data stream

从数据流中查找中值，是[leetcode295](https://leetcode.com/problems/find-median-from-data-stream/description/)题。这道题完全就是两个堆数据结构堆实现算法。理解原理，记住就好。注意最后的结果需要是小数，所以最后的除数需要也是小数。

解题步骤：

- 将数据分为两个堆数据结构，一个是前半部分的最大堆，一个是后半部分的最小堆。
- 插入数据，有后半部分的最小堆，并且该数据大于它的最小值，插入。反之插入前半部分的最大堆。
- 插入后，调整两个堆堆长度，相差不超过一。
- 取值，哪个堆长度长则取它的最大或者最小值。
- 取值，如果两个堆长度相同，则各取一个，取均值。
- 整个过程的最大堆注意用 -1 调整值的正负。

```python
import heapq


class MedianFinder(object):

    def __init__(self):
        self.small, self.large = [], []

    def addNum(self, num):
        """
        :type num: int
        :rtype: None
        """
        
        if self.large and num > self.large[0]:
            heapq.heappush(self.large, num)
        else:
            heapq.heappush(self.small, -1 * num)

        # balance the length
        if len(self.small) > len(self.large) + 1:
            val = -1 * heapq.heappop(self.small)
            heapq.heappush(self.large, val)
        if len(self.large) > len(self.small) + 1:
            val = -1 * heapq.heappop(self.large)
            heapq.heappush(self.small, val)

    def findMedian(self):
        """
        :rtype: float
        """
        if len(self.small) > len(self.large):
            return -1 * self.small[0]
        elif len(self.large) > len(self.small):
            return self.large[0]
        return (-1 * self.small[0] + self.large[0]) / 2.0
```

学习笔记：以上是我力扣的题解，但这这次我做这道题，它将数据结构拆分成了另有最大堆和最小堆的重构，所以写的很长，贴在最后的附录部分。（然后它题解部分给的是三个class的框架，解答给的是和我上面一样的直接使用 heapq），其实我自己觉的，不需要额外构架大小堆，因为两个很多重复的地方。上面的方法就足够了。

时间复杂度上插入操作为O(logn)，虽然如果将每次插入都加起来的话，在对数部分，会有一个n的阶乘，但是使用斯特林估计，总体来说是logn。在查找上为常数时间O(1)因为可以直接在堆顶取得最大或者最小值。空间复杂度为O(1)，因为是将原有的数组进行了重构不需要其他的空间，在调整的过程中也只是使用了需要操作的单个数据，所以需要的是常数的数据空间。

### 问题3: Maximize Capital

这是一道leetcode题，原名叫IPO[leetcode502](https://leetcode.com/problems/ipo/description/)目的是最大化投资的利润。

首先理解题意。两个数组，一个是利润数组profits，一个是资金数组captical，这两个数组是一一对应关系。两个变量，一个k是投资轮次，一个w是初始资金。就是投资人，拿着初始资金w进行投资，投资后得到的利润会直接加进现在的w中进行下一轮投资，通过k轮投资，想办法让最终手中的w最大化。可以通过最小堆最大堆实现，让（资金，利润）加进最小堆，这样就可以确保每次投入资金最小，让利润在每次投资的时候放入最大堆，这样就可以让得到的利润最大。最终在k轮中，如果手中资金小于需要的投资资金，那就要提前停止投资了。

题解步骤：

- 创建一个最小堆来存储资金。
- 确定现有资金范围内可以投资的项目。
- 选择利润最高的项目。
- 将赚取的利润添加进当前资本中。
- 重复此操作，直至选择了k项目。

题解练习。

```python
from heapq import *

def maximum_capital(c, k, capitals, profits):
    maxProfit = []
    minCapital = [(cap, pro) for cap, pro in zip(capitals, profits)]
    heapify(minCapital)

    for i in range(k):

        while minCapital and minCapital[0][0] <= c:
            cap, pro = heappop(minCapital)
            heappush(maxProfit, -1 * pro)

        if not maxProfit:
            break

        c += -1 * heappop(maxProfit)

    return c
```

因为在lc做过所以没什么问题。

附上题解：我觉得这个题解还是写的有点麻烦的，它使用了capitals数组中的index同时作为profits数组的索引，来取得对应的profit，只能说也是一种方法OKK。
```python
from heapq import heappush, heappop


def maximum_capital(c, k, capitals, profits):
    current_capital = c
    capitals_min_heap = []
    profits_max_heap = []

    for x in range(0, len(capitals)):
        heappush(capitals_min_heap, (capitals[x], x))

    for _ in range(k):

        while capitals_min_heap and capitals_min_heap[0][0] <= current_capital:
            c, i = heappop(capitals_min_heap)
            heappush(profits_max_heap, (-profits[i]))
        
        if not profits_max_heap:
            break

        j = -heappop(profits_max_heap)
        current_capital = current_capital + j

    return current_capital


def main():
    input = (
              (0, 1, [1, 1, 2], [1 ,2, 3]),
              (1, 2, [1, 2, 2, 3], [2, 4, 6, 8]),
              (2, 3, [1, 3, 4, 5, 6], [1, 2, 3, 4, 5]),
              (1, 3, [1, 2, 3, 4], [1, 3, 5, 7]),
              (7, 2, [6, 7, 8, 10], [4, 8, 12, 14]),
              (2, 4, [2, 3, 5, 6, 8, 12], [1, 2, 5, 6, 8, 9])
            )
    num = 1
    for i in input:
        print(f"{num}.\tProject capital requirements:  {i[2]}")
        print(f"\tProject expected profits:      {i[3]}")
        print(f"\tNumber of projects:            {i[1]}")
        print(f"\tStart-up capital:              {i[0]}")
        print("\n\tMaximum capital earned: ",
              maximum_capital(i[0], i[1], i[2], i[3]))
        print("-" * 100, "\n")
        num += 1


if __name__ == "__main__":
    main()
          
```

另外附上在leetcode中我的题解：
```python
class Solution(object):
    def findMaximizedCapital(self, k, w, profits, capital):
        """
        :type k: int
        :type w: int
        :type profits: List[int]
        :type capital: List[int]
        :rtype: int
        """
        import heapq
        # a maxheap to get the max profit
        maxProfit = []
        # a minheap to get all the valiable project
        minCapital = [(c, p) for c, p in zip(capital, profits)]
        # make the minCapital a minheap
        heapq.heapify(minCapital)
        for i in range(k):
            while minCapital and minCapital[0][0] <= w:
                # put all the profit to maxProfit in order to get the max profit
                c, p = heappop(minCapital)
                heapq.heappush(maxProfit, -1 * p)
            if not maxProfit:
                break
            # every time in k loops get the max profit
            w += -1 * heapq.heappop(maxProfit)
        return w
```

学习笔记：将资金推入最小堆的时间复杂度是O(nlogn)，这里的n是所有的项目数量。从最大堆中选取最大资金的时间复杂度是O(klogn)，这里的k是投资轮次。所以总的来说，时间复杂度就是二者相加，由于n加上k依然是一个线性时间，考虑最坏的情况就都是n，也就是2n，省略常数，结果为O(nlogn)。空间复杂度我们使用了两个额外的堆，用于存储数据，一个是n长度一个是k长度，最坏的情况k的数量达到n，也就是2n，去掉常数，空间复杂度就是O(n)。

### 附录

问题2，找中位的三个类版本的写法代码。

```python
from heapq import *
class min_heap:
    def __init__(self):
        self.min_heap_list = []
        
    def insert(self, x):
        heappush(self.min_heap_list, x)

    def pop(self):
        return heappop(self.min_heap_list)

    def get_len(self):
        return len(self.min_heap_list)

    def get_min(self):
        return self.min_heap_list[0]

    def __str__(self):
        out = "["
        for i in self.min_heap_list:
            out+=str(i) + ", "
        out = out[:-2] + "]"
        return out


class max_heap:
    def __init__(self):
        self.max_heap_list = []

    def insert(self, x):
        heappush(self.max_heap_list, -x)

    def pop(self):
        return heappop(self.max_heap_list)

    def get_len(self):
        return len(self.max_heap_list)

    def get_max(self):
        return -self.max_heap_list[0]

    def __str__(self):
        out = "["
        for i in self.max_heap_list:
            out+=str(i) + ", "
        out = out[:-2] + "]"
        return out


class MedianOfStream:
    def __init__(self):
        self.small = max_heap()
        self.large = min_heap()

    # This function should take a number and store it
    def insert_num(self, num):
        if self.large.get_len() > 0 and num > self.large.get_min():
            self.large.insert(num)
        else:
            self.small.insert(num)

        # balance the length
        if self.small.get_len() > self.large.get_len() + 1:
            val = -1 * self.small.pop()
            self.large.insert(val)
        if self.large.get_len() > self.small.get_len() + 1:
            val = self.large.pop()
            self.small.insert(val)

    # This function should return the median of the stored numbers
    def find_median(self):
        if self.small.get_len() > self.large.get_len():
            return self.small.get_max()
        elif self.small.get_len() < self.large.get_len():
            return self.large.get_min()
        return (self.small.get_max() + self.large.get_min()) / 2.0
```
