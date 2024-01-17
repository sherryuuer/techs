## 堆算法：两个堆（Two Heaps）

---

### 它是什么

简单的说就是一种数据结构，为了实现动态中位数的算法。一个数组将它调整为由两个堆组成的数据结构，一个是最大堆，一个是最小堆。

通过动态插入，大小和长度调整，实现快速获得中位数的数据结构。

使用python的heapq库实现。因为heapq是最小堆，所有要在大小堆之间交换数字的时候，都把val乘以-1就可以了，很清晰。

使用两个堆可以实现什么问题？通过做leetcode502的IPO问题，看出，当想要最大化一个数组，同时最小化一个数组的量，取得权衡的时候，两个堆可以很好的发挥作用。同时，利用python的heapq库，可以很好的解决数据结构的问题，他是一个最小堆，但是-1出奇迹啊。

### 代码实现

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

### leetcode 逐行解析

- 从数据流中查找中值[leetcode295 题目描述](https://leetcode.com/problems/find-median-from-data-stream/description/)

完全就是两个堆数据结构堆实现算法。理解原理，记住就好。注意最后的结果需要是小数，所以最后的除数需要也是小数。

- [leetcode480 题目描述](https://leetcode.com/problems/sliding-window-median/description/)

- leetcode想要IPO[leetcode502 题目描述](https://leetcode.com/problems/ipo/description/)

通过最小堆最大堆实现，每次都能找到最小资金可以实现的项目。

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
