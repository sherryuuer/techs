## 数据结构的组合应用 Custom Data Structure

### 什么是自定义数据结构

自定义数据结构指的是在编程中根据特定需求而创建的数据类型和数据组织方式。通常情况下，编程语言提供了一些内置的数据结构，如数组、链表、栈、队列等，但有时候这些数据结构可能无法满足特定的需求，或者有更高效的实现方式。

一个最简单的例子比如哈希表是键值对结构，在值的地方你可以自定义各种数据结构。

再比如，设计一个堆栈数据结构以在 O(1) 时间内检索最小值。这种优化数据结构的行为也是一种自定义。

这些都是为了解决具体问题而重新对数据结构的优化组合。所以一般有两种形式，一种是修改现有数据结构，一种是使用多种数据结构。

不过我自己看来，与其称之为自定义数据结构，不如叫做对数据结构的优化组合，因为它更像是一种对基础数据机构的灵活应用。

这一篇就是针对这样的问题而进行的练习。

### 问题1:Snapshot Array

要求创建数据结构一个快照数组。实现如下方法：

- Constructor(length)初始化数据结构为该数组的长度。
- set_value(idx, val)设置idx处的值为val。
- snapshot()没有参数，但是从0开始更新快照id。
- get_value(idx, snapid)取得idx处的值，在snapid的地方。

示例解释：Input: ["SnapshotArray","set","snap","set","get"], [[3],[0,5],[],[0,6],[0,0]], 输出Output: [null,null,0,null,5]

- 一开始是初始化该数据结构的长度为 3。
- snapshotArr.set(0,5)表示array[0] = 5。
- snapshotArr.snap()表示进行一次快照，这时候snap_id = 0，之后不断增加，第n次的id为n-1。
- snapshotArr.get(0,0)表示取得array[0]的值，在snap_id = 0的位置, return 5。

代码做了如下尝试后，在测试中通过了。

```python
class SnapshotArray:
    # Constructor
    def __init__(self, length):
        self.length = length
        self.snap = {}
        self.array = [0] * self.length
        self.curid = -1

    # Function set_value sets the value at a given index idx to val. 
    def set_value(self, idx, val):
        self.array[idx] = val
    
    # This function takes no parameters and returns the snapid.
    # snapid is the number of times that the snapshot() function was called minus 1. 
    def snapshot(self):
        self.curid += 1
        self.snap[self.curid] = self.array[:]
        return self.curid
    
    # Function get_value returns the value at the index idx with the given snapid.
    def get_value(self, idx, snapid):
        return self.snap[snapid][idx]
```

题解给的带条件判断的代码：

```python
import copy
class SnapshotArray:
    # Constructor
    def __init__(self, length):
        self.snapid = 0
        self.node_value = dict()
        self.node_value[0] = dict()
        self.ncount = length

    # Function set_value sets the value at a given index idx to val.
    def set_value(self, idx, val):
        if idx < self.ncount:
            self.node_value[self.snapid][idx] = val

    # This function takes no parameters and returns the snapid.
    # snapid is the number of times that the snapshot() function was called minus 1.
    def snapshot(self):
        self.node_value[self.snapid+1] = copy.deepcopy(self.node_value[self.snapid])
        self.snapid += 1
        return self.snapid - 1

    # Function get_value returns the value at the index idx with the given snapid.
    def get_value(self, idx, snapid):
        if snapid < self.snapid and snapid >= 0 and idx < self.ncount:
            return self.node_value[snapid][idx] if idx in self.node_value[snapid] else 0
        else:
            return None
```

但是事情还没结束，在力扣中，它memory不够了（不愧是网友的测试用例），需要优化空间复杂度，只保存更新的idx和值，优化后的力扣题代码如下：

```python
class SnapshotArray(object):

    def __init__(self, length):
        """
        :type length: int
        """
        self.array = [[[-1, 0]] for _ in range(length)]
        self.snapid = 0

    def set(self, index, val):
        """
        :type index: int
        :type val: int
        :rtype: None
        """
        self.array[index].append([self.snapid, val])

    def snap(self):
        """
        :rtype: int
        """
        self.snapid += 1
        return self.snapid - 1

    def get(self, index, snap_id):
        """
        :type index: int
        :type snap_id: int
        :rtype: int
        """
        import bisect
        # print(self.array)
        i = bisect.bisect(self.array[index], [snap_id + 1]) - 1
        return self.array[index][i][1]
```

其中，代码中 `bisect` 函数用于在 **已排序** 的列表中插入元素。

比如`bisect.bisect(self.array, id])`是指要插入的元素的 id。他会在列表中查找 id 的插入位置。它会返回一个整数索引，该索引指示 id 在 self.array 中的插入位置，使得插入后列表仍然保持排序。如果 id 已存在于列表中，则会返回其现有索引。

这道题，因为这里不是要插入，而是要找到快照的索引，或者它之前的最新的快照位置，因此减 1 后得到的就是最新的快照位置。

总而言之，这行代码的作用是：**在已排序的列表 `self.array` 中找到 `id` 的插入位置，并返回该位置**。

`bisect` 函数的时间复杂度为 O(log n)，其中 n 是列表的长度。这意味着，即使列表很长，也能高效地进行查找或插入操作。

这个解法使用了外部包，来实现快照更新只更新新的部分的目的。

学习笔记：一般来说自定义数据结构的目的，都是为了更快更高效查找，这道题也不例外，基本查找，更新插入都是常数时间，唯一注意点就是上面的内存不足问题。无限更新快照会导致内存不足，但是我想，实际应用中应该会将快照储存在外部设备，或者有版本数量上限。上面的增分更新也是不错的操作实践。

### 问题2:Time-Based Key-Value Store

设计一个基于时间的键值数据结构，可以存储同一个键在不同时间戳的多个值，并在某个时间戳检索该键的值。

- init()初始值和时间戳dict。
- set_value(key, value, timestamp)用于存储键和值在对应的时间戳位置。
- get_value(key, timestamp)用于取回键和时间戳的值。这个值被之前的set过, 并且 timestamp_prev <= timestamp. 如果有多个值, 则返回最新的timestamp_prev的值，如果没有值，返回"".

比如：

- timeMap = new TimeStamp();
- timeMap.set("foo", "bar", 1);  // store the key "foo" and value "bar" along with timestamp = 1.
- timeMap.get("foo", 1);         // return "bar"
- timeMap.get("foo", 3);         // return "bar", since there is no value corresponding to foo at timestamp 3 and timestamp 2, then the only value is at timestamp 1 is "bar".
- timeMap.set("foo", "bar2", 4); // store the key "foo" and value "bar2" along with timestamp = 4.
- timeMap.get("foo", 4);         // return "bar2"
- timeMap.get("foo", 5);         // return "bar2"

这个题在本质上和上一道题很像。在题解中还给出了二分查找的辅助函数，是一道二分查找和hashmap的结合题。

代码如下：

```python
class TimeStamp:
    def __init__(self):
        self.values_dict = {}

    #  Set TimeStamp data variables
    def set_value(self, key, value, timestamp):
        if key not in self.values_dict:
            self.values_dict[key] = []
        self.values_dict[key].append([timestamp, value])

    # Get TimeStamp data variables
    def get_value(self, key, timestamp):
        res = ""
        values = self.values_dict.get(key, [])
        left, right = 0, len(values) - 1
        while left <= right:
            mid = (left + right) // 2
            if values[mid][0] <= timestamp:
                left = mid + 1
                res = values[mid][1]
            else:
                right = mid - 1
        return res
```

以及题解给的答案：这个答案的不同在于将值和时间戳分为两个hashmap分别存储，通过索引查找，使用的二分查找函数是一样的。

```python
import random

class TimeStamp:
    def __init__(self):
        self.values_dict = {}
        self.timestamps_dict = {} 

    #  Set TimeStamp data variables
    def set_value(self, key, value, timestamp):
        if key in self.values_dict:
            if value != self.values_dict[key][len(self.values_dict[key])-1]:
                self.values_dict[key].append(value)
                self.timestamps_dict[key].append(timestamp)
        else:
            self.values_dict[key] = [value]
            self.timestamps_dict[key] = [timestamp]

    # Find the index of right most occurrence of the given timestamp
    # using binary search
    def search_index(self, n, key, timestamp):
        left = 0
        right = n
        mid = 0

        while left < right:
            mid = (left + right) >> 1 # bit位运算，相当于除以2取整
            if self.timestamps_dict[key][mid] <= timestamp:
                left = mid + 1
            else:
                right = mid
        return left - 1

    # Get time_stamp data variables
    def get_value(self, key, timestamp):
        if key not in self.values_dict:
            return ""
        else:
            index = self.search_index(len(self.timestamps_dict[key]),
                                      key, timestamp)
            if index > -1:
                return self.values_dict[key][index]

            return ""
```
学习笔记：这道题的关键，在于在时间戳列表中，找到目标时间戳或者它前面的那个位置，也就是使用二分查找，最后就是可以满足查找条件的数据结构。时间复杂度上set是O(1)，get是二分查找的O(logn)。空间复杂度上是数字键值对的长度O(n)。

### 问题3:LRU Cache

LRU实现问题，是力扣的146题，中等难度。

LRU（Least Recently Used，最近最少使用）是一种常见的缓存淘汰算法，用于管理缓存中的数据。LRU 数据结构维护了一个有序列表，记录了最近访问过的数据项，当缓存空间满时，会优先淘汰最近最少使用的数据项。

LRU 数据结构的实现通常基于双向链表和哈希表：

- 双向链表：LRU 数据结构使用双向链表来维护数据项的访问顺序。链表的头部表示最近访问过的数据项，尾部表示最久未访问的数据项。每次访问一个数据项时，将其移动到链表的头部。
- 哈希表：为了快速定位数据项在链表中的位置，LRU 数据结构使用哈希表来存储数据项的键和对应的链表节点。
- 访问数据项：当访问一个数据项时，如果它已经在缓存中，则将其移到链表的头部；如果不在缓存中，则将其添加到链表的头部，并将其存储在哈希表中。如果缓存已满，则将链表尾部的数据项淘汰，并从哈希表中删除对应的条目。
- 淘汰数据项：当缓存空间满时，淘汰链表尾部的数据项，即最近最少使用的数据项。

在这道题中：

- 用容量大小初始化数据结构。
- 用get方法返回key的值，如果不存在返回-1。
- 用set方法更新在key处的值，如果不存在则加在最后面。如果这个操作使得容量到达了上限，则删除最旧的key。
- 要求操作他们的时间复杂度都为O(1)。

这是一道我之前做过的重要的题，代码如下：

```python
class Node:
    def __init__(self, key, value):
        self.key, self.value = key, value
        self.prev = self.next = None

class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = {}
        # LRU dummy endpoint
        self.left = Node(0, 0)
        # MRU dummy endpoint
        self.right = Node(0, 0)
        self.left.next = self.right
        self.right.prev = self.left

    def remove(self, node):
        # remove a node from the linked list
        prev, nxt = node.prev, node.next
        prev.next, nxt.prev = nxt, prev

    def insert(self, node):
        # insert the node to the point of before-right
        prev, nxt = self.right.prev, self.right
        prev.next = nxt.prev = node
        node.prev, node.next = prev, nxt

    def get(self, key):
        if key in self.cache:
            # update the linked list
            self.remove(self.cache[key])
            self.insert(self.cache[key])
            return self.cache[key].value
        return -1

    def set(self, key, value):
        if key in self.cache:
            self.remove(self.cache[key])
        self.cache[key] = Node(key, value)
        self.insert(self.cache[key])
        # check the capacity
        if len(self.cache) > self.capacity:
            lru = self.left.next
            self.remove(lru)
            del self.cache[lru.key]
```
学习笔记：这道题如题要求，时间复杂度和空间复杂度都是O(1)，因为他们使用了操作上都很有优势的链表和hashmap。空间复杂度上是缓存的长度O(n)。这是一道非常棒的题。
