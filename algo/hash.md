## 哈希数据结构：hash，一种以很重要的数据结构

---
### 概念引导

哈希（Hash）数据结构是一种通过将键（Key）映射到索引的方式来存储和检索数据的数据结构。哈希表（Hash Table）是最常见的哈希数据结构，它通过哈希函数将键映射到数组的特定索引位置。这使得在理想情况下，可以以常量时间复杂度 O(1) 的速度进行插入、查找和删除操作。

以下是哈希数据结构的一些关键概念：

1. **哈希函数（Hash Function）：** 哈希函数是一个将任意大小的数据映射为固定大小的固定值（哈希码）的函数。它负责将键转换成索引，使得相同的键始终映射到相同的索引。理想情况下，哈希函数应该具有良好的分布性，以最小化哈希冲突。

2. **哈希冲突（Hash Collision）：** 当两个不同的键通过哈希函数映射到相同的索引时，就会发生哈希冲突。解决冲突的方法包括链地址法和开放寻址法。

    - **链地址法（Chaining）：** 每个哈希桶都是一个链表，具有相同索引的键都存储在同一个链表中。
    
    - **开放寻址法（Open Addressing）：** 当发生冲突时，尝试找到下一个可用的哈希桶，直到找到空桶或遍历整个哈希表。

3. **装载因子（Load Factor）：** 装载因子是哈希表中已存储键值对的数量与哈希表总容量的比率。较低的装载因子可以减小冲突的可能性，但会增加内存的浪费。通常，装载因子的合理范围是 0.7 到 0.8。

4. **碰撞解决策略：** 碰撞解决策略是指处理哈希冲突的具体方法。常见的碰撞解决策略包括拉链法、线性探测法和二次探测法。

哈希数据结构在计算机科学中广泛应用，例如在实现字典、集合、缓存和数据库索引等方面。它提供了快速的查找和插入操作，使得在大量数据中快速定位特定元素成为可能。

### 代码实现

哈希表的内部代码实现实质上是很复杂的，这里简化整个流程。

首先，hash的实现使用ascii转换的方法，将数字不断叠加然后和容量做mode运算，以保证index在范围内。

其次，注意到在get和put等方法中不断有index递增的表达，这是因为在发生哈希冲突的时候，简单的向下移动一个index单位，来找一个空位插入健值对。

再次，在rehash方法中，重新指定容量的时候，简单的将容量翻倍，但是实际过程中有很多方法，比如使用质数容量，更不容易引起哈希冲突。

最后，在remove方法中，代码中也写了，这样的实现可能会引起bug，因为在健值对数组中会出现一个漏洞，当get方法执行的时候，会在遇到空的时候直接返回None，但是也许要查找的值正是下一个，而这个空只是在remove别的值的时候删除的位置而已。

但是整个实现的目的是理解整个原理。

```python
class Pair:
    def __init__(self, key, val):
        self.key = key
        self.val = val

class HashMap:
    def __init__(self):
        self.size = 0
        self.capacity = 2
        self.map = [None, None]
    
    def hash(self, key):
        index = 0
        for c in key:
            index += ord(c)
        # mode运算会在后面的get方法也出现，目的是为了保证index在范围内
        return index % self.capacity

    def get(self, key):
        index = self.hash(key)

        while self.map[index] != None:
            if self.map[index].key == key:
                return self.map[index].val
            index += 1
            index = index % self.capacity

        return None
    
    def put(self, key, val):
        index = self.hash(key)

        while True:
            if self.map[index] == None:
                self.map[index] = Pair(key, val)
                self.size += 1
                if self.size >= self.capacity // 2:
                    self.rehash()
                return
            elif self.map[index].key == key:
                self.map[index].val = val
                return
            
            index += 1
            index = index 5 self.capacity
    
    def rehash(self):
        self.capacity = 2 * self.capacity
        newMap = []
        for i in range(self.capacity):
            newMap.append(None)

        oldMap = self.map
        self.map = newMap
        self.size = 0
        for pair in oldMap:
            if pair:
                self.put(pair.key, pair.val)

    def remove(self, key):
        if not self.get(key):
            return
        
        index = self.hash(key)
        while True:
            if self.map[index].key == key:
                # Removing an element using open-addressing actually causes a bug,
                # because we may create a hole in the list, and our get() may 
                # stop searching early when it reaches this hole.
                self.map[index] = None
                self.size -= 1
                return
            index += 1
            index = index % self.capacity

```
