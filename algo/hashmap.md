## Hashmap相关算法问题

---
### 概念和适用范围

人人都知道的哈希表（Hash Table），也称为哈希映射（Hash Map）或字典（Dictionary），是一种数据结构，用于存储键值对（Key-Value pairs）。哈希表通过哈希函数将键映射到特定的索引位置，从而实现快速的查找、插入和删除操作。最显著的特征是他查找元素的时间复杂度为O(1)，这是他优于数组的地方。

哈希表适用于许多算法问题，特别是在需要高效地进行数据查找、插入和删除操作的情况下：

- 查找问题：在给定一组数据中查找特定元素的问题，如查找数组中是否存在某个元素、查找字符串中是否存在某个子串等。
- 去重问题：去除数组、列表或字符串中的重复元素。（遍历数据，将元素作为键存储在哈希表中，然后提取哈希表中的键，即可实现去重。）
- 计数问题：统计数据中每个元素的出现次数，如统计字符串中每个字符出现的次数、统计数组中每个元素出现的频率等。（遍历数据，使用哈希表存储每个元素的出现次数，然后查询哈希表即可获取每个元素的计数。）
- 求交集、并集、差集问题：给定两个集合，计算它们的交集、并集、差集等运算。
- 检测重复问题：检测给定数据中是否存在重复元素，如查找数组中是否存在重复元素、判断字符串中是否有重复字符等。（遍历数据，将元素作为键存储在哈希表中，如果元素已存在于哈希表中，则存在重复。）
- 字典、映射问题：实现字典或映射结构，将键映射到值，并支持快速的插入、查找和删除操作。
- 缓存问题：实现高效的缓存结构，用于存储和检索之前计算过的结果，以加速后续的计算过程。（使用哈希表作为缓存结构，存储计算过的结果，并在需要时直接从哈希表中检索结果。）
- 散列表问题：解决与散列表相关的问题，如实现散列函数、解决散列冲突、实现开放地址法或链地址法等。

### 问题1:Isomorphic Strings

"同构字符串"（Isomorphic Strings）是一道经典的字符串处理问题，题目要求判断两个给定的字符串是否是同构的。同构字符串指的是两个字符串中的字符可以被一一映射，并且保持字符的相对顺序不变。换句话说，如果一个字符在字符串 A 中出现了，那么它在字符串 B 中的对应字符的位置也必须相同，且相同字符要被映射到相同的字符上。

例如，"egg" 和 "add" 是同构字符串，因为 "e" 在 "egg" 中映射到 "a" 在 "add" 中， "g" 在 "egg" 中映射到 "d" 在 "add" 中。又如，"paper" 和 "title" 也是同构字符串，因为 "p" 在 "paper" 中映射到 "t" 在 "title" 中， "e" 在 "paper" 中映射到 "i" 在 "title" 中，以此类推。

解决这道问题的一种常见方法就是使用哈希表来记录字符的映射关系。具体步骤如下：

- 遍历两个字符串的对应位置的字符。
- 使用两个哈希表分别记录字符在字符串 A 到字符串 B 的映射关系和字符串 B 到字符串 A 的映射关系。
- 检查当前字符在两个哈希表中的映射关系是否相符。

如果两个字符串是同构的，则它们的每个字符都会有相应的一对一映射关系，否则它们不是同构的。

代码：

```python
def is_isomorphic(string1, string2):
    if len(string1) != len(string2):
        return False
    hashmap1 = {}
    hashmap2 = {}

    for i in range(len(string1)):
        
        if string1[i] in hashmap1:
            if hashmap1[string1[i]] != string2[i]:
                return False
        else:
            hashmap1[string1[i]] = string2[i]

        if string2[i] in hashmap2:
            if hashmap2[string2[i]] != string1[i]:
                return False
        else:
            hashmap2[string2[i]] = string1[i]

    return True
```

题解给出的答案我很喜欢感觉很清晰：先进性两个判断，然后再添加，这样思路很清晰。

```python
def is_isomorphic(string1, string2):

    map_str1_str2 = {}
    map_str2_str1 = {}

    for i in range(len(string1)):
        char1 = string1[i]
        char2 = string2[i]

        if char1 in map_str1_str2 and map_str1_str2[char1] != char2:
            return False

        if char2 in map_str2_str1 and map_str2_str1[char2] != char1:
            return False

        map_str1_str2[char1] = char2
        map_str2_str1[char2] = char1

    return True
```

学习笔记：时间复杂度是O(n)因为遍历了整个字符串长度，空间复杂度为O(1)因为我们使用的hashmap的大小是固定的ascii字符集。

### 问题2:Logger Rate Limiter

Logger Rate Limiter问题是一个经典的算法设计问题，题目描述如下：

假设你在开发一个应用程序，其中需要记录日志。你希望设计一个日志记录器，这个日志记录器在同一秒钟内不会重复记录相同的消息。如果在同一秒钟内收到相同的消息，它只会记录一次，并忽略其他相同消息。而对于不同的消息，即使在同一秒钟内，也应该记录下来。

具体来说，Logger Rate Limiter问题要求设计一个Logger类，其包含两个方法：

1. `shouldPrintMessage(timestamp: int, message: str) -> bool`: 这个方法接收两个参数，一个是时间戳 `timestamp`，另一个是消息 `message`。方法返回一个布尔值，表示是否应该打印出这条消息。如果该方法返回True，则表示应该打印消息，同时记录该消息及时间戳，否则返回False，表示应该忽略该消息。

2. `__init__(self)`: 这是Logger类的构造函数，用于初始化Logger对象。

例如，假设Logger对象在某一时刻调用了`shouldPrintMessage(1, "foo")`，则应该返回True，并记录消息"foo"及时间戳1；如果在时刻2调用了`shouldPrintMessage(2, "foo")`，则应该返回False，因为消息"foo"在同一秒钟内已经被记录过了；但是如果在时刻3调用了`shouldPrintMessage(3, "foo")`，则应该返回True，因为此时消息"foo"的时间戳不同于之前的记录。这里的时刻，是时间戳的意思，一个秒的单位内会有很多时间戳。

在这道具体的算法题中，会给出一个具体输入`time_limit`表示在这个时间范围内，之需要记录同一条记录一次。

输入数据的格式如下：第一个数字是时间，第二个是信息。

```
[[1,"good morning"],[5,"hello world"],[6,"good morning"],[7,"good morning"],[15,"hello world"]]
time_limit = 7
```

步骤解析：

- 在之后的接收信息的过程中，使用所有的信息列表构建一个hashmap，信息本身是key，时间戳是value。
- 进行信息接收，当收到一个新消息，查看它是否在hashmap中，如果没出现过，则加入hashmap。标记为True。
- 如果这个新消息已经在hashmap中，那么查看他们的时间差是否大于给定的time_limit，如果大于则更新hashmap。标记为True。
  - 反之拒绝这个消息。标记为False。

编码尝试：

```python
class RequestLogger:
    def __init__(self, time_limit):
        self.logger_map = {}
        self.time_limit = time_limit

    # This function decides whether the message request should be accepted or rejected
    def message_request_decision(self, timestamp, request):
        if request not in self.logger_map:
            self.logger_map[request] = timestamp
            return True
        else:
            if timestamp - self.logger_map[request] >= self.time_limit:
                self.logger_map[request] = timestamp
                return True
            else:
                return False
```
简单尝试通过了测试，看来是一道比较简单的题了。

题解答案写的更简洁：

```python
class RequestLogger:

    # initailization of requests hash map
    def __init__(self, time_limit):
        self.requests = {}
        self.limit = time_limit

    # function to accept and deny message requests
    def message_request_decision(self, timestamp, request):
        if request not in self.requests or timestamp - self.requests[request] >= self.limit:
            self.requests[request] = timestamp
            return True
        else:
            return False
```

学习笔记：在计算中因为使用了hashmap，所以时间复杂度只有O(1)，空间复杂度，因为将所有的n个消息存储在表中，所以为O(n)。

### 问题3:Next Greater Element

是一道经典的算法问题。题中给出两个数组，nums1，和nums2，第一个数组是第二个数组的子集。要求返回一个ans结果，它和nums1的长度一致，每一个元素是对应的nums1中的元素，在nums2中的同一个元素的右侧的第一个最大值。

举个例子：

```
nums1 = [5, 4, 7]
nums2 = [4, 5, 7, 3]
```
那么对应的输出结果就是：[7, 5, -1]

第一个元素 5 对应的在第二个数组中右侧第一个最大的是 7，第二个元素 4 对应的在第二个数组中右侧第一个最大的是 5，第三个元素 7 由于在第二个数组中右侧没有比自己大的，所以返回 -1。

解题思路：

- 创建一个空的 stack 和一个空的 hashmap。
- 遍历第二个数组，对于每一个元素都和栈顶的元素进行比较（这里是一个while循环，因为遍历的当前元素可以是 stack 中多个元素的右侧最大值）。如果该元素大于栈顶的元素，弹出栈顶元素作为 key，当前元素作为 value，推进 hashmap。
- 将当前元素加入栈顶。
- 重复上述步骤结束遍历后，检查 stack 中剩余的元素，将他们的作为 key，value 为 -1 加入hashmap。
- 遍历第一个数组，从hashmap中取出结果即可。

代码尝试：

```python
def next_greater_element(nums1, nums2):
    stack = []
    hashmap = {}
    ans = []
    
    for num in nums2:
        while stack and num > stack[-1]:
            hashmap[stack.pop()] = num
        stack.append(num)
    
    while stack:
        hashmap[stack.pop()] = -1

    for num in nums1:
        ans.append(hashmap[num])

    return ans
```
通过测试。

学习笔记：参考答案和自己写的基本没差异，就不贴了。时间复杂度上来说，经历了三次遍历，所以加起来是O(n)线性时间复杂度。空间复杂度上因为使用了两个容器，长度不会超过 n 所以空间复杂度也为O(n)。

另外这道题是用了两个数组，原题其实可以是一个数组，找到每个元素右侧的最大元素。这道题还可以用反向遍历的方法，省掉一个 hashmap 的空间，直接在结果列表上进行操作。但是使用 hashmap 的好处就在于容易明白，所以暂且抛去不谈。
