## hashmap tracking 的相关问题

---
### 应用场景

在算法中使用哈希表进行跟踪（tracking）通常是指利用哈希表数据结构来记录和跟踪某些信息，以便在算法的执行过程中快速地查找、更新或检索这些信息。这种技术在许多算法和数据结构中都有广泛的应用，特别是在需要频繁进行查找、删除或更新操作的情况下。比如两数之和，力扣第一题这种。它就像一本电话黄页，将你要查找的东西存储起来，以便后续使用，没有很复杂的逻辑，构建结构，加以应用，但是hashmap的优秀的查找时间，让这个算法变的强大而应用广泛。

### 问题1:Palindrome Permutation

这是一道经典的字符串处理问题，要求判断一个给定的字符串是否可以通过重新排列组合成一个回文字符串。回文字符串是指正着读和反着读都一样的字符串。例如，对于字符串 "code"，无法通过重新排列组合成一个回文字符串；而对于字符串 "aab"，可以通过重新排列组合成 "aba"，是一个回文字符串。

解决这个问题的一种常见方法是统计字符串中每个字符出现的次数，然后检查字符出现次数的奇偶性。如果字符串中最多只有一个字符出现奇数次，那么它就可以通过重新排列组合成一个回文字符串。

解决步骤：总之很简单

- 遍历字符串统计字符串中每个字符的出现次数，并将统计结果存储在哈希表中。
- 遍历哈希表中每个字符的出现次数，统计出现奇数次的字符的数量。
- 如果出现奇数次的字符的数量大于 1，则无法通过重新排列组合成一个回文字符串，返回 False；否则返回 True。

尝试代码：
```python
class hash_map:
    def __init__(self):
        self.hash_map_dict = {}
        
    def insert(self, x):
        if x in self.hash_map_dict.keys():
            self.hash_map_dict[x] += 1
        else:
            self.hash_map_dict[x] = 1

def permute_palindrome(st):

    hashmap = hash_map()
    for char in st:
        hashmap.insert(char)

    count = 0
    for value in hashmap.hash_map_dict.values():
        if value % 2:
            count += 1
    if count > 1:
        return False
    return True
```
通过测试。但是题目给出的是有class的模板，解答却是函数，并且函数构筑的还不够优雅，下面是一个优化的题解。

```python
def can_permute_palindrome(s):
    char_count = {}
    for char in s:
        char_count[char] = char_count.get(char, 0) + 1

    odd_count = 0
    for count in char_count.values():
        if count % 2:
            odd_count += 1
        
    return odd_count <= 1
```
学习笔记：这道题的时间复杂度是O(n)虽然经过了两次遍历，但是最坏的情况，其中一次遍历的长度也只有26个字母的长度，所以总体是O(n)，空间复杂度是一个hashmap，同样这个hashmap的最坏情况是充斥了26个字母，这是一个常数，所以空间复杂度是O(1)。

### 问题2:Valid Anagram

有效字母异位词问题。题意来说，给定两个字符串str1，和str2，判断str2是不是str1的有效异位词。字母异位词是指通过重新排列原字符串中的字符而产生的新字符串，两个字符串中包含的字母相同，但顺序不同。换句话说，两个字符串中的字符种类和数量必须完全相同，只是排列顺序不同。

解决这道题只需要统计两个字符串中字母出现的次数，如果相同就是有效异位词。很简单。顺便，暴力破解可以排序两个字符串，然后逐个检查每个位置的字符是否相等。

编码步骤：

- 检查字符串长度，如果不同直接返回False。
- 初始化hashmap用于存储。
- 遍历字符串1，统计每次字符数量。
- 遍历字符串2，对每个字符串递减数量。
- 计算hashmap中的所有字符的次数的和，为0就是True，反之为False。

```python
def is_anagram(str1, str2):
    if len(str1) != len(str2):
        return False

    hashmap = {}
    for char in str1:
        hashmap[char] = hashmap.get(char, 0) + 1

    for char in str2:
        if char not in hashmap:
            return False
        hashmap[char] -= 1

    return all(c == 0 for c in hashmap.values())
```
简单通过。

学习笔记：时间复杂度是遍历的O(n)，空间复杂度是使用了有限长度的hashmap的O(1)。

### 问题3:Maximum Frequency Stack

题意是要求设计一种数据结构，支持两种操作：
- init()：初始化这个类似stack的数据结构。
- Push(value)：将一个元素压入栈中。
- Pop()：弹出栈中频率最高的元素。如果有多个频率相同的元素，则弹出最近压入栈的那个元素。

常见方法是使用哈希表和堆栈结合的方式，使用hashmap来记录每个元素的频率，使用一个优先队列（堆）来记录当前栈中的元素，按照元素的频率和压入顺序进行排序。对于 Push 操作，将元素压入栈中，并更新哈希表中对应元素的频率。对于 Pop 操作，从优先队列中弹出频率最高的元素，如果有多个频率相同的元素，则弹出最近压入栈的那个元素，并更新哈希表中对应元素的频率。

这道题的具体步骤：
- 创建hashmap用于存储频率。
- 对于输入序列，将频率值存储在hashmap中。同时将相应元素押入堆栈。
- 删除元素时，删除频率最高的元素。
- 如果频率相同则删除最新入栈的元素。

代码尝试：
```python
import heapq

class FreqStack:
    def __init__(self):
        self.hashmap = {}
        self.queue = []
        self.count = 0 # 押入顺序

    def push(self, value):
        self.count += 1
        self.hashmap[value] = self.hashmap.get(value, []) + [self.count]
        heapq.heappush(self.queue, [-1 * len(self.hashmap[value]), -1 * self.hashmap[value][-1], value])

    def pop(self):
        _, _, value = heapq.heappop(self.queue)
        self.hashmap[value].pop()
        return value
```
通过测试，在过程中返回结果简单，但是一开始我没处理出来相同频率的情况，后来加入了一个 count 变量，作为押入的频率，一起计入最小堆，就可以了。

**值得注意的是**我的这个方法最后pop的更新是基于hashmap结构的，hashmap中不是想象中的每一个value都有一个自己的list，而是每次push都会产生一个list，这个要注意，给出一个直观的输入输出就明白了：

```python
fs = FreqStack()
fs.push(1)
fs.push(2)
fs.push(3)
fs.push(3)
fs.push(2)
fs.push(2)
fs.pop()

# output:
# [[-1, -1, 1]]
# [[-1, -2, 2], [-1, -1, 1]]
# [[-1, -3, 3], [-1, -1, 1], [-1, -2, 2]]
# [[-2, -4, 3], [-1, -3, 3], [-1, -2, 2], [-1, -1, 1]]
# [[-2, -5, 2], [-2, -4, 3], [-1, -2, 2], [-1, -1, 1], [-1, -3, 3]]
# [[-3, -6, 2], [-2, -4, 3], [-2, -5, 2], [-1, -1, 1], [-1, -3, 3], [-1, -2, 2]]
# [[-2, -5, 2], [-2, -4, 3], [-1, -2, 2], [-1, -1, 1], [-1, -3, 3]]
```
可以看到在每次push后，value是有重复的，记录了每次操作后的结果，所以pop后只是回到了hashmap的上上次的状态而已，这样就保证了代码的正确性。


另外参考题解的答案如下：很长，但是很具象化的表达了每一步的数量增删过程。
```python
from collections import defaultdict


# Declare a FreqStack class containing frequency and group hashmaps
# and maxFrequency integer
class FreqStack:

    # Use constructor to initialize the FreqStack object
    def __init__(self):
        self.frequency = defaultdict(int)
        self.group = defaultdict(list)
        self.max_frequency = 0

    # Use push function to push the value into the FreqStack
    def push(self, value):
        freq = self.frequency[value] + 1
        self.frequency[value] = freq
        
        if freq > self.max_frequency:
            self.max_frequency = freq
        
        self.group[freq].append(value)

    def pop(self):
        value = ""

        if self.max_frequency > 0:    
            value = self.group[self.max_frequency].pop()
            self.frequency[value] -= 1

            if not self.group[self.max_frequency]:
                self.max_frequency -= 1
        else:
            return -1

        return value
```
学习笔记：时间复杂度为常数时间O(1)因为每次都是对一个元素进行增删处理，空间复杂度为O(n)因为使用了额外的线性空间。
