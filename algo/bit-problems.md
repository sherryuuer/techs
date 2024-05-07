## 位运算相关问题

### 位运算和现实世界问题

位操作问题是指在计算机科学和编程中涉及使用二进制位运算符（如 AND（&）、OR（|）、XOR（^）和位移（<<、>>））来操作数据的一类问题。这些问题通常涉及对整数的二进制表示进行操作，以实现某种特定的算法或逻辑。位操作通常用于优化算法，提高程序的效率，并且在一些特定的问题领域中非常有用，比如密码学、图形处理、网络编程等。

现实世界的问题：

- 压缩算法：按位算法是霍夫曼编码等压缩技术的基础，有助于在位级别上有效地编码和解码数据。它们通过连接位以及优化存储和传输来促进可变长度代码的紧凑表示。通过采用按位“与”、“或”和移位运算，压缩算法可以在不丢失信息的情况下实现数据压缩，这对于资源受限的环境至关重要。
- 状态寄存器：在计算机处理器的状态寄存器中，每一位都传达不同的含义。例如，状态寄存器的起始位表示算术运算的结果是否为零，称为零标志。可以使用相同长度的掩码来检查、更改或清除该位的值，其中相关位设置为 1。在提供的场景中，选择的掩码将为 10000000，与位于第一个位置的零标志对齐。
- 密码学：密码算法中通常采用循环移位来引入混乱和扩散，从而增强安全性。通过应用循环移位，输入和输出数据之间的关系变得复杂，使攻击者更难破译原始信息。此外，循环移位会产生雪崩效应，确保即使输入的微小变化也会导致输出的显着变化，从而增强密码算法抵御各种攻击的能力。
- 哈希函数：按位运算用于计算循环冗余校验 (CRC) 和 Adler-32 等哈希函数中的校验和。这些校验和用于错误检测和数据完整性验证。

### 问题1:Find the Difference

是一道力扣的简单题，简单意味着可以有更多的方法解决。先理解题。

给出两个字符串s和t。字符串t是用s的字母随机组成的。然后在一个随机的位置，加了一个多余的字符。找到这个多余的字符。

直觉上我们可以用hashmap立刻解决。两种情况，该多余字符不在s中，另一种情况，是重复使用了已经有的字符。代码如下：

```python
class Solution(object):
    def findTheDifference(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: str
        """
        from collections import Counter
        count_s, count_t = Counter(s), Counter(t)
        for c in count_t:
            if c not in count_s:
                return c
            if count_t[c] > count_s[c]:
                return c
```

但是这道题在bitwise领域就要用数字解决。ASCII字符就是字母到数字的映射。两个只差一个字母的字符串，他们的ASCII字符的数字变换的和之差就是那个多余的字符。这里只需要用到两个函数就可以做到了。

```python
class Solution(object):
    def findTheDifference(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: str
        """
        sum_s, sum_t = 0, 0
        for c in s:
            sum_s += ord(c)
        for c in t:
            sum_t += ord(c)
        
        return chr(sum_t - sum_s)
```

不过这个还不是bit运算。这里运用bit运算要用到XOR运算也就是小三角 `^`，XOR操作可以对数字进行取消操作。

XOR具有这种性质：

- 一个数和0进行XOR运算会等于它本身
- 一个数和自身进行XOR操作则会等于0。
- 两个不同的数字进行XOR操作则会基于二进制进行不同位上的二进制操作。

这道题，如果从0开始对所有的数字进行XOR操作，则会通过连续计算不断取消已经乘过的数字，最后留下的就是目标元素。

```python
class Solution(object):
    def findTheDifference(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: str
        """
        res = 0
        for c in s:
            res = res ^ ord(c)
        for c in t:
            res = res ^ ord(c)
        return chr(res)
```

如果要求的是字母的index，题解代码如下：

```python
def extra_character_index(str1, str2):

    result = 0
    str1_length = len(str1)
    str2_length = len(str2)

    for i in range(str1_length):
        result = result ^ (ord)(str1[i])

    for j in range(str2_length):
        result = result ^ (ord)(str2[j])

    if len(str1) > len(str2):
        index = str1.index((chr)(result))
        return index
    else:
        index = str2.index((chr)(result))
        return index
```

学习笔记：bit运算是计算机的运算，是一种很聪明的运算，所以才会叫bitwise吗，哈哈，时间复杂度只有遍历的线性时间O(n)，空间复杂度只用了一个res额外空间O(1)所以为常数时间。

### 问题2:Complement of Base 10 Number

这道题目是在讨论二进制数的补数（complement）。在计算机科学中，补数是一个很常见的概念，特别是在处理负数时。

什么是二进制数的补数。在二进制中，一个数的补数是另一个数相对于某个固定的位数的表示。常见的补数包括原码、反码和补码。在这里，我们主要讨论的是二进制的补码。

对于一个二进制数，它的补码是通过将该数的每一位取反（0变成1，1变成0），然后再加上1来得到的。举个例子，假设我们有一个4位的二进制数1101，它的补码可以这样计算：

- 将每一位取反：0010
- 加1：0010 + 1 = 0011

所以，1101的补码是0011。

但是这道题没有这么复杂，只需要每一位取反就可以了。同样的可以用XOR计算，比如101如果和111取XOR就可以得到010。

比如 5 就是"101", 取反为"010", 结果就可以得到2。

```python
class Solution(object):
    def bitwiseComplement(self, n):
        """
        :type n: int
        :rtype: int
        """
        if n == 0: return 1
        x = 1
        # 当x小于n的时候，不断乘以2
        while x <= n:
            x <<= 1
        return (x - 1) ^ n
```

题解给的答案如下：

```python
from math import log2, floor


def find_bitwise_complement(num):
    if num == 0:
        return 1

    bit_count = floor(log2(num)) + 1
    all_bits_set = (1 << bit_count) - 1
    return num ^ all_bits_set
```

学习笔记：题解给出的答案相对比较复杂，但是原则上也是将数字转换为比他更高一位的二进制，然后减去1，使得所有位都为0，最后进行异或运算。这道题的时间复杂度和空间复杂度都是常数。
