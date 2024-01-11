## 数组算法：双指针（two-pointers）

---

### 双指针和滑动窗口的区别是什么

两种算法都有两个指针，但是滑动窗口关注两个点之间的区间，双指针主要关注两个点。

### 双指针的相关概念

双指针需要数组是排列好的。

双指针关注左右两端的计算结果和目标的比较，通过条件，判断是否移动指针。

while 的条件通常是 L 小于 R，让指针遍历整个数组。

### leetcode 逐行解析

- 判断是否是回文[leetcode125 题目描述](https://leetcode.com/problems/valid-palindrome/description/)

一条字符串是一条英文语句，踢掉里面除了数字，和字母的其他字符，然后进行判断，是非常典型的双指针题目。

```python
class Solution(object):
    def isPalindrome(self, s):
        """
        :type s: str
        :rtype: bool
        """
        # 对字符串进行处理，去掉除了字母和数字以外的字符
        s = "".join(char for char in s if char.isalpha()
                    or char.isdigit()).lower()
        # 设定左右指针在头尾
        L, R = 0, len(s) - 1
        # 判断当左指针小于右边的指针的时候，while循环
        while L < R:
            # 当发现左指针的字符和右边的不一样的时候，返回false
            if s[L] != s[R]:
                return False
            # 当条件尚且满足的时候，移动左右指针，缩小范围
            L += 1
            R -= 1
        # 一切可以顺利结束的话，返回true
        return True
```

- 两数之和之有序数组[leetcode167 题目描述](https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/description/)

给定一个 target，和一个有序数组，返回和为这个 target 的数字的 index。典型的双指针题目。解决起来应该很丝滑。

```python
class Solution(object):
    def twoSum(self, numbers, target):
        """
        :type numbers: List[int]
        :type target: int
        :rtype: List[int]
        """
```

- 去重之有序数组[leetcode26 题目描述](https://leetcode.com/problems/remove-duplicates-from-sorted-array/)

80 题的简单版本，去重并且按照原先的顺序排列，返回长度 k 和前 k 是有效元素的有效数组。
两个指针同时从 0 出发，左指针控制答案顺序，负责插入，右指针负责遍历。

```python
class Solution(object):
    def removeDuplicates(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
```

- 去重之有序数组 2[leetcode80 题目描述](https://leetcode.com/problems/remove-duplicates-from-sorted-array-ii/description/)

是上面 26 题的复杂版本，可以保留一个重复，多出的就不可以了。

```python
class Solution(object):
    def removeDuplicates(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
```

- 装水最多的容器[leetcode11 题目描述](https://leetcode.com/problems/container-with-most-water/description/)

感觉题目描述不如这张图片来的实在：

[题目解析图](https://s3-lc-upload.s3.amazonaws.com/uploads/2018/07/17/question_11.jpg)

给定的数组长度想象为容器的底座，数组中的数字，是分割容器的墙壁。

根据示例：height = [1, 8, 6, 2, 5, 4, 8, 3, 7]

那么第一个 8 的 index 是 1，最后一个 7 的 index 是 8，底边长就是 8-1，很明显。

如何确定 8 和 7 呢。看到是两个端点，自然可以想到双指针。设定一个最大初始化的容量，不断移动指针更新容量，最终找到最大容量。一开始会以为两个两个隔板之间的数字有没有什么关系，其实没有，只需要关注两个隔板的高度就可以了。

hint：如果使用双层 forloop 会被力扣网友跑出超时结果。更新左右指针的时候，如果那个数字比原本当前的高度最短的还短就没有计算的必要了。

其实我觉得这个题更像是一个滑动窗口！

```python
class Solution(object):
    def maxArea(self, height):
        """
        :type height: List[int]
        :rtype: int
        """
```

- 收集雨水[leetcode42 题目描述](https://leetcode.com/problems/trapping-rain-water/description/)

这道题真的有趣，依然是看图就能明白题意：

[题目解析图](https://assets.leetcode.com/uploads/2018/10/22/rainwatertrap.png)

比方说给的这个高度数组 height = [0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]

实质上给的是一个地形图，那么就是问这地形能兜住多少的雨水。例题如图答案是 6。这也是一道很棒的题了。

hint: min(maxL, maxR) - height[i]

```python
class Solution(object):
    def trap(self, height):
        """
        :type height: List[int]
        :rtype: int
        """
```
