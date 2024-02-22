## 数组算法：滑动窗口（sliding window）

---

### 固定尺寸窗口 fixed size

固定窗口算法，通常在一个数组中，用于解决字符串，数组子串的问题。通常这个子串窗口是连续的，原本需要双层嵌套，甚至三层嵌套的数组问题，通过滑动窗口的方法，通常可以将时间复杂度将为n。是一种好用的算法。

固定尺寸就是问题中的窗口大小通常是固定的。

考虑这样一个问题，给定一个数组，在窗口大小为k的子数组中，是否存在重复的元素。（感觉自己描述问题的能力挺差的）这就是一个典型的固定尺寸窗口问题。

如果使用暴力破解的方法，简单直接就是两层嵌套：

```python
def closeDuplicatesBruteForce(nums, k):
    for L in range(len(nums)):
        for R in range(L + 1, min(len(nums), L + k)):
            if nums[L] == nums[R]:
                return True
    return False
```

如果使用滑动窗口，则动态的进行比较从而将时间复杂度降低为O(n)。

```python
def closeDuplicates(nums, k):
    # 设定一个set集合用于存储所在范围的元素
    window = set()
    # 初始化左侧指针为0
    L = 0
    # 动态地使用右指针R遍历数组作为窗口的右边边界
    for R in range(len(nums)):
        # 当窗口大小超出了k的时候将左侧指针的元素从window删除并且移动一个位置
        if R - L + 1 > k:
            window.remove(nums[L])
            L += 1
        # 检测当前的R位置的元素是否在window中，在的话则表示有重复直接结束函数执行
        if nums[R] in window:
            return True
        # 每次都将新的R加入window以便后续比较使用
        window.add(nums[R])
    # 以上都不满足，则返回False
    return False
```


- 包含重复 2[leetcode219 题目描述](https://leetcode.com/problems/contains-duplicate-ii/description/)
- 大小为 k 且均值大于等于阈值的数组[leetcode1343 题目描述](https://leetcode.com/problems/number-of-sub-arrays-of-size-k-and-average-greater-than-or-equal-to-threshold/description/)

### 变动尺寸窗口 variable size

给定一个数组，找到一个长度最长的子数组，该数组拥有相同的数字，返回它的长度。

```python
def longestSubarray(nums):
    length = 0
    L = 0
    # 不断移动右指针
    for R in range(len(nums)):
        # 当左右指针的数字不相等， 将左指针更新为右边指针所在的位置
        if nums[L] != nums[R]:
            L = R
        # 每次移动右指针，更新最长的子串长度
        length = max(length, R - L + 1)
    return length
```

注意，每次右指针遍历的时候，内部一定是先判断左右是否相等，再进行长度计算，如果先进行计算长度，可能会出现错误的计算，因为这个时候，可能右指针已经遇到了不同元素。

另一个例子：

给定一个数组，和一个target目标数字，找到一个最小长度的子数组，该数组的数字之和等于或者大于target数字。（假设该数组中的元素都正数, 如果数组中的元素有负数，则是另一种情况）。

```python
def shortestSubarray(nums, target):
    L, total = 0, 0
    length = float("inf")

    for R in range(len(nums)):
        total += nums[R]
        while total >= target:
            length = min(length, R - L + 1)
            total -= nums[L]
            L += 1
    return 0 if length == float("inf") else length
```

该算法的时间复杂度是O(n)，虽然中间残套了一层while循环，但是循环内的计算并不是每次都会被执行，是作为条件判断存在的，所以只有一层时间复杂度。

力扣：

- 最小子数组和[leetcode209 题目描述](https://leetcode.com/problems/minimum-size-subarray-sum/description/)
- 无重复最长子串[leetcode3 题目描述](https://leetcode.com/problems/longest-substring-without-repeating-characters/description/)
- 可重复可替换最长重复字符串[leetcode424 题目描述](https://leetcode.com/problems/longest-repeating-character-replacement/description/)
