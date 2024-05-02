## 排序算法：循环排序 Cyclic Sort

### 概念解析

Cyclic Sort（循环排序）是一种原地排序算法，用于对包含从 1 到 n 的连续整数的数组进行排序。这种排序算法的主要思想是将数组元素按照它们的值进行排序，并将每个元素放在其正确的位置上，即将元素 i 放在索引为 i-1 的位置上。

Cyclic Sort 的步骤如下：

1. 遍历数组，对于每个元素 nums[i]：
   - 如果 nums[i] 的值等于 i+1，则将其放在正确的位置上，继续遍历下一个元素。
   - 否则，将 nums[i] 与位于索引为 nums[i]-1 的元素进行交换，使得 nums[i] 放在正确的位置上。

2. 继续遍历数组，直到所有元素都被放置在它们的正确位置上。

Cyclic Sort 的时间复杂度为 O(n)，空间复杂度为 O(1)。它是一种简单且高效的排序算法，适用于排序包含连续整数的数组，例如 [1, n]。然而，Cyclic Sort 仅适用于特定类型的输入数据，对于包含重复元素或其他类型的数据，可能需要其他更复杂的排序算法。

尽管 Cyclic Sort 是一种简单且高效的排序算法，但它也存在一些缺点：

1. 仅适用于特定类型的输入数据：Cyclic Sort 只适用于包含从 1 到 n 的连续整数的数组。对于其他类型的数据或包含重复元素的数组，Cyclic Sort 不再适用，需要使用其他排序算法。

2. 不稳定性：Cyclic Sort 在交换元素时，并没有考虑元素的相对顺序。因此，在排序过程中可能会改变相同元素之间的顺序，导致排序结果不稳定。

3. 不适用于大规模数据集：尽管 Cyclic Sort 的时间复杂度为 O(n)，但它需要遍历整个数组多次，因此对于大规模数据集，其性能可能不如一些更高级的排序算法，如快速排序或归并排序。

4. 需要原地排序：Cyclic Sort 是一种原地排序算法，即不需要额外的空间来存储排序结果。然而，这也意味着它会修改原始数组，而不是生成一个新的排序数组，这可能不适用于某些应用场景。

所以尽管 Cyclic Sort 在某些特定情况下非常有效，但它也有一些限制和缺点，需要根据具体的问题和数据集选择合适的排序算法。

根据它的特性，Cyclic Sort 适合解决以下问题：

1. 排序包含连续整数的数组：Cyclic Sort 最适合用于对包含从 1 到 n 的连续整数的数组进行排序。由于数组中的元素是连续的，因此可以利用元素的值与索引的关系进行排序，而不需要使用比较排序算法。

2. 查找缺失的元素：在排序过程中，如果发现某个元素没有被放置在它正确的位置上，就意味着这个元素是缺失的。因此，通过 Cyclic Sort 可以快速查找数组中缺失的元素。

3. 查找重复的元素：类似地，如果在排序过程中发现某个位置上已经存在相同的元素，则意味着这个元素是重复的。因此，通过 Cyclic Sort 也可以快速查找数组中的重复元素。

总的来说，Cyclic Sort 主要适用于解决排序连续整数数组、查找缺失元素和重复元素等问题，特别是在对简单数据结构进行排序和查找时非常有效。

### 问题1:Missing Number

力扣268，难度为easy。

给定一个数组 nums，包括 n 个不同的数字在 [0, n] 的范围内，返回这中间唯一确实的数字。比如 nums = [3, 0, 1]， 因为一共有 len(nums) 即 3 个数字，所以应该是 [0, 3] 内的数字，缺失的就是 2。

解题思路：

- 从第一个数字开始遍历。
- 如果数字和索引不匹配，就将它和正确位置上的数字调换。
- 如果位置正确或者大于数组长度，就跳过。
- 迭代数组后，将每个数字和他的索引比较。
- 第一个出现不等于索引的数字的索引，就是缺失数字。

代码如下：我保留了我一开始做错的部分，就是用for循环得到了错误的逻辑，单用一层for循环无法找到正确解，这道题必须用while循环，不断遍历直到每一个位上的数字都满足无法再移动了为止。如果选择之后罗列的高斯求和之类的方法，会更简单和清晰，但是我们就是来犯错和进步的是吧，就是喜欢受苦（bushi

```python
def missingNumber(self, nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    # n = len(nums)
    # for i in range(n):
    #     val = nums[i]
    #     if val < n and val != nums[val]:
    #         nums[val], nums[i] = nums[i], nums[val]
    #     else:
    #         continue
    #
    # for i in range(n):
    #     if nums[i] != i:
    #         return i

    # return n
    len_nums = len(nums)
    index = 0

    while index < len_nums:
        value = nums[index]

        if value < len_nums and value != nums[value]:
            nums[index], nums[value] = nums[value], nums[index]

        else:
            index += 1

    for x in range(len_nums):
        if x != nums[x]:
            return x
    return len_nums
```

当然这道题还有很多其他简单解法，比如通过高斯求和公式：

```python
def missingNumber(nums):
    n = len(nums)
    
    # 计算数组中所有元素的和
    nums_sum = sum(nums)
    
    # 计算从 1 到 n 的连续整数的和
    expected_sum = (n * (n + 1)) // 2
    
    # 返回缺失的数字
    return expected_sum - nums_sum

nums = [9, 6, 4, 2, 3, 5, 7, 0, 1]
print(missingNumber(nums))
```

或者其他通过数学技巧得到的结果，比如：

```python
def find_missing_number(nums):
    res = len(nums)
    for i in range(len(nums)):
        res += (i - nums[i])
    return res
```
学习笔记：这道题的重点就是不利用额外空间得到结果。时间复杂度为O(n)，空间复杂度O(1)。

### 问题2:Find the Corrupt Pair

这道题和上面的题很像。给出1 - n个数字的数组，当然数组长度就是n，比如 [4, 1, 3, 4, 5]，返回其中缺失的数字和重复的数字：（missing，duplicated）。

解题思路仍然是先进行 cyclic 排序，然后遍历找到位置错误的数字，最后返回结果即可。

代码尝试：基本将上一道题的题解复制过来进行修改，就可以得到结果了。

```python
def find_corrupt_pair(nums):
    len_nums = len(nums)
    index = 0

    while index < len_nums:
        value = nums[index]

        if value <= len_nums and value != nums[value - 1]:
            nums[index], nums[value - 1] = nums[value - 1], nums[index]

        else:
            index += 1
    print(nums)
    for x in range(len_nums):
        if (x + 1) != nums[x]:
            return [x + 1, nums[x]]
    return -1
```

学习笔记：时间复杂度为O(n)，空间复杂度为O(1)。

### 问题3:First Missing Positive

找到一个数组中的最小的非负数。并且要使用O(n)的复杂度。

比如[3, 4, -1, 2]那么最小的缺失正数就是1，比如[3, 4, 2, 1]那么最小的缺失正数就是5，比如[7, 8, 9, 10]那么最小的缺失正数就是1。

解题思路同样是用 cyclic 排序，然后再次遍历找到第一个和index不符合的数字，如果全都符合那么就是长度加一。

代码尝试：

```python
def smallest_missing_positive_integer(nums):
    for i in range(len(nums)):
        correct_spot = nums[i] - 1
        
        while 1 <= nums[i] <= len(nums) and nums[i] != nums[correct_spot]:
            nums[i], nums[correct_spot] = nums[correct_spot], nums[i]
            correct_spot = nums[i] - 1

    for x in range(len(nums)):
        if (x + 1) != nums[x]:
            return x + 1
    
    return len(nums) + 1
```

或者：

```python
def smallest_missing_positive_integer(nums):
    i = 0
    while i < len(nums):
        correct_spot = nums[i] - 1
        if 0 <= correct_spot < len(nums) and nums[i] != nums[correct_spot]:
            nums[i], nums[correct_spot] = nums[correct_spot], nums[i]
        else:
            i += 1

    for i in range(len(nums)):
        if i + 1 != nums[i]:
            return i + 1
    return len(nums) + 1
```

只是尽心cyclic排序的方式稍有不同而已。

学习笔记：如题要求的时间复杂度O(n)，空间复杂度为O(1)。但是有固定的模式，因此有限定的问题。
