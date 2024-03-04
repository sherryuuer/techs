## 回溯算法：排列（Permutations）

---

### 数学含义

简单说就是一种实现数学全排列的算法。

分别使用递归和迭代的方法实现。递归的方法比较好理解，迭代的方法比较符合数学直觉。

递归的算法不断call出helper函数，直到遍历到最后一位的最后，return一个base情况也就是空数组。拿到空数组后，从最后一位有效位数字开始，不断在数组的各个子数组的，所有index上插入这个有效数字，直到处于第一位的有效数字也被插入所有的数组。

讲起来很难懂，只要看一下代码实现的print打印结果就可以很好理解了。

```python
# Time: O(n^2 * n!)
def permutationsRecursive(nums):
    return helper(0, nums)

def helper(i, nums):
    if i = len(nums):
        return [[]]

    resPerms = []
    # 通过这一行每次都取回迭代的结果
    perms = helper(i + 1, nums)
    for p in perms:
        for j in range(len(p) + 1):
            pcopy = p[:]
            pcopy.insert(j, nums[i])
            resPerms.append(pcopy)
    return resPerms
```

迭代算法刚好相反，先给出base情况空数组，然后从一地位开始迭代遍历，每次试图插入一个数字，直到最后一个数字也被完全插入所有的位置。

```python
# Time: O(n^2 * n!)
def permutationsIterative(nums):
    perms = [[]]

    for n in nums:
        nextPerms = []
        for p in perms:
            for j in range(len(p) + 1):
                pcopy = p[:]
                pcopy.insert(j, nums[i])
                resPerms.append(pcopy)
        perms = nextPerms
    return perms
```

总体上感觉这个数学概念的代码实现还是挺难的，可以多做几遍，也可以尝试用树等方法，多探索其他的方法可以加深理解。

另外虽然将这个算法归于回溯算法上面却没有这个代码，但是下面的leetcode的47题，是用回溯的算法很好的实现。

下面也是一种用回溯进行实现的方法：我在直觉上对它挺费解的。

```python
def permutations(nums):
    def backtrack(start, end):
        if start == end:
            result.append(nums[:])
        else:
            for i in range(start, end):
                # 交换元素
                nums[start], nums[i] = nums[i], nums[start]
                # 递归调用
                backtrack(start + 1, end)
                # 恢复原始状态，以便下一次迭代
                nums[start], nums[i] = nums[i], nums[start]

    result = []
    backtrack(0, len(nums))
    return result

# 示例
nums = [1, 2, 3]
result = permutations(nums)
print("全排列结果：", result)
```

总之这个主题还需要再探索一下。

都没有llm聪明，问他实现算法：他说

以下是使用Python实现的全排列算法示例：

```python
import itertools

def permutations(nums):
    return list(itertools.permutations(nums))

# 示例
nums = [1, 2, 3]
result = permutations(nums)
print("全排列结果：", result)
```

这段代码使用了Python标准库中的`itertools`模块的`permutations`函数来生成给定列表的所有排列。然后将结果转换为列表并返回。

### leetcodes

- 排列[leetcode46 题目描述](https://leetcode.com/problems/permutations/description/)

- 排列2[leetcode47 题目描述](https://leetcode.com/problems/permutations-ii/description/)

因为有重复数字的存在，使用树-回溯算法-以及Counter字典记数。说实话感觉没有完全在脑内重现整个流程，还需要进一步深入理解（2024/1/20）！
