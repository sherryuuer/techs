## Kadane算法

---
### 概念引导

Kadane算法是用于求解最大子数组和问题的经典算法。最大子数组和问题是指在一个数组中找到一个连续的子数组，使其元素之和最大。Kadane算法的思想相对简单，但非常有效，其时间复杂度为O(n)，其中n是数组的长度。

下面是Kadane算法的基本思想：

1. 遍历数组，同时维护两个变量：当前最大子数组和（`max_so_far`）和当前最大子数组的结束位置（`end_here_max`）。

2. 对于数组中的每个元素，更新当前最大子数组的结束位置和当前最大子数组和，同时更新当前最大子数组和如果加上当前元素后是否更大，如果是，则将当前元素作为新的子数组的起始位置。

3. 在遍历过程中不断更新全局最大子数组和（`max_so_far`）。

4. 遍历完成后，`max_so_far`中存储的就是最大子数组的和。

下面是一个简单的伪代码表示Kadane算法：

```plaintext
initialize:
    max_so_far = 0
    end_here_max = 0

loop over each element in the array:
    end_here_max = max(current_element, end_here_max + current_element)
    max_so_far = max(max_so_far, end_here_max)

return max_so_far
```

这个算法的关键在于理解如何在遍历数组的过程中动态更新当前最大子数组和和全局最大子数组和。通过仔细观察，可以发现这个算法确保了在每一步都考虑了截止到当前位置的最大子数组和，并不断更新全局最大子数组和。这种方法保证了算法的正确性，并且在遍历完成后得到了全局的最优解。

总之，Kadane算法是解决最大子数组和问题的一种高效方法，通过动态规划的思想在线性时间内求解问题，适用于处理大规模的数据集。

### Python代码实现

从一个数组中找到一个非空子集，拥有最大和，求这个sum。（前提条件是连续元素子集）

暴力破解法，使用双层嵌套loop检查每一个sum和maxsum比较。

```python
# Brute Force: O(n^2)
def bruteForce(nums):
    maxSum = nums[0]

    for i in range(len(nums)):
        curSum = 0
        for j in range(i, len(nums)):
            curSum += nums[j]
            maxSum = max(maxSum, curSum)
    return maxSum
```

Kadane算法中心思想是，当curSum出现负数的时候，直接放弃前面所有的元素，重新开始累加。

```python
# Kadane's Algorithm: O(n)
def kadanes(nums):
    maxSum = nums[0]
    curSum = 0

    for n in nums:
        curSum = max(curSum, 0)
        curSum += n
        maxSum = max(maxSum, curSum)
    return maxSum
```

Kadane算法是滑动窗口算法的基础，通过小的修改，可以改编成滑动窗口算法，用于求出该窗口的index位置。

```python
# Return the left and right index of the max subarray sum,
# assuming there's exactly one result (no ties).
# Sliding window variation of Kadane's: O(n)
def slidingWindow(nums):
    maxSum = nums[0]
    curSum = 0
    maxL, maxR = 0, 0
    L = 0

    for R in range(len(nums)):
        if curSum < 0:
            curSum = 0
            L = R

        curSum += nums[R]
        if curSum > maxSum:
            maxSum = curSum
            maxL, maxR = L, R 

    return [maxL, maxR]
```
