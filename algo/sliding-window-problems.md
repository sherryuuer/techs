## 滑动窗口

### 概念复习

滑动窗口算法是一种用于解决序列型问题的常用技巧，通常用于求解连续子序列或子数组的最优解。该算法通过维护一个固定大小的窗口（通常是一个子数组或子序列），在序列上滑动窗口并不断更新窗口内的状态，以解决特定的问题。

适用范围：滑动窗口算法适用于一类需要对连续子序列或子数组进行求解的问题，通常是在数组、字符串或链表等序列型数据结构上。一般来说，如果问题可以被转化为在固定大小的窗口上进行操作，并且可以通过滑动窗口来更新窗口状态以达到最优解的目的，那么滑动窗口算法就可以被应用。

典型问题：

- 最大子数组和（Maximum Subarray Sum）：给定一个整数数组，找到一个连续的子数组，使得子数组的和最大。
- 最小覆盖子串（Minimum Window Substring）：给定一个字符串 S 和一个字符串 T，找到 S 中包含 T 中所有字符的最短连续子串。
- 长度最小的子数组（Minimum Size Subarray Sum）：给定一个整数数组和一个整数 target，找到数组中满足其和 ≥ target 的长度最小的连续子数组。
- 无重复字符的最长子串（Longest Substring Without Repeating Characters）：给定一个字符串，找到其中不含有重复字符的最长子串的长度。

这些典型问题都可以使用滑动窗口算法来解决。通过维护一个窗口，根据问题的要求不断调整窗口的大小和位置，可以高效地解决这些问题，并在时间复杂度上取得较优的效果。

### 问题1:Find Maximum in Sliding Window

力扣中是239. Sliding Window Maximum题hard难度。

题中给出一个数组，和一个窗口大小 k，窗口将从最左侧滑动到最右侧，每次只能判断 k 个元素，找到每个窗口的最大元素加入结果数组。每次窗口移动的步长为 1。最后返回这个结果数组。

例如，nums = [1, 3, -1, -3, 5, 3, 6, 7]，窗口长度为k = 3，那么输出结果就是[3, 3, 5, 5, 6, 7]。

代码如下：

```python
def find_max_sliding_window(nums, w):
    def quick_sort(arr):
        if len(arr) <= 1:
            return arr
        pivot = arr[len(arr) // 2]  # 选择中间元素作为基准值
        left = [x for x in arr if x < pivot]
        middle = [x for x in arr if x == pivot]
        right = [x for x in arr if x > pivot]
        return quick_sort(left) + middle + quick_sort(right)

    # 获取数组最大值
    def get_max(arr):
        sorted_arr = quick_sort(arr)
        return sorted_arr[-1] if sorted_arr else None

    left, right = 0, w
    res = []
    
    while right <= len(nums):
        res.append(get_max(nums[left:right]))
        left += 1
        right += 1

    return res
```

在我的学习用例上过了，然后败给了力扣网友的测试用例。当然了！这是必然环节。怎么可能让我一直排序。

引入**单调递减队列（monotonically decreasing queue）**是一种数据结构，用于存储元素，并确保队列中的元素按照单调递减的顺序排列。换句话说，队列中的元素从队首到队尾是依次递减的。

单调递减队列通常用于解决一些与滑动窗口相关的问题，特别是在需要在滑动窗口中找到最大值或最小值的情况下。通过使用单调递减队列，我们可以在常数时间内获取当前滑动窗口的最大值。

单调递减队列的常见操作包括：

1. **push**：向队尾添加一个元素，并确保队列的单调递减性质不变。
2. **pop**：从队首弹出一个元素。
3. **get_max**：获取当前队列中的最大元素，通常位于队首。

在实际应用中，通常会结合双端队列（deque）来实现单调递减队列，因为双端队列支持在队首和队尾高效地进行添加和删除操作，非常适合实现单调递减队列的需求。

重新优化，代码如下：

```python
class Solution(object):
    def maxSlidingWindow(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: List[int]
        """
        from collections import deque
        output = []
        q = deque()  # from max to min
        left = right = 0

        while right < len(nums):
            # pop smaller value from queue
            while q and nums[q[-1]] < nums[right]:
                q.pop()
            q.append(right)

            # remove left value if it is out of bounds
            if q[0] < left:
                q.popleft()

            # detect the right is increased to the window size k
            if (right + 1) >= k:
                output.append(nums[q[0]])
                left += 1
            
            right += 1
        
        return output
```

学习笔记：

这是一段很高效的代码，使用单调递减queue的思想。关注主代码的部分。

第一个逻辑段落，保证每次加入deque的都是最小的元素，如果deque中有小于新加入的元素的元素，则全部丢弃。这保证了queue中剩下的元素没有小于新元素的。通过这个逻辑就维护了queue中一定保存一个最大的元素。也就是 q[0] 第一个元素。我们最后总会加入right的元素。所以q中的元素，要么只有两个，要么是包括最大和大于right元素的其他所有元素的一个queue。右侧的right元素一定是最小的。整个queue是一个单调递减状态的维护。

第二个逻辑段落，保证了q内最大元素在窗口内，如果最左的元素出了左侧边界，就丢弃它。

第三个逻辑段落保证在窗口到达指定大小后将结果存入output数组，最后移动指针。

我很喜欢这个解法。它的时间复杂度达到了O(n)，空间复杂度也只有窗口大小O(k)。覆盖了所有力扣的测试用例，耶耶耶。

### 问题2:Minimum Window Subsequence

最小窗口子序列问题。一个经典的字符串处理问题。问题的目标是在给定的字符串S中找到包含另一个字符串T的最小窗口子序列。换句话说，要找到S中的最短连续子字符串，其中包含T的所有字符，顺序不必完全一致，但相对顺序必须一致。

例如，如果S = "abcdebdde"，T = "bde"，那么结果应该是"bcde"，因为它是S的一个最小窗口子字符串，包含T的所有字符。

这个问题可以通过使用滑动窗口技巧来解决。算法步骤大致如下：

- 初始化两个指针i和j分别指向S和T的开头。
- 用指针i遍历字符串S，当找到包含T中所有字符的子序列后，记录该子序列的起始和终止位置。同时反向遍历j以找到小于最小长度的起始位置。
- 重制j指针进行下一次遍历，以便找到更小的窗口。比如这个`min_window("abcdbebe", "bbe")`这个例子，找到位于index-1上的b之后可以得到一个长度为5的子窗口，但是如果进行下一轮遍历，可以找到一个更短的窗口长度为4的子串。
- 最后返回最短子窗口的字符串。

```python
def min_window(S, T):
    n = len(S)
    m = len(T)

    start = -1  # 记录最小窗口子序列的起始位置
    minLen = float('inf')  # 记录最小窗口子序列的长度

    i = 0
    j = 0

    while i < n:
        if S[i] == T[j]:
            j += 1  # 在T中找到一个字符，向后移动T的指针
            if j == m:  # 如果T的指针已经到达末尾，即找到一个包含T的子序列
                end = i  # 记录当前窗口的结束位置
                j -= 1  # 回退T的指针，以便找到最小窗口子序列

                # 向前移动T的指针，直到不再满足条件
                while j >= 0:
                    if S[i] == T[j]:
                        j -= 1
                    i -= 1
                i += 1  # 退回到满足条件的位置

                # 计算当前窗口子序列的长度，并更新最小窗口的起始位置和长度
                if end - i + 1 < minLen:
                    minLen = end - i + 1
                    start = i
                # 这里很重要，要把j重制到开头，以便下一次遍历和比较
                j = 0
        i += 1

    if start == -1:
        return ""
    else:
        return S[start: start + minLen]
```

学习笔记：

这道题的过程就像是i和j两面墙不断的压缩，以求找到更短的子串的过程。总结一下包括如下要点：

在外层i的遍历中，重要的是，找到j所在字符串的所有的字符，这意味着我们总能找到`S[i] == T[j]`的条件，并且一直遍历到T的结尾。当找到之后，进行反向遍历，以求找到这个情况下的起点。找到起点start后，这轮遍历结束。我们初始化的子串的长度是无穷大，所以第一轮我们总能找到一个小窗口。

然后我们需要初始化起点i为刚刚找到的窗口的起点，因为你需要一个更小的窗口，你必须比刚刚的起点走的更远才行。然后初始化j到T的开头，重新开始新一轮的搜索。直到找到另一个符合条件的窗口，进行窗口长度的比较。

最终找到最小的窗口，得到最后的结果。这是一个很巧妙的过程。一开始单纯的思考很难想到。但是一旦清楚了过程，就会发现这里，滑动窗口，是怎样一种灵活的解决问题的方式。

时间复杂度是内外两层嵌套的乘积也就是O(nxm)，空间复杂度是优秀的O(1)。
