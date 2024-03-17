## 链表算法：快慢指针（fast and slow pointers）

---

### 快慢指针

使用场景：**检测是否有循环，是列表或者链表，可以遍历，需要找到第x个元素或者第x%元素。**

例子：文件系统的符号连接验证，程序之间的依赖关系是否有循环。（比如我在Dataform转换中发生的互相依赖就是循环。）

主要用于链表的算法，比如找到链表的中间节点。链表分为奇数链表和偶数链表，假设偶数链表的情况下中间节点是，后半段的第一个，也就是中间两个节点的第二个，那么用快慢指针算法，如何计算。

快慢指针的起点是一样的，都是最初的节点，如何找到中间节点的代码如下。时间复杂度是O(n)，空间复杂度是O(1)。

```python
def midddleOfList(head):
    fast, slow = head, head
    while fast and fast.next:
        fast = fast.next.next
        slow = slow.next
    return slow
```

只要通过next不断遍历链表，当快指针到达结尾（奇数长度），或者超过了最后一个节点（偶数长度）的时候，慢指针所在的位置就是中点位置。

同时，上面的代码的前提是链表没有循环圈。

那么如何判断链表中是否有循环，代码如下，时间复杂度是O(n)，空间复杂度是O(1)。如果链表中存在循环，则快慢指针一定会在某处相遇，这就是判断条件。

```python
def hasCycle(head):
    fast, slow = head, head
    while fast and fast.next:
        fast = fast.next.next
        slow = slow.next
        if fast == slow:
            return True
    return False
```

那么在有循环的情况下，如何找到循环圈的起点，代码如下。时间复杂度是O(n)，空间复杂度是O(1)。

实现过程如下：

首先进行正常的遍历，和判断是否有循环的代码，是一样的。在快慢指针相遇的地方打断循环，此时慢指针处于循环相遇的节点。或者没有触发该条件，也就是链表没有循环圈，这会在下一步进行判断。

然后判断此时快指针的位置，如果位于结尾直接返回没有结果。

初始化二号慢指针，从头开始遍历，同时也启动慢指针，当两个慢指针相遇的地方就是循环开始的地方。数学推导省略。

```python
def cycleStart(head):
    fast, slow = head, head
    while fast and fast.next:
        fast = fast.next.next
        slow = slow.next
        if slow == fast:
            break
    
    if not fast or not fast.next:
        return None

    slow2 = head
    while slow != slow2:
        slow = slow.next
        slow2 = slow2.next
    return slow
```

### 经典案例：Happy number问题

题目：Identify if repeatedly computing the sum of squares of the digits of number 19 results in 1。对一个数字的每一个位置的数字进行平方和，重复该步骤，如果能得到1就是happy number，反之可能是循环，使用快慢指针就可以判断出该循环。

暴力破解可以使用一个hashset存储所有的已经算过的结果，然后重复判断当前结果是否在hashset中，但是这将花费额外的空间。

要使用快慢指针方法解决这个问题，可以将该过程看作是一个链表的循环检测问题。具体步骤如下：

1. 定义一个函数，该函数接受一个整数作为输入，并返回该整数的各个数字的平方和。
2. 使用快慢指针的方法，在循环中迭代计算输入数字的各个数字的平方和，直到出现以下两种情况之一：
   - 平方和等于1，则返回true。
   - 出现一个之前出现过的平方和，则返回false，这表明出现了循环。
3. 当快指针的平方和等于1时，返回true，否则返回false。

下面是用Python实现的代码：

```python
def compute_square_sum(n):
    square_sum = 0
    while n > 0:
        digit = n % 10
        square_sum += digit ** 2
        n //= 10
    return square_sum

def is_happy_number(num):
    slow = num
    fast = num
    while True:
        slow = compute_square_sum(slow)
        fast = compute_square_sum(compute_square_sum(fast))
        if slow == 1:
            return True
        if slow == fast:
            return False

# 测试
if is_happy_number(19):
    print("The number 19 results in 1.")
else:
    print("The number 19 does not result in 1.")
```

这段代码首先定义了一个函数`compute_square_sum`，用于计算输入数字的各个数字的平方和。然后定义了一个函数`is_happy_number`，该函数使用快慢指针方法来检测输入数字是否会最终得到1。最后进行了一个测试，输出结果表明数字19最终会得到1。

时间复杂度：

在分析时间复杂度时，首先考虑到 `compute_square_sum` 函数，它的时间复杂度取决于输入数字的位数。假设输入数字是 n，则其位数为 O(log n)。

在 `is_happy_number` 函数中，我们使用了快慢指针的方法来检测循环。在最坏情况下，当输入数字不是快乐数时，快指针会在循环中移动，但由于我们使用了快慢指针，所以快指针每次移动的距离是慢指针的两倍。这意味着快指针每次迭代时，循环长度至少减少一半。因此，最多会进行 O(log n) 次迭代。

每次迭代中，`compute_square_sum` 函数都会被调用两次。因此，`is_happy_number` 函数的总时间复杂度可以表示为 O(log n) * O(1)，即 O(log n)。

综上所述，整个算法的时间复杂度为 O(log n)。空间复杂度是常数。因为使用双指针，每次只需指向两个数字而已。

### leetcodes

- 链表的中间[leetcode876 题目描述](https://leetcode.com/problems/middle-of-the-linked-list/description/)

典型的链表快慢指针热身题。

- 链表的最大孪生和[leetcode2130 题目描述](https://leetcode.com/problems/maximum-twin-sum-of-a-linked-list/description/)

重点在于正确的反转前半段链表，让他反过来。是一道非常好的题，可以帮助想清楚，找中点的方法和指针移动的位置和程序。

- 链表循环[leetcode141 题目描述](https://leetcode.com/problems/linked-list-cycle/description/)

典型的链表循环题。

- 链表循环 2[leetcode142 题目描述](https://leetcode.com/problems/linked-list-cycle-ii/description/)

找循环链表节点位置的题。

- 找出重复数字[leetcode287 题目描述](https://leetcode.com/problems/find-the-duplicate-number/description/)

找出数组中唯一重复的数字。弄清楚 index 和 num 的关系，发现其实是一个寻找循环链表的循环节点的问题。正常人没见过的很难一下子解决。
