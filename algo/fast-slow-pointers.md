## 链表算法：快慢指针（fast and slow pointers）

---

### 快慢指针

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

### leetcode 逐行解析

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
