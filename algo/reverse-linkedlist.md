## 反转单链表及其变体问题

---

### 原地反转Linked list问题（in-place reversal linked list）

何种情况需要这种算法，当你需要反转链表但是不想花费额外的内存空间。一般来说该问题的变种，如果情况复杂的话，时间复杂度可能会达到n的平方，（如果使用stack方法，也可以达到时间复杂度n，但是将花费额外的空间）但是使用原地反转之需要n时间复杂度。空间复杂度同样，在普通方法下需要额外的空间储存整个新的反转链表，但是使用原地反转之需要常数空间复杂度。

从直觉上理解原地反转法，需要做的只是，将指针转个头，指向下一个node的指针，指向前一个即可。

这个算法的很多变体，比如反转后半段链表，都可以使用相似的技巧，关键点，就是灵活操作节点的指针。

适用范围：

如果满足以下条件：适用该技巧
1. 问题需要反转给定的链表，无论是作为最终目标，还是作为解决方案的中间步骤。
2. 修改链表必须原地进行，也就是说，不能使用超过 O(1) 的额外内存空间。
3. 此模式也适用于问题需要反转给定链表的选定部分。

如果满足以下任一条件：则不使用
1. 输入数据不是链表形式。
2. 我们明确需要使用额外的内存。
3. 我们不允许修改输入的链表。

代码：其中最需要注意的，一个事while后的条件，另一个就是赋值时候的顺序，prev和cur的顺序如果反转就无法得到正确的结果，因为这两行都用到了cur变量，要注意，nxt不需要赋值，因为在下一轮loop中的开头就会进行赋值了。

```python
from linked_list import LinkedList
from linked_list_node import LinkedListNode
            
def reverse(head):

    cur = head
    prev, nxt = None, None

    while cur:
        nxt = cur.next
        cur.next = prev
        prev = cur
        cur = nxt
    
    head = prev
    return head
```

### 原地反转问题变体：Reverse Nodes In Even Length Groups（leetcode 2074 median）

这是一道描述起来很麻烦的问题，但是描述问题很重要。

比如如下链表：

8 → 13 → 9 → 3 → 12 → 2 → 7 → 4 → 5 → 1 → 11 → 0 → 14 → 6 → 10

首先要对这个链表分组分为如下五组（如果最后的不够就那么放着就好了）：

- 8
- 13 → 9
- 3 → 12 → 2
- 7 → 4 → 5 → 1
- 11 → 0 → 14 → 6 → 10

目标是对于是偶数长度的子链表进行反转。

这个代码把我搞的晕头转向哈哈哈。自己也是通过观察educative网上的动图理解了前后的步骤。

```python
def reverse_even_length_groups(head):
    # 定义一个函数，用于对偶数长度的链表组进行反转
    # 参数 head 是指向链表头节点的指针
    
    # 初始化 prev 为 head，用于迭代遍历链表中的节点
    prev = head  
    # 初始化组的长度为 2，因为第一个组只有一个元素不必进行反转
    l = 2

    # 使用 while 循环遍历链表，只要当前节点 prev 有下一个节点，就进行循环
    while prev.next:
        # 将 prev 赋值给 node，并初始化计数器 n 为 0
        node = prev
        n = 0
        # 使用 for 循环遍历长度为 l 的组
        for i in range(l):
            # 如果没有下一个节点，则跳出循环
            if not node.next:
                break
            # 统计节点数量
            n += 1
            # 移动到下一个节点
            node = node.next
        
        # 以下判断如果节点数量为偶数，则进行反转操作
        if n % 2:  # 输出有余数，为真不进行反转
            prev = node
        else:      # 进行反转
            # 记录当前组的下一个节点
            reverse = node.next
            # 初始化 curr 为 prev 的下一个节点，此时prev还是上一个组结尾
            curr = prev.next
            # 遍历当前组的节点进行反转操作
            for j in range(n):
                curr_next = curr.next
                curr.next = reverse
                reverse = curr
                curr = curr_next
            # 将 prev 的下一个节点指向当前组的最后一个节点
            prev_next = prev.next
            prev.next = node
            # 移动 prev 到当前组的第一个节点之前
            prev = prev_next
        # 组的长度增加 1，准备处理下一个组
        l += 1

    # 循环结束后，返回链表的头节点
    return head
```

### 原地反转问题变体：Reverse Nodes in k-Group（leetcode 25 hard）

这道问题虽然leetcode是hard难度的，但是从题目描述上，似乎比上面那道描述起来简单，一个链表，每k个元素，进行一个反转操作。如果最后不足k个就不操作了。

代码部分：我自己在理解上面一题的基础上，做了下面的答案：

```python
def reverse_k_groups(head, k):
    # 定义一个函数，用于对 k 长度的链表组进行反转
    # 参数 head 是指向链表头节点的指针
    
    # 初始化 prev 为 head 之前的 None，用于迭代遍历链表中的节点
    prev = LinkedListNode(None)
    # prev 的下一个就是 head
    prev.next = head
    new_head = None

    # 使用 while 循环遍历链表，只要当前节点 prev 的下一个还有就可以继续遍历 
    while prev.next:
        # 将 prev 赋值给 node，并初始化计数器 n 为 0
        node = prev
        n = 0
        # 使用 for 循环遍历长度为 k 的组
        for i in range(k):
            # 如果没有下一个节点，则跳出循环
            if not node.next:
                break
            # 统计节点数量
            n += 1
            # 移动到下一个节点
            node = node.next
        # 最终 node 会移动到当前组的最后一个节点, 也就是未来的头
        if not new_head:
            new_head = node
        
        # 如果长度不构成 k 了，说明到头了也，跳出循环返回结果
        if n != k:
            break
        # 以下判断如果节点数量为k，则进行反转操作
        else:
            # 记录当前组的下一个节点
            reverse = node.next
            # 初始化 curr 为 prev 的下一个, 也就是这 k 个元素的头
            curr = prev.next
            
            # 遍历当前组的节点进行反转操作，总之就是转圈指向
            for j in range(n):
                curr_next = curr.next
                curr.next = reverse
                reverse = curr
                curr = curr_next
            
            # 将 prev 的下一个节点指向当前组的最后一个节点，这是为了将两个组连起来
            prev_next = prev.next
            prev.next = node
            # 移动 prev 到当前组的第一个节点
            prev = prev_next
    
    
    # 循环结束后，返回链表的头节点
    return new_head
```

中间我因为没有追踪链表的头节点，所以我的输出是从第一组的最后一个元素（也就是原始的head）开始。

如下错误：

```python
Input
[3,4,5,6,2,8,7,7] , 3
Output
[3,8,2,6,7,7]
Expected
[5,4,3,8,2,6,7,7]
```

所以我增加以下操作，更新了最新的head，这里注意缩紧位置，一定是遍历完第一个组以后。不然head还是老head。

```python
new_head = node
# 最终 node 会移动到当前组的最后一个节点, 也就是未来的头
if not new_head:
    new_head = node
```

如此就通过了test case了。

其实如果使用额外的空间，使用stack的方法其实最好理解，每次推入k个元素，倒序取出来，就很方便。

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def reverse_k_groups(head, k):
    # 定义一个辅助函数，用于将栈中的元素依次连接到链表中
    def connect_stack_to_list(prev, stack):
        while stack:
            prev.next = stack.pop()
            prev = prev.next
        prev.next = None  # 将最后一个节点的 next 指针置为 None
    
    # 创建一个哑节点作为链表的头节点
    dummy = ListNode(0)
    dummy.next = head
    prev = dummy

    while True:
        # 判断是否有足够的节点可以反转
        curr = prev
        for _ in range(k):
            curr = curr.next
            if not curr:
                return dummy.next  # 不足 k 个节点，无需反转

        # 使用栈来存储需要反转的节点
        stack = []
        for _ in range(k):
            stack.append(prev.next)
            prev = prev.next

        # 将栈中的元素依次连接到链表中
        connect_stack_to_list(curr, stack)

    return dummy.next
```

该方法使用一个辅助函数 `connect_stack_to_list` 将栈中的元素依次连接到链表中。然后，主函数 `reverse_k_groups` 中，我们遍历链表，每次检查是否有足够的节点可以反转，如果有，则将需要反转的节点存储在栈中，然后调用辅助函数进行反转。最后返回反转后的链表头节点。

回到这道题，使用辅助函数，用tracker的题解：追踪tracker的数量，足够k个就进行反转操作。

```python
def reverse_k_groups(head, k):
    dummy = LinkedListNode(0)
    dummy.next = head
    ptr = dummy
 
    while(ptr != None):

        tracker = ptr

        for i in range(k):
            if tracker == None:
                break
       
            tracker = tracker.next

        if tracker == None:
            break
    
        previous, current = reverse_linked_list(ptr.next, k)

        last_node_of_reversed_group = ptr.next
        last_node_of_reversed_group.next = current
        ptr.next = previous
        ptr = last_node_of_reversed_group

    return dummy.next
```

辅助函数：

```python
def reverse_linked_list(head, k):
	previous, current, next = None, head, None
	for _ in range(k):
		# temporarily store the next node
		next = current.next
		# reverse the current node
		current.next = previous
		# before we move to the next node, point previous to the
        # current node
		previous = current
		# move to the next node 
		current = next
	return previous, current
```
