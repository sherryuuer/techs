## 链表结构：单链表，双链表，队列

---

### 单向链表（Singly linked list）

由一个值和一个指针组成。和数组不一样，每一个节点在内存中不是按照顺序排列的，可能是随机的，需要查询的时候，你需要按照指针顺序一个一个去走，直到走到你要的节点，拿到你的数值。

常用的循环是使用while进行遍历，以指针为索引，不断的执行`cur = cur.next`。

和数组不同，链表不需要每次操作移动大量数据，比如在结尾加上一个节点，时间复杂度是常数时间。因为不需要调整在内存中的前后顺序，只需要将最后一个指针指向新的节点。但是查找需要O(n)复杂度，因为你需要遍历所有节点来找到目标。

数据结构的python实现：

**注意**：在初始化头节点的时候，使用一个dummy的节点。在单链表的实现中，将头节点初始化为虚拟节点（dummy node）是一种常见的技巧，也称为哑节点、哨兵节点或标兵节点。这个虚拟节点并不存储实际的数据，它的目的是为了简化链表的操作，特别是在处理边界情况和插入/删除操作时。

一些初始化头节点为虚拟节点的优势包括：

1. **简化插入和删除操作：** 当链表为空时，或者在链表头部插入/删除时，无需特殊处理。始终存在一个虚拟节点，这样就不需要检查链表是否为空。这简化了代码逻辑，减少了对边界情况的处理。

2. **一致性：** 使用虚拟节点可以确保链表中始终存在一个节点，即使链表为空。这种一致性使得代码更加一致和易于理解。

3. **简化代码逻辑：** 在进行插入和删除操作时，使用虚拟节点可以**避免对头节点和其他节点进行特殊处理**。无论在链表的哪个位置插入或删除节点，都可以统一处理。

4. **避免空指针异常：** 当头节点为空时，如果试图访问头节点可能会导致空指针异常。使用虚拟节点可以确保链表的头始终存在，减少了出现异常的可能性。


```python
# 初始化节点对象
class ListNode:
    def __init__(self, val):
        # 注意一个单链表的两个组成部分是储存值的部分和指针next部分
        self.val = val
        self.next = None

# 初始化这个单链表
class LinkedList:
    def __init__(self):
        # 初始化头和尾节点，他们都指向一个dummy的节点
        self.head = ListNode(-1)
        self.tail = self.head
    
    def insertEnd(self, val):
        # 在链表结尾处增加一个值为val的节点
        # 将tail指针的下一个位置初始化一个新的节点
        self.tail.next = ListNode(val)
        # 更新tail指针的指向位置
        self.tail = self.tail.next

    def remove(self, index):
        # 删除index位置处的节点
        i = 0
        curr = self.head
        while i < index and curr:
            i += 1
            curr = curr.next
        
        # Remove the node ahead of curr
        if curr and curr.next:
            if curr.next == self.tail:
                self.tail = curr
            curr.next = curr.next.next

    def print(self):
        curr = self.head.next
        while curr:
            print(curr.val, " -> ", end="")
            curr = curr.next
        print()

```

### 双向链表（Doubly linked list）

和单向链表相比，双向的链表增加了一个prev的前向指针，增加了一些灵活性，但是也增加了一些空间开销（毕竟每一个节点都多了一个指针要存储）维护起来也相对复杂，看python实现的代码长度就可以发现。

增加一个前向节点的优势有很多：可以双向遍历，在插入和删除节点的时候，不需要通过从头遍历来找到前驱节点。

每一个数据结构的存在，都是因为在它擅长的地方使用它，下面几个场景就是双向链表的应用！

文本编辑器的Undo和Redo功能，正是因为记录了你的每个操作几点的前后操作。浏览器的前进后退。音乐播放器的前后操作。任何你想到的需要往前又需要往后的东西，也许里面正使用了双向链表结构。

注意，和单向链表一样，双向链表同样进行dummy头尾设置。这样就可以实现在所有的节点都方便的操作。

```python
class ListNode:
    def __init__(self, val):
        self.val = val
        self.next = None
        self.prev = None

# Implementation for Doubly Linked List
class LinkedList:
    def __init__(self):
        # Init the list with 'dummy' head and tail nodes which makes 
        # edge cases for insert & remove easier.
        self.head = ListNode(-1)
        self.tail = ListNode(-1)
        self.head.next = self.tail
        self.tail.prev = self.head
    
    def insertFront(self, val):
        newNode = ListNode(val)
        newNode.prev = self.head
        newNode.next = self.head.next

        self.head.next.prev = newNode
        self.head.next = newNode

    def insertEnd(self, val):
        newNode = ListNode(val)
        newNode.next = self.tail
        newNode.prev = self.tail.prev

        self.tail.prev.next = newNode
        self.tail.prev = newNode

    # Remove first node after dummy head (assume it exists)
    def removeFront(self):
        self.head.next.next.prev = self.head
        self.head.next = self.head.next.next

    # Remove last node before dummy tail (assume it exists)
    def removeEnd(self):
        self.tail.prev.prev.next = self.tail
        self.tail.prev = self.tail.prev.prev

    def print(self):
        curr = self.head.next
        while curr != self.tail:
            print(curr.val, " -> ")
            curr = curr.next
        print()

```

### 队列（Queue）

队列的实现通常使用单链表，因为队列本身的特性使得单链表足以满足操作的需求，并且在某些情况下更为简单和高效。以下是使用单链表实现队列的一些原因：

1. **先进先出（FIFO）的特性：** 队列是一种先进先出的数据结构，元素按照插入的顺序排列。单链表自然符合这个特性，每个节点都包含一个指向下一个节点的指针，元素可以依次从队头进入，从队尾出去，保持了先进先出的顺序。

2. **简单高效：** 单链表的实现相对简单，操作包括在队尾插入元素（入队）和在队头删除元素（出队）。在单链表中，出队操作只需修改头指针，入队操作只需在尾部插入新节点，这样的操作是高效的。

3. **内存分配效率：** 单链表中的节点是动态分配的，这意味着可以根据需要动态调整存储空间，避免了固定大小的数组可能导致的空间浪费。

4. **简化实现：** 单链表的实现相对来说比较简单，不需要考虑双向链表中的额外指针，使得代码更为清晰和易于理解。

5. **常见用途：** 在许多应用中，队列主要用于按照顺序处理任务、事件或数据。单链表在这种情况下提供了足够的功能，而不需要引入双向链表的额外复杂性。

尽管单链表在队列的实现中具有这些优势，但在特定的应用场景中，如果需要支持更多复杂的操作，例如在队列中间删除或插入元素，或者需要支持双端队列的功能，那么双向链表可能更适合。选择数据结构取决于具体的需求和操作模式。

```python
class ListNode:
    def __init__(self, val):
        self.val = val
        self.next = None

class Queue:
    # Implementing this with dummy nodes would be easier!
    def __init__(self):
        self.left = self.right = None
    
    def enqueue(self, val):
        newNode = ListNode(val)

        # Queue is non-empty
        if self.right:
            self.right.next = newNode
            self.right = self.right.next
        # Queue is empty
        else:
            self.left = self.right = newNode

    def dequeue(self):
        # Queue is empty
        if not self.left:
            return None
        
        # Remove left node and return value
        val = self.left.val
        self.left = self.left.next
        if not self.left:
            self.right = None
        return val

    def print(self):
        cur = self.left
        while cur:
            print(cur.val, ' -> ', end ="")
            cur = cur.next
        print() # new line

```
