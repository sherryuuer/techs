## 数组：静态动态数组以及堆栈（Arrays）

### 静态数组

在Python中更多的是以list的形式存在的。是最熟悉的数据结构，之后熟悉的应该是字典里吧。数组在读写删除和插入上，只有读是O(1)其他都是O(n)。

在读取数据的时候，因为数据存储在RAM中，而我们又需要索引来快速读取数据，所以读取的时候就可以做到立刻得到指定索引上的数据，而在删除插入的时候，就需要对其他位置的数据进行移动。由于在大O表示法中，我们只考虑最坏的情况，也就是O(n)了，n代表数组的长度。

除此之外，还有一些特殊情况要考虑。针对各种情况的代码解析如下。

在数组的结束位置插入一个元素的情况：大O为O(1)，假定长度没有超过数组使用的内存限制。将n插入数组arr。
同样删除最后一个元素，大O也是O(1)，假定数组不为空。

```python
def insertEnd(arr, n, length, capacity):
    if length < capacity:
        arr[length] = n

def removeEnd(arr, length):
    if length > 0:
        # 将原本的数据重写为一些规定的default数据
        arr[length - 1] = 0
```

相对的在数组中间插入和删除元素，就需要递归地对其他元素进行移动，所以O(n)。

```python
def insertMiddle(arr, i, n, length):
    # 反向遍历，从最后一个元素开始到i（因为range不包括结束位置，所以这里用i-1）的元素都向后移动一位，
    for index in range(length - 1, i - 1, -1):
        arr[index + 1] = arr[index]
    # 移动结束后在i处插入n
    arr[i] = n

def removeMiddle(arr, i, n, length):
    for index in range(i, length - 1):
        arr[index] = arr[index + 1]
```

### 动态数组

基本的Python的数组其实就是动态数组，我们不需要在意它的容量，它是动态扩展的，但是知道数组的真正结构依然很重要。

逐行进行解析构建数组的过程。因为在计算机内部其实一开始是为数组分配了内存容量的，当数组长度超过了内存，内存就会进行重新分配，只是由于Python在背后帮我们都做了我们不知道而已。

```python
# 初始化数组，容量，长度，和默认值。初始容量为2。默认值设为0。由于构架数据结构是为了更好的理解，所以可以自行设定。
class Array:
    def __init__(self):
        self.capacity = 2
        self.length = 0
        self.arr = [0] * 2

    # 在数组长度的最后一位增加一个元素
    def pushback(self, n):
        # 如果数组的长度和达到了容量上限就进行，新的容量再分配resize
        if self.length == self.capacity:
            self.resize()
            
        # 拓展操作
        self.arr[self.length] = n
        self.length += 1

    # 容量再分配的resize操作
    def resize(self):
        # 重新分配内存区域，进行二倍容量拓展，并为每个位置设定默认值
        self.capacity = 2 * self.capacity
        newArr = [0] * self.capacity 
        
        # 从老数组将每个元素重新放进新的二倍容量的数组
        for i in range(self.length):
            newArr[i] = self.arr[i]
        # 再次将数组设定了新数组
        self.arr = newArr
        
    # 移除数组的最后一个元素，直接在长度上摘掉最后一个，变成一个无法触达的领域哈哈
    def popback(self):
        if self.length > 0:
            self.length -= 1
    
    # 读取第i个位置的元素，此处可以立刻得到
    def get(self, i):
        if i < self.length:
            return self.arr[i]

    # 插入操作，在i处插入元素n（因为是动态数组，所以内部的位移交给了计算机进行）
    def insert(self, i, n):
        if i < self.length:
            self.arr[i] = n
            return     

    # 打印数组
    def print(self):
        for i in range(self.length):
            print(self.arr[i])
        print()
```

### 堆栈stack

之所以堆栈和数组在一起，是因为其实堆栈是一种特殊的数组，它和数组的不同，仅仅是，它规定只能从最后进行push，pop等操作。

使用动态数组，快速构建一个堆栈：因为堆栈只需要两个操作，一个是在结尾增加一个元素，一个是从结尾弹出一个元素。他们的大O都是O(1)。

堆栈是一种很好的数据结构，在很多力扣题的解法中，需要加入一个堆栈来辅助，比如二叉树的遍历问题，就需要使用堆栈控制元素的遍历顺序。即使如此，理解它却很容易，就是堆叠东西，你只能放在上面，也只能从上面拿。

```python
class Stack:
    def __init__(self):
        self.stack = []

    def push(self, n):
        self.stack.append(n)

    def pop(self):
        return self.stack.pop()
```
