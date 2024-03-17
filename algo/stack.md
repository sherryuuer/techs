## 数组结构：栈数据结构stack

---
### 适用范围

stack的主要特征只有一句就是后进先出。

在解决问题中主要用于和顺序有关的问题。在现实世界中比如：

- 检查代码文件：在编译器中，堆栈被用来检查代码文件中的括号是否平衡。
- 撤销/重做功能：在编辑过程中，堆栈常被用于撤销/重做操作。
- 递归调用堆栈：编译器在存储递归调用信息时内部调用堆栈本身。堆栈有助于递归子程序，其中每个调用的状态都被收集在一个堆栈帧中，然后放在堆栈上。著名的网站stackoverflow就是这么来的。

### 问题1:Remove All Adjacent Duplicates In String

示例我选择这个例子string = “azxxzy”，移除后的结果是“ay”，中间看似两个z是不相邻的，但是移除了x之后两个就相邻了这也是满足条件的。

实现步骤：

- 初始化一个stack，然后遍历整个字符串。
- 在每一轮比较中，如果字符和stack顶端的字符一致，则从stack弹出字符丢弃。
- 如果字符和stack顶端的字符不一致，则将字符推入stack。
- 结束遍历后，将stack中的字符重新结合为字符串。

代码实现：

```python
def remove_duplicates(string):
    stack = []
    
    for c in string:
        # need to check if stack is not empty
        if stack and c == stack[-1]:
            stack.pop()
        else:
            stack.append(c)
    
    return "".join(stack)
```

学习笔记：

在题解中谈到，naive的做法，是找到所有的小写字母的双对比如aa，bb，之类的，所有的二十六个字母的组合，每次删掉其中的重复后继续遍历删除，直到无法删除为止。虽然所有组合也只有26个，但是时间复杂度会达到n的平方，相比较stack是一种很简单并且容易想到的方法，所以并没有必要这么做。总之这是一道很简单的题，它的时间复杂度和空间复杂度都是O(n)。

### 问题2:Implement Queue Using Stacks

使用stack结构构架一个队列。

队列的性质，先入先出。

需要实现以下方法：

- Void Push(int x): 将元素推入队列的末尾。（在stack中就需要将它放在最下面了）
- Int Pop(): 从队列中移除最前面的元素。（只要实现了第一步，就可以简单的pop）
- Int Peek(): 返回队列的第一个元素，并不弹出。
- Boolean Empty(): 检查队列是否为空，返回布尔值。

实现push方法的步骤：两个stack

- 将stack1中的元素移动到stack2中。
- 将新的元素推入stack1中确保它在最下面。
- 将stack2中的剩余元素重新返回stack1中。

已有stack：

```python
class Stack:
    def __init__(self):
        self.stack_list = []

    def is_empty(self):
        return len(self.stack_list) == 0

    def top(self):
        if self.is_empty():
            return None
        return self.stack_list[-1]

    def size(self):
        return len(self.stack_list)

    def push(self, value):
        self.stack_list.append(value)

    def pop(self):
        if self.is_empty():
            return None
        return self.stack_list.pop()
```

主代码练习：

```python
from stack import Stack

class MyQueue(object):

    def __init__(self):
        self.queue = Stack()

    def push(self, x):
        stack = Stack()
        if self.queue.is_empty():
            self.queue.push(x)
        
        else:
            for i in range(self.queue.size()):
                stack.push(self.queue.pop())
            self.queue.push(x)
            for i in range(stack.size()):
                self.queue.push(stack.pop())

    def pop(self):
        return self.queue.pop()

    def peek(self):
        return self.queue.top()

    def empty(self):
        return self.queue.is_empty()
```

学习笔记：很顺利，基本没卡壳，活用给出的stack数据结构就行了。答案题解如下，直接在初始中构架两个stack，默认stack1为结果队列。中间遍历使用了while，一开始我也想用的，但还是选了for哈哈。最后那个判断空的结构，如果是用了while就不需要的，答案很好！时间复杂度除了push是n，其他都是常数。

```python
from stack import Stack

class MyQueue(object):

    # constructor to initialize two stacks
    def __init__(self):
        self.stack1 = Stack()
        self.stack2 = Stack()

    def push(self, x):
        while not self.stack1.is_empty():
            self.stack2.push(self.stack1.pop())
        self.stack1.push(x)

        while not self.stack2.is_empty():
            self.stack1.push(self.stack2.pop())

    def pop(self):
        return self.stack1.pop()

    def peek(self):
        return self.stack1.top()

    def empty(self):
        return self.stack1.is_empty()
```

### 问题3:Basic Calculator

计算器计算问题。给一个有效的计算公式字符串s，计算结果。

- s 代表一个连续的字符串，包括数字, “+”, “-”, “(”, 和 “)”。
- 加号不能作为一元运算符，减号可以，也就是说减号可以使得数字为负数。
- 不能有连续的运算符。

操作顺序：首先将连续的数字字符转化为数字，然后处理加号和减号，最后处理括号。

过程梳理：

- 初始化一个stack和三个变量：number（current number），sign_value（符号值），result
- 遍历字符串判断每个字符。
- 遇到一个digit类型，则更新number，将它加到number上去。
- 遇到了加号和减号，则计算左边的结果，使用符号值和数字，计算左边的结果，将它加到result中去。
- 遇到了一个括号，就将当前的计算结果和和符号值存储在里面，等遇到另一半括号，则计算栈中的结果。然后更新result。
- 遍历结束后，返回result。

代码练习：

```python
def calculator(s):
    stack = []
    number = 0
    sign_value = 1
    result = 0
    
    for char in s:
        if char.isdigit():
            number = number * 10 + int(char)
        elif char == '+':
            result += sign_value * number
            number = 0
            sign_value = 1
        elif char == '-':
            result += sign_value * number
            number = 0
            sign_value = -1
        elif char == '(':
            stack.append(result)
            stack.append(sign_value)
            # reset
            result = 0
            sign_value = 1
        elif char == ')':
            result += sign_value * number
            number = 0
            result *= stack.pop()  # sign_value
            result += stack.pop()  # previous result
            
    result += sign_value * number
    
    return result  
```

答案题解代码：

```python
def calculator(expression):
    number = 0
    sign_value = 1
    result = 0
    operations_stack = []

    for c in expression:
        if c.isdigit():
            number = number * 10 + int(c)
        if c in "+-":
            result += number * sign_value
            sign_value = -1 if c == '-' else 1
            number = 0
        elif c == '(':
            operations_stack.append(result)
            operations_stack.append(sign_value)
            result = 0
            sign_value = 1

        elif c == ')':
            result += sign_value * number
            pop_sign_value = operations_stack.pop()
            result *= pop_sign_value

            second_value = operations_stack.pop()
            result += second_value
            number = 0
    
    return result + number * sign_value
```

学习笔记：我觉得这个题对我算法小菜鸡来说挺难的，尤其是中间的多次符号重新初始化操作，需要很清晰的逻辑。由于是遍历了整个字符串，所以时间复杂度是O(n)。
