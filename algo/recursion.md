## 递归算法：宇宙的开始和终结

---

### 在一开始想说的

这个主题真的是我觉得所有算法主题里面最喜欢的主题之一。

从一个初始开始，不断调用自己，然后终于在一个终结条件上，那回了最终的答案，然后一直回溯，交给上一个自己。就像一个磅礴的大气的史诗神话。就像是宇宙从一个节点开始，不断的变大，增熵的过程，然后终于在最后的条件上，打碎了所有的循环，重新回到了bigbang的最初原点，拿回了那个真理。如果空间不足，就会stackoverflow，要么破灭，要么圆满。太中二，太美好了，完全就是宇宙的缩影。

代码简介而优雅，只要你懂得了这个宇宙的因果链就能一下子理解。

### 什么是递归思想

当我们说 "递归算法" 时，通常指的是一个函数在解决问题的过程中调用自身的一种技巧。递归算法涉及到将一个大问题拆分为更小的相似子问题，通过解决这些子问题最终得到整个问题的解。

递归函数通常有两部分：

一个是基本情况（Base Case）： 在递归算法中，必须有一个或多个基本情况，它们是递归终止的条件。在阶乘的例子中，基本情况是当 n 等于 0 或 1 时，阶乘的值是 1。另一个是递归调用： 在函数内部，通过调用自身解决规模更小的子问题。在阶乘的例子中，递归调用是计算 (n-1)!。

在树结构的遍历，分治问题中都有广泛应用。

提到递归，不可避免的会和迭代进行比较。为了避免堆栈溢出问题，迭代使用更新变量的方式，重复执行代码。

### 阶乘的递归

```python
# Recursive implementation of n! (n-factorial) calculation
def factorial(n):
    # Base case: n = 0 or 1
    if n <= 1:
        return 1

    # Recursive case: n! = n * (n - 1)!
    return n * factorial(n - 1)

```

### 斐波那契递归

```python
# Recursive implementation to calculate the n-th Fibonacci number
def fibonacci(n):
    # Base case: n = 0 or 1
    if n <= 1:
        return n

    # Recursive case: fib(n) = fib(n - 1) + fib(n - 2)
    return fibonacci(n - 1) + fibonacci(n - 2)

```
