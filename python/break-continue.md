## 关于Python的continue和break再次巩固

### break和continue我并不懂

在做[拓扑排序题](algo/toplogical-sort-problems.md)第一题Verifying an Alien Dictionary的时候遇到了双层嵌套循环的时候需要打断和继续的情况，总是无法按照想要的样子进行逻辑循环。

所以重新查了一下进行了总结。

### 双层嵌套中的动作方式

首先是continue我们知道是在循环中跳过这一次循环中的后续部分，继续下一次循环，但是在双层循环的时候，要记住，跳过的是**内层循环**即可，代码如下：

```python
for i in range(3):
    print("Outer loop:", i)
    for j in range(3):
        if j == 1:
            continue  # 跳过当前内层循环的剩余代码
        print("- Inner loop:", j)

# Outer loop: 0
# - Inner loop: 0
# - Inner loop: 2
# Outer loop: 1
# - Inner loop: 0
# - Inner loop: 2
# Outer loop: 2
# - Inner loop: 0
# - Inner loop: 2
```

可以看到输出结果，在每次到1的时候，内层就跳过了，外层并不影响。

关于break，表示完全打断循环，在双层循环的时候，打断的也是**内层循环**对外层不会有影响。代码如下：

```python
for i in range(3):
    print("Outer loop:", i)
    for j in range(3):
        if j == 1:
            break  # 打断内层循环继续下一层外层循环
        print("- Inner loop:", j)
    print("work?")

# Outer loop: 0
# - Inner loop: 0
# Outer loop: 1
# - Inner loop: 0
# Outer loop: 2
# - Inner loop: 0
```

可以看到在1之后的内层都没有被执行，但是外层都被执行了。

到这里我们知道了不管是continue和break，刚刚虽然写的是内层循环，但是如果有多层呢，只要记住，他们都**只作用于一层循环**就可以了。

**flag**的作用：我在需要实现的逻辑中，想要在内层循环没有被break的时候，执行最后的if语句，但刚刚我们看到如果内层被break了后面的就都不会被执行，这时候使用一个flag，标记如果触发了内层break就翻转flag不执行最后的if，如果没有被break就可以执行最后的if语句。

通过flag就可以更加灵活的处理这种情况了。代码如下：

```python
for i in range(3):
    print("Outer loop:", i)
    inner_break = False  # 用于记录内层循环是否被break
    for j in range(3):
        if j == 5:
            inner_break = True  # 标记内层循环被break
            break
        print("- Inner loop:", j)
    if not inner_break:
        print("Print after inner loop")
```

当然还有非常聪明的人有更多的好办法，慢慢学习和总结。

### 最后附上这道题

是leetcode的原题，虽然自己写的代码不是很精炼，但是是自己写的就是最可爱的。

```python
def verify_alien_dictionary(words, order):
    char_dict = {c: idx for idx, c in enumerate(order)}
    for i in range(len(words) - 1):
        word1 = words[i]
        word2 = words[i + 1]
        for j in range(min(len(word1), len(word2))):
            if word1[j] == word2[j]:
                continue
            elif char_dict[word1[j]] < char_dict[word2[j]]:
                inner_break = True # set the flag
                break  # break current inner loop
            else:
                return False

        if not inner_break and len(word2) < len(word1):
            return False

    return True
```
