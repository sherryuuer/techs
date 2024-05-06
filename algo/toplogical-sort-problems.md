## 拓扑排序相关问题

### 简单复习定义

拓扑排序是一种用于有向无环图（DAG）的排序方法，它能够将图中的节点按照一种特定的顺序进行排列，使得图中任意一条有向边的起点在排序中都排在终点之前。换句话说，拓扑排序能够将一个有向图的所有节点排成一个线性序列，使得对于图中的每一条有向边 (u, v)，节点 u 在排序中都出现在节点 v 的前面。

适用范围：拓扑排序常常用于任务调度、依赖关系分析等领域。例如，在软件工程中，如果有一系列任务，其中一些任务必须在另一些任务完成后才能开始，那么可以使用拓扑排序来确定任务的执行顺序，确保所有的依赖关系都被满足。

算法来源：拓扑排序算法的来源可以追溯到图论领域。其中，一种常见的算法是基于深度优先搜索（DFS）的拓扑排序算法。这种算法首先对图进行深度优先搜索，然后在回溯的过程中将节点加入到结果序列中，最终得到拓扑排序的结果。另一种常见的算法是基于队列的拓扑排序算法，它使用了图中节点的入度信息来进行排序，每次选择入度为0的节点加入到结果序列中，并更新其邻接节点的入度。

图论方面的东西都不简单，但也是现代最重要的基础学科之一。

### 问题1:Verifying an Alien Dictionary

在外星语言中，他们也使用英文小写字母，但可能是不同的order。order字母表中的 是小写字母的某种排列。

words给定一个用外语语言书写的单词列表 ，返回true当且仅当给定的words列表在此外语中按字典顺序排序。

情况1，例如，words = ["hello", "leetcode"], order = "hlabcdefgijkmnopqrstuvwxyz"，输出true，因为 h 在 l 前面，所以这个词典是排序的。

情况2，如果是，words = ["word", "world", "row"], order = "worldabcefghijkmnpqstuvxyz"，那么就是false，因为看到 d 应该在 l 后面。

情况3，对于words = ["apple", "app"], order = "abcdefghijklmnopqrstuvwxyz"，则返回false，因为要考虑长度了，对于空它比任何字母都小。

解题思路：

- 将order中的每个字母的index存储在数据结构中，字典最好。
- 遍历单词表中的每两个相邻的单词。
- 如果后一个单词比前一个先遍历结束那么返回False，这是上面的情况3的时候，前面的prefix相同，但是长度短的应该在前面。
- 否则遇到不同的字母顺序正确，则退出该遍历，继续遍历下两个相邻单词。反之返回False。
- 如果遍历顺利都结束了，那么最终返回True。

代码尝试：

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
                inner_break = True
                break  # break current inner loop
            else:
                return False

        if not inner_break and len(word2) < len(word1):
            return False

    return True
```

题解代码：

```python
def verify_alien_dictionary(words, order):
    if len(words) == 1:
        return True
    
    order_map = {}
    
    for index, val in enumerate(order):
        order_map[val] = index
    
    for i in range(len(words) - 1):
        for j in range(len(words[i])):
            # 当顺次可以比较到这个位置的时候，会触发长度判断
            if j >= len(words[i + 1]):
                return False
            
            if words[i][j] != words[i + 1][j]:
                if order_map[words[i][j]] > order_map[words[i + 1][j]]:
                    return False
                # 否则打断循环继续下一组单词比较
                break
    
    return True
```
学习笔记：题解真的很聪明，比我自己的代码好很多，但是通过自己的代码我重新认识了一下break和continue的用法。以及如何使用flag实现一些效果。时间复杂度上，第一次循环的长度是有限的，上限26忽略不计，第二次循环是整个列表的长度，所以是O(n)。空间复杂度上，使用了额外的空间，同样的这个空间有上限所以为常数O(1)。

### 问题2:Compilation Order

拓扑排序的典型问题。这里的编译顺序意味着各个元素之间有依存关系，所以要找到一个顺序，该顺序可以满足所有的元素按顺序正常编译结束。但是如果检测到循环，则返回空列表。

这道题和[课程排序题](https://leetcode.com/problems/course-schedule-ii/description/)的第二题是一样的逻辑。

例如，[A, B], [B, C], [A, D]，意味着A依存于B，B依存于C，A依存于D，那么被依存的就要先完成。于是[C, B, D, A]或者[D, C, B, A]都是有效解。

解题思路：

- 初始化：所有的邻接表转化成：src:dist_list 的形式，即起点和终点列表。
- 初始化一个hashset储存已经访问过的节点。
- 初始化一个结果列表topSort。
- 遍历每个节点进行深度优先搜索，直到没有邻居为止将结果append进结果列表。

代码如下：

```python
def find_compilation_order(dependencies):
    adj = {}
    for src, dst in dependencies:
        if src not in adj:
            adj[src] = []
        if dst not in adj:
            adj[dst] = []
        adj[src].append(dst)

    topSort = []
    cycle, visit = set(), set()
    def dfs(i):
        if i in cycle:
            return False
        if i in visit:
            return True
        
        cycle.add(i)
        for neighbor in adj[i]:
            if not dfs(neighbor): return False
        cycle.remove(i)
        visit.add(i)
        topSort.append(i)
        return True
    for j in adj:
        if not dfs(j): return []
    return topSort
```

学习笔记：循环的检测需要一个hashset，访问过与否依然需要一个hashset。拓扑排序对我来说不是一类容易的题。还是需要重新做一下课程安排三部曲。V是所有的节点的数量，E是所有的边的数量。这道题的时间复杂度是O(V+E)，空间复杂度是O(V)。

### 问题3:Alien Dictionary

力扣premium题，hard难度。和第一道题题目一看就是一个系统的，这次反过来出题，给出的是按照外星人的单词表排列的单词列表，要求得到一个有效的外星人的字典也就是问题一中的order。

在理解了第一题和第二题的基础上，这道题就是两道题的拼接而已。

解题思路来说，首先明白如何得到字母和字母之间的邻接表，单词的比较遵循一定的规律，遍历每两个单词，当排序不符合规则的时候，直接返回空集，当符合规则的时候，进行每个index位置的字母比较，字母不同就可以得到前后顺序，将结果存储在邻接表中。

当构造完邻接表，剩下的就是第二题中的排序策略了。

以下是两段代码，第一部分的单词判断部分完全相同，第二部分的根据邻接表排序部分使用了两种不同的方法，第二个函数单纯是用问题2的方法修改，使用两个hashset，第一个函数使用了哈希表标记布尔值来判断循环。

```python
def AlienDictionary(words):
    adj = {char:set() for word in words for char in word}

    for i in range(len(words) - 1):
        w1, w2 = words[i], words[i + 1]
        minlen = min(len(w1), len(w2))
        if len(w1) > len(w2) and w1[:minlen] == w2[:minlen]:
            return ""

        for j in range(minlen):
            if w1[j] != w2[j]:
                adj[w1[j]].add(w2[j])
                break

    visit = {}
    res = []

    def dfs(char):
        if char in visit:
            return visit[char]
        
        visit[char] = True
        for nei in adj[char]:
            if dfs(nei):
                return True

        visit[char] = False
        res.append(char)

    for char in adj:
        if dfs(char):
            return ""
    
    res.reverse()
    return "".join(res)


def AlienDictionary(words):
    adj = {char:set() for word in words for char in word}

    for i in range(len(words) - 1):
        w1, w2 = words[i], words[i + 1]
        minlen = min(len(w1), len(w2))
        if len(w1) > len(w2) and w1[:minlen] == w2[:minlen]:
            return ""

        for j in range(minlen):
            if w1[j] != w2[j]:
                adj[w1[j]].add(w2[j])
                break

    topSort = []
    cycle, visit = set(), set()
    def dfs(i):
        if i in cycle:
            return False
        if i in visit:
            return True
        
        cycle.add(i)
        for neighbor in adj[i]:
            if not dfs(neighbor): return False
        cycle.remove(i)
        visit.add(i)
        topSort.append(i)
        return True
    for char in adj:
        if not dfs(char): return ""
    return "".join(topSort)
```

学习笔记：第二部分的排序部分的时间空间复杂度在问题二中进行了分析，单词比较构造邻接表的部分是遍历单词的线性时间。
