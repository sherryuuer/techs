## 前缀树或者叫做字典树（Trie）

---

### 概念

顾名思义就是一个字符合集的树形结构。想象一个单词表。树的每一个节点都是字母。

同样的功能似乎可以用哈希集合实现，他和前缀树的查找和插入的时间复杂度都是常数时间。但是前缀树的优势是，它共享单词的前缀，可以方便地查找单词的前缀。

数据结构的实现分为两个部分，一个是节点，一个是数据结构本身。

trie的节点，使用hashset实现，是由于他的灵活性，同时定义一个布尔判定word，用于判断**到当前的字母为止，是否是一个单词**。（这个标记让我想起，自然语言处理的Seq2seq中的EOS的标记，单词来说的话，或者自然语言，都需要有一个标记来表明这个语句结束了没有。）有了节点就可以进行整个数据结构的构建了。

search和startWith方法其实差异不大，都是在Trie中检索字符串是否存在，区别只有最后的判断，判断单词存在与否只需要提取布尔值，而判断前缀，如果都能检索到的话，简单的返回True即可。

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.word = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        cur = self.root
        for c in word:
            if c not in cur.children:
                cur.children[c] = TrieNode()
            cur = cur.children[c]
        cur.word = True

    def search(self, word):
        cur = self.root
        for c in word:
            if c not in cur.children:
                return False
            cur = cur.children[c]
        return cur.word

    def startWith(self, prefix):
        cur = self.root
        for c in prefix:
            if c not in cur.children:
                return False
            cur = cur.children[c]
        return True
```

### leetcode 逐行解析

- 前缀树代码实现[leetcode208 题目描述](https://leetcode.com/problems/implement-trie-prefix-tree/description/)

一个典型的前缀树实现题。

- 设计添加和搜索单词数据结构[leetcode211 题目描述](https://leetcode.com/problems/design-add-and-search-words-data-structure/description/)

在 search 的部分，需要判断如果是点的情况，所有的字母都适用，这个时候需要递归深度优先搜索。注意最后一个字符是点的情况如何避免 bug。

- 单词查找 2[leetcode212 题目描述](https://leetcode.com/problems/word-search-ii/description/)

在一个单词 board 板子上，查找单词，很想连词游戏的那种板子，很有意思，对于网格题，真是，欲罢不能。
**这道题是标记为 hard 难度的，在做的时候确实是解出来了，然后有一种做 aoc 的感觉，原来 aoc 的题都是难题啊！真是越来越期待每年的十二月盛宴了！**
这道题经历了五个版本，真的非常好玩。

- 前缀和后缀搜索[leetcode745 题目描述](https://leetcode.com/problems/prefix-and-suffix-search/description/)

还没做
