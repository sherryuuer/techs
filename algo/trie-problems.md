## 前缀树or字典树相关问题

### 实现 implement

字典树（Trie树）是一种树形数据结构，用于存储动态集合，其中键通常是字符串。它允许在时间复杂度为 O(1) 的情况下查找、插入和删除字符串。字典树的名称来自于“检索”（retrieve）这个单词的发音。（想起了最近做的RAG的R就是这个词）

在字典树中，每个节点代表一个字符串的字符，从根节点到某个节点的路径表示了一个字符串。通常，根节点不包含字符，每个节点有若干子节点，这些子节点对应着可能的下一个字符。另外，通常在每个节点上还会有一个标志，表示从根节点到当前节点的路径是否构成了一个完整的字符串。

字典树的主要特点包括：

1. **高效的插入和查询操作**：由于字典树的特殊结构，插入和查询操作的时间复杂度与字符串的长度成线性关系，而与字典树中存储的字符串数量无关。

2. **前缀匹配**：字典树可以高效地查找具有特定前缀的字符串集合，例如，可以快速找到所有以给定前缀开头的单词。

3. **空间开销**：尽管字典树在插入和查询方面非常高效，但它可能会占用大量内存，尤其是当存储大量相似字符串时。

字典树在自然语言处理、编译器设计、数据压缩、拼写检查等领域都有广泛的应用。

之前虽然学习过但是这里还是想再implement一下这个结构。主要包括对节点和树本身对初始化，以及对插入，搜索，搜索前缀等方法的实现。

代码如下：

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

整个结构很清晰简单，写着也是很开心。后面的题目也会基于这个数据结构。

### 问题1:Search Suggestions System

力扣的1268题，是一道中等难度的题。给定一个字符串数组 products 和一个字符串 searchWord。

设计一个系统，在输入 searchWord 的每个字符后，建议至多三个产品名称，这些产品名称与 searchWord 具有相同的前缀。如果具有相同前缀的产品超过三个，则返回按字典序最小的三个产品。

在输入 searchWord 的每个字符后，返回建议产品的列表。这题读完我也不知道在说什么，看看例子理解一下。

Input: products = ["mobile","mouse","moneypot","monitor","mousepad"]是搜索用的单词列表。

searchWord = "mouse"是要用于搜索的单词。

Output: [["mobile","moneypot","monitor"],["mobile","moneypot","monitor"],["mouse","mousepad"],["mouse","mousepad"],["mouse","mousepad"]]

输出是一个列表中的五个列表，意思是那么我们每打出一个字母都会更新这个prefix，每个prefix都会有一个对应的结果列表。

一开始没有看清题的话代码会简单的写成这样：结果会报错，因为没有按照单词的字典顺序，所以字典顺序是关键。

```python
def startWith(word, prefix):
    if len(prefix) > len(word):
        return False
    for i in range(len(prefix)):
        if prefix[i] != word[i]:
            return False
    return True


def suggested_products(products, search_word):
    result = []
    for end in range(1, len(search_word) + 1):
        prefix = search_word[0:end]
        lst = []
        count = 0
        for product in products:
            if startWith(product, prefix):
                lst.append(product)
                count += 1
                if count == 3:
                    break
        result.append(lst)
    return result
```


解题思路：使用上一部分基础的字典树的数据结构，增加排序方法，和搜索前缀部分的方法，然后就可以用于这道题。前提当然是对所有的单词进行数据结构的重构。

- 使用上述的insert方法，重构数据结构。
- 遍历搜索单词长度的loop次数，每次根据该prefix，遍历找到产品列表中符合条件的前三个单词。
- 返回结果。

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True

    def dfs(self, node, prefix, result):
        if node.is_end_of_word:
            result.append(prefix)
        for char, child in sorted(node.children.items()):
            self.dfs(child, prefix + char, result)

    def sort_words(self):
        result = []
        self.dfs(self.root, "", result)
        return result

    def startWith(self, word, prefix):
        if len(prefix) > len(word):
            return False
        for i in range(len(prefix)):
            if prefix[i] != word[i]:
                return False
        return True


def suggested_products(products, search_word):
    trie = Trie()
    for word in products:
        trie.insert(word)
    words = trie.sort_words()
    result = []
    for end in range(1, len(search_word) + 1):
        prefix = search_word[0:end]
        lst = []
        count = 0
        for word in words:
            if trie.startWith(word, prefix):
                lst.append(word)
                count += 1
                if count == 3:
                    break
        result.append(lst)
    return result
```

看一下题解给的参考答案有什么可以学习的：解说已经标注在代码中

```python
class TrieNode(object):
    def __init__(self):
        # 在每个节点中，添加一个列表来存储匹配的单词
        self.search_words = []
        # 子节点字典
        self.children = {}

class Trie(object):
    def __init__(self):
        # 创建 Trie 树的根节点
        self.root = TrieNode()

    def insert(self, data):
        # 插入单词到 Trie 树中
        node = self.root
        idx = 0
        for char in data:
            # 如果字符不在当前节点的子节点中，则添加一个新的子节点
            if char not in node.children:
                node.children[char] = TrieNode()
            # 移动到下一个子节点
            node = node.children[char]
            # 限制每个节点最多存储三个匹配的单词（这里节省了内存）
            if len(node.search_words) < 3:
                node.search_words.append(data)
            idx += 1

    def search(self, word):
        # 搜索与给定单词匹配的单词
        result, node = [], self.root
        for i, char in enumerate(word):
            if char not in node.children:
                # 如果当前字符不在 Trie 中，则返回空列表
                temp = [[] for _ in range(len(word) - i)]
                return result + temp
            else:
                node = node.children[char]
                # 将当前节点存储的匹配单词列表添加到结果中
                result.append(node.search_words[:])
        return result


def suggested_products(products, search_word):
    # 对产品列表进行排序，怀疑这里的计算速度
    products.sort()
    # 创建 Trie 树
    trie = Trie()
    # 将产品插入到 Trie 树中
    for x in products:
        trie.insert(x)
    # 搜索匹配的产品
    return trie.search(search_word)

```

说实话这个答案最后用了products.sort()的话我直接用这一条就可以改善第一个错误答案的代码了。我似乎用了数据结构来重组字典顺序。anyway result allright。另外这个答案中，主要逻辑集中在search方法上。但是这段代码中对变量的应用非常让我费解，比如在search方法中的search_words，是在最后的函数中出现的而不是在类中的。下面是对题解代码的优化：

```python
class TrieNode:
    def __init__(self):
        self.matching_words = []  # 存储匹配的单词列表
        self.children = {}  # 子节点字典


class Trie:
    def __init__(self):
        self.root = TrieNode()  # 创建 Trie 树的根节点

    def insert(self, word):
        # 将单词插入到 Trie 树中
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
            # 限制每个节点最多存储三个匹配的单词
            if len(node.matching_words) < 3:
                node.matching_words.append(word)

    def search(self, prefix):
        # 搜索与给定前缀匹配的单词
        result, node = [], self.root
        for char in prefix:
            if char not in node.children:
                # 如果前缀不在 Trie 中，则返回空列表
                return result
            node = node.children[char]
        # 返回匹配的单词列表
        return node.matching_words


def suggested_products(products, search_word):
    # 对产品列表进行排序
    products.sort()
    # 创建 Trie 树
    trie = Trie()
    # 将产品插入到 Trie 树中
    for product in products:
        trie.insert(product)
    # 搜索匹配的产品并返回
    result = []
    prefix = ""
    for char in search_word:
        prefix += char
        result.append(trie.search(prefix))
    return result
```
学习笔记：

每道题一开始只是想要实现它的数据结构的时候，总是简单清晰，但是一旦到应用就开始进行乐高了，各种复杂的地方都需要去解决，不过这也是乐趣所在。

分析一下这道题的时间和空间复杂度。

插入操作的时间复杂度：插入操作主要发生在 `insert` 方法中。在 Trie 树中插入一个单词的过程需要遍历该单词的每个字符，并将其插入到 Trie 树的相应位置。假设单词的平均长度为 m，Trie 树的高度不会超过单词的长度，因此插入操作的时间复杂度为 O(m)。

搜索操作的时间复杂度：搜索操作主要发生在 `search` 方法中。在 Trie 树中搜索与给定前缀匹配的单词的过程需要遍历该前缀的每个字符，并在 Trie 树中向下移动。最坏情况下，如果 Trie 树的高度为 h，则搜索操作的时间复杂度为 O(h)。但由于 Trie 树是基于前缀的数据结构，因此在大多数情况下，搜索的时间复杂度将取决于前缀的长度，即 O(m)，其中 m 是前缀的长度。

空间复杂度：空间复杂度取决于 Trie 树所需的存储空间。假设有 n 个单词，每个单词的平均长度为 m，则 Trie 树的空间复杂度为 O(nm)，因为每个字符都需要一个节点来存储。除了存储单词本身外，每个节点还需要存储指向子节点的指针，这会增加额外的存储空间。

但是！在给定的代码中，有一个排序操作，它在 suggested_products 函数中使用了 Python 的内置排序函数 sort() 来对产品列表进行排序。

Python 中的排序算法通常是使用 TimSort，其平均时间复杂度为 O(nlogn)，其中 n 是列表的长度。因此，排序操作的时间复杂度为 O(nlogn)。

小知识：

TimSort 是由 Tim Peters 在 Python 标准库中实现的一种混合排序算法。它结合了归并排序（Merge Sort）和插入排序（Insertion Sort），并具有以下特点：

1. **归并排序的思想**：TimSort 使用了归并排序的思想，将列表分割成小块，然后对这些小块进行排序，最后合并成一个有序的列表。

2. **插入排序的优化**：TimSort 对归并排序的每个小块应用了插入排序的思想。在小块的大小小于某个阈值时，采用插入排序来提高效率。

3. **稳定性**：TimSort 是一种稳定的排序算法，即对于相等的元素，排序后它们的相对顺序保持不变。

4. **适应性**：TimSort 对于已经部分有序的列表效果良好。它会尽可能利用已经有序的部分，减少不必要的比较和交换操作。

5. **低内存消耗**：TimSort 通过在排序过程中利用辅助数组来减少内存消耗，因此具有较低的内存使用率。

TimSort 在 Python 中被广泛应用，它被用作 Python 内置的排序算法，并且在 Java 的 Arrays.sort() 中也有类似的实现。

### 问题2:Design Add and Search Words Data Structure

力扣题211，同时多加了一点要求的一道题。要求设计一个数据结构：单词字典。包括如下方法：

add方法：将单词加入词典。search方法，查找单词是否在字典里，返回bool，这里的查找对象不仅包括单词本身，而且用`.`替代的字母，也算作该字母存在，也就是说如果查找`.in`，如果`bin`在单词表里那么返回 True，但是`.n`就不可以，一个点代表一个字母。get方法返回所有的单词。

总的来说还是在原数据结构上的修改和增加。

代码如下：经过了多次修改，以及在`get_words`方法中对回溯算法的各种尝试，收获还是很多的，尤其是最后debug成功后，很开心。

```python
class TrieNode:
    def __init__(self, val=None):
        self.children = {}
        self.word = False
        self.val = val


class WordDictionary:
    def __init__(self):
        self.root = TrieNode()

    def add_word(self, word):
        cur = self.root
        for c in word:
            if c not in cur.children:
                cur.children[c] = TrieNode(c)
            cur = cur.children[c]
        cur.word = True

    def search_word(self, word):
        def dfs(j, root):
            cur = root
            for i in range(j, len(word)):
                c = word[i]
                if c == '.':
                    for child in cur.children.values():
                        if dfs(i + 1, child):
                            return True
                    return False

                else:
                    if c not in cur.children:
                        return False
                    cur = cur.children[c]
            return cur.word

        return dfs(0, self.root)

    def get_words(self):
        def dfs(cur, curset, subsets):
            curset.append(cur.val)
            if cur.word:
                subsets.append(''.join(curset))
            # 嵌套字典
            for child in cur.children.values():
                dfs(child, curset, subsets)
            curset.pop()

        result = []
        for cur in self.root.children.values():
            dfs(cur, [], result)
        return result


obj = WordDictionary()
obj.add_word("bad")
obj.add_word("dad")
obj.add_word("mad")
print(obj.get_words())
print(obj.search_word(".ad"))
```
下面是题解给的参考答案：

```python
class TrieNode():
  
  # Initialize TrieNode instance
  def __init__(self):
    self.children = []
    self.complete = False
    for i in range(0, 26):
      self.children.append(None)

class WordDictionary:
    # Initialize the root with TrieNode and set 
    # the 'can_find' boolean to FALSE
    def __init__(self):
        self.root = TrieNode()
        self.can_find = False


    # Function to add a new word to the dictionary
    def add_word(self, word):
        n = len(word)
        cur_node = self.root
        for i, val in enumerate(word):
            index = ord(val) - ord('a')
            if cur_node.children[index] is None:
                cur_node.children[index] = TrieNode()
            cur_node = cur_node.children[index]
            if i == n - 1:
                if cur_node.complete:
                    print("\tWord already present!")
                    return
                cur_node.complete = True
        print("\tWord added successfully!")


    # Function to search for a word in the dictionary
    def search_word(self, word):
        self.can_find = False
        self.search_helper(self.root, word, 0)
        return self.can_find


    def search_helper(self, node, word, i):
        if self.can_find:
            return
        if not node:
            return
        if len(word) == i:
            if node.complete:
                self.can_find = True
            return

        if word[i] == '.':
            for j in range(ord('a'), ord('z') + 1):
                self.search_helper(node.children[j - ord('a')], word, i + 1)
        else:
            index = ord(word[i]) - ord('a')
            self.search_helper(node.children[index], word, i + 1)


    # Function to get all words in the dictionary
    def get_words(self):
        words_list = []
        if not self.root:
            return []
        return self.dfs(self.root, "", words_list)

    def dfs(self, node, word, words_list):
        if not node:
            return words_list
        if node.complete:
            words_list.append(word)

        for j in range(ord('a'), ord('z') + 1):
            prefix = word + chr(j)
            words_list = self.dfs(node.children[j - ord('a')], prefix, words_list)
        return words_list
```

学习笔记：时间复杂度上来说，添加单词和搜索单词都使用单词长度的时间O(m)，而取得所有单词的时间复杂度取决于节点数量n所以是O(n)。在空间复杂度上，如果节点数量是n那么最坏情况来说空间为O(n*26)了。

### 问题3:Word Search II

力扣212，hard难度，给定一个矩阵：比如这个矩阵。

```python
[["C","S","L","I","M"],
 ["O","I","L","M","O"],
 ["O","L","I","E","O"],
 ["R","T","A","S","N"],
 ["S","I","T","A","C"]]
```

以及一个字符串列表，["SLIME","SAILOR","MATCH","COCOON"]找出所有在这个单词board上可以找到的单词。单词的意思是它需要在横向或者竖向上是相邻的。

暴力解法是通过对每个单词进行深度优先搜索。但是这种方法我在力扣提交中败给了网友的例题，超时了。所以需要的是前缀树数据结构的优化。

代码如下：代码中对res列表使用了set结构，目的是为了去重，set结构是一种很好的去重的方法，省了很多麻烦，如果不去重，还需要对每次找到的word，进行再次isWord的重新标记。

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.isWord = False

    def addWord(self, word):
        cur = self
        for c in word:
            if c not in cur.children:
                cur.children[c] = TrieNode()
            cur = cur.children[c]
        cur.isWord = True

def find_strings(grid, words):
    root = TrieNode()
    for w in words:
        root.addWord(w)

    ROWS, COLS = len(grid), len(grid[0])
    res, visit = set(), set()

    def dfs(r, c, node, word):
        if (
            r < 0 or c < 0 or
            r == ROWS or c == COLS or
            (r, c) in visit or grid[r][c] not in node.children
        ):
            return
        
        visit.add((r, c))
        node = node.children[grid[r][c]]
        word += grid[r][c]
        if node.isWord:
            res.add(word)

        directions = [(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)]
        for d in directions:
            dfs(d[0], d[1], node, word) 
            
        visit.remove((r, c))
    
    for r in range(ROWS):
        for c in range(COLS):
            dfs(r, c, root, "")
    return list(res)
```
学习笔记：时间复杂度上，在进行查找的时候我们从网格(假设有n个点)的每个点开始，同时每次都遍历三个方向（第四个方向是来的地方），单词长度假设为l，那么时间复杂度就是O(n * 3^l)，空间上，需要一个存储单词的前缀树空间，和一个遍历空间，前缀树最坏的情况是所有单词的字母都不重复，我们假设为m好了，遍历空间最坏的情况，一个单词的长度遍布了整个网格n，这时候的空间复杂度总和为O(m + n)。
