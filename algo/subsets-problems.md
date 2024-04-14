## 回溯算法：子集相关问题

---
### 问题描述

回溯算法的子集问题是指在给定一个集合，要求找出该集合的所有子集，包括空集和本身集合。这个问题可以通过回溯算法来解决，其基本思路是通过递归的方式，不断向当前部分解中添加元素，直到生成一个完整的子集。在添加元素的过程中，需要对每个元素进行两种选择：选择将该元素添加到当前部分解中，或者选择不添加该元素。通过回溯的方式，不断探索所有可能的选择，最终生成所有可能的子集。

回溯算法的子集问题适用于那些需要列举出集合的所有子集的情况，例如：

1. 组合优化问题：寻找一个集合的所有可能的组合，以便在解决问题时考虑不同的选择组合。
2. 排列问题：寻找一个集合的所有可能的排列方式，以便对集合中的元素进行重新排序。
3. 子集问题：寻找一个集合的所有可能的子集，以便对集合中的元素进行组合和排列。
4. 集合覆盖问题：寻找一个集合的所有可能的子集，以便覆盖给定的目标集合。
5. 组合数学问题：研究集合的各种组合和排列的性质和特征。
6. 排序问题：对一个集合中的元素进行排序，以便按照某种顺序进行处理或展示。

总之回溯算法的子集问题适用于那些需要穷举所有可能情况的问题，以便找到最优解或者满足特定条件的解的情况。

### 问题1:Permutations

简而言之是一个字符串全排列的问题。

比如针对字符串 xyz 得出的结果是三个字母的全排列结果 [“xyz”, “xzy”, “yxz”, “yzx”, “zyx”, “zxy”]。

解题思路：

- 递归函数设计：设计一个递归函数，用于生成字符串的全排列。函数需要考虑当前位置字符和其后字符的交换排列情况。
- 递归终止条件：确定递归函数的终止条件。通常是当处理到字符串的最后一个字符时，将当前排列结果加入结果集合中。
- 交换字符位置：在递归函数中，通过不断交换当前位置字符与后面字符的位置，实现字符串的全排列。
- 回溯操作：在每一轮递归后，需要回溯到上一步的状态，保证下一轮递归的字符位置正确。
- 去重处理：如果字符串中有重复字符，需要在递归过程中进行去重处理，避免生成重复的排列结果。

```python
def permute_word(word):
    result = []

    def backtracking(s, start, end, result):
        # s is a list of the word
        if start == end:
            result.append(''.join(s))
            print(result)
        else:
            for i in range(start, end + 1):
                s[start], s[i] = s[i], s[start]
                print(s)
                backtracking(s, start + 1, end, result)
                s[start], s[i] = s[i], s[start]

    backtracking(list(word), 0, len(word) - 1, result)
    return result
```

学习笔记：

交换字符串位置可以实现全排列的原因在于每次交换都相当于在当前位置确定一个字符，然后递归地处理剩余字符的全排列。具体来说，这个算法通过不断地将当前位置的字符与后面位置的字符交换，实现了对字符串中所有字符位置的全排列。

假设有一个字符串 "abc"，通过交换字符位置可以生成如下的全排列：

1. 当起始位置为 0 时，固定字符 "a"，对剩余字符 "bc" 进行全排列，得到 "abc" 和 "acb"。
2. 当起始位置为 1 时，固定字符 "b"，对剩余字符 "ac" 进行全排列，得到 "bac" 和 "bca"。
3. 当起始位置为 2 时，固定字符 "c"，对剩余字符 "ab" 进行全排列，得到 "cab" 和 "cba"。

可以看出，通过不断地交换字符位置并递归处理剩余字符，最终可以得到字符串的所有全排列。这种方法利用了递归和回溯的思想，是一种常见的全排列生成算法。

这是一个比较大的主体，因为它是一个完整的数学概念，除了这种做法，还有其他的做法，比如插值，因为做法比较多，准备另开主题。

时间复杂度正是数学中排列算法的n的阶乘O(n!)，空间复杂度是递归的时候调用的栈的大小O(n)。

### 问题2:Letter Combinations of a Phone Number

也是一个经典的编程问题，通常在算法和数据结构的学习中遇到。这个问题的目标是给定一个数字字符串，如"23"，输出所有可能的由这些数字映射的字母组合。

在电话按键上，每个数字都与一些字母对应。例如：

- 数字 2 对应字母 'a', 'b', 'c'
- 数字 3 对应字母 'd', 'e', 'f'
- 以此类推

所以，对于输入"23"，可能的字母组合包括 "ad", "ae", "af", "bd", "be", "bf", "cd", "ce", "cf"。

具体步骤：

- 定义一个数字到字母的映射表。
- 使用回溯算法递归地构建所有可能的组合。
- 在递归函数中，对于当前数字对应的每个字母，都尝试将其添加到当前组合中，并递归地处理下一个数字。
- 当处理完所有数字后，将当前组合添加到结果集中。

代码尝试：顺利通过。

```python
def letter_combinations(digits):

    result = []
    if not digits:
        return []

    mapping = {
        '2': ['a', 'b', 'c'],
        '3': ['d', 'e', 'f'],
        '4': ['g', 'h', 'i'],
        '5': ['j', 'k', 'l'],
        '6': ['m', 'n', 'o'],
        '7': ['p', 'q', 'r', 's'],
        '8': ['t', 'u', 'v'],
        '9': ['w', 'x', 'y', 'z']
    }

    def backtracking(s, subset, start, end, result):

        if len(subset) == len(s):
            result.append(''.join(subset))
        else:
            for char in mapping[s[start]]:
                subset.append(char)
                backtracking(s, subset, start + 1, end, result)
                subset.pop()

    digits = str(digits)
    backtracking(list(digits), [], 0, len(digits) - 1, result)
    return result
```

参考答案如下：判断条件后return我很喜欢。

```python
# Use backtrack function to generate all possible combinations
def backtrack(index, path, digits, letters, combinations):
    if len(path) == len(digits):
        combinations.append(''.join(path))
        return 
    possible_letters = letters[digits[index]]
    if possible_letters:
        for letter in possible_letters:
            path.append(letter)
            backtrack(index + 1, path, digits, letters, combinations)
            path.pop()
            

def letter_combinations(digits):
    combinations = []
    
    if len(digits) == 0:
        return []

    digits_mapping = {
        "1": [""],
        "2": ["a", "b", "c"],
        "3": ["d", "e", "f"],
        "4": ["g", "h", "i"],
        "5": ["j", "k", "l"],
        "6": ["m", "n", "o"],
        "7": ["p", "q", "r", "s"],
        "8": ["t", "u", "v"],
        "9": ["w", "x", "y", "z"]}


    backtrack(0, [], digits, digits_mapping, combinations)
    return combinations
```
学习笔记：数字的长度是n，每个数字对应的字母数量是k，那么时间复杂度就是O(n * k^n)，空间复杂度为O(nk)。

### 问题3:Generate Parentheses

经典的编程问题，问题的目标是生成所有有效的括号组合。

具体来说，给定一个整数 n，表示生成括号的对数，要求输出所有可能的由 n 对括号组成的合法组合。

合法的括号组合需要满足以下条件：

- 每个左括号都必须有一个相应的右括号与之配对。
- 左括号必须先于其相应的右括号出现。
- 在任何一个位置，左括号的数量都必须大于等于右括号的数量。

例如，当 n = 3 时，有效的括号组合包括：
```
"((()))"
"(()())"
"(())()"
"()(())"
"()()()"
```
这道题我解的不好，所以不贴自己的代码了，直接上答案的代码，以及另一种我找到的优化代码。

答案代码：

```python
def back_track(n, left_count, right_count, output, result):
    if left_count >= n and right_count >= n:
        result.append("".join(output))

    if left_count < n:
        output.append('(')
        back_track(n, left_count + 1,
                   right_count, output, result)
        output.pop()

    if right_count < left_count:
        output.append(')')
        back_track(n, left_count,
                   right_count + 1, output, result)
        output.pop()


def generate_combinations(n):
    result = []
    output = []
    back_track(n, 0, 0, output, result)

    return result
```

这是答案的代码，使用常规的列表output，以及最终结果列表result。因为每个分支都是if，所以要做三次判断，第二个判断结束后，要去第三个if做判断，所以最后需要pop掉最后一个符号。下面是一种优化的解法：

```python
def generate_combinations(n):
    def backtrack(s='', left=0, right=0):
        if len(s) == 2 * n:
            result.append(s)
            return
        if left < n:
            backtrack(s + '(', left + 1, right)
        if right < left:
            backtrack(s + ')', left, right + 1)

    result = []
    backtrack()
    return result
```

这里通过在递归调用时传递更新后的字符串。当我们在递归调用时，将更新后的字符串传递给下一个递归调用，相当于传递了一个副本，因此不需要在递归调用结束后手动将字符串恢复到之前的状态，也就不需要使用 pop 函数来移除最后一个字符。在 Python 中，字符串是不可变的，每次对字符串的操作都会创建一个新的字符串对象，因此传递更新后的字符串并不会导致额外的内存消耗。

这种做法的优点是简化了代码，不需要手动处理字符串的恢复操作，使得代码更加清晰简洁。

学习笔记：

时间复杂度：假设输入为 n 对括号。

- 生成所有有效括号组合的数量：对于每一对括号，都有两种选择：添加左括号或右括号。因此，总共有 (2^(2n)) 种可能的括号组合。
- 每种组合的长度为 2n。
- 生成所有有效括号组合的时间复杂度为 O(2^(2n) x 2n) = O(4^n x n)。

空间复杂度：

- 在递归调用过程中，会产生最多 2n 层递归调用（每次递归调用都会增加一个左括号或右括号），因此需要额外的 O(n) 的空间用于存储递归调用栈。
- 存储所有有效括号组合的空间复杂度取决于结果的数量。由于共有 2^(2n) 种可能的括号组合，因此需要额外的 O(2^(2n)) 的空间来存储结果。

综上所述，总的空间复杂度为 O(n + 2^(2n))，其中 2^(2n) 项的空间复杂度占主导。
