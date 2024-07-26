## Python 单元测试（unittest&pytest）

---

### 什么是单元测试

单元测试，顾名思义，是进行小型代码单元的测试代码。写 python 代码来测试其他代码，通常是函数。单环测试的特性有这么几个：隔离性，高性能，快速，可重复。和单元测试不同，还有集成测试功能测试等，也就是整个业务流程整体的测试。在日语环境我理解为，结合测试。单元测试经常和敏捷开发结合起来。

通过测试，会让代码进行进化，修改，变成能更加泛用的代码。这让我想到了写 leetcode 的时候，经常会因为其他用户测试用例，不断的修改我的代码，虽然挺烦的，但是这也是一种进化代码的方式。

### 测试边界和方法

对于外部库，我们假定他们有自己的测试。不在我门的范围内。对于所用的工具，主要有两个，一个是 unittest，一个是 pytest。当涉及到我们的代码的测试场景时，unittest 单独使用很可能就足够了，因为它有很多帮助器。然而，对于更复杂的系统，我们有多个依赖项、与外部系统的连接，并且可能需要修补对象、定义固定装置和参数化测试用例，那么 pytest 看起来是一个更完整的选项。

### unittest 框架

`unittest` 是 Python 标准库中的测试框架，提供了一套强大的工具来编写和组织测试用例。`unittest` 受到 Java 中的 JUnit 的影响，因此在使用上有一些类似之处。

以下是 `unittest` 的一些主要特点：

1. **面向对象：** `unittest` 使用面向对象的方式组织测试，测试用例是继承自 `unittest.TestCase` 的类，测试方法以 `test_` 开头。
2. **丰富的断言：** `unittest` 提供了多种断言来验证测试条件，包括相等、不相等、包含、异常等。
3. **测试装置：** `unittest` 提供了 `setUp` 和 `tearDown` 方法，分别用于在测试用例执行前后进行初始化和清理工作。
4. **测试发现：** `unittest` 能够自动发现和执行项目中的测试用例，也可以通过命令行工具或测试套件手动执行。
5. **测试套件：** 可以使用 `TestLoader` 和 `TestSuite` 来组织和运行多个测试用例。
6. **丰富的插件支持：** 通过扩展 `TestCase` 类，可以实现自定义的测试装置和行为。

一个简单的 `unittest` 测试用例的例子：

```python
import unittest

class TestMathOperations(unittest.TestCase):
    def test_addition(self):
        self.assertEqual(1 + 1, 2)

    def test_subtraction(self):
        self.assertEqual(3 - 1, 2)

if __name__ == '__main__':
    unittest.main()
```

你可以通过运行上述测试文件或使用 `unittest` 命令行工具来执行测试。

```bash
python test_sample.py
```

虽然 `unittest` 是 Python 的标准库测试框架，但在一些项目中，开发者也选择使用第三方框架，如 `pytest`，因为它提供了更简洁、灵活和功能丰富的测试体验。选择使用哪个测试框架通常取决于个人或团队的偏好和项目需求。

### pytest 框架

`pytest` 是一个用于编写和运行 Python 测试的框架。功能强大而灵活，被广泛用于 Python 项目中的单元测试、集成测试和功能测试。

以下是 `pytest` 的一些特点：

1. **简单易用：** 编写测试用例非常简单，测试用例的文件以 `test_` 开头，测试函数以 `test_` 开头。
2. **自动发现：** `pytest` 能够自动发现项目中的测试用例，无需额外配置。
3. **丰富的断言：** `pytest` 提供了丰富的断言（assertions）用于验证测试条件，包括标准的 `assert`，也包括丰富的 `assert` 扩展，提供更多的信息以便更容易调试测试失败。
4. **丰富的插件：** `pytest` 可以通过插件进行扩展，允许你自定义测试行为和输出。
5. **支持参数化测试：** 你可以使用装饰器 `@pytest.mark.parametrize` 来定义参数化测试，使得你可以使用不同的输入值运行相同的测试。
6. **并行执行：** `pytest` 可以并行运行测试用例，提高测试的执行效率。

一个简单的 `pytest` 测试用例的例子：

```python
# content of test_sample.py
def test_addition():
    assert 1 + 1 == 2

def test_subtraction():
    assert 3 - 1 == 2
```

可以通过运行 `pytest` 命令来执行测试，`pytest` 将自动查找和执行项目中的测试用例。

```bash
pytest test_sample.py
```

由于我工作中用的是 pytest 所以重点梳理这方面的。

### pytest 的参数化测试

参数化测试是一种测试技术，允许你以不同的参数运行同一个测试用例。在 `pytest` 中，你可以使用 `@pytest.mark.parametrize` 装饰器来实现参数化测试。

通过参数化测试，你可以用不同的输入数据多次运行相同的测试函数，从而更全面地验证函数的行为。这对于测试函数在不同输入情况下的正确性非常有用。

以下是一个简单的例子：

```python
import pytest

def add(a, b):
    return a + b

@pytest.mark.parametrize("input_a, input_b, expected", [
    (1, 2, 3),
    (0, 0, 0),
    (-1, 1, 0),
    (10, -5, 5),
])
def test_addition(input_a, input_b, expected):
    result = add(input_a, input_b)
    assert result == expected
```

在这个例子中，`test_addition` 函数被标记为参数化测试，它接受三个参数：`input_a`、`input_b` 和 `expected`。`@pytest.mark.parametrize` 装饰器定义了不同的输入组合，并为每组输入运行一次测试。

在这个例子中，`test_addition` 函数将以 `(1, 2, 3)`、`(0, 0, 0)`、`(-1, 1, 0)` 和 `(10, -5, 5)` 这四组参数分别运行四次。如果其中任何一次运行失败，`pytest` 将报告哪一组参数导致了失败，帮助你更容易地定位问题。

参数化测试使得能够更容易地覆盖不同的测试场景，减少代码重复，同时保持测试的可读性。

### pytest 的 fixture

在 `pytest` 中，fixture 是一种用于为测试提供预配置的机制。允许你在测试函数运行之前或之后执行一些代码，并将其结果传递给测试函数。Fixtures 可以用来模拟测试环境、提供测试数据或执行其他与测试相关的任务。

一般情况下，你需要定义 fixture 函数，并在测试函数的参数中引用它。`pytest` 将自动检测和调用相关的 fixture 函数。Fixture 函数可以返回一个值，该值会被传递给测试函数，或者它可以在测试运行之前和之后执行一些操作。

以下是一个简单的例子：

```python
import pytest

# 定义一个简单的 fixture，返回一个字符串
@pytest.fixture
def greeting():
    return "Hello, "

# 使用 fixture，在测试函数参数中引用它
def test_greet_person(greeting):
    person = "Alice"
    result = greeting + person
    assert result == "Hello, Alice"
```

在这个例子中，`greeting` 是一个简单的 fixture，返回一个字符串 "Hello, "。测试函数 `test_greet_person` 接受 `greeting` 作为参数，并使用它来构造问候语。当测试运行时，`pytest` 会自动调用 `greeting` fixture 并将其返回值传递给测试函数。

常见的用途包括：

1. **设置和清理测试环境：** 在测试开始前创建必要的资源，测试结束后进行清理。
2. **提供测试数据：** 为测试函数提供一些初始数据。
3. **模拟外部依赖：** 通过 fixture 实现对外部依赖的模拟，例如数据库连接、网络请求等。

可以通过命令行选项 `-s` 来查看 fixture 的调用过程，以便更好地理解它是如何在测试中起作用的：

```bash
pytest -s test_example.py
```

`-s` 选项允许 fixture 打印输出，以帮助调试。

### 实践中所得

- 函数名的设置，最好使用动词开头，显示功能。

比如检查 table 是否存在的函数 check_table_exsits 诸如此类。

- 格式的统一

f 方法的统一：文字列统一使用 f 方法而不是 format 方法。

import 改行：import 第三方的库，以及 import 自己的 function 之前添加改行。

功能改行：test 函数中，在准备，执行，确认中添加改行，方便阅读和确认。

生产环境独立：为了和生产环境相互独立，不影响生产环境，比如上面的 table check，最好不使用生产环境的 table，而是独立准备测试用例。

例如：

```python
@pytest.fixture
def test_table(client):
    """create test table"""
    query = f"""
        create or replace table temp.test_table
        (
            id string
            ,record_type string
            ,created_date date
        )
    """

    client.query(query).result()

    yield

    drop_query = f"""
        drop table if exists temp.test_table
    """

    client.query(drop_query).result()
```

- .gitignore 添加如下标记可以递归地忽视所有的同类文件夹

```
**/__pycache__/**
```

- pytest.init 的设置方法内容

```
# pytest.init
[pytest]
testpaths = tests
pythonpath = src
addopts =
    --strict-markers
```
