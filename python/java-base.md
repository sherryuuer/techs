学习[菜鸟教程](https://www.runoob.com/java/java-tutorial.html)！

## Java特征

Java编程语言有许多显著的特征，使其成为广泛应用的语言。以下是Java的五个最主要特征：

1. **平台无关性 (Platform Independence)**

Java编写的程序可以在不同的操作系统上运行，而不需要修改源代码。这个特性归功于Java虚拟机 (Java Virtual Machine, JVM)。Java代码首先被编译成平台无关的字节码 (Bytecode)，然后通过JVM在不同的平台上解释和执行。这一特性使Java成为一种“编写一次，随处运行”的语言。

2. **面向对象编程 (Object-Oriented Programming, OOP)**

Java是一种严格的面向对象的编程语言。它的设计基于对象和类的概念，促进了代码的可重用性和模块化。主要的OOP特性包括：
   - **封装 (Encapsulation)**: 数据和行为被封装在对象中，提供了良好的数据保护机制。
   - **继承 (Inheritance)**: 类可以继承其他类的特性和行为，使代码重用更加容易。
   - **多态 (Polymorphism)**: 对象可以以多种形式存在，使代码更加灵活和易于扩展。

3. **自动内存管理 (Automatic Memory Management)**

Java具有垃圾收集 (Garbage Collection) 功能，它可以自动管理内存的分配和回收。开发者不需要手动释放内存，减少了内存泄漏和其他内存管理问题的风险。这使得Java程序在内存管理方面更容易和更安全。

4. **丰富的标准库 (Rich Standard Library)**

Java提供了丰富且功能强大的标准库 (Java Standard Library)，它涵盖了从数据结构、网络编程、数据库连接、图形用户界面 (GUI) 到多线程等各种应用。这些库大大简化了开发过程，开发者可以专注于应用逻辑而不是底层实现。

5. **高安全性 (High Security)**

Java从设计之初就强调安全性。它的安全机制包括：
   - **字节码验证**: JVM会在执行字节码之前验证其安全性，防止恶意代码的执行。
   - **沙盒模型 (Sandbox Model)**: 限制应用程序对系统资源的访问，尤其是在网络环境中。
   - **强类型检查 (Strong Type Checking)**: Java的严格类型检查有助于在编译时发现潜在的安全漏洞和错误。
   - **内置的安全API**: 提供了丰富的API来处理加密、鉴权等安全相关的任务。

这些特性使Java成为开发安全、稳定和跨平台应用的理想选择。

## Java Application Structure

Java应用程序结构包括包、类和接口、方法、成员变量、构造函数、主方法、异常处理、文件结构、外部依赖和配置文件。理解这些组件及其组织方式是构建和维护Java应用程序的基础。

- 类class，是实例的蓝图，对象object=实例instance，class包括从属于类的属性和方法，和从属于实例的属性和方法
- class members（两种）
  - Part of the instance：可以有不同的outcome
    - Instance members：Fields，Methods
  - Part of the class：对于所有的instance都相同，因为它是类的一部分，static members
- package 用于将*类和接口*分组到*逻辑命名空间中*，通常对应于物理文件系统中的目录结构，相当于python的模块和包，包可以包含子包和模块，提供类似于文件夹的层次结构，java创建包要在顶部生明，`package com.example.myapp;`，使用的时候要用如此，`import java.util.List;`，相对的python的import：`from mypackage import submodule`，他们的目的都是为了，便于组织，唯一的命名，和import方便
- Constructors：new

## public static void main(String[] args)

`public static void main(String[] args)`

- 这是一个访问修饰符，表示 main 方法是公共的，可以被外部调用。由于 main 方法必须从 Java 运行时环境（JRE）调用，因此它必须是 public 的。
- static 关键字表示该方法*属于类，而不是类的实例*。因为 main 方法在程序启动时被调用，而没有任何类实例，所以它必须是 static 的。
- void 表示该方法*没有返回值*。main 方法是程序的入口点，不需要返回任何值，所以它定义为 void。
- *main* 是方法的名称，这是一个*特别的方法名*，Java 运行时环境会寻找这个方法作为程序的入口点。
- 这是 main 方法的参数，它是一个 String 类型的数组，用于接收从命令行传递给程序的参数。这些参数在程序启动时传递给 main 方法，可以在程序中使用它们。

- 编译：javac MainExample.java
- 运行：java MainExample arg1 arg2

- 不能在同一个类中定义多个 main 方法。Java 不允许在一个类中有两个完全相同的方法签名（包括 main 方法），因为这会导致方法重载冲突和编译错误。但是可以指定不同的class，就可以使用同样的名为main的方法作为接口
- 为了代码的清晰和组织性，建议将每个公共类放在单独的文件中。并且这个文件名必须与类名匹配。

- 独立文件：将每个类放在独立文件中，提升了代码的可读性、可维护性、模块化和版本控制的效率。
- 实践中的例外：在某些特殊情况下（例如内部类、快速原型开发、小型项目），将多个类放在同一个文件中是可以接受的。

## 一切皆是类，一切皆是对象

- 在练习中，真正体会到了一切皆是类，一切皆是对象。
- 连main方法也要放在类定义中。
- 所有的定义都要加上type，这些type也是类，可以用别的类作为另一个类的type加property进行定义。String等也是类。
- 对象作为类的实例：在 Java 中，类是一个模板，而对象是这个模板的实例。所有操作和数据都需要通过对象来进行。
- 引用和操作：程序中的数据和操作都是通过对象和对对象的操作实现的。即使基本数据类型也可以通过包装类来被视为对象。
- 对象的生命周期：对象从创建到销毁都遵循一定的生命周期管理，通过堆内存和垃圾收集器（Garbage Collector）来管理。
- 多态性：对象可以根据它们的类和子类的关系，表现出不同的行为（多态性）。
- 但是它不能多重继承。
- 垃圾收集功能。

*对象的广泛使用*
- 对象创建和使用：大部分程序都是通过创建和操作对象来实现功能的。例如，new Car("Toyota", 2020) 创建了一个新的 Car 对象。
- 集合和容器：Java 提供了丰富的集合类（如 ArrayList、HashMap），这些集合类都是对象，用于存储和管理其他对象。
- 接口和多态：Java 通过接口和多态支持对象的多种形式和行为。例如，List 接口的实现可以是 ArrayList 或 LinkedList，而我们可以通过 List 类型来操作它们。

*面向对象编程（OOP）原则*

- Java 的这些理念深刻地反映了面向对象编程（OOP）的核心原则：
- 封装（Encapsulation）：数据和行为被封装在类和对象中，隐藏了实现细节。
- 继承（Inheritance）：类可以继承其他类的特性和行为，支持代码的复用。
- 多态性（Polymorphism）：对象可以表现出多种行为形式，根据其实际的类型决定运行时的行为。
- 抽象（Abstraction）：类和接口提供了对现实世界概念的抽象表示，简化了复杂系统的设计。

*FP*
- 函数式编程（Functional Programming，FP）是一种编程范式，它强调使用函数来进行计算。与面向对象编程（Object-Oriented Programming，OOP）不同，FP 更加关注函数的纯粹性、不可变性和高阶函数（可以接受其他函数作为参数或返回函数作为结果的函数）。
- 在函数式编程中，函数可以像其他数据类型一样被传递、赋值和返回。
- 惰性计算指的是在需要的时候才进行计算。惰性计算允许定义无限数据结构和延迟计算的值。
- Scala

## 变量

在 Java 编程语言中，变量可以分为两种主要类型：**基本类型**（Primitive Types）和**引用类型**（Reference Types）。

基本类型基本以小写字母开头，引用类型基本以大写字母开头。

### 1. 基本类型（Primitive Types）

基本类型是 Java 中最简单的数据类型。它们存储的是实际的值，而不是指向值的引用。Java 提供了 8 种基本数据类型，每一种类型都有固定的大小和范围：

根据bit位数：

- **整数类型**：
  - `byte`：8位，表示范围为 -128 到 127。
  - `short`：16位，表示范围为 -32,768 到 32,767。
  - `int`：32位，表示范围为 -2^31 到 2^31 - 1。
  - `long`：64位，表示范围为 -2^63 到 2^63 - 1。

- **浮点类型**：
  - `float`：32位，单精度浮点数。
  - `double`：64位，双精度浮点数。

- **字符类型**：
  - `char`：16位，表示一个单一的 Unicode 字符，范围为 0 到 65,535。

- **布尔类型**：
  - `boolean`：表示真 (`true`) 或假 (`false`) 两个值。

这些基本类型都具有直接存储的值，并且内存消耗固定且较小，因此它们是效率最高的数据存储方式之一。

### 2. 引用类型（Reference Types）

引用类型用于存储对象的引用，而不是对象的值本身。Java 中的引用类型包括：

- **类（Class）**：
  - 类是创建对象的蓝图。每个对象都是一个类的实例。例如，`String` 是 Java 中的一个类，当你创建一个 `String` 类型的变量时，它实际上是对 `String` 对象的引用。
  - 示例：
    ```java
    String text = "Hello, World!";
    ```

- **接口（Interface）**：
  - 接口是 Java 中一种特殊的类型，它定义了类必须实现的方法，而不包含这些方法的实现细节。
  - 示例：
    ```java
    List<String> list = new ArrayList<>();
    ```

- **数组（Array）**：
  - 数组是一种特殊的引用类型，它保存同类型数据的集合。
  - 示例：
    ```java
    int[] numbers = {1, 2, 3, 4, 5};
    ```

- **枚举（Enum）**：
  - 枚举类型是一种特殊的类，表示一组常量（例如，方向、状态）。
  - 示例：
    ```java
    enum Day { MONDAY, TUESDAY, WEDNESDAY }
    Day today = Day.MONDAY;
    ```

引用类型变量实际存储的是对象在内存中的地址，而不是对象本身。对象的操作通过这些引用来间接地访问实际的对象。

单引号和双引号在 Java 中的不同用途和规则是编写和调试 Java 程序的重要基础。单引号用于字符常量，而双引号用于字符串常量，它们分别对应 char 和 String 类型。

### 基本类型与引用类型的区别

1. **内存分配**：
   - 基本类型内存占用较小，值直接存储在栈内存中，分配固定大小的内存空间。
   - 引用类型的引用（地址）存储在栈内存中，而对象本身存储在堆内存中，引用指向对象的实际内存地址。

2. **操作方式**：
   - 基本类型的变量是对实际值的直接操作。
   - 引用类型的变量是通过引用来间接操作对象。

3. **默认值**：
   - 基本类型有明确的默认值（如 `int` 默认值是 `0`，`boolean` 默认值是 `false`）。
   - 引用类型的默认值是 `null`，表示它们没有指向任何对象。

### 示例代码

以下示例展示了基本类型和引用类型的使用：

```java
public class Main {
    public static void main(String[] args) {
        // 基本类型
        int num = 10; // 整型基本类型
        double pi = 3.14; // 双精度浮点型基本类型

        // 引用类型
        String greeting = "Hello, World!"; // String 类
        int[] numbers = {1, 2, 3}; // 数组类型
        MyClass obj = new MyClass(); // 自定义类的对象

        System.out.println("Number: " + num);
        System.out.println("Pi: " + pi);
        System.out.println("Greeting: " + greeting);
        System.out.println("First number in array: " + numbers[0]);
        System.out.println("Object reference: " + obj);
    }
}

class MyClass {
    // 自定义类
}
```
## homebrew install

```bash
# 更新
brew update
# 搜索
brew search openjdk
# 开始安装
brew install openjdk@17
# 根据安装后的提示，run了如下命令
If you need to have openjdk@17 first in your PATH, run:
  echo 'export PATH="/opt/homebrew/opt/openjdk@17/bin:$PATH"' >> ~/.zshrc

For compilers to find openjdk@17 you may need to set:
  export CPPFLAGS="-I/opt/homebrew/opt/openjdk@17/include"

# 使更新生效
source ~/.zshrc
# 验证版本
java -version

# 确认安装路径
brew --prefix openjdk@17
# /opt/homebrew/opt/openjdk@17

# found the home
/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home
```

编辑VScode的配置文件setting.json
```json
  "java.configuration.runtimes": [
    {
      "name": "JavaSE-17",
      "path": "/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home",
      "default": true
    }
  ]
```

为了配置项目根目录，我配置了如下设置：
```json
  "java.project.sourcePaths": ["src"],
```
### javac

在 Java 中，通过 `javac` 编译后生成的文件通常是字节码文件（bytecode files），而不是二进制文件。Java 编译器 (`javac`) 将 Java 源代码编译成 Java 字节码，而不是将其编译成本地机器的二进制代码。

Java 字节码是一种中间代码（intermediate code），它不直接运行在计算机的硬件上，而是运行在 Java 虚拟机（JVM）上。这种中间代码的好处是跨平台性，即你可以将同样的字节码文件在不同的操作系统上运行，只要这些系统都安装了相同版本的 Java 运行时环境（JRE）或 Java 开发工具包（JDK）。

Java 字节码文件的扩展名是 `.class`，每个 `.class` 文件对应一个 Java 类。例如，编译后生成的 `Main.class` 文件就是一个 Java 字节码文件。

运行方法：

```bash
javac Main.java
java Main
```

复杂一点的：

**编译代码**

从项目的根目录（`project_root`）下执行编译命令。编译器会根据 `src` 目录下的包结构生成对应的类文件。

1. **打开终端（命令行）**。
2. **导航到项目根目录**。假设你的项目目录是 `project_root`：

   ```bash
   cd /path/to/project_root
   ```

3. **执行编译命令**：

   ```bash
   javac -d out src/main/MainAccount.java src/bank/BankAccount.java
   ```

   这个命令的解释：
   - `javac` 是 Java 编译器命令。
   - `-d out` 表示将编译后的类文件输出到 `out` 目录，并保持包结构。
   - `src/main/MainAccount.java` 和 `src/bank/BankAccount.java` 是需要编译的 Java 源文件。

   编译完成后，目录结构应为：

   ```
   project_root/
   ├── out/
   │   ├── MainAccount.class
   │   └── bank/
   │       └── BankAccount.class
   ├── src/
   │   ├── MainAccount.java
   │   └── bank/
   │       └── BankAccount.java
   ```

**运行代码**

编译完成后，你可以运行 `MainAccount` 类。确保你在项目的根目录下执行运行命令：

1. **在终端中**：

   ```bash
   java -cp out main.MainAccount
   ```

   这个命令的解释：
   - `java` 是 Java 运行时命令。
   - `-cp out` 指定类路径为 `out` 目录，Java 虚拟机会从这个目录加载编译后的类文件。
   - `main.MainAccount` 是类的全限定名，`main` 是包名，`MainAccount` 是类名。

2. **输出结果**：

   运行后，你应该看到如下输出：

   ```
   Account number: 78986
   Account holder: Saally
   Account balance: 1.0E12
   ```

## Static

是静态方法的意思，表明这个member，它属于类，不属于实例，如果一个函数有static，那么可以在类中直接使用该方法，如果没有static关键字，则需要对类进行实例化。

Python中也有staticmethod修饰符。表示就是可以直接靠类使用的方法。

## extends

继承其他的class，对于变量可以直接使用，对于方法需要是public/protected属性。

## 构造方法（constructor）

构造方法重载指的是在一个类中可以定义多个构造方法，它们具有相同的名字（类的名字），但参数列表不同（参数的类型、数量或顺序不同）。通过这种方式，可以根据不同的需求初始化对象，提供灵活性和方便性。

相当于Python中的初始化函数，但是Python的初始化函数不能有多个，只能靠类方法等实现。

## 类型转换

`++x`：先自增，再使用。`x++`：先使用，再自增。

对象的比较：`boolean same = s.equals(s1);` 也就是使用自身的方法

字符串取切片区间：`String substring = s.substring(1, 2);` 区间和Python一样也是包含左边不包含右边

以某字符为prefix：`boolean startswith = s.startsWith("H");`

长度：`s.length()`

`s.charAt(3)`返回index位于3的字符

Cast，类型转换：

```java
long l = 123;
int x = (int) l;

double d = 1.2;
float f = (float) d;
```

在 Java 中，字符类型 `char` 使用 16 位无符号整数表示 Unicode 字符，这意味着它可以表示的最小值是 `0`，最大值是 `65535`。如果超过了这个范围，`char` 类型就会溢出（overflow）。

1. **`char` 类型的范围**：
   - `char` 类型在 Java 中表示一个 16 位的无符号整数。
   - 它的取值范围是 `0` 到 `65535`，也就是 `2^16 - 1`。

2. **Overflow（溢出）**：
   - 当一个无符号整数超过它的最大值时，继续增加会导致它回到最小值，这被称为溢出。
   - 对于 `char` 类型，当值达到 `65535` 时，再加 `1`，就会超出它的最大值，发生溢出。

3. **加法操作和溢出**：
   - 如果对 `char` 类型的最大值 `65535` 加 `1`，它就会溢出。
   - 因为 `char` 类型是无符号的，所以它不能表示负数。溢出后，值会从头开始，也就是从 `0` 开始。
   - 因此，`65535 + 1` 在 `char` 类型下会变成 `0`。

```java
public class CharOverflowExample {
    public static void main(String[] args) {
        char maxChar = 65535; // char 的最大值
        System.out.println("Initial maxChar value: " + (int) maxChar); // 输出 65535

        // 对 maxChar 加 1
        maxChar++;

        // 输出溢出后的值
        System.out.println("maxChar after increment: " + (int) maxChar); // 输出 0
    }
}
```
## Primitives 和 Objects

- 基本数据类型（primitives）适合用于需要高性能和低内存占用的场景，特别是在处理简单的数值计算时。
  - 内存占用较小
  - immutable不可变
  - *stack*存储

- 对象（objects）提供了更丰富的功能和灵活性，适合表示复杂的数据结构和行为。
  - 内存占用较大
  - default是mutable可变的
  - 可以包含基本数据类型，用于定义object的比如属性
  - *对象可以包含对象*
    - 这是一个很有趣的思想，比如一只猫有一个主人，主人有一辆车，一辆车有价格，价格就是一个基本数字double类型
    - 从一个对象，检索到其他对象的过程是一种高维度的思考方式
  - 也是前面说的reference类型
  - *heap*存储

- 代入函数，其实是一种copy：当对基本类型的data进行copy和修改后，原本的data不会改变，因为它immutable，但是对object进行copy和修改，只是在其引用上修改，原本object也会被修改。

## **栈**（Stack）和**堆**（Heap）

### 栈（Stack）

栈内存用于存储局部变量、方法调用和方法参数。它是一个LIFO（Last In, First Out）结构，意味着最后一个被压入栈的元素最先被弹出。栈的主要特点和用途包括：

1. **存储内容**：
   - 局部变量：在方法内部定义的变量。
   - 方法调用：每次方法调用都会在栈上创建一个栈帧（Stack Frame），存储该方法的参数、局部变量和返回地址。
   - 方法参数：传递给方法的参数也存储在栈上。

2. **生命周期**：
   - 栈上的内存由方法的执行控制。一个方法一旦结束，栈上的相应栈帧会被弹出并释放内存。
   - 局部变量的生命周期也与方法相同，当方法完成时，局部变量被销毁。

3. **访问速度**：
   - 由于栈是LIFO结构，访问速度非常快，因为每次操作只涉及栈顶的元素。

4. **自动管理**：
   - Java 自动管理栈内存，无需手动分配和释放。

5. **空间限制**：
   - 栈的空间较小，通常有限。过深的递归调用或过多的局部变量可能会导致栈溢出（StackOverflowError）。

#### 栈的例子

```java
public class StackExample {
    public static void main(String[] args) {
        int a = 5;   // 局部变量a
        int b = 10;  // 局部变量b
        int result = add(a, b);  // 方法调用，result存储返回值
        System.out.println(result);
    }

    public static int add(int x, int y) {
        int sum = x + y;  // 局部变量sum
        return sum;       // 返回sum
    }
}
```

在上述代码中，`a`、`b`、`x`、`y` 和 `sum` 都是存储在栈上的局部变量。`add` 方法调用时，会在栈上创建一个新的栈帧来存储其参数和局部变量。

### 堆（Heap）

堆内存用于动态分配存储在Java中的所有对象和类实例。与栈不同，堆内存的管理更为复杂和灵活。堆的主要特点和用途包括：

1. **存储内容**：
   - Java中的所有对象和数组。
   - 类的实例变量（字段）。

2. **生命周期**：
   - 堆上的对象有更长的生命周期，它们的内存由Java的垃圾回收机制（Garbage Collection）管理。
   - 垃圾回收器会自动识别和清理不再使用的对象，从而释放内存。

3. **访问速度**：
   - 由于堆内存不遵循特定的顺序，其访问速度比栈慢。

4. **手动管理**：
   - 尽管不需要手动释放内存，开发者仍然需要小心管理对象的生命周期，避免内存泄漏（Memory Leak）。

5. **空间优势**：
   - 堆空间比栈大得多，适合存储大对象和长期存在的数据。

#### 堆的例子

```java
public class HeapExample {
    public static void main(String[] args) {
        Person person = new Person("John", 30); // 创建一个Person对象
        System.out.println(person.getName() + " is " + person.getAge() + " years old.");
    }
}

class Person {
    private String name; // 实例变量，存储在堆上
    private int age;     // 实例变量，存储在堆上

    public Person(String name, int age) {
        this.name = name;
        this.age = age;
    }

    public String getName() {
        return name;
    }

    public int getAge() {
        return age;
    }
}
```

在上述代码中，`Person` 对象存储在堆上，`name` 和 `age` 是该对象的实例变量，它们也存储在堆上。

### 栈与堆的对比

| 特性            | 栈（Stack）                                    | 堆（Heap）                                   |
|-----------------|------------------------------------------------|---------------------------------------------|
| 用途            | 存储局部变量、方法调用和方法参数                | 存储所有对象和数组                          |
| 内存管理        | 自动管理（LIFO）                                | 由垃圾回收器自动管理                         |
| 访问速度        | 快（由于LIFO结构）                              | 相对较慢（由于需要动态分配和垃圾回收）       |
| 生命周期        | 与方法调用周期一致                              | 对象可在方法结束后继续存在                   |
| 大小限制        | 较小，受限于栈大小（可能导致StackOverflowError）| 较大，适合存储大量和长期存在的数据           |
| 内存释放        | 方法结束时自动释放                              | 通过垃圾回收机制自动释放                     |

理解栈和堆在Java中的角色和区别，对于有效地管理内存和优化应用程序性能非常重要。在编写代码时，注意局部变量和对象的使用方式，可以帮助避免常见的内存管理问题，如栈溢出和内存泄漏。

## Access Modifiers 访问修饰符

- public：任何地方都可以访问。
- protected：同包内和子类可以访问。
- default（包级私有）：仅同包内可以访问。（啥也不用写）
- private：仅类内部可以访问。
  - 针对私有变量，设置公有的set和get就可以从外部访问了

## String Class

String不可变immutable，这意味着，一个字符串对象的值是不可改变的，因为任何新的字符串值会被存储在一个分开的新的对象中，变量会指向这个新的对象的引用。

在Java中，字符串（`String`类）是线程安全的。线程安全性意味着多个线程可以同时访问同一个对象，而不会导致数据不一致或其他并发问题。Java中的`String`类具有以下几个特性，使其成为线程安全的：

**不可变性（Immutability）**：
- `String`对象是不可变的。一旦创建了一个`String`对象，它的值就不能被改变。任何对字符串的操作（如拼接、替换等）都会生成一个新的`String`对象，而不是修改原来的对象。这种不可变性确保了在多线程环境下，一个线程对`String`对象的修改不会影响到其他线程。

**内部实现**：
- `String`类的底层实现使用了一个`final`的字符数组来存储字符串的值。这个字符数组在`String`对象创建后也不能被修改。即使多个线程同时访问这个字符数组，也不会有并发问题。

**常量池**：
- Java有一个特殊的字符串常量池（String Pool），用于存储字符串字面量。每次创建一个新的字符串字面量时，JVM会先检查常量池中是否已经存在相同的字符串。如果存在，直接返回该字符串的引用；如果不存在，则将其添加到常量池中。由于字符串常量池中的字符串也是不可变的，所以它们的线程安全性也得到了保证。

尽管`String`本身是线程安全的，但在多线程环境下操作字符串时，仍需要注意一些其他问题。例如，如果在多个线程中频繁地拼接字符串，最好使用`StringBuilder`或`StringBuffer`类来代替`String`。其中，`StringBuilder`是非线程安全的，但性能较高，而`StringBuffer`是线程安全的，可以在多线程环境中使用。他们操作起来就像是在操作数组。

如果使用equals()，则是他们的content被比较。当字符串在常量池中指向同一个对象的时候，==可以得到True的结果。

## Dates & Times

- `java.time.LocalDate`
- `java.time.LocalTime`
- `java.time.LocalDateTime`
- `java.time.ZonedDateTime` set by ZoneId
- `java.time.Duration`：秒和微秒
- `java.time.Period`：年月日
- 加减区间：plus/minus duration/period

- `.now()`, `.of()`, `.parse()`
- `.getYear()`, `.getDayOfWeek()`, and so on
- `.minusWeeks()`, `.plusDays()`, and so on

- `java.time.format.DateTimeFormatter`： `.ofPattern("MM/dd/yyyy")`
  - 创建了formatter实例后，使用实例调用`.format()`
  - string to date, `.parse(<string>, <the instance of the formatter>)`
- `java.time.format.DateTimeFormatterBuilder`： 类似于 String 的 Builder，可以进行append等类似列表的操作，很神奇

- 其他的还有util里的：参考[菜鸟教程](https://www.runoob.com/java/java-date-time.html)

## OOP

Java的面向对象编程（OOP）有四大核心特性：封装、继承、多态和抽象。这些特性使得Java程序具有模块化、可维护性和可扩展性。

### 1. 封装（Encapsulation）
封装是将对象的属性和方法封装在一个类中，通过访问控制（如private, protected, public）来限制对这些属性和方法的直接访问。这样可以保护数据不被随意修改，同时也隐藏了对象的内部实现细节。
- **优点：** 提高了代码的安全性和可维护性，提供了清晰的接口。
- **示例：**
  ```java
  public class Person {
      private String name;
      private int age;

      public String getName() {
          return name;
      }

      public void setName(String name) {
          this.name = name;
      }

      public int getAge() {
          return age;
      }

      public void setAge(int age) {
          this.age = age;
      }
  }
  ```

### 2. 继承（Inheritance）
继承是通过从已有的类（称为父类或超类）中创建一个新的类（称为子类或派生类），使子类可以继承父类的属性和方法，并可以新增自己的属性和方法。继承实现了代码的复用和扩展。
- **优点：** 代码复用、增强类的功能、提高代码的扩展性。
- **示例：**
  ```java
  public class Animal {
      public void eat() {
          System.out.println("This animal eats food.");
      }
  }

  public class Dog extends Animal {
      public void bark() {
          System.out.println("The dog barks.");
      }
  }
  ```

### 3. 多态（Polymorphism）
多态是指同一操作在不同对象上具有不同表现形式的能力。多态性通过方法重载和方法重写实现。它允许对象在不同的上下文中以不同的方式进行响应。
- **优点：** 提高代码的灵活性和可扩展性。
- **示例：**
  ```java
  public class Animal {
      public void makeSound() {
          System.out.println("Some generic animal sound");
      }
  }

  public class Dog extends Animal {
      @Override
      public void makeSound() {
          System.out.println("Bark");
      }
  }

  public class Cat extends Animal {
      @Override
      public void makeSound() {
          System.out.println("Meow");
      }
  }

  public class Main {
      public static void main(String[] args) {
          Animal myDog = new Dog();
          Animal myCat = new Cat();
          myDog.makeSound(); // 输出：Bark
          myCat.makeSound(); // 输出：Meow
      }
  }
  ```

### 4. 抽象（Abstraction）
抽象是指将对象的复杂实现隐藏起来，只保留对象的必要特性和行为。通过抽象类和接口，可以定义对象的抽象行为。
- **优点：** 减少代码复杂度，提供清晰的接口，促进代码的设计。
- **示例：**
  ```java
  abstract class Animal {
      public abstract void makeSound();

      public void sleep() {
          System.out.println("This animal sleeps.");
      }
  }

  class Dog extends Animal {
      @Override
      public void makeSound() {
          System.out.println("Bark");
      }
  }

  class Main {
      public static void main(String[] args) {
          Animal myDog = new Dog();
          myDog.makeSound(); // 输出：Bark
          myDog.sleep(); // 输出：This animal sleeps.
      }
  }
  ```

通过这四大特性，Java实现了面向对象编程的理念，帮助开发者构建更为结构化、模块化和可维护的代码。

### Override，Overload，Hide

- Override（重写）： 子类重写父类的非静态方法，方法签名必须一致，实现多态性。
- Overload（重载）： 同一个类中方法名相同，但参数列表不同的方法。
- Hide（隐藏）： 子类定义了一个与父类中静态方法同名且参数列表相同的方法，隐藏父类的静态方法。

## Constructor

- 当你没有创建Constructor的时候，系统会在编译的时候为你创建一个default的Constructor，一个空的
- Custom Constructor可以设置在初始化的时候传入的参数，一个Class可以有多个不同的Constructor，他们有不同的参数
- 当你设置了Custom Constructor，系统就不会给它的default的Constructor了
- **注意**：Constructor创建的时候，是没有返回类型的，不能加void，不然就是method了

### super() and this()

- super是引用了父类的Constructor
- this是引用了自己的类中的其他Constructor
- 如果没有显式地call父类的Constructor，那么java会自己添加super，默认call了父类的Constructor，并且是没有参数的，这时候如果给父类设置了参数，那么会出现compile错误
