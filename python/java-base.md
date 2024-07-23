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

- **final**：不可改变的变量，不可override的方法，和不能继承的类

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

## 枚举(enum)

- Java 枚举是一个特殊的类，一般表示一组常量，比如一年的 4 个季节，一年的 12 个月份，一个星期的 7 天，方向有东南西北等。
- 在Java中，每个枚举类型在编译时都会自动生成一个对应的类。这意味着每个枚举类型实际上是一个类，这个类继承自java.lang.Enum。
- Java 枚举类使用 enum 关键字来定义，各个常量使用逗号 , 来分割。
- 常用于switch中
- 类型安全，可读性强，内存运算效率高

* values() 返回枚举类中所有的值。
* ordinal()方法可以找到每个枚举常量的索引，就像数组索引一样。
* valueOf()方法返回指定字符串值的枚举常量。
* toString()转换为字符串常量

**每个枚举都是通过 Class 在内部实现的，且所有的枚举值都是 public static final 的。**

1. 枚举是通过类（`Class`）在内部实现的

在Java中，每个枚举类型在编译时都会自动生成一个对应的类。这意味着每个枚举类型实际上是一个类，这个类继承自`java.lang.Enum`。例如，定义如下的枚举：

```java
public enum Day {
    SUNDAY, MONDAY, TUESDAY, WEDNESDAY, THURSDAY, FRIDAY, SATURDAY
}
```

编译后会生成一个类似以下结构的类：

```java
public final class Day extends Enum<Day> {
    public static final Day SUNDAY = new Day("SUNDAY", 0);
    public static final Day MONDAY = new Day("MONDAY", 1);
    public static final Day TUESDAY = new Day("TUESDAY", 2);
    public static final Day WEDNESDAY = new Day("WEDNESDAY", 3);
    public static final Day THURSDAY = new Day("THURSDAY", 4);
    public static final Day FRIDAY = new Day("FRIDAY", 5);
    public static final Day SATURDAY = new Day("SATURDAY", 6);

    private static final Day[] VALUES = {SUNDAY, MONDAY, TUESDAY, WEDNESDAY, THURSDAY, FRIDAY, SATURDAY};

    private Day(String name, int ordinal) {
        super(name, ordinal);
    }

    public static Day[] values() {
        return VALUES.clone();
    }

    public static Day valueOf(String name) {
        for (Day day : VALUES) {
            if (day.name().equals(name)) {
                return day;
            }
        }
        throw new IllegalArgumentException("No enum constant " + name);
    }
}
```

2. 枚举值是`public static final`

枚举类型的每个枚举常量都是`public static final`的，这意味着：

- **public**：枚举常量可以被外部代码访问。
- **static**：枚举常量属于枚举类型本身，而不是某个特定的实例。这使得我们可以通过类名直接访问这些常量，例如`Day.MONDAY`。
- **final**：枚举常量是不可变的，一旦创建，就不能被更改。

每个枚举常量实际上是枚举类型的一个实例。由于这些常量是`static`的，它们在类加载时就被创建并初始化。因此，可以在任何地方通过类名直接访问它们。

3. 遍历 for-each

```java
public enum Day {
    SUNDAY, MONDAY, TUESDAY, WEDNESDAY, THURSDAY, FRIDAY, SATURDAY
}

public class EnumExample {
    public static void main(String[] args) {
        // 使用for-each循环遍历所有的枚举常量
        for (Day day : Day.values()) {
            System.out.println(day);
        }
    }
}
```

## Interface

在Java编程语言中，`Interface`（接口）是一种抽象类型，定义了一组方法，但不提供这些方法的实现。接口指定了类必须遵循的协议，它是实现多态性和解耦代码的关键机制。

以下是Java接口的一些关键特性：

1. **方法声明**：
   - 接口中的方法默认是`public`和`abstract`，不包含方法体。
   - 在Java 8中，引入了`default`方法和`static`方法。`default`方法提供了方法的默认实现，`static`方法则是接口中的静态方法，`private`是只能在接口内部使用的方法，隐藏方法的内部结构

2. **常量**：
   - 接口不能包含普通的实例变量（fields）。这是因为接口是抽象的行为定义，不能有具体的实现细节，而实例变量属于具体实现的一部分。接口中可以包含常量，默认是`public static final`。即，它们是公共的、静态的、不可变的常量。

3. **多重继承**：
   - Java类可以实现多个接口，克服了Java类只能单继承的限制。这使得接口在设计灵活性和可扩展性方面非常有用。

4. **实现接口**：
   - 一个类通过使用`implements`关键字来实现一个接口，并且必须提供接口中所有抽象方法的实现。

5. **标记接口**：
   - 一些接口没有方法或常量，例如`Serializable`接口。这些被称为标记接口，用于表示类具有某种属性。

下面是一个简单的接口示例，以及一个实现该接口的类：

```java
// 定义一个接口
public interface Animal {
    // 抽象方法，没有花括号
    void eat();
    void sleep();
}

// 实现该接口的类
public class Dog implements Animal {
    @Override
    public void eat() {
        System.out.println("Dog is eating");
    }

    @Override
    public void sleep() {
        System.out.println("Dog is sleeping");
    }

    public static void main(String[] args) {
        Dog dog = new Dog();
        dog.eat();
        dog.sleep();
    }
}
```

在这个例子中，`Animal`接口定义了两个抽象方法`eat`和`sleep`。`Dog`类实现了`Animal`接口，并提供了这些方法的具体实现。在`main`方法中，创建了一个`Dog`对象并调用了其`eat`和`sleep`方法。

接口在Java编程中有许多应用场景，包括：

- **定义规范**：接口可以定义一组方法，要求实现这些接口的类必须提供这些方法的具体实现。
- **解耦和模块化**：接口可以帮助分离代码的实现和使用者，从而实现解耦和模块化。
- **多态性**：接口使得不同类可以实现相同的接口，从而可以用相同的方式对待不同的对象，实现多态性。
- **回调机制**：接口可以用于定义回调方法，以便在特定事件发生时调用。

**接口冲突**：如果一个类有两个接口，两个接口有相同的`default`方法，会产生冲突conflict，解决它必须在类中`override`这个方法，让类有定义的方法，也就是覆盖这两个冲突的默认方法。

## Abstract Class & Method

在Java中，抽象类和抽象方法是面向对象编程中的重要概念，它们用于定义通用行为的框架，而不提供具体实现。以下是它们的定义和用法：

### 抽象类（Abstract Class）

**抽象类**是不能被实例化的类，它通常包含一个或多个抽象方法（没有方法体的方法），以及可以包含具体的方法（有方法体的方法）。抽象类提供了一种定义类层次结构和共享代码的机制。

- **定义抽象类**：使用关键字`abstract`定义。
- **不能实例化**：不能直接创建抽象类的实例。
- **可以包含具体方法**：除了抽象方法外，抽象类还可以包含具体方法（具有方法体的方法）。
- **可以有成员变量**：抽象类可以有成员变量。
- **可以继承**：抽象类可以被其他类继承，子类必须实现所有的抽象方法，或者本身也是抽象类。

### 抽象方法（Abstract Method）

**抽象方法**是没有方法体的方法，仅声明方法签名。它们必须在抽象类中定义，并且必须在非抽象子类中实现。

- **定义抽象方法**：使用关键字`abstract`定义。
- **没有方法体**：抽象方法只有方法签名，没有方法体。
- **必须在子类中实现**：如果一个类继承了抽象类，必须实现所有的抽象方法，除非该子类也是抽象类。

```java
// 定义一个抽象类
public abstract class Animal {
    // 抽象方法
    public abstract void eat();
    public abstract void sleep();

    // 具体方法
    public void breathe() {
        System.out.println("Animal is breathing");
    }
}

// 实现抽象类的子类
public class Dog extends Animal {
    @Override
    public void eat() {
        System.out.println("Dog is eating");
    }

    @Override
    public void sleep() {
        System.out.println("Dog is sleeping");
    }

    public static void main(String[] args) {
        Dog dog = new Dog();
        dog.eat();
        dog.sleep();
        dog.breathe();
    }
}
```

在这个例子中：

- `Animal`是一个抽象类，包含两个抽象方法`eat`和`sleep`，以及一个具体方法`breathe`。
- `Dog`类继承了`Animal`类，并实现了所有的抽象方法。
- 在`main`方法中，创建了一个`Dog`对象，并调用了其`eat`、`sleep`和`breathe`方法。

抽象类和抽象方法在Java编程中有许多应用场景，包括：

- **定义共同行为**：抽象类可以定义一组共同行为，所有子类共享这些行为。
- **代码复用**：抽象类可以包含具体方法，子类可以复用这些方法。
- **模板设计模式**：抽象类可以用来定义算法的骨架，而具体实现由子类提供。这种设计模式称为模板方法模式（Template Method Pattern）。
- **接口与实现分离**：抽象类可以作为接口的替代方案，提供一些默认实现，同时要求子类实现特定的方法。

通过使用抽象类和抽象方法，Java开发者可以创建更灵活和可维护的代码结构，实现代码的复用和扩展。

## Generics and Collections

Java的泛型（Generics）和集合（Collections）是Java编程语言中两个重要的概念，广泛用于编写类型安全和灵活的代码。

### 泛型（Generics）

**泛型**是一种编程语言的特性，允许在定义类、接口和方法时使用类型参数。它们使得代码可以用于不同的数据类型，而不必重新编写代码，同时提供了编译时类型检查，提高了代码的安全性和可维护性。

#### 泛型的优点

1. **类型安全**：在编译时检查类型，防止类型转换错误。
2. **代码重用**：编写一次代码，可以用于不同的数据类型。
3. **提高代码的可读性和可维护性**：代码更简洁、明确。

#### 泛型的使用

1. **泛型类**：

```java
public class Box<T> {
    private T t;

    public void set(T t) {
        this.t = t;
    }

    public T get() {
        return t;
    }

    public static void main(String[] args) {
        Box<Integer> integerBox = new Box<>();
        integerBox.set(10);
        System.out.println("Integer Value: " + integerBox.get());

        Box<String> stringBox = new Box<>();
        stringBox.set("Hello Generics");
        System.out.println("String Value: " + stringBox.get());
    }
}
```

2. **泛型方法**：

```java
public class GenericsExample {
    public static <T> void printArray(T[] array) {
        for (T element : array) {
            System.out.println(element);
        }
    }

    public static void main(String[] args) {
        Integer[] intArray = {1, 2, 3, 4, 5};
        String[] strArray = {"A", "B", "C", "D", "E"};

        printArray(intArray);
        printArray(strArray);
    }
}
```

3. **有界类型参数**：

```java
public class GenericsExample {
    public static <T extends Number> void printNumber(T number) {
        System.out.println("Number: " + number);
    }

    public static void main(String[] args) {
        printNumber(10); // Integer
        printNumber(10.5); // Double
        // printNumber("10"); // 编译错误，String不是Number的子类
    }
}
```

### 集合（Collections）

**集合框架**（Collections Framework）是Java提供的一组类和接口，用于存储和操作数据集合。它们提供了对数据结构的高效操作，包括列表（List）、集合（Set）、队列（Queue）和映射（Map）。

#### 集合框架的主要接口

1. **List**：有序集合，允许重复元素。常用实现类有`ArrayList`、`LinkedList`。

```java
List<String> list = new ArrayList<>();
list.add("A");
list.add("B");
list.add("C");
for (String s : list) {
    System.out.println(s);
}
```

2. **Set**：不允许重复元素的集合。常用实现类有`HashSet`、`LinkedHashSet`、`TreeSet`。

```java
Set<String> set = new HashSet<>();
set.add("A");
set.add("B");
set.add("A"); // 重复元素不会被添加
for (String s : set) {
    System.out.println(s);
}
```

3. **Queue**：先进先出的集合。常用实现类有`LinkedList`、`PriorityQueue`。

```java
Queue<String> queue = new LinkedList<>();
queue.add("A");
queue.add("B");
queue.add("C");
System.out.println(queue.poll()); // A
System.out.println(queue.poll()); // B
```

4. **Map**：键值对映射，不允许重复键。常用实现类有`HashMap`、`LinkedHashMap`、`TreeMap`。

```java
Map<String, Integer> map = new HashMap<>();
map.put("A", 1);
map.put("B", 2);
map.put("A", 3); // 键"A"的值被更新
for (Map.Entry<String, Integer> entry : map.entrySet()) {
    System.out.println(entry.getKey() + ": " + entry.getValue());
}
```

#### 集合工具类

**Collections**类提供了许多静态方法，用于操作或返回集合。常用方法有：

- `sort(List<T> list)`: 对列表进行排序。
- `shuffle(List<?> list)`: 对列表进行随机排序。
- `reverse(List<?> list)`: 反转列表中的元素顺序。

```java
List<Integer> numbers = new ArrayList<>(Arrays.asList(1, 2, 3, 4, 5));
Collections.shuffle(numbers);
System.out.println("Shuffled List: " + numbers);

Collections.sort(numbers);
System.out.println("Sorted List: " + numbers);

Collections.reverse(numbers);
System.out.println("Reversed List: " + numbers);
```

### 总结和廖雪峰老师网站

[廖雪峰老师Java学习网站的解释](https://www.liaoxuefeng.com/wiki/1252599548343744/1265102638843296)

- **泛型**：通过类型参数提高代码的类型安全性和复用性。
- **集合**：提供了一组类和接口，用于高效地存储和操作数据集合。

通过使用泛型和集合，Java开发者可以编写更灵活、高效和易于维护的代码。

## Comparable and Comparator

在Java中，`Comparable` 和 `Comparator` 是用于对象比较和排序的两个重要接口。

### Comparable 接口

`Comparable` 接口是Java类库提供的一种内部比较器接口，它允许实现了该接口的类的对象自行进行比较。实现了 `Comparable` 接口的类必须实现 `compareTo()` 方法，该方法返回一个整数值，用于表示对象的顺序关系。

- **方法签名**：

  ```java
  public interface Comparable<T> {
      int compareTo(T o);
  }
  ```

- **用途**：

  - 当一个类实现了 `Comparable` 接口后，它的对象可以通过 `Collections.sort()` 方法进行排序。
  - 实现了 `Comparable` 接口的类的对象可以作为元素存储在 `SortedSet` 和 `SortedMap` 的实现类中。

- **示例**：

  ```java
  public class Person implements Comparable<Person> {
      private String name;
      private int age;

      // 构造函数、getter和setter等省略

      @Override
      public int compareTo(Person otherPerson) {
          // 比较逻辑，根据需要定义比较的方式
          return this.age - otherPerson.age;
      }
  }
  ```

### Comparator 接口

`Comparator` 接口是一个外部比较器接口，它允许创建独立的比较器实现来进行对象的比较。`Comparator` 接口不会影响类的实现，可以在需要时创建多个不同的比较规则。

- **方法签名**：

  ```java
  public interface Comparator<T> {
      int compare(T o1, T o2);
  }
  ```

- **用途**：

  - `Comparator` 接口适用于需要在不同情况下使用不同的比较逻辑的场景。
  - 可以使用 `Comparator` 来对类的对象进行排序，而不需要修改类本身或者使用其默认的比较方式。

- **示例**：

  ```java
  public class PersonAgeComparator implements Comparator<Person> {
      @Override
      public int compare(Person p1, Person p2) {
          return p1.getAge() - p2.getAge();
      }
  }
  ```

### 区别和适用场景

- **Comparable vs Comparator**：

  - `Comparable` 接口是对象自身的内部比较方式，类必须实现它来定义对象之间的默认比较规则。
  - `Comparator` 接口是一个独立的比较器，允许定义多种不同的比较规则，并在需要时动态选择和应用这些规则。

- **适用场景**：

  - 使用 `Comparable` 接口当类的自然顺序（默认的排序方式）已经被明确定义时。
  - 使用 `Comparator` 接口当需要定义额外的、非默认的比较规则，或者在不同的场景下使用不同的比较方式时。

总结来说，`Comparable` 和 `Comparator` 是Java中用于对象比较和排序的两种不同方式，每种方式都有其适用的场景和优势，开发者可以根据具体需求选择合适的接口来实现对象的比较和排序功能。

## 异常处理 Exception

Java的异常处理机制旨在提高代码的健壮性和可维护性，通过捕获和处理异常来防止程序在运行时崩溃。Java使用`try`, `catch`, `finally` 和 `throw` 关键字来实现异常处理。

### 异常的分类

Java中的异常主要分为两大类：

1. **受检异常（Checked Exception）**：
   - 这些是由编译器强制检查的异常，必须被捕获或在方法签名中声明。
   - 例如：`IOException`, `SQLException`

2. **非受检异常（Unchecked Exception）**：
   - 这些是由程序错误引起的异常，编译器不强制检查。
   - 包括运行时异常（Runtime Exception）和错误（Error）。
   - 例如：`NullPointerException`, `ArrayIndexOutOfBoundsException`

### 异常处理机制

1. **try-catch**：

   用于捕获和处理异常。在`try`块中放置可能会抛出异常的代码，在`catch`块中处理该异常。

   ```java
   try {
       // 可能抛出异常的代码
   } catch (ExceptionType e) {
       // 处理异常的代码
   }
   ```

2. **finally**：

   `finally`块用于执行一些重要的清理代码，不论是否抛出异常，该块中的代码都会执行。

   ```java
   try {
       // 可能抛出异常的代码
   } catch (ExceptionType e) {
       // 处理异常的代码
   } finally {
       // 总是执行的代码
   }
   ```

3. **throw**：

   用于显式抛出异常对象。

   ```java
   public void someMethod() throws Exception {
       if (someCondition) {
           throw new Exception("错误信息");
       }
   }
   ```

4. **throws**：

   用于在方法签名中声明该方法可能抛出的异常，提醒调用者处理这些异常。

   ```java
   public void someMethod() throws IOException {
       // 可能抛出IOException的代码
   }
   ```

### 示例代码

以下是一个简单的例子，展示了如何使用`try`, `catch`, `finally`和`throw`来处理异常：

```java
import java.io.*;

public class ExceptionExample {
    public static void main(String[] args) {
        try {
            // 可能抛出IOException的代码
            readFile("test.txt");
        } catch (IOException e) {
            // 处理IOException
            System.out.println("An error occurred: " + e.getMessage());
        } finally {
            // 总是执行的代码
            System.out.println("Execution finished.");
        }
    }

    public static void readFile(String fileName) throws IOException {
        FileReader file = new FileReader(fileName);
        BufferedReader fileInput = new BufferedReader(file);

        // 打印文件内容
        for (int counter = 0; counter < 3; counter++) {
            System.out.println(fileInput.readLine());
        }
        fileInput.close();
    }
}
```

### 自定义异常

Java允许开发者创建自定义异常类，通常用于特定业务逻辑中的错误处理。自定义异常需要继承`Exception`或`RuntimeException`。

```java
public class CustomException extends Exception {
    public CustomException(String message) {
        super(message);
    }
}

public class TestCustomException {
    public static void main(String[] args) {
        try {
            validateAge(15);
        } catch (CustomException e) {
            System.out.println("Caught custom exception: " + e.getMessage());
        }
    }

    static void validateAge(int age) throws CustomException {
        if (age < 18) {
            throw new CustomException("Age must be 18 or above.");
        }
    }
}
```

## `try-with-resources`

`try-with-resources` 是Java 7引入的一种资源管理机制，用于简化资源（如文件、数据库连接等）关闭的代码。它确保了任何实现了 `AutoCloseable` 接口的资源在使用完后会自动关闭，从而减少资源泄漏的风险。

### 语法和用法

`try-with-resources` 的基本语法如下：

```java
try (ResourceType resource = new ResourceType()) {
    // 使用资源的代码
} catch (ExceptionType e) {
    // 处理异常的代码
}
```

其中，`ResourceType` 必须是实现了 `AutoCloseable` 或 `Closeable` 接口的类。

### 工作原理

当`try-with-resources`语句结束时，无论是否抛出异常，都会自动调用资源的 `close()` 方法。这确保了资源的正确释放。

### 示例

下面是一个使用 `try-with-resources` 读取文件的例子：

```java
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

public class TryWithResourcesExample {
    public static void main(String[] args) {
        try (BufferedReader br = new BufferedReader(new FileReader("test.txt"))) {
            String line;
            while ((line = br.readLine()) != null) {
                System.out.println(line);
            }
        } catch (IOException e) {
            System.out.println("An error occurred: " + e.getMessage());
        }
    }
}
```

在这个例子中：

- `BufferedReader` 和 `FileReader` 都实现了 `AutoCloseable` 接口。
- 当`try`块结束时，不论是否发生异常，`BufferedReader` 和 `FileReader` 的 `close()` 方法都会被自动调用。

### 多个资源

`try-with-resources` 语句可以管理多个资源，多个资源之间用分号分隔：

```java
try (BufferedReader br = new BufferedReader(new FileReader("test.txt"));
     PrintWriter pw = new PrintWriter(new FileWriter("output.txt"))) {
    String line;
    while ((line = br.readLine()) != null) {
        pw.println(line);
    }
} catch (IOException e) {
    System.out.println("An error occurred: " + e.getMessage());
}
```

在这个例子中，`BufferedReader` 和 `PrintWriter` 都会在`try`块结束时自动关闭。

### 自定义资源类

任何实现了 `AutoCloseable` 接口的类都可以使用 `try-with-resources` 语句。以下是一个自定义资源类的例子：

```java
public class CustomResource implements AutoCloseable {
    public void useResource() {
        System.out.println("Using resource");
    }

    @Override
    public void close() {
        System.out.println("Closing resource");
    }
}

public class TryWithResourcesCustomExample {
    public static void main(String[] args) {
        try (CustomResource resource = new CustomResource()) {
            resource.useResource();
        }
    }
}
```

在这个例子中，当`try`块结束时，`CustomResource` 的 `close()` 方法会被自动调用。

### 优点

- **简洁**：减少了显式的资源关闭代码，使代码更简洁和易读。
- **安全**：确保资源被正确关闭，减少资源泄漏的风险。
- **简化异常处理**：自动处理在关闭资源时可能抛出的异常。

### 总结

`try-with-resources` 是Java中处理资源管理的一种简洁高效的方式。通过自动管理资源的关闭，它不仅简化了代码，还提高了程序的安全性和可靠性。

## 文件读写（字符和字节）

在Java中，文件读写操作主要通过`java.io`包中的类来实现。以下是一些常用类和方法，分别用于读取和写入文件。

### 文件读取

#### 使用 `FileReader` 和 `BufferedReader`
`FileReader` 类用于读取字符文件。`BufferedReader` 提供缓冲读取，提高了读取效率。

```java
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

public class FileReadExample {
    public static void main(String[] args) {
        try (BufferedReader br = new BufferedReader(new FileReader("example.txt"))) {
            String line;
            while ((line = br.readLine()) != null) {
                System.out.println(line);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
```

#### 使用 `FileInputStream` 和 `InputStreamReader`
`FileInputStream` 类用于读取字节文件，通常与 `InputStreamReader` 结合使用，将字节流转换为字符流。

```java
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.io.BufferedReader;
import java.io.IOException;

public class FileReadExample2 {
    public static void main(String[] args) {
        try (BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream("example.txt")))) {
            String line;
            while ((line = br.readLine()) != null) {
                System.out.println(line);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
```

### 文件写入

#### 使用 `FileWriter` 和 `BufferedWriter`
`FileWriter` 类用于写入字符文件。`BufferedWriter` 提供缓冲写入，提高了写入效率。

```java
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;

public class FileWriteExample {
    public static void main(String[] args) {
        try (BufferedWriter bw = new BufferedWriter(new FileWriter("example.txt"))) {
            bw.write("Hello, World!");
            bw.newLine();
            bw.write("Java File Writing Example.");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
```

#### 使用 `FileOutputStream`
`FileOutputStream` 类用于写入字节文件。

```java
import java.io.FileOutputStream;
import java.io.IOException;

public class FileWriteExample2 {
    public static void main(String[] args) {
        try (FileOutputStream fos = new FileOutputStream("example.txt")) {
            String content = "Hello, World!\nJava File Writing Example.";
            fos.write(content.getBytes());
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
```

### 使用 NIO 包中的类
Java NIO（New IO）提供了更加高效的文件操作方法。

#### 文件读取
```java
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;
import java.io.IOException;

public class FileReadNIOExample {
    public static void main(String[] args) {
        try {
            List<String> lines = Files.readAllLines(Paths.get("example.txt"));
            for (String line : lines) {
                System.out.println(line);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
```

#### 文件写入
```java
import java.nio.file.Files;
import java.nio.file.Paths;
import java.io.IOException;
import java.nio.charset.StandardCharsets;

public class FileWriteNIOExample {
    public static void main(String[] args) {
        String content = "Hello, World!\nJava NIO File Writing Example.";
        try {
            Files.write(Paths.get("example.txt"), content.getBytes(StandardCharsets.UTF_8));
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
```

### 总结
- `FileReader` 和 `BufferedReader` 适合读取字符文件。
- `FileInputStream` 和 `InputStreamReader` 适合读取字节文件，并可以将字节流转换为字符流。
- `FileWriter` 和 `BufferedWriter` 适合写入字符文件。
- `FileOutputStream` 适合写入字节文件。
- Java NIO 提供了更加高效和简便的文件读写方法。

在选择使用哪个类时，可以根据具体需求和文件类型来决定。

## Java的Lambda表达式和Functional接口

### Lambda表达式

**定义**：Lambda表达式是Java 8引入的一种简洁的方式，用于实现匿名函数。它可以使代码更加简洁、可读性更强，特别是在需要使用短小的代码段来实现接口方法时。

**语法**：
```java
(parameters) -> expression
或
(parameters) -> { statements; }
```

**示例**：
```java
// 使用Lambda表达式实现Runnable接口
Runnable r = () -> System.out.println("Hello, Lambda!");
r.run();
```

### Functional接口

**定义**：Functional接口是指仅包含一个抽象方法的接口。这种接口可以隐式地转换为Lambda表达式。Java 8引入了`@FunctionalInterface`注解，用于显式地声明一个接口为Functional接口，但这不是强制的，任何满足条件的接口都可以作为Functional接口。

**示例**：
```java
@FunctionalInterface
interface MyFunctionalInterface {
    void myMethod();
}
```

**常见的Functional接口**：
- `java.lang.Runnable`：只有一个`run`方法。
- `java.util.concurrent.Callable`：只有一个`call`方法。
- `java.util.function.Predicate`：只有一个`test`方法，用于条件判断。
- `java.util.function.Function`：只有一个`apply`方法，用于将一个值转换为另一个值。
- `java.util.function.Consumer`：只有一个`accept`方法，用于处理一个输入值而不返回结果。
- `java.util.function.Supplier`：只有一个`get`方法，用于提供一个值。

Java的Functional接口提供了许多通用的接口类型，每个接口在不同的场景中使用，用于解决特定类型的问题。以下是常见的Functional接口及其典型使用场景：

#### 1. `Runnable` 接口
**使用场景**：用于在新线程中执行代码块。

**示例**：
```java
Runnable task = () -> System.out.println("Running in a separate thread");
new Thread(task).start();
```

#### 2. `Callable<V>` 接口
**使用场景**：与`Runnable`类似，但可以返回结果或抛出异常，常用于需要返回结果的异步任务。

**示例**：
```java
import java.util.concurrent.Callable;

Callable<Integer> task = () -> {
    return 123;
};
```

#### 3. `Predicate<T>` 接口
**使用场景**：用于进行条件判断，返回`true`或`false`。常用于过滤操作。

**示例**：
```java
import java.util.function.Predicate;

Predicate<String> isLongerThan5 = s -> s.length() > 5;
System.out.println(isLongerThan5.test("Hello")); // false
System.out.println(isLongerThan5.test("Hello, World!")); // true
```

**典型应用**：在集合过滤中使用
```java
List<String> names = Arrays.asList("Alice", "Bob", "Charlie");
List<String> longNames = names.stream()
                              .filter(isLongerThan5)
                              .collect(Collectors.toList());
```

#### 4. `Function<T, R>` 接口
**使用场景**：用于将一种类型的数据转换为另一种类型，常用于映射操作。

**示例**：
```java
import java.util.function.Function;

Function<Integer, String> intToString = i -> "Number: " + i;
System.out.println(intToString.apply(5)); // "Number: 5"
```

**典型应用**：在集合映射中使用
```java
List<Integer> numbers = Arrays.asList(1, 2, 3, 4, 5);
List<String> numberStrings = numbers.stream()
                                    .map(intToString)
                                    .collect(Collectors.toList());
```

#### 5. `Consumer<T>` 接口
**使用场景**：用于对单个输入执行某些操作，但不返回结果，常用于遍历操作或打印操作。

**示例**：
```java
import java.util.function.Consumer;

Consumer<String> printUpperCase = s -> System.out.println(s.toUpperCase());
printUpperCase.accept("hello"); // "HELLO"
```

**典型应用**：在集合遍历中使用
```java
List<String> names = Arrays.asList("Alice", "Bob", "Charlie");
names.forEach(printUpperCase);
```

#### 6. `Supplier<T>` 接口
**使用场景**：用于提供或生成一个结果，不接受任何输入，常用于延迟计算或对象实例化。

**示例**：
```java
import java.util.function.Supplier;

Supplier<Double> randomValue = () -> Math.random();
System.out.println(randomValue.get()); // 随机数
```

**典型应用**：延迟加载或默认值提供
```java
public class LazyInitialization {
    private Supplier<HeavyObject> heavyObjectSupplier = () -> createHeavyObject();

    private HeavyObject createHeavyObject() {
        // 创建一个重型对象
        return new HeavyObject();
    }

    public HeavyObject getHeavyObject() {
        return heavyObjectSupplier.get();
    }
}
```

#### 7. `UnaryOperator<T>` 接口
**使用场景**：用于对单个操作数进行操作，并返回与操作数相同类型的结果。它是`Function`的特殊化形式。

**示例**：
```java
import java.util.function.UnaryOperator;

UnaryOperator<Integer> square = x -> x * x;
System.out.println(square.apply(5)); // 25
```

**典型应用**：在集合操作中使用
```java
List<Integer> numbers = Arrays.asList(1, 2, 3, 4, 5);
List<Integer> squaredNumbers = numbers.stream()
                                      .map(square)
                                      .collect(Collectors.toList());
```

#### 8. `BinaryOperator<T>` 接口
**使用场景**：用于对两个操作数进行操作，并返回与操作数相同类型的结果。它是`BiFunction`的特殊化形式。

**示例**：
```java
import java.util.function.BinaryOperator;

BinaryOperator<Integer> sum = (a, b) -> a + b;
System.out.println(sum.apply(2, 3)); // 5
```

**典型应用**：在归约操作中使用
```java
List<Integer> numbers = Arrays.asList(1, 2, 3, 4, 5);
int total = numbers.stream()
                   .reduce(0, sum);
System.out.println(total); // 15
```

#### 9. `BiFunction<T, U, R>` 接口
**使用场景**：用于将两个输入类型的数据转换为另一种类型，常用于需要两个输入参数的情况。

**示例**：
```java
import java.util.function.BiFunction;

BiFunction<String, String, Integer> compareLengths = (a, b) -> a.length() - b.length();
System.out.println(compareLengths.apply("hello", "world!")); // -1
```

Java 的 Functional 接口广泛应用于各种场景中，特别是在集合操作、并发处理、条件判断、数据转换和遍历操作中。Lambda 表达式与 Functional 接口的结合，使得代码更加简洁、易读和可维护。在实际开发中，根据需求选择适当的 Functional 接口，可以大大提高代码的效率和质量。

### 总结

- **Lambda表达式** 提供了一种简洁的方式来实现Functional接口的方法，使代码更加简洁和易读。**它是Functional接口的一个实例**
- **Functional接口** 是*只包含一个*抽象方法的接口，Java 8提供了许多预定义的Functional接口，使得Lambda表达式的使用更加广泛和灵活。
- 通过结合Lambda表达式和Functional接口，可以显著减少代码的冗余，提高代码的可维护性和可读性。

## Stream API

在Java中，Lambda表达式和方法引用（Method Reference）都是简化代码的一种方式。方法引用是Lambda表达式的一种简写形式，用于直接引用已有的方法，而不需要显式地写出Lambda表达式。

Java的Stream API是从Java 8引入的一个新特性，提供了一种高效且易于使用的方式来处理集合（如List、Set、Map等）中的数据。Stream API允许你以声明性编程的方式进行集合的操作，比如过滤、映射、归约等。

### 核心概念

1. **Stream**：不是数据结构，而是从支持数据处理操作的源生成的元素序列。可以通过集合、数组、生成器等获取Stream。
2. **中间操作**：返回Stream本身，因此可以链式调用多个操作。常见的中间操作有`filter`、`map`、`flatMap`、`sorted`、`distinct`等。
3. **终端操作**：触发Stream操作并生成结果，如`forEach`、`collect`、`reduce`、`count`等。终端操作后Stream不再可用。

### 创建Stream

Stream可以通过多种方式创建：

1. **从集合**：
   ```java
   List<String> list = Arrays.asList("a", "b", "c");
   Stream<String> stream = list.stream();
   ```

2. **从数组**：
   ```java
   String[] array = {"a", "b", "c"};
   Stream<String> stream = Arrays.stream(array);
   ```

3. **从值**：
   ```java
   Stream<String> stream = Stream.of("a", "b", "c");
   ```

4. **从生成器**：
   ```java
   Stream<Double> stream = Stream.generate(Math::random).limit(10);
   ```

5. **从迭代器**：
   ```java
   Stream<Integer> infiniteStream = Stream.iterate(0, n -> n + 1).limit(10);
   ```

### 中间操作示例

1. **filter**：过滤符合条件的元素
   ```java
   List<Integer> numbers = Arrays.asList(1, 2, 3, 4, 5);
   List<Integer> evenNumbers = numbers.stream()
                                      .filter(n -> n % 2 == 0)
                                      .collect(Collectors.toList());
   ```

2. **map**：将元素转换为另一种形式
   ```java
   List<String> strings = Arrays.asList("a", "b", "c");
   List<String> upperStrings = strings.stream()
                                      .map(String::toUpperCase)
                                      .collect(Collectors.toList());
   ```

3. **sorted**：排序
   ```java
   List<String> strings = Arrays.asList("d", "a", "c", "b");
   List<String> sortedStrings = strings.stream()
                                       .sorted()
                                       .collect(Collectors.toList());
   ```

4. **distinct**：去重
   ```java
   List<Integer> numbers = Arrays.asList(1, 2, 2, 3, 3, 3);
   List<Integer> uniqueNumbers = numbers.stream()
                                        .distinct()
                                        .collect(Collectors.toList());
   ```

### 终端操作示例

终端操作符后，才被执行。

1. **forEach**：对每个元素执行操作
   ```java
   List<String> list = Arrays.asList("a", "b", "c");
   list.stream().forEach(System.out::println);
   ```

2. **collect**：将Stream转换为其他形式
   ```java
   List<String> list = Arrays.asList("a", "b", "c");
   List<String> upperList = list.stream()
                                .map(String::toUpperCase)
                                .collect(Collectors.toList());
   ```

3. **reduce**：将元素组合起来
   ```java
   List<Integer> numbers = Arrays.asList(1, 2, 3, 4, 5);
   int sum = numbers.stream()
                    .reduce(0, Integer::sum);
   ```

4. **count**：计算元素个数
   ```java
   List<String> list = Arrays.asList("a", "b", "c");
   long count = list.stream().count();
   ```

### 并行Stream

Stream API还支持并行处理，可以通过`parallelStream()`方法或`parallel()`方法将Stream转换为并行流，以利用多核处理器的优势提高性能。

```java
List<Integer> numbers = Arrays.asList(1, 2, 3, 4, 5);
int sum = numbers.parallelStream()
                 .reduce(0, Integer::sum);
```

### 示例

以下是一个综合示例，展示了如何使用Stream API对集合进行一系列操作：

```java
List<String> names = Arrays.asList("Alice", "Bob", "Charlie", "David", "Edward");

List<String> filteredAndSortedNames = names.stream()
                                           .filter(name -> name.length() > 3)
                                           .sorted()
                                           .map(String::toUpperCase)
                                           .collect(Collectors.toList());

filteredAndSortedNames.forEach(System.out::println);
```

以上代码首先过滤掉长度小于等于3的名字，然后对剩下的名字进行排序，将名字转换为大写，最后收集到一个新的列表中，并打印出来。

## Method Reference 方法引用的四种类型

1. **引用静态方法**：
   - 语法：`ClassName::staticMethodName`
   - 示例：
     ```java
     Function<Integer, String> func = String::valueOf;
     ```

2. **引用实例方法**：
   - 语法：`instance::instanceMethodName`
   - 示例：
     ```java
     String str = "Hello";
     Supplier<Integer> func = str::length;
     ```

3. **引用特定类型实例方法**：
   - 语法：`ClassName::instanceMethodName`
   - 示例：
     ```java
     Function<String, Integer> func = String::length;
     ```

4. **引用构造器**：
   - 语法：`ClassName::new`
   - 示例：
     ```java
     Supplier<List<String>> func = ArrayList::new;
     ```

**Lambda表达式与方法引用的对比**：

1. **Lambda表达式**：
   ```java
   List<String> list = Arrays.asList("a", "b", "c");
   list.forEach(s -> System.out.println(s));
   ```

2. **方法引用**：
   ```java
   List<String> list = Arrays.asList("a", "b", "c");
   list.forEach(System.out::println);
   ```
因为它是一种引用，所以方法必须事先存在。
