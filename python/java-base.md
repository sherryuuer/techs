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
   - 基本类型的值直接存储在栈内存中，分配固定大小的内存空间。
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
   │   ├── main/
   │   │   └── MainAccount.class
   │   └── bank/
   │       └── BankAccount.class
   ├── src/
   │   ├── main/
   │   │   └── MainAccount.java
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

是静态方法的意思，如果一个函数有static，那么可以在类中直接使用该方法，如果没有static关键字，则需要对类进行实例化。

Python中也有staticmethod修饰符。表示就是可以直接靠类使用的方法。

## extends

继承其他的class，对于变量可以直接使用，对于方法需要是public/protected属性。

## 构造方法（constructor）

构造方法重载指的是在一个类中可以定义多个构造方法，它们具有相同的名字（类的名字），但参数列表不同（参数的类型、数量或顺序不同）。通过这种方式，可以根据不同的需求初始化对象，提供灵活性和方便性。

相当于Python中的初始化函数，但是Python的初始化函数不能有多个，只能靠类方法等实现。
