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
