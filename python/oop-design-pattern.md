[代码地址](https://github.com/sherryuuer/cheatsheets/tree/main/core_code)

包含Java和Python写法。

## Creational Patterns

### Factory Method Pattern

工厂方法模式(Factory Method Pattern)。

工厂方法模式是一种创建型设计模式，它提供了一种创建对象的最佳方式。

在代码的例子中，它定义了一个抽象类Vehicle和三个具体的子类Car、Bike和Truck。同时还定义了一个抽象工厂类VehicleFactory和三个具体的工厂类CarFactory、BikeFactory和TruckFactory。

通过不同的工厂类，可以创建出不同类型的Vehicle对象。每个具体的工厂类都实现了createVehicle()方法，用于创建对应类型的Vehicle对象。
最后，代码展示了如何使用这些工厂类来创建不同类型的Vehicle对象，并调用它们的getType()方法来获取对象类型。

`@abstractmethod`是Python中用于定义抽象方法的装饰器。它来自于abc(Abstract Base Class)模块，用于标记某个方法为抽象方法。
抽象方法是指在基类中声明的方法，但没有实现具体的功能，只定义了方法签名。抽象方法必须在子类中**被重写(override)**，否则在实例化子类时将引发TypeError异常。

### Singleton Pattern

是一种创建型设计模式，它确保一个类只有一个实例，并提供一个全局访问点。主要解决的问题是保证一个类只有一个实例，并提供一个全局访问点。

Singleton模式的优点是:

- 对于需要频繁创建和销毁的对象可以提高性能。
- 由于单例模式中只存在一个实例，所以可以节省内存。
- 避免对资源的多重占用。
- 可以设置全局访问点，优化和共享资源访问。

但是你要是改变心意了咋办，这不是一个好的设计方法，所以这个模式经常被当成反面教材。

Singleton模式在现实开发中有广泛的应用，例如*线程池、缓存、日志对象、配置对象*等都可以使用单例模式进行设计。

该模式确保了，无论调用多少次实例，总是返回同一个实例。

在Python版本中，使用__new__方法来控制实例的创建。__new__是在创建实例时由Python解释器自动调用的方法。通过重写__new__方法，我们可以做到只创建一个实例，并将其存储在_instance类属性中。
当第一次调用Singleton()时，_instance为None，会创建一个新的实例并赋值给_instance。后续调用则直接返回已创建的实例。这样就实现了单例模式。

### Builder Pattern

生成器模式是一种创建型设计模式，旨在简化复杂对象的创建过程。它通过将对象的构建过程与其表示分离，使得相同的构建过程可以创建不同的表示。Builder Pattern 对于那些具有多个可选参数或构造步骤的对象特别有用。

Builder Pattern 通常包括以下几个部分：

- Builder（生成器）：定义构建对象的步骤，并且提供方法来设置对象的各个部分。
- ConcreteBuilder（具体生成器）：实现 Builder 接口，构建并装配各个部分。
- Product（产品）：要创建的复杂对象。
- Director（指挥者）：使用 Builder 对象构建 Product 的各个部分。Director 知道如何使用 Builder 来构建对象。

Builder Pattern（生成器模式）是一种创建型设计模式，旨在简化复杂对象的创建过程。它通过将对象的构建过程与其表示分离，使得相同的构建过程可以创建不同的表示。Builder Pattern 对于那些具有多个可选参数或构造步骤的对象特别有用。

```java
// 使用示例
public class BuilderPatternExample {
    public static void main(String[] args) {
        Computer computer = new Computer.Builder("Intel i7", "16GB")
                                .setUSBPorts(4)
                                .setHasGraphicsCard(true)
                                .build();
        System.out.println(computer);
    }
}
```

Builder Pattern 是一种创建复杂对象的有效方法，特别适用于那些具有多个可选参数的对象。通过将对象的构建过程与其表示分离，Builder Pattern 提高了代码的可读性和可维护性，同时增加了对象创建的灵活性。

### Prototype Pattern

Prototype（原型）是原型范式（Prototype Paradigm）的一部分，主要与基于原型的编程（Prototype-based Programming）有关。这种编程范式特别常见于面向对象编程（OOP）的一种实现方式，与传统的基于类的面向对象编程不同。

1. **对象而非类**:
    - 在原型范式中，编程的基本单位是对象而不是类。对象是具体实例，而类在这种范式中并不显式存在。

2. **对象复制**:
    - 新的对象通过复制现有的对象（即原型）来创建，而不是通过类的实例化。这种方式允许灵活地创建和修改对象。

3. **动态行为修改**:
    - 对象可以动态地修改其结构和行为。你可以在运行时向对象添加属性和方法，或改变它们的属性和方法。

4. **继承机制**:
    - 继承是通过委托（delegation）机制实现的。一个对象可以指向另一个对象作为其原型，从而继承其属性和方法。

原型范式的一个典型示例是 JavaScript。在 JavaScript 中，每个对象都有一个内部链接指向另一个对象（即其原型）。当试图访问一个对象的属性时，如果该对象自身没有这个属性，JavaScript 会沿着原型链向上查找，直到找到该属性或者到达原型链的末端。

```javascript
// 创建一个原型对象
let person = {
    type: 'human',
    sayHello: function() {
        console.log(`Hello, I am a ${this.type}`);
    }
};

// 通过复制原型对象创建一个新对象
let john = Object.create(person);
john.type = 'developer'; // 修改新对象的属性
john.sayHello(); // 输出: Hello, I am a developer

// 创建另一个对象
let jane = Object.create(person);
jane.type = 'designer';
jane.sayHello(); // 输出: Hello, I am a designer
```

在这个例子中，`person` 是一个原型对象。`john` 和 `jane` 是通过 `Object.create(person)` 创建的新对象，并继承了 `person` 的属性和方法。


- **Prototype（原型）** 是 **原型范式**（Prototype Paradigm）的核心概念。
- 原型范式强调对象的直接使用和**深度复制**，而不是依赖类的定义和实例化。
- 通过对象的复制和动态修改，原型范式提供了灵活的对象创建和继承机制。

原型范式提供了一种不同于基于类的面向对象编程的方法，更加灵活和动态，适合某些特定的编程需求和环境。
