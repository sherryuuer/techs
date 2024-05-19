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
