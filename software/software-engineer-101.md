- 涉及的是一个工程问题，不是一个小的代码，而是一种大规模开发。包括团队组织，代码开发组织，未来易于维护性等
- Bad Code和错误的开发实践（development practices），导致金钱浪费和项目失败
- 最初的设计，非常重要
- 从**范畴**大小来说：
  - 最里层的是*具体的开发，实践，设计模式，代码编程，构架，运维等技术方面*
  - 然后是统筹这些的*SDLC*框架，用合适的框架进行开发
  - 最终支持整个交付的是*项目管理PM*

## Software Lifecycle

- **软件开发生命周期（SDLC, Software Development Life Cycle）**是一个系统化的方法论，用于定义软件项目的开发过程。它包括从概念到交付和维护的各个阶段，帮助开发团队在项目中保持一致性和质量
  - 关注*技术和方法*
  - 最终目标是*交付高质量的软件产品*
- **项目管理（PM, Project Management）**是应用知识、技能、工具和技术来执行项目活动，以满足项目要求。项目管理关注的是项目的总体成功，包括资源的有效利用、风险管理和项目交付的按时完成
  - *范围，时间，成本，质量，沟通*
  - 关注*项目整体的管理和交付*，支持*SDLC*的执行
  - 目标是*成功交付项目，满足时间、成本和质量要求*

- Requirements = what
- Design = how
- Implementation = build
- Verification = test(do they meet the requirements)
- Maintenane = new cycle(debug, new features)
- [Reference](https://www.tutorialspoint.com/sdlc/sdlc_overview.htm)

## Requirements / WRSPM

- **Requirements**是一种谁都可以懂的东西，客户要达到何种目的和功能，是一种**用户user**的要求
- **Specifications**则是一种规格说明，不会用太多的专业词汇，用于指导之后*开发者*的设计，是一种**系统system**的要求

- **Functional vs Non-Functional**：功能性是系统能做什么，非功能性要求则是基于产品，组织，外部的一些基准的要求，必须加密用什么方式之类的

- **WRSPM**：
  - *Environment*：
  - world：周围的所有环境需求，电力，网络，温度
  - requirements：各方用户所需的功能要求
  - *Interface*：
  - specifications：规格要求，芯片技术，加密方式，传输协议，付款方式
  - *System*：
  - program：编程需求
  - machine：硬件需求

  - 可以看出他们是对应的关系，现实的世界**world**和机器的世界**machine**，现实的需求**requirements**对应**program**程序代码的实现，中间的接口是**specifications**

- **WRSPM Variables**:
  - *Environment*：
  - Eh：Elements of the environment which are hidden from the system，必须一个ATM机器来说，你的银行卡就是这样的元素
  - *Interface*：
  - Ev：Elements of the environment which are visible to the system，比如银行卡的芯片，通过这个芯片的内容，系统才可以进行交互，什么元素，可以让系统识别到世界中的内容，比如机器学习中，对文字来说，机器不可识别，但是可以识别数字，所以需要embedding
  - Sv：Elements of the system which are visible to the environment，系统中对环境来说也是可见的东西，比如按钮，屏幕，UI等
  - *System*：
  - Sh：Elements of the system which are hidden from the environment，系统中对环境来说不可见的东西，比如你的网络应用的后端，对用户是不可见的

- **WRSPM：从左到右就是环境，接口，和系统的顺序**，called *world-machine model*

## Design：Architecture

- 高层次的设计
- 想法是否可以落实到现实的蓝图计划
- *错误的构架无法靠代码弥补*，因为它设计基础的地方，当工程开始，基础很难改变
- 一个好的构架师，一个好的PM，都应该首先是一个优秀的技术者，从细节到上层全面把握

- 系统构架和软件构架是两件事，这里谈*软件构架*
- **软件构架都是关于，如何将一个大的系统，分解成小的子系统和其中的模块**
- *解藕*和*交互API*非常重要
- 好的构架很难，错误的构架在之后很难修正，有时候你甚至需要重建
- 好的构架在之后很容易维护和进化

- 好的设计构架都会有好的doc，作为使用接口的人，**阅读文档**太重要了

- 构架例子：
  - 前端 - 逻辑 - 服务器（贪吃蛇游戏如果做成一个应用服务，是如何的）
  - 一个系统都有什么*组件*，哪些组件之间需要*交互*，哪些不需要，如何交互
- architecture of the Online Learning Platform:
```mermaid
graph LR;

    subgraph UI[User Interface]
        A1[Web Application]
        A2[Mobile Application]
    end

    subgraph UMS[User Management System]
        B1[Authentication]
        B2[Profile Management]
        B3[Notifications]
    end

    subgraph CMS[Course Management System]
        C1[Course Creation]
        C2[Content Management]
        C3[Course Enrollment]
    end

    subgraph CDN[Content Delivery Network]
        D1[Video Streaming]
        D2[File Hosting]
        D3[Load Balancing]
    end

    subgraph AES[Assessment & Evaluation System]
        E1[Quiz Module]
        E2[Automated Grading]
        E3[Peer Review]
    end

    subgraph DCP[Discussion & Communication Platform]
        F1[Forums]
        F2[Live Chat]
        F3[Webinars]
    end

    subgraph AR[Analytics & Reporting]
        G1[Progress Tracking]
        G2[Engagement Metrics]
        G3[Dashboard]
    end

    subgraph PSS[Payment & Subscription System]
        H1[Payment Gateway]
        H2[Subscription Management]
        H3[Coupons]
    end

    subgraph BI[Backend Infrastructure]
        I1[Database]
        I2[API Services]
        I3[Microservices]
    end

    subgraph SC[Security & Compliance]
        J1[Data Encryption]
        J2[Compliance]
        J3[Audits]
    end

    UI --> UMS
    UI --> CMS
    UI --> CDN
    UI --> AES
    UI --> DCP
    UI --> AR
    UI --> PSS
    UI --> SC

    UMS --> BI
    CMS --> BI
    CDN --> BI
    AES --> BI
    DCP --> BI
    AR --> BI
    PSS --> BI
    SC --> BI
```

### Pipe and Filter

**管道和过滤器（Pipe and Filter）**是在软件架构和设计模式中，一种非常常见的设计模式。它被用来处理数据流，并将复杂的处理过程分解为一系列简单的步骤。

- *管道（Pipe）*：在这个模式中，管道用于连接一系列的过滤器。管道负责传递数据，从一个过滤器的输出端到下一个过滤器的输入端，类似于管道中水的流动。
- *过滤器（Filter）*：过滤器是独立的处理组件，每个过滤器接收数据流，处理数据，然后将其输出传递给下一个过滤器。
  - 独立性：每个过滤器作为一个独立的单元，可以被重用和独立开发。
  - 数据处理：每个过滤器负责执行特定的数据处理任务，例如转换、过滤、排序等。
  - 无状态性：大多数过滤器是无状态的，这意味着它们不会存储处理过的数据，处理过程不依赖于之前的数据状态。
- 管道和过滤器的工作流程：
  - 数据输入：初始数据被输入到第一个过滤器中。
  - 数据处理：数据流经第一个过滤器进行处理，处理后的数据通过管道传递到下一个过滤器。
  - 逐步处理：这一过程在每个过滤器中重复，直到数据流通过所有的过滤器。
  - 数据输出：最终处理完成的数据被输出或存储。

- 现实案例，比如shell的管道符号｜，大数据处理的ETL，编排服务的数据流等，Airflow编排服务也是这种软件构架也许

### Client-Server

**Client-Server**（客户端-服务器）是一种经典的软件设计模型。它是*分布式计算架构*的一种，其中任务和负载被分配到两个主要的组件：客户端和服务器。这种模型被广泛用于各种网络应用程序和系统设计中。

1. **请求-响应循环**：
  - 客户端向服务器发送请求，服务器接收到请求后进行处理，然后将结果返回给客户端。

2. **网络通信**：
  - 客户端和服务器通过网络进行通信，常用的协议有HTTP、HTTPS、FTP、TCP/IP等。

3. **异步处理**：
  - 在许多现代应用中，客户端可以异步地发送请求，而无需等待服务器的响应才能继续进行其他操作。

Client-Server 的应用示例
- Web 应用：浏览器作为客户端，Web 服务器（如Apache、Nginx）作为服务器。
- 数据库系统：应用程序作为客户端，数据库服务器（如MySQL、PostgreSQL）作为服务器。
- 电子邮件系统：邮件客户端（如Outlook、Gmail）和邮件服务器（如SMTP服务器）。

Client-Server 与其他模型的对比
- **对等网络（Peer-to-Peer，P2P）**：
  - 在P2P网络中，每个节点既可以是客户端，也可以是服务器。这种架构更适合于资源共享和分布式计算。
- **微服务架构**：
  - 微服务架构将应用程序拆分成多个独立的小服务，每个服务负责特定的功能。这与传统的Client-Server模型不同，因为它强调服务的独立性和去中心化。

### Master-Slave

**Master-Slave 模式**是一种分布式计算架构，广泛应用于数据库、计算任务调度和系统管理中。它将任务和数据在主服务器（Master）和从服务器（Slave）之间进行分配，以提高系统的可用性、性能和容错性。

1. **主从角色**：
   - **主服务器（Master）**：负责管理和协调，从属于控制中心。它分配任务、存储数据，并对系统进行整体控制。
   - **从服务器（Slave）**：接收来自主服务器的任务，并执行指定的操作。它可以将结果返回给主服务器或独立存储数据。

2. **任务分配**：
   - 主服务器分配任务，从服务器执行。通常在计算任务中，主服务器会根据任务的复杂度和从服务器的负载情况进行分配。

3. **数据复制**：
   - 在数据库系统中，主服务器将数据更新推送给从服务器，从而实现数据的冗余和备份。这样可以提高数据读取的效率和系统的容错能力。

4. **容错性和高可用性**：
   - 从服务器的存在提高了系统的容错能力。如果主服务器出现故障，从服务器可以作为备份进行故障切换（Failover）。

5. **读写分离**：
   - 通过将写操作集中在主服务器，而将读操作分散到从服务器，从而提高系统的性能。

Master-Slave 模式的应用示例

1. **数据库系统**：
   - 在 MySQL、PostgreSQL 等数据库中，Master-Slave 模式用于实现数据复制和负载均衡。主数据库负责写操作，从数据库负责读操作。

2. **分布式计算**：
   - 在 Hadoop 中，NameNode 作为主节点，DataNode 作为从节点。NameNode 负责元数据管理，DataNode 负责实际的数据存储。

3. **任务调度系统**：
   - 诸如 Apache Kafka 之类的消息队列中，主节点负责管理主题和分区，从节点负责实际的消息存储和传递。

**优点**:

- 性能优化：通过读写分离和任务分配，提高了系统的整体性能。
- 高可用性：从服务器提供冗余和备份，提高了系统的可靠性。
- 易于扩展：可以通过增加从服务器来扩展系统的处理能力。

**缺点**:

- 单点故障：主服务器故障可能导致整个系统的停滞，需要额外的机制来实现主服务器的故障切换。
- 数据一致性：数据复制延迟可能导致暂时的不一致性，需要仔细设计数据同步机制。

Master-Slave 模式通过角色分配和任务调度，实现了高效的数据处理和系统管理。它在性能优化和高可用性方面表现出色，但也面临单点故障和数据一致性等挑战。通过合理的设计和实施，Master-Slave 模式能够为复杂的分布式系统提供强大的支持。

想起了k8s和Airflow的scheduler

### Layered Model

分层模型是一种软件架构设计模式，通过将系统划分为多个层次，每个层次承担不同的职责，以实现更好的结构化和模块化。这种模型通常用于设计复杂的软件系统，使系统更易于开发、维护和扩展。

1. **分层结构**：
   - 系统被划分为多个层，每一层在其上层之下并依赖于其下层。
   - 每层负责特定的功能，并通过接口与其他层进行交互。

2. **模块化**：
   - 每一层可以独立开发、测试和修改，降低系统复杂度。
   - 层与层之间的相互依赖减少，增强了系统的灵活性和可维护性。

3. **职责分离**：
   - 不同层次负责不同的职责，如数据访问、业务逻辑、用户界面等。
   - 职责的分离有助于明确系统的功能边界，提高代码的可读性和可维护性。

4. **可扩展性**：
   - 通过增加或修改特定层的功能，可以轻松扩展系统，而不影响其他层。

5. **可移植性**：
   - 底层的实现细节可以独立于高层，增强了系统的可移植性。

Layered Model 通常由以下几个主要层次组成：

1. **Presentation Layer（表示层）**：
   - 负责与用户的交互，包括用户界面的显示和输入处理。
   - 例如：Web 应用中的 HTML/CSS/JavaScript，桌面应用中的 GUI 组件。

2. **Application Layer（应用层）或 Business Logic Layer（业务逻辑层）**：
   - 负责处理应用程序的核心业务逻辑和规则。
   - 将用户的请求转化为具体的操作，通过与数据层交互来处理数据。

3. **Data Access Layer（数据访问层）**：
   - 负责与数据库或其他持久化存储系统进行交互。
   - 包括数据的查询、插入、更新和删除操作。

4. **Data Layer（数据层）或 Persistence Layer（持久层）**：
   - 负责实际的数据存储和管理，包括数据库系统和文件存储。
   - 管理数据的物理存储，确保数据的持久性和一致性。

以一个电子商务应用为例，Layered Model 的各层可能包含以下内容：

1. **Presentation Layer（表示层）**：
   - 负责显示产品信息、处理用户输入和结账流程。
   - 通过 *RESTful API* 或者 *GraphQL* 与应用层进行交互。

2. **Application Layer（应用层）**：
   - 实现产品搜索、购物车管理和订单处理等业务逻辑。
   - 负责用户认证和授权，确保用户的操作符合业务规则。

3. **Data Access Layer（数据访问层）**：
   - 负责从数据库中读取产品信息、订单状态，并进行数据持久化。
   - 使用 ORM（对象关系映射）工具或 SQL 查询与数据库交互。

4. **Data Layer（数据层）**：
   - 使用 MySQL 或 MongoDB 等数据库存储用户信息、产品目录和订单记录。
   - 维护数据库的架构和索引，确保数据的完整性和查询效率。

**优点**：

- 易于理解和维护：分层结构清晰，职责分离明确，使得系统易于理解和维护。
- 提高可重用性：各层可以独立开发并重用于不同项目，减少重复劳动。
- 增强可测试性：每层的功能可以单独测试，提高测试的覆盖率和效率。

**缺点**：

- 性能开销：层与层之间的交互可能带来额外的性能开销，特别是在高频调用的情况下。
- 灵活性不足：严格的分层可能导致一些特定需求难以实现，需要在架构上进行调整。
- 复杂性增加：对于简单的应用，分层模型可能引入不必要的复杂性。

Layered Model 被广泛应用于各种软件系统，包括但不限于：

- **Web 应用开发**：常见于 *MVC（Model-View-Controller）*架构中。
- **企业级应用**：如 ERP 和 CRM 系统，常采用三层或多层架构。
- **操作系统设计**：如 OSI 模型，将网络协议栈划分为多个层次。

**Layered Model 与其他架构的对比**：

- **Microservices Architecture（微服务架构）**：
  - 微服务架构将应用拆分为多个独立的服务，强调服务的独立性和去中心化。
  - Layered Model 强调职责分离和模块化，适用于系统内部的组织结构。

- **Event-Driven Architecture（事件驱动架构）**：
  - 事件驱动架构通过事件触发系统中的动作，适用于需要高度响应性和解耦的系统。
  - Layered Model 强调各层的交互和依赖，适合于逻辑清晰的业务系统。

- **Client-Server Architecture（客户端-服务器架构）**：
  - 客户端-服务器架构强调请求-响应机制，适用于网络应用和分布式系统。
  - Layered Model 可以作为客户端或服务器内部的结构化方式。

Layered Model 是一种经典的架构模式，通过分层实现系统的职责分离和模块化，适用于大多数复杂的业务系统。虽然在性能和灵活性上存在一些挑战，但其易于理解、维护和扩展的特点，使得它在软件设计中具有重要地位。通过合理的分层和接口设计，Layered Model 可以为复杂的系统提供清晰的结构和强大的支持。

### 其他构架

**Microservices Architecture（微服务架构）**

- 微服务架构是一种构建应用的架构模式，它将应用分解为一系列小型、独立的服务，每个服务都可以独立部署和扩展。这些服务通常通过轻量级的协议（如 HTTP/REST、gRPC）进行通信。
- 特点：
- 独立部署：各服务可以独立于其他服务进行部署。
- 可独立扩展：根据需要扩展特定服务。
- 强大的技术栈：每个服务可以选择适合自身的技术栈和编程语言。
- 应用示例：
- 大型互联网公司，如 Netflix、Amazon。
- 需要灵活扩展和快速迭代的企业应用。

**Event-Driven Architecture（事件驱动架构）**

- 事件驱动架构是一种基于事件的架构模式，强调通过事件来驱动系统行为。
- 特点：
- 异步处理：事件驱动的系统通常是异步的。
- 解耦：生产者和消费者之间没有直接联系。
- 可伸缩性：事件可以在多个消费者之间分发。
- 应用示例：
- 实时数据处理系统。
- 物联网（IoT）系统。
- 系统通知和消息传递。

**Service-Oriented Architecture（面向服务架构）**

- SOA 是一种基于服务的设计模式，类似于微服务架构，但通常采用较重的通信协议（如 SOAP），并更注重企业级的应用集成。
- 特点：
- 组件松耦合：服务独立于平台和技术。
- 标准化接口：使用标准协议定义服务接口。
- 高复用性：服务可重用性高，支持不同应用集成。
- 应用示例：
- 企业级应用集成，特别是在传统企业中广泛应用。

**Repository Pattern（仓储模式）**

- Repository Pattern 是一种用于数据访问逻辑分离的模式，通过定义一个接口来实现对数据访问的抽象。
- 特点：
- 隔离数据访问逻辑：通过仓储接口来操作数据源。
- 提高测试性：业务逻辑与数据访问逻辑分离，更易于单元测试。
- 应用示例：
- 数据访问层设计，如在 DDD（领域驱动设计）中应用广泛。

### 资源链接

- [微软构架模型文档](https://learn.microsoft.com/en-us/azure/architecture/patterns/)

## Design：Modularity

**模块化**是一种软件设计原则，旨在将软件系统分解为若干个相对独立的模块（Module），以便于开发、理解、维护和重用。这些模块可以是代码、组件、功能模块或子系统，它们通过清晰的接口进行交互。

**和上一部分的系统构架不同，这部分专注于系统内部的，功能分离的设计模式**

**何为设计？**：设计行动本身，和设计文档

**Design is not coding, coding is not design.What important is the real world solutions.**

*模块化的基本概念*：
- **独立性**：每个模块应该尽可能独立运作，具有明确的功能和责任，减少对其他模块的依赖。
- **高内聚、低耦合**：模块内部的元素应紧密相关（高内聚），而模块之间应尽量少的依赖（低耦合）。
- **接口**：模块之间通过接口进行通信，这些接口定义了模块提供的服务和可供其他模块调用的功能。

### 应用领域

- **代码层面的模块化**：在代码层面上，模块化通常体现在类、函数和库的设计中：
  - *类和方法*：在面向对象编程中，类和方法是模块化设计的基本单位。每个类或方法应该负责单一功能或责任。
  - *模块和包*：在许多编程语言中，模块和包用于组织和管理代码库中的相关功能。例如，Python 的 `module` 和 Java 的 `package`。
- **系统架构层面的模块化**：在系统架构层面上，模块化体现在微服务架构、插件架构等：
  - *微服务架构*：将应用程序拆分为一组小型、独立部署的服务，每个服务负责特定的业务功能。
- **插件架构**：允许开发者通过插件来扩展或修改系统功能，而无需更改核心系统。浏览器的插件系统允许用户自定义浏览器的功能
- **框架和库的模块化**：许多软件框架和库也采用模块化设计，以提高可扩展性和易用性
  - *前端框架（如 React、Angular）*：这些框架通常使用组件（Component）作为模块化单元，每个组件封装了特定的 UI 逻辑和样式
  - *后端框架（如 Django、Spring）*：通过模块化设计，开发者可以使用或替换特定功能模块，如身份验证、数据库访问等

## Implementation & Deployment
## Testing
## Software Development
## Agile
## Scrum Deep Dive