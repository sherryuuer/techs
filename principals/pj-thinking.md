## 总结一下项目中得到的心得

### 关于项目推进上的感想

- 交流很重要，一开始的沟通，工具，会议，人员配置，都是学问。如果你能承担起一个项目，也许就能搭建自己的系统，从要素型生产变为系统型生产者。这可能就是项目管理的魅力。
- 之前的案件我和上司两个人没有认真的设计也没有认真的测试，直接就走到了运用阶段。这是不太正规的行为。
- 虽然不喜欢做资料，但是要理解的是，做资料只是一个整理的行为，他们是工程中必不可少的过程的一种指导和证明。那么就留下更多的template作为自己的武器吧。同时更加是一种训练自己思考方式的方法。
- 思考更上层的东西，不只是这一个案件，而是将整个过程抽象化，那么无论做什么项目，都可以进行框架的套用，升级，修改。
- 工程师的觉悟的培养，离不开琐碎的工作。开发者和项目管理是两个不同的概念，他们考虑的内容不同，同时也有交集，那就是对技术的理解，开发者关注的是系统或者应用的性能，反馈，结构，最佳实践。而项目管理，是对整个工程的协调，项目管理也需要有技术才能判断这样的管理是否合适，同时信任开发者。
- 一个人只有先有技术，然后再去做项目管理。
- 这么来思考前一个项目和现在项目的关系，前一个项目的经验就好像是PoC，然后将那个经验拿过来放到现在的项目中来用。但是如果本身就是一个新的不懂的项目，就可能需要一个PoC的过程了。


### 关于权限申请/阅读doc的实践经验

这次（24/04/12）发生的问题前置情况：

- 整个项目的云环境的权限是通过向管理者申请相对应服务的权限实现的。
- 我们需要从AWS环境的Redshift，传送数据到GCP的Bigquery。GCP的Bigquery有一个功能data transfer。
- 管理者是通过Terraform进行权限管理的，如果发生环境改变，他会自动修复为定义好的环境。
- Bigquery data transfer内部默认使用内部的服务账号进行作业。
- 这导致我们虽然在定义作业的时候没有看到内部的服务账号显示，但是在执行的时候，内部系统自动为内部的服务账号进行了权限赋予，方便它对BQ对dataset进行数据写入的作业。
- 对方的管理者检测到了这一行为，对我们提出了警告。

最终解释清楚了但是实质上是我们没有调查清楚，要请求内部服务账号的权限才可以。

总结起来有如下经验：

- 对于官方的Documentation的阅读，不要只看片面的一页，如果没有时间阅读所有，则向周围或者上司请求复查。
- 即是是复查也可能漏掉，所以可以从如下步骤进行检查：
  - 权限行为主体：是人还是service也就是代理。
  - 主体行为如何：主要是代理行为，如何进行认证和授权的，整个流程如何，要自己理解清楚。
  - 数据的流动路径：从A数据库到B数据库，那么两边都需要权限吗。