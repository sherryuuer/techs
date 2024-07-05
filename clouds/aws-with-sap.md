## Identity & Federation

**IAM Policies精细控制用户权限**
- IAM Role和Resource Based Policies：
  - IAM Role是用户赋予一个Role进行资源操作（这个时候，完全无法使用user原有的权限了）
  - RBP是在资源中进行Polies限制（比如S3 Bucket Policies）（这个时候user不需要放弃自己的其他权限）
  - 后者似乎应用场景更灵活
- IAM Permission Boundary：权限边界，这是一个高级功能，限制了用户的权限边界，对用户定义的任何权限如果超过了这个设置，都是不可用的
  - 三者结合：组织SCP + IAM Permission Boundary + 用户个人权限
- IAM Access Analyzer：分析任何针对资源的*外部用户* -- 若有，通过finding进行记录
  - 设置Zone of Trust：表示对一个资源的信任主体集合，超出这个集合的，都会被finding记录
  - 自动生成Policy的功能构架：通过Cloud Trail记录过去90天的资源的API call记录，来自动生成Policy，这种构架默认你90天内的活动是合规的，但是如何保证是合规的呢，如果有人错误使用了该服务进行了外部连接？
    - 不过说回来，是可以通过人工检查进行细节调整的

**STS认证**
- 跨账户的临时用户权限，通过发行token，使得其他账户的用户可以使用自己账户的临时Role进行操作
- 对于第三方账户用户，可以通过*ExternalID*进行精准控制，保证进来的是特定的用户，该ID是秘密的
- CLI流程：assume-role命令，然后通过得到的临时Credentials，设置环境变量
  - 当有使用MFA的时候，使用[官方手顺](https://repost.aws/knowledge-center/authenticate-mfa-cli)，get-session-token参数为账号信息，你的设备名和token code（当时显示的），然后同样得到临时认证信息后，设置环境变量，就可以使用了
