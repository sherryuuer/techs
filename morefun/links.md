[snowflake该要理解油管视频](https://www.youtube.com/watch?v=VIJH7TZXkaA&list=WL&index=8)

- 全新的数据仓库形态使用虚拟数据仓库作为分散处理引擎，同时具有pipeline功能，以及对非结构化数据的处理，简洁的UI，可以使用SQL和Python进行快速的数据处理，坐于云上，不断增加的工具

[快速构建ETL管道(dbt, Snowflake, Airflow)油管视频](https://www.youtube.com/watch?v=OLXkGB7krGo&list=WL&index=8)

- warehouse，database.schema，role都是snowflake一开始要设置的东西
- dbt初始化pj后，可以连接的数据库：postgre，bigquery，snowflake，redshift
- dbt搭建起来后，在vscode中可以看到*项目文件夹，中间会有整套组件，比如models（SQL），test测试，分析，宏工具，快照，seeds（存放不变的文件）等，然后还有一个yaml文件（dbt_project.yml）用于写pipeline*，它的写法，因为是yml，所以和workflow，以及CloudFormation很像
- dbt中*packages.yml*用于安装需要的组件，第三方的包等，这个也和dataform有点像
- dbt中的相当于dataform的sqlx的文件，被叫做*models*，当你创建source表的时候也需要用yml定义，写SQL有大量jinja语法可以用，同样，它用*ref语法*，引用其他的table
- dbt的*macro*用jinja定义计算公式，保持sql的DRY
- dbt的*test*需要用yml定义，同时用sql进行测试case的编写，但是我没明白为什么yml要放在model文件夹中，他们似乎是两种测试方式
- Astro是一个管理Airflow的平台！
- dbt的整个项目包竟然是扔进了dags文件夹！
- cosmos是一个操作dbt的package，在Ariflow中似乎很好用
