## base
(Clear everything and start from begining.)

- DBMS：数据库管理系统
- RDBMS：关系型数据库管理系统
- SQL：结构型查询语言/StructuredQuery Language

**类型：**

- Relational
- Document：MongoDB，Firebase
- Key-Value：Redis，DynamoDB
- Graph：Neo4j，Amazon Neptune
- Wide Columnar：Google Bigtable

**Imperative vs Declarative:**

- Imperative：命令式编程是一种编程范式，通过明确的指令来改变程序状态，描述*怎么做*，常用的语言有C、Java；
- Declarative：声明式编程则侧重于描述*是什么*，例如SQL和HTML，它们更关注于逻辑和表达，而不是具体的实现步骤。
- Python主要是一种命令式编程语言，但它也支持声明式编程风格，特别是在函数式编程和使用声明式库（如SQLAlchemy）时。


**SQLStandards**是指SQL（结构化查询语言）的标准规范，由国际标准化组织（ISO）和美国国家标准协会（ANSI）制定，这些标准定义了SQL的语法、功能和行为，确保不同数据库系统的兼容性和一致性，主要版本包括SQL-86、SQL-89、SQL-92、SQL:1999、SQL:2003、SQL:2008、SQL:2011和SQL:2016。

**Columns**：又可以叫做degrees，attributes，一个列可以叫做Column，Domain，Constraint（约束）。同一个东西。

**OLTP&OLAP**：前者是事务性处理，后者是分析目的的处理。前者的构架主要是应用端的客户信息记录，后者的构架则是在ETL中充当数据存储的数据库，然后供下游分析使用。

## Database Models

Database Models是指用于组织、存储和管理数据的结构和方法。

*Hierarchical Model*

Hierarchical模型是一种数据库模型，其中数据按*树形结构*组织，每个记录（节点）有一个单一的父节点和多个子节点，也就是*一对多*关系，类似于*文件系统的目录结构*。它常用于早期的数据库系统和某些特定应用，如IBM的IMS数据库，适合处理层次关系明显的数据，但在灵活性和查询复杂关系方面有限。

*Network Model*

Network Model是一种数据库模型，它通过图结构来表示数据和它们之间的关系。在这种模型中，数据实体被表示为节点，实体之间的关系被表示为边（链接）。这种模型*允许多对多*关系，并且比层次模型更灵活。Network Model的一个典型实现是CODASYL（Conference on Data Systems Languages）数据库模型，早期数据库系统如IDMS（Integrated Database Management System）就是基于这种模型。Network Model适合复杂关系的数据，但查询语言通常较为复杂，不如关系模型直观。

*Relational Model*

Relational Model是一种数据库模型，通过二维表格（关系）来组织数据，每个表由行（记录）和列（字段）组成。行代表数据项，列代表数据属性。关系模型的核心概念包括表（relation）、键（key）、关系（relationship），以及使用SQL（结构化查询语言）进行数据查询和操作。这种模型提供了高度的数据独立性和灵活性，是目前最广泛使用的数据库模型，常见的关系数据库管理系统（RDBMS）有MySQL、PostgreSQL、Oracle、SQL Server等。

*OLTP&OLAP*：一个是事务处理，一个是分析处理。前者主要是数据行为记录，后者主要是数据分析用途。

## Postgre

- DCL(data control): grant, revoke
- DDL(data defination): create, drop, alter, rename, trancate, comment
- DQL(data query): select
- DML(data modification): insert, delete, update, merge, call, explain plan, lock table 

## aggregate functions**（聚合函数）和scalar functions（标量函数）

聚合函数对一组行进行操作，并返回一个单一的值。它们常用于数据汇总和分析，尤其是在`GROUP BY`子句中，帮助从一组数据中计算总体值。常见的聚合函数有：

1. **COUNT()**: 计算特定列或行的数量。
   - `SELECT COUNT(*) FROM Employees;` // 计算`Employees`表中的行数。

2. **SUM()**: 计算数值列的总和。
   - `SELECT SUM(Salary) FROM Employees;` // 计算`Employees`表中`Salary`列的总和。

3. **AVG()**: 计算数值列的平均值。
   - `SELECT AVG(Salary) FROM Employees;` // 计算`Employees`表中`Salary`列的平均值。

4. **MAX()**: 返回列中的最大值。
   - `SELECT MAX(Salary) FROM Employees;` // 查找`Employees`表中`Salary`列的最大值。

5. **MIN()**: 返回列中的最小值。
   - `SELECT MIN(Salary) FROM Employees;` // 查找`Employees`表中`Salary`列的最小值。

6. **GROUP_CONCAT()**: 将多行的值合并成一个字符串，通常使用逗号或指定的分隔符。
   - `SELECT GROUP_CONCAT(Name) FROM Employees;` // 将`Employees`表中`Name`列的所有值合并成一个字符串。

标量函数对单个值（而不是一组值）进行操作，并返回一个单一的值。它们通常用于转换数据或从数据中提取信息。常见的标量函数有：

1. **UPPER()**: 将字符串转换为大写。
   - `SELECT UPPER(Name) FROM Employees;` // 将`Name`列的值转换为大写。

2. **LOWER()**: 将字符串转换为小写。
   - `SELECT LOWER(Name) FROM Employees;` // 将`Name`列的值转换为小写。

3. **LEN() 或 LENGTH()**: 返回字符串的长度。
   - `SELECT LENGTH(Name) FROM Employees;` // 返回`Name`列中每个值的长度。

4. **ROUND()**: 对数值进行四舍五入。
   - `SELECT ROUND(Salary, 2) FROM Employees;` // 将`Salary`列的值四舍五入到小数点后两位。

5. **NOW()**: 返回当前的日期和时间。
   - `SELECT NOW();` // 返回当前的日期和时间。

6. **DATE()**: 从日期或日期时间值中提取日期部分。
   - `SELECT DATE(HireDate) FROM Employees;` // 从`HireDate`列提取日期部分。

7. **CONCAT()**: 连接两个或多个字符串。
   - `SELECT CONCAT(FirstName, ' ', LastName) AS FullName FROM Employees;` // 将`FirstName`和`LastName`列的值连接成一个完整的名字。

- **Aggregate Functions**: 操作一组行并返回一个单一的值。常用于数据汇总和报告。
- **Scalar Functions**: 操作单个值并返回一个单一的值。常用于数据转换和处理。
- 我觉得聚合函数就像是reduce，标量函数就像是map方法。

## 使用双引号，而不是单引号

```sql
select first_name as "first name" from "Users";
```


## resources

- [DB-Fiddle](https://www.db-fiddle.com/)
- [SQLZOO练习册](https://sqlzoo.net/wiki/SQL_Tutorial)
- [w3schools-SQL-Playground](https://www.w3schools.com/sql/trysql.asp?filename=trysql_op_in)
