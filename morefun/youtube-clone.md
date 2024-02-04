## full-stack构架一个油管克隆网站的学习笔记

---
### 全栈

全栈最初来自是web开发方面，包括前端后端所有部分所以叫做全栈。开发一个油管网站，不只是用youtube的api实现，而是从0构架，所有部分的流程，通过学习可以得到一个对全栈开发的全面理解。

### 所需工具

- github的仓库
- vscode
- node.js
- 最好有一个Linux系统，Mac继续，Win可以安装WSL
- Docker

### 创建视频处理服务

使用Express.js和TypeScript创建这个服务。

1. 创建工作台克隆远程仓库，使用npm初始化工作台。

```js
mkdir video-processing-service
cd video-processing-service
npm init -y
```

2. 安装express，typescript和ts开发服务node。

```js
npm install express
npm install --save-dev typescript ts-node
```

3. 安装type定义。

TypeScript会自动使用`@types`包为Node.js和Express提供类型定义。

基本上，TypeScript代码会经过一个构建步骤，在这个过程中它会被转换成JavaScript。在构建步骤中，会使用类型定义来确保代码是类型安全的（没有类型不匹配）。如果存在错误，构建将失败。

**使用 @types 包的自动类型定义：**
   - TypeScript依赖于`@types`包，自动包含外部库和框架（如Node.js和Express）的类型定义。
   - 这些`@types`包包含相应JavaScript库的预定义TypeScript声明，使TypeScript能够理解和检查代码中的类型。

**TypeScript构建步骤：**
   - 当编写TypeScript代码时，它需要被转译或转换成JavaScript，因为浏览器和Node.js运行JavaScript。
   - TypeScript编译器会执行一个构建步骤，将TypeScript代码转换为JavaScript代码。

```
npm install --save-dev @types/node @types/express
```

4. 创建一个tsconfig.json对TypeScript进行设置。

```js
{
  "compilerOptions": {
    "target": "es6",
    "module": "commonjs",
    "rootDir": "src",
    "outDir": "dist",
    "strict": true,
    "esModuleInterop": true},
  "include": ["src/**/*.ts"],
  "exclude": ["node_modules"]
}
```

5. 更新package.json。

```js
"scripts": {
  "start": "ts-node src/index.ts",
  "build": "tsc",
  "serve": "node dist/index.js"
}
```

6. 创建src/index.ts文件。

```js
import express from 'express';

const app = express();
const port = 3000;

app.get('/', (req, res) => {
  res.send('Hello World!');
});

app.listen(port, () => {
  console.log(`Server running at http://localhost:${port}`);
});
```

7. 此时服务器就可以用命令`npm run start`启动了。
