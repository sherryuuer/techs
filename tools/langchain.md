## LangChain大模型框架应用

---

### 五大模块

ModelIO -> Data Connection -> Chains -> Memory -> Agents

所有代码练习内容都放在了仓库`projects-drafts`的`langchain`中。各个大标题各对应一个notebook。

### ModelIO

可以看成是大模型的接口，通过改变一些参数，就可以接入到各种模型中，不管是OpenAI还是谷歌之类的各种模型。和模型的输入和输出进行交互和处理。看起来和OpenAI很相似。但是不要急。

内部有两种模型接口：一种是**LLM**用于文字完成text completion，另一种是聊天模型**Chat**。chat模型内部有强化学习系统，通过和人类来回聊天，进化聊天内容。

1. **环境**：使用openai的话需要一个apikey。在官网注册登陆信用卡后可用。

使用python进行环境信息搭建。执行安装代码`!pip install openai`然后使用os设置环境变量。也可以将key存储在文件中导入。

使用官方的接口接入openai模型。

其实还有很多其他类型的模型可以使用，在官网都有可以直接使用的接入代码。使用langchain的生成器，可以pass一个提示列表给模型，输出一个生成器结果，从生成器中提取想要的文本非常方便。

2. **LLM model 和 Chat models**

Langchain可以定义三种信息给模型：系统信息（system message），人类信息（human message），AI信息（AI message）。

系统信息是给系统设置的各种参数比如个性，语气等，人类信息是人类提示信息和回复，AI信息是AI的回复。

3. **Cache answer**：可以使用cache将回复结果进行缓存，因为每次执行代码都会花费金额。

4. **Prompt Template**：提供了一种提示的模板，方便将prompt的各个部分作为变量，提供给接口。因为在构建大型的系统的时候你不能一条一条写信息吧。

**Few shot prompt template**: 中文叫少量用例，加入两个模板，给出input和output的示例，传送给chat_prompt。一开始看的不是很明白，说白了应该就是，你教AI用什么样的方式进行回答固定模式的问题。

5. **Output parsers**: 上面为止是**输入**方式的话，那么既然是IO肯定也有输出的助攻也就是解释器。帮助你输出你想要输出的格式。

看例子感觉就是，在他们的包里，有给模型的指导也就是`get_format_instructions()`，如果打印，是可以输出对机器说的人话的。比如`CommaSeparatedListOutputParser`就是一句告诉机器，你的结果要用逗号分隔开的指导。

还有修复你的输出的解释器，比如可以帮你把输出的结果一起，重新送回模型重新修正答案再次输出。但是内部也是含有指导信息的一种方法。

其实通过设定很强的`System Prompt`也可以修复很多问题，这是一个使用内部指导还是你的外部指导的区别。

还有一种自动修复的解析器`OutputFixingParser`可以将你的输出的错误结果和你的解析器，以及模型一起重新送回去，让大模型自己修正，但是并不是每次都可以保证修复，所以可以将这个和system提示一起使用。

比较特殊的是`PydanticOutputparser`使用python的类来自己定义自己想要类，包括属性等，输出一个合适的json格式结果。

6. **Save and load prompt**: 提示模板的保存和载入，一种可以模板复用和分享的方式。

最后有一个整合的小的项目，使用以上IO部分的知识，写一个历史知识日期竞答的bot。在听解说的时候，听到说哦你可以去问Chatgpt答案，可以的呀！让AI教你操作AI，这才是正确打开方式，哈哈。

项目本身的流程是：AI给问题，AI给答案，人给答案，给出结果比较。

**这部分的感受**就是对lc的一个整体的理解。

### Data Connections

是LangChain和外界数据的交互。外界数据比如CSV文件，PDF文件，HTML文件，以及各种网页，云存储还有很多第三方（比如他还有接入hacker news的loader）的数据源，都可以作为信息载体接入使用。他们对LC来说是一个document对象。并且要知道，很多操作需要安装对应的库进行支持。比如HTML需要Beautiful soup，pdf需要python的pypdf包。

一个案例是外接一个wiki的页面，然后根据这个页面和问题，得到答案。
