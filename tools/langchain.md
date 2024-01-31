## LangChain大模型框架应用

---

### 五大模块

ModelIO -> Data Connection -> Chains -> Memory -> Agents

所有代码练习内容都放在了仓库`projects-drafts`的`langchain`中。各个大标题各对应一个notebook。

### ModelIO

可以看成是大模型的接口，通过改变一些参数，就可以接入到各种模型中，不管是OpenAI还是谷歌之类的各种模型。和模型的输入和输出进行交互和处理。看起来和OpenAI很相似。但是不要急。

内部有两种模型接口：一种是**LLM**用于文字完成text completion，另一种是聊天模型**Chat**。chat模型内部有强化学习系统，通过和人类来回聊天，进化聊天内容。

**环境**：使用openai的话需要一个apikey。在官网注册登陆信用卡后可用。

使用python进行环境信息搭建。执行安装代码`!pip install openai`然后使用os设置环境变量。也可以将key存储在文件中导入。

使用官方的接口接入openai模型。

其实还有很多其他类型的模型可以使用，在官网都有可以直接使用的接入代码。使用langchain的生成器，可以pass一个提示列表给模型，输出一个生成器结果，从生成器中提取想要的文本非常方便。

**LLM model 和 Chat models**

Langchain可以定义三种信息给模型：系统信息（system message），人类信息（human message），AI信息（AI message）。

系统信息是给系统设置的各种参数比如个性，语气等，人类信息是人类提示信息和回复，AI信息是AI的回复。

**Cache answer**：可以使用cache将回复结果进行缓存，因为每次执行代码都会花费金额。

**Prompt Template**：提供了一种提示的模板，方便将prompt的各个部分作为变量，提供给接口。因为在构建大型的系统的时候你不能一条一条写信息吧。

**Few shot prompt template**: 中文叫少量用例，加入两个模板，给出input和output的示例，传送给chat_prompt。一开始看的不是很明白，说白了应该就是，你教AI用什么样的方式进行回答固定模式的问题。

**Output parsers**: 上面为止是
