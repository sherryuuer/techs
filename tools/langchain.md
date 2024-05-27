[官方LangChain大模型框架应用](https://github.com/langchain-ai/langchain)

### 五大模块

ModelIO -> Data Connection -> Chains -> Memory -> Agents

所有代码练习内容都放在了仓库`projects-drafts`的`langchain`中。各个大标题各对应一个notebook。

### ModelIO

可以看成是大模型的接口，通过改变一些参数，就可以接入到各种模型中，不管是OpenAI还是谷歌双子座各种模型。和模型的输入和输出进行交互和处理。看起来和OpenAI很相似。

内部有两种模型接口：一种是**LLM**用于文字完成text completion，另一种是聊天模型**Chat**。chat模型内部有强化学习系统，通过和人类来回聊天，进化聊天内容。

1. **环境**：使用openai的话需要一个apikey。在官网注册登陆信用卡后可用。使用python进行环境信息搭建。执行安装代码`!pip install openai`然后使用os设置环境变量。也可以将key存储在文件中导入。然后就可以使用官方的接口接入openai模型。

使用langchain的生成器，可以pass一个提示列表给模型，输出一个生成器结果，从生成器中提取想要的文本非常方便。

2. **LLM model 和 Chat models**

Langchain可以定义三种信息给模型：系统信息（system message），人类信息（human message），AI信息（AI message）。

系统信息是给系统设置的各种参数比如个性，语气等，人类信息是人类提示信息和回复，AI信息是AI的回复。

3. **Cache answer**：可以使用cache将回复结果进行缓存，因为每次执行代码都会花费token的金额。

4. **Prompt Template**：提供了一种提示的模板，方便将prompt的各个部分作为变量，提供给接口。因为在构建大型的系统的时候你不能一条一条写信息。

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

1. **Document Loader & Integration**：LC提供了很多文件载入器，可以将外部文件直接载入模型，一个案例是外接一个wiki的页面，然后根据这个页面和问题，得到答案。

2. **Document Transformer**：LC内置文本转换工具，因为模型可以吃的token是有上限的，因此将文本分割成小块，一点点喂给模型是一个好的选择。

3. **Text & Document Embeddings**：句子本文的向量嵌入，比如openai的使用的是OpenAIEmbeddings的类接口，对文本进行直接的嵌入。注意不同模型的文本嵌入方法不可以共用，他们可能不可兼容。

4. **Vector Store**：向量数据库存储，和一般的DB一样，向量也可以有数据库来保存向量，以便删除，更新，插入，查询等DB等功能。在查询上，使用余弦相似度来查找数据库中的向量和新的向量的相似度，以便拿出最相近的那些向量来使用。`Chroma`是一个很好用的开源的向量数据库。话说它和浏览器只差一个字母咩。

一般的流程是载入文档-->分割文档-->文本嵌入-->存储进数据库-->查询。

在Chroma中的存储形式是一个文件夹，其中包括文本parquet文件，向量parquet文件，和index文件，index和其他两个文件是一一对应的关系。

注意向量数据库的作用是找到相似度最高的文本，而不是直接进行提问的。

5. **Vector Store Retriever**：向量数据库检索器，是为了将向量数据库和大模型进行结合的使用方式。该接口公开一个`get_relevant_documents`方法，该方法接受查询（字符串）并返回文档列表。但是要注意到，给出的结果只是根据，余弦相似度给出的，相关性文本，可能不是你写的问题的回答。

6. **Context Compression**：上下文压缩检索器，会只返回文档中的，和提问相关的内容。通过这一步就可以实现，连接数据库-->使用模型提问关于文档中的问题-->并返回最相关的文档内容，这整个步骤了。

### Chains

chain的意思是连接的意思，这也是langchain名字来源的地方，不只是使用一个模型，而是将多个模型的输入和输出连接起来的功能，就是chain。甚至可以实现将chain和chain连接起来。

1. **LLM Chain**：简单的chain可以将一个chat_template和一个model作为参数pass给chain。原本在ModelIO部分是，chat模型里直接将chat_prompt作为输入放进去，而使用chain的情况，是将模型和template两个作为参数pass给chain，然后执行chain。

打比方就是，原本是method(input)，现在是chain(method, input)。

2. **Simple Sequential Chain**：简单的序列化chain，就是将上面的多个LLMchain，放进列表，作为`SimpleSequentialChain`的参数。从而达到，第一个chain的输出是第二个chain的输入的效果。但是如同表题所说，因为是一个简单的序列化chain，所以一般一个输出一个输入，真的是最简单的那种。

3. **Sequential Chains**：和上面的Simple的序列化chain不同的是，会有个`output_key`关键字被pass进chain的参数。明确指出，需要输入给下一个chain的是什么输入。

4. **LLMRouterChain**：如果说上面的序列化chain是一条线的话，这里的router就是一种**条件选择**的关系，设置多种case，比如你可以设置一个数学通道，一个物理通道，内部大模型会根据input检测，将input输入哪个case比较合适，然后继续下一步处理。还是挺酷的。

你要给出多个目的地的chain，同时还有一个default的chain，可能进化的版本不需要default的LLMchain，然后将prompt的信息和chat模型一起pass给多提示chain对象，run这个对象的时候将问题输入即可返回答案。

5. **TransformChain**：考虑到有时候你想要用自己的python函数对文本进行一些处理的情况。

当然你可以让模型帮你处理，但是那样会浪费钱，因为每一个token都要花钱。所以使用这个功能，就可以pass自己写的处理函数，是一个callback的功能，当你call了那个Transformchain的时候，就会自动用你写的函数帮你处理了。写出来就是这样：`TransformChain(input_variables=['text'], output_variables=['output'], transform=transformer_fun)`。

6. **Using OpenAI Functions API**：使用OpenAI的函数API可以实现多种功能，比如可以自定义输出的json格式传递给chain，从而满足后续的处理需求。

>PS:在学习过程中，我发现中文的document真的不太行，缺失和版本低等，还是需要官方的英文版的document来学习。

7. **其他的chain**：除此之外还有很多的chain。

比如math数学chain问他数学问题，可能答案不会正确，我为什么不直接用numpy哈哈。sqlchain，针对sql数据库提出问题。QAchain`load_qa_chain`针对向量数据库提出问题。总结summary文本用的chain等。

### Memory

是一种保存和模型交互历史记录的功能。在我们使用浏览器界面的chatgpt的时候，左边会显示一个历史记录，和那个很相似。

1. `ChatMessageHistory`可以创建一个很基础的history对象，用于存储或者用户，或者ai的message列表。

2. `ConversationBufferMemory`的对象可以在`ConversationChain`类对象的参数中，用来存储历史记录对话，然后通过pickle包就可以保存和复用。

3. `ConversationBufferWindowMemory`和上面的buffermemory一样，但是可以设定k值，当`load_memory_variables({})`的时候，只会得到k条对话结果。

4. `ConversationSummaryMemory`是对对话的内容进行一个摘要，而不是显示具体的对话内容，比如你说hi，AI说hi，摘要会load说你们进行了一个打招呼的操作。

### Agent

理解起来就是一种强化学习的机制，通过**观察--思考--行动**的循环，不断提升表现。同时这一部分的Agent非常容易构建，但是是在结合了前四部分的基础上，前面的理解过后就比较容易理解Agent的整个机制了。

1. **具体过程：**

input --> decide tool --> output(observation) --> decide action by history tools, tool inputs, and observations 

then repeat.

最终实质上，还是一种对大模型的插件，或者说就是chain。将Agent嵌入模型的参数中而已。以`agent.run(input)`的形式。

如果你讲verbose设置为true可以看到代理的如下思考过程。

```
> Entering new  chain...
I can use the search tool to find the year Albert Einstein was born. Then I can use the calculator to multiply that year by 5.
Action: Search
Action Input: "Albert Einstein birth year"
Observation: March 14, 1879
Thought:Now I can use the calculator to multiply 1879 by 5.
Action: Calculator
Action Input: 1879 * 5
Observation: Answer: 9395
Thought:I now know the final answer
Final Answer: The year Albert Einstein was born is 1879. When multiplied by 5, the result is 9395.
```

2. 使用`PythonREPLTool`和`create_python_agent`你甚至可以打造自己的程序员助理。

注意有些外部的api比如google search需要一些key，有些可能需要付费。

3. 使用`@tool`修饰符可以写custom的函数作为custom tool交给llm使用。定义自己想要的答案。

4. 对话代理`Conversation Agents`则需要一个memory对象作为参数，来记录和user之间的聊天记录。

