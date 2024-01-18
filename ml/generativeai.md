## 生成式 AI（Genarative AI）

---

### 什么是生成式 AI

"生成式 AI"（Generative AI）通常指的是一类人工智能系统，这些系统能够生成新的、原创性的内容，而不仅仅是对输入数据的分析、分类或预测。

生成式 AI 通常基于深度学习模型，特别是生成对抗网络（GANs，Generative Adversarial Networks）和变分自编码器（VAEs，Variational Autoencoders）等模型。这些模型能够学习并生成与输入数据类似的全新数据，如图像、文本、音频等。

**对现状的巨大冲击**：这是一个有趣的时代，AI还没有完全发展起来没有成熟，我们处在一个可以见证它慢慢成长的时代。未来很多现代科幻书籍的内容可能会成为现实，百分之五十的人面临失业，现代来说，受到最大影响的是开发者。会不会用工具，开始影响人和人的差别。在这个被冲击的时代，周围可能没有什么大变化，这只是看起来的，有时候，巨大的变革是一瞬间的。做好准备吧。

应用领域主要有以下方面：

1. **图像生成：** 生成对抗网络（GANs）可以生成逼真的图像，这些图像在视觉上难以与真实图像区分。

2. **文本生成：** 使用循环神经网络（RNNs）或变分自编码器（VAEs）等模型，可以生成自然语言文本，包括文章、散文、甚至对话。

3. **音频生成：** WaveNet 和类似的模型可以生成逼真的语音，这对于语音合成等任务非常有用。

4. **艺术创作：** 生成式 AI 被用于创建艺术品，包括绘画、音乐、甚至是电影剧本。

5. **图像风格迁移：** 将一个图像的风格应用于另一个图像，使其看起来像是由相同的艺术家创作。

6. **虚拟现实与增强现实：** 在这两个领域中，生成式 AI 被用于创建逼真的虚拟环境或增强现实应用。

生成式 AI 的发展推动了艺术创意、内容生成、虚拟环境模拟等领域的创新，并为计算机生成的内容赋予更多的创造性和原创性。然而，它也引发了一些伦理和法律上的问题，例如深度假图像的制作和传播可能引发虚假信息和欺骗等问题。

另外很多你在网上查不到的东西，如果想要头脑风暴，很适合和LLM一起生成答案，很有意思。

### 领域划分

AI-->ML-->DeepLearning&NLP-->LLM

生成式 AI 是深度学习和自然语言处理的交集领域，出现的大语言模型就是这两个领域交集在语言领域的模型。宗旨在于模拟人的行为，学习模式，完成仿佛人类做的 task。

### 擅长做什么？？

生成式AI和其他AI技术一样，无法特定地说它擅长什么领域，适合做什么，要探索的还有很多，比如你可以说微波炉适合热食物，你却没法说清生成式AI适合做什么，就像问你，电适合做什么？一样，它是一种新的**生产力**。这也是它的可怕之处。

现状如何通过它提升自己的生产力？？

我总结有以下几点：

1，头脑风暴-writing能力：人类的特长就是思考出新的观点，而不只是通过概率进行联系和生成，通过和生成式AI配合，想出新的创意，是提升脑力的好方法。

2，内容分类和总结-reading能力：将内容扔个生成式AI，他会帮你总结该要，阅读重点，分类垃圾邮件等，说白了还是一种文本处理能力。

### 生成式AI快速搭建项目的能力

GanAI让项目搭建变得更加迅速。在之前还需要非常多的代码，进行环境构建，数据训练。但是这之后只需要如同以下的几行代码而已。
（代码来自Coursera吴恩达课程）

整个代码看起来就是写了几行简单的python代码而已。

同时它的整个生命周期循环，和一般的ai项目一样，需要通过用户反馈进行优化。

```python
import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

def llm_response(prompt):
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=[{'role':'user','content':prompt}],
        temperature=0
    )
    return response.choices[0].message['content']

prompt = '''
    Classify the following review 
    as having either a positive or
    negative sentiment:

    The banana pudding was really tasty!
'''

response = llm_response(prompt)
print(response)

all_reviews = [
    'The mochi is excellent!',
    'Best soup dumplings I have ever eaten.',
    'Not worth the 3 month wait for a reservation.',
    'The colorful tablecloths made me smile!',
    'The pasta was cold.'
]

all_sentiments = []
for review in all_reviews:
    prompt = f'''
        Classify the following review 
        as having either a positive or
        negative sentiment. State your answer
        as a single word, either "positive" or
        "negative":

        {review}
        '''
    response = llm_response(prompt)
    all_sentiments.append(response)

num_positive = 0
num_negative = 0
for sentiment in all_sentiments:
    if sentiment == 'positive':
        num_positive += 1
    elif sentiment == 'negative':
        num_negative += 1
print(f"There are {num_positive} positive and {num_negative} negative reviews.")
```

### 检索和生成相结合的能力RAG

在生成式AI领域，RAG 通常指的是"Retrieval-Augmented Generation"，即检索增强生成。这是一种结合了检索（retrieval）和生成（generation）方法的AI模型。RAG 模型的目标是利用检索方法提供的信息来增强生成模型的效果，特别是在自然语言处理任务中。

RAG 模型的核心思想是将检索模块与生成模块结合起来，以提高生成模型在生成文本或回答问题等任务上的性能。一种常见的应用是在问答系统中，其中 RAG 模型可以通过检索阶段获得可能的候选答案，然后在生成阶段进一步完善和生成最终的答案。

RAG 模型通常包含以下关键组件：

1. **Retrieval Module（检索模块）：** 该模块负责从大型文本数据库或知识库中检索相关的信息。这可以通过检索相似问题、相关文本片段等方式实现。

2. **Ranking Module（排序模块）：** 对检索到的信息进行排序，以确定哪些信息更相关，更有可能对生成任务有帮助。

3. **Generation Module（生成模块）：** 该模块负责生成最终的文本或答案。生成模块可以使用预训练的语言模型，例如 GPT（Generative Pre-trained Transformer）。

RAG 模型的优势在于结合了检索和生成的优点，既能利用大规模文本库中的知识，又能根据任务的具体要求生成更具体、更个性化的文本。这种方法常常用于解决开放领域问答、文本生成和对话生成等任务，提高了模型在处理真实场景中的效果。

### 通过迁移学习和微调进行更针对具体任务的AI生成

比如调节成特殊的说话方式，学习更专业的特殊领域知识（比如法律文件，财务文件等）。使用迁移学习，只需要很少的数据量，就可以进行训练了。相比较pretraining，也就是从头训练一个模型，会非常便宜。在kaggle，huggingface等平台就可以得到个人可用的模型进行微调和使用。

相比较来说只有非常大的公司，数据，资金的拥有者，才会进行，pretraining模型。
