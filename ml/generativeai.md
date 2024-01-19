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

### 文本到文本

```python
# Text2Text_GPT
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load pre-trained model tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Load pre-trained model
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Set model to eval mode
model.eval()

# Encode context to input ids
context = "In the morning, I drink a cup of coffee to start my day. Next, I"
input_ids = tokenizer.encode(context, return_tensors='pt')

# Generate text
generated = model.generate(input_ids, max_length=100, do_sample=True)

# Decode generated text
text = tokenizer.decode(generated[0], skip_special_tokens=True)

print(text)
```

这段代码使用Hugging Face的transformers库来加载预训练的GPT-2模型，并生成一段文本。下面是对代码的详细解释：

```python
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load pre-trained model tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
```

首先，导入PyTorch和Hugging Face的transformers库。然后，使用`GPT2Tokenizer.from_pretrained('gpt2')`加载GPT-2模型的预训练分词器。

```python
# Load pre-trained model
model = GPT2LMHeadModel.from_pretrained('gpt2')
```

接着，使用`GPT2LMHeadModel.from_pretrained('gpt2')`加载GPT-2的预训练模型。

```python
# Set model to eval mode
model.eval()
```

将模型设置为评估（evaluation）模式，这是因为在生成文本时，不需要进行模型的训练。

```python
# Encode context to input ids
context = "In the morning, I drink a cup of coffee to start my day. Next, I"
input_ids = tokenizer.encode(context, return_tensors='pt')
```

使用已加载的分词器（tokenizer）将输入文本编码成模型可接受的输入张量（input tensor）。在这里，文本"In the morning, I drink a cup of coffee to start my day. Next, I"被编码为模型的输入张量（input_ids）。

```python
# Generate text
generated = model.generate(input_ids, max_length=100, do_sample=True)
```

使用已加载的模型生成文本。`model.generate`函数接收一个输入张量（input_ids）和一些生成文本的参数，如`max_length`表示生成文本的最大长度，`do_sample=True`表示使用采样的方式生成文本。

```python
# Decode generated text
text = tokenizer.decode(generated[0], skip_special_tokens=True)
```

解码生成的文本，将模型生成的输出张量（generated）通过分词器的`decode`方法转换为可读的文本。`skip_special_tokens=True`表示跳过特殊标记，比如`[PAD]`和`[CLS]`。

```python
print(text)
```

最后，打印生成的文本。这段代码的作用是生成一个以给定文本开头的、最大长度为100字符的文本，采用了模型的自由生成（sampling）方式。

---

```python
# DistilBERT- Text2Text
from transformers import pipeline
import time

# Use a pipeline as a high-level helper
pipe = pipeline("question-answering", model="distilbert-base-cased-distilled-squad", device=0)


def time_wrapper(func):
    def inner_func(*args, **kwargs):
        s = time.time()
        res = func(*args, **kwargs)
        return time.time() - s

    return inner_func


@time_wrapper
def predict(**kwargs):
  print(pipe(**kwargs))

times = []
for i in range(len(questions)):
  times.append(predict(question=questions[i % len(questions)], context=context))
```

这段代码使用Hugging Face的transformers库中的pipeline模块，结合DistilBERT模型，在给定的上下文（context）中回答一系列问题。下面是对代码的详细解释：

```python
from transformers import pipeline
import time

# Use a pipeline as a high-level helper
pipe = pipeline("question-answering", model="distilbert-base-cased-distilled-squad", device=0)
```

首先，导入Hugging Face的transformers库中的pipeline模块以及计时用的time库。然后，创建一个名为`pipe`的pipeline对象，通过这个对象可以方便地使用预训练模型进行问答任务。在这里，使用的是DistilBERT模型（"distilbert-base-cased-distilled-squad"）。

```python
def time_wrapper(func):
    def inner_func(*args, **kwargs):
        s = time.time()
        res = func(*args, **kwargs)
        return time.time() - s

    return inner_func
```

定义了一个装饰器函数`time_wrapper`，用于计时其他函数的执行时间。这个装饰器接受一个函数作为参数，返回一个新的函数，该新函数在执行原始函数的同时计时，并返回执行时间。

```python
@time_wrapper
def predict(**kwargs):
  print(pipe(**kwargs))
```

定义了一个函数`predict`，使用了刚刚定义的装饰器`time_wrapper`。这个函数通过`pipe`对象进行问答预测，接收任意关键字参数，并打印出问答的结果。

```python
times = []
for i in range(len(questions)):
  times.append(predict(question=questions[i % len(questions)], context=context))
```

创建了一个空列表`times`，然后使用循环对一系列问题进行问答，并记录每次问答的执行时间。循环中，通过`predict`函数传递问题（question）和上下文（context），并将计时结果添加到`times`列表中。

在这里，通过`question-answering`的pipeline进行问答，其中`model="distilbert-base-cased-distilled-squad"`表示使用DistilBERT模型。`device=0`表示使用第一个GPU设备，如果有的话。 `predict`函数打印了问答的结果，而`times`列表则包含了每次问答的执行时间。

### 图像到文本，文本到图像

```python
# Image2Text
from transformers import pipeline
import matplotlib.pyplot as plt
import urllib
import numpy as np
from PIL import Image
import torch

# @title Model Selection  { display-mode: "form" }
model = "microsoft/git-base" #@param ["Salesforce/blip2-opt-2.7b", "microsoft/git-base"]

model_pipe = pipeline("image-to-text", model=model)

# @title Prediction  { display-mode: "form" }
image_path = 'https://m.media-amazon.com/images/I/71cXwgCUrYL.jpg' #@param {type:"string"}

if image_path.startswith('http'):
  img = np.array(Image.open(urllib.request.urlopen(image_path)))
else:
  img = plt.imread(image_path)

caption = model_pipe(image_path)[0]['generated_text']
print('Caption:', caption)

plt.axis('off')
plt.imshow(img)

import lavis
from lavis.models import load_model_and_preprocess
from lavis.processors import load_processor
from lavis.common.gradcam import getAttMap
from lavis.models.blip_models.blip_image_text_matching import compute_gradcam

# @title Setup Gradcam model {display-mode: "form"}
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
model, vis_processors, text_processors = load_model_and_preprocess("blip_image_text_matching", "base", device=device, is_eval=True)

# @title Compute Gradcam  { display-mode: "form" }
def visualize_attention(img, full_caption):
    raw_image = Image.fromarray(img).convert('RGB')

    dst_w = 720
    w, h = raw_image.size
    scaling_factor = dst_w / w

    resized_img = raw_image.resize((int(w * scaling_factor), int(h * scaling_factor)))
    norm_img = np.float32(resized_img) / 255

    img = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    txt = text_processors["eval"](full_caption)

    txt_tokens = model.tokenizer(txt, return_tensors="pt").to(device)
    gradcam, _ = compute_gradcam(model, img, txt, txt_tokens, block_num=7)
    gradcam[0] = gradcam[0].numpy().astype(np.float32)

    num_image = len(txt_tokens.input_ids[0]) - 2
    fig, ax = plt.subplots(num_image, 1, figsize=(15, 5 * num_image))

    gradcam_iter = iter(gradcam[0][2:-1])
    token_id_iter = iter(txt_tokens.input_ids[0][1:-1])

    for i, (gradcam, token_id) in enumerate(zip(gradcam_iter, token_id_iter)):
        word = model.tokenizer.decode([token_id])
        gradcam_image = getAttMap(norm_img, gradcam, blur=True)
        gradcam_image = (gradcam_image * 255).astype(np.uint8)
        ax[i].imshow(gradcam_image)
        ax[i].set_yticks([])
        ax[i].set_xticks([])
        ax[i].set_xlabel(word)

visualize_attention(img, caption)
```
这段代码实现了对图像和文本之间的注意力可视化（attention visualization），其中使用了Hugging Face的transformers库中的pipeline，以及LAVIS（Language-Agnostic Vision and Language Interoperability System）库中的一些模块。下面是对代码的逐行解释：

```python
from transformers import pipeline
import matplotlib.pyplot as plt
import urllib
import numpy as np
from PIL import Image
import torch
import lavis
from lavis.models import load_model_and_preprocess
from lavis.processors import load_processor
from lavis.common.gradcam import getAttMap
from lavis.models.blip_models.blip_image_text_matching import compute_gradcam
```

导入所需的库和模块，包括Hugging Face的transformers库、matplotlib、urllib、numpy、PIL（Python Imaging Library）、torch、以及LAVIS库中的相关模块。

```python
model = "microsoft/git-base" #@param ["Salesforce/blip2-opt-2.7b", "microsoft/git-base"]
model_pipe = pipeline("image-to-text", model=model)
```

选择图像到文本的预训练模型，这里可以选择使用"Salesforce/blip2-opt-2.7b"或"microsoft/git-base"，并创建一个图像到文本的pipeline对象 `model_pipe`。

```python
image_path = 'https://m.media-amazon.com/images/I/71cXwgCUrYL.jpg' #@param {type:"string"}
if image_path.startswith('http'):
    img = np.array(Image.open(urllib.request.urlopen(image_path)))
else:
    img = plt.imread(image_path)
```

指定输入图像的路径，如果是以'http'开头，说明是网络上的图片，使用`urllib`库加载。否则，使用`matplotlib`加载本地图片。

```python
caption = model_pipe(image_path)[0]['generated_text']
print('Caption:', caption)
```

通过`model_pipe`对输入图像进行预测，获取生成的文本描述（caption）。

```python
import lavis
from lavis.models import load_model_and_preprocess
from lavis.processors import load_processor
from lavis.common.gradcam import getAttMap
from lavis.models.blip_models.blip_image_text_matching import compute_gradcam
```

导入LAVIS库的相关模块，包括模型加载、预处理、图像文本匹配的GradCAM（Gradient-weighted Class Activation Mapping）计算等。

```python
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
model, vis_processors, text_processors = load_model_and_preprocess("blip_image_text_matching", "base", device=device, is_eval=True)
```

设置设备为GPU（cuda）或CPU，并加载图像文本匹配模型以及相应的处理器。

```python
def visualize_attention(img, full_caption):
    raw_image = Image.fromarray(img).convert('RGB')
    dst_w = 720
    w, h = raw_image.size
    scaling_factor = dst_w / w
    resized_img = raw_image.resize((int(w * scaling_factor), int(h * scaling_factor)))
    norm_img = np.float32(resized_img) / 255

    img = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    txt = text_processors["eval"](full_caption)

    txt_tokens = model.tokenizer(txt, return_tensors="pt").to(device)
    gradcam, _ = compute_gradcam(model, img, txt, txt_tokens, block_num=7)
    gradcam[0] = gradcam[0].numpy().astype(np.float32)

    num_image = len(txt_tokens.input_ids[0]) - 2
    fig, ax = plt.subplots(num_image, 1, figsize=(15, 5 * num_image))

    gradcam_iter = iter(gradcam[0][2:-1])
    token_id_iter = iter(txt_tokens.input_ids[0][1:-1])

    for i, (gradcam, token_id) in enumerate(zip(gradcam_iter, token_id_iter)):
        word = model.tokenizer.decode([token_id])
        gradcam_image = getAttMap(norm_img, gradcam, blur=True)
        gradcam_image = (gradcam_image * 255).astype(np.uint8)
        ax[i].imshow(gradcam_image)
        ax[i].set_yticks([])
        ax[i].set_xticks([])
        ax[i].set_xlabel(word)
```

定义了`visualize_attention`函数，用于可视化注意力。首先，对输入的图像进行预处理，然后加载文本处理器和模型。接下来，通过LAVIS库的`compute_gradcam`函数计算GradCAM，并通过`matplotlib`库进行可视化。最终，使用`visualize_attention`函数对图像和生成的文本描述进行注意力可视化。

下面是对`visualize_attention`函数的逐行解析：

```python
def visualize_attention(img, full_caption):
    # 将输入图像转换为PIL格式，并转换为RGB模式
    raw_image = Image.fromarray(img).convert('RGB')
    
    # 目标宽度设定为720
    dst_w = 720
    # 获取原始图像的宽度和高度
    w, h = raw_image.size
    # 计算缩放因子，以便将图像缩放到目标宽度
    scaling_factor = dst_w / w
    # 缩放图像
    resized_img = raw_image.resize((int(w * scaling_factor), int(h * scaling_factor)))
    # 将缩放后的图像转换为范围在[0, 1]的浮点数
    norm_img = np.float32(resized_img) / 255

    # 将原始图像传递给图像处理器，并在评估模式下处理（eval）
    img = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    # 使用文本处理器处理完整的标题
    txt = text_processors["eval"](full_caption)

    # 使用模型的分词器将处理后的文本转换为模型可接受的输入格式（PyTorch张量）
    txt_tokens = model.tokenizer(txt, return_tensors="pt").to(device)
    # 计算GradCAM
    gradcam, _ = compute_gradcam(model, img, txt, txt_tokens, block_num=7)
    # 将GradCAM的数据类型转换为NumPy数组，同时将数据类型转换为32位浮点数
    gradcam[0] = gradcam[0].numpy().astype(np.float32)

    # 获取文本序列的长度，减去2是因为开头和结尾的特殊标记
    num_image = len(txt_tokens.input_ids[0]) - 2
    # 创建一个子图数组，用于显示GradCAM的图像
    fig, ax = plt.subplots(num_image, 1, figsize=(15, 5 * num_image))

    # 迭代GradCAM和文本标记，以显示每个标记的GradCAM图像
    gradcam_iter = iter(gradcam[0][2:-1])
    token_id_iter = iter(txt_tokens.input_ids[0][1:-1])

    for i, (gradcam, token_id) in enumerate(zip(gradcam_iter, token_id_iter)):
        # 将标记ID解码为单词
        word = model.tokenizer.decode([token_id])
        # 获取经过模糊处理的GradCAM图像
        gradcam_image = getAttMap(norm_img, gradcam, blur=True)
        # 将图像值缩放到[0, 255]范围，并转换为无符号8位整数
        gradcam_image = (gradcam_image * 255).astype(np.uint8)
        # 在子图中显示GradCAM图像
        ax[i].imshow(gradcam_image)
        ax[i].set_yticks([])
        ax[i].set_xticks([])
        ax[i].set_xlabel(word)
```

这个函数主要用于可视化图像和文本之间的注意力分布。它首先对输入图像进行预处理，然后使用图像处理器和文本处理器进行处理。接着，通过调用`compute_gradcam`函数计算GradCAM，最后使用matplotlib库在子图中显示GradCAM图像。

### 文本到图像

```python
# Text2Image1
from min_dalle import MinDalle
import matplotlib.pyplot as plt
import numpy as np
import torch

# loading pre-trained model
model = MinDalle(
    models_root='./pretrained',
    dtype=torch.float32,
    device='cuda',
    is_mega=True,
    is_reusable=True
)
# @title String fields
text_input = 'flying batman with colorful sky' # @param {type:"string"}
# e.g. 'flying animals with blue sky and white clouds'
plt.figure(figsize=(10,5))
max_images = 5
for x in range(max_images):
  image = model.generate_image(
      text=text_input,
      seed=-1,
      grid_size=1
  )
  image = np.asarray(image)
  plt.subplot(1,5,x+1); plt.imshow(image); plt.axis('off')
  print(x+1,'/',max_images,' --- generated')
plt.suptitle(text_input)
plt.tight_layout()
```
这段代码使用 `min_dalle` 库中的 `MinDalle` 类生成基于文本描述的图像。以下是代码的逐行解释：

```python
from min_dalle import MinDalle
import matplotlib.pyplot as plt
import numpy as np
import torch
```

导入所需的库和模块，包括 `min_dalle` 库、`matplotlib`、`numpy` 和 `torch`。

```python
# loading pre-trained model
model = MinDalle(
    models_root='./pretrained',
    dtype=torch.float32,
    device='cuda',
    is_mega=True,
    is_reusable=True
)
```

加载预训练的 `MinDalle` 模型。参数说明：
- `models_root='./pretrained'`：指定预训练模型的存储路径。
- `dtype=torch.float32`：指定模型使用的数据类型。
- `device='cuda'`：指定在GPU上运行模型。
- `is_mega=True`：表示使用MEGA模型，这可能是模型的一个变种或配置。
- `is_reusable=True`：表示模型是可重用的，可以生成多个图像。

```python
# @title String fields
text_input = 'flying batman with colorful sky' # @param {type:"string"}
# e.g. 'flying animals with blue sky and white clouds'
```

通过注释 (`# @title String fields`) 定义了一个文本输入字段，用户可以在这里输入一个描述图像的文本。在这个例子中，文本描述是 'flying batman with colorful sky'。

```python
plt.figure(figsize=(10,5))
max_images = 5
```

创建一个图形，并设定图形的大小。设置变量 `max_images` 为 5，表示生成 5 张图像。

```python
for x in range(max_images):
  image = model.generate_image(
      text=text_input,
      seed=-1,
      grid_size=1
  )
  image = np.asarray(image)
  plt.subplot(1, 5, x + 1); plt.imshow(image); plt.axis('off')
  print(x + 1, '/', max_images, ' --- generated')
```

通过循环生成图像，并在每次迭代中显示生成的图像。使用 `model.generate_image` 函数生成图像，其中的参数包括：
- `text=text_input`：用于生成图像的文本描述。
- `seed=-1`：随机数生成器的种子，设置为 -1 表示使用默认的随机种子。
- `grid_size=1`：生成的图像的网格大小。

生成的图像被转换为 NumPy 数组，并使用 `plt.imshow` 在子图中显示，同时关闭坐标轴。打印生成的图像序号。

```python
plt.suptitle(text_input)
plt.tight_layout()
```

设置整个图形的标题为输入的文本描述，然后调整布局以确保子图之间的合适间距。

---

```python
# Text2Image2
import numpy as np
import matplotlib.pyplot as plt
import pickle
import glob
import pandas as pd
import os
from PIL import Image
import torch
import diffusers
from diffusers import VQDiffusionPipeline
from diffusers import StableDiffusionPipeline
import matplotlib.pyplot as plt

# loading pre-trained model
pipeline = VQDiffusionPipeline.from_pretrained("microsoft/vq-diffusion-ithq")
#pipeline = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
pipeline = pipeline.to("cuda")

# @title String fields
text_input = 'flying batman with colorful sky' # @param {type:"string"}
# e.g. 'flying animals with blue sky and white clouds'

plt.figure(figsize=(10,5))
max_images = 5
for x in range(max_images):
  prompt = text_input
  image = pipeline(prompt).images[0]
  plt.subplot(1,5,x+1); plt.imshow(image); plt.axis('off')
  print(x+1,'/',max_images,' --- generated')
plt.suptitle(text_input)
plt.tight_layout()
```
这段代码使用了 Diffusers 库中的 VQ-Diffusion 模型来生成图像，并使用 Matplotlib 对生成的图像进行可视化。下面是代码的逐行解释：

```python
import numpy as np
import matplotlib.pyplot as plt
import pickle
import glob
import pandas as pd
import os
from PIL import Image
import torch
import diffusers
from diffusers import VQDiffusionPipeline
from diffusers import StableDiffusionPipeline
import matplotlib.pyplot as plt
```

导入所需的库和模块，包括 NumPy、Matplotlib、Pickle、Glob、Pandas、OS、PIL、PyTorch、Diffusers 等。

```python
# loading pre-trained model
pipeline = VQDiffusionPipeline.from_pretrained("microsoft/vq-diffusion-ithq")
#pipeline = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
pipeline = pipeline.to("cuda")
```

加载预训练的 VQ-Diffusion 模型，或者也可以使用 Stable-Diffusion 模型（注释掉的部分）。模型被加载到 GPU 上进行推理。

```python
# @title String fields
text_input = 'flying batman with colorful sky' # @param {type:"string"}
# e.g. 'flying animals with blue sky and white clouds'
```

通过注释 (`# @title String fields`) 定义了一个文本输入字段，用户可以在这里输入一个描述图像的文本。在这个例子中，文本描述是 'flying batman with colorful sky'。

```python
plt.figure(figsize=(10,5))
max_images = 5
```

创建一个图形，并设定图形的大小。设置变量 `max_images` 为 5，表示生成 5 张图像。

```python
for x in range(max_images):
  prompt = text_input
  image = pipeline(prompt).images[0]
  plt.subplot(1,5,x+1); plt.imshow(image); plt.axis('off')
  print(x+1,'/',max_images,' --- generated')
```

通过循环生成图像，并在每次迭代中显示生成的图像。使用 `pipeline(prompt).images[0]` 函数生成图像，其中 `prompt` 是文本描述。生成的图像在 Matplotlib 的子图中显示，同时关闭坐标轴。打印生成的图像序号。

```python
plt.suptitle(text_input)
plt.tight_layout()
```

设置整个图形的标题为输入的文本描述，然后调整布局以确保子图之间的合适间距。

### 文本到视频

```python
import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video
from IPython.display import HTML
from base64 import b64encode
import datetime

pipe = DiffusionPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float16, variant="fp16")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()
pipe.enable_vae_slicing()

!mkdir -p /content/videos
prompt = 'flying bird' #@param {type:"string"}
negative_prompt = "low quality" #@param {type:"string"}
num_frames = 30 #@param {type:"raw"}
video_frames = pipe(prompt, negative_prompt=negative_prompt, num_inference_steps=25, num_frames=num_frames).frames
output_video_path = export_to_video(video_frames)

new_video_path = f'/content/videos/{datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")}.mp4'
!ffmpeg -y -i {output_video_path} -c:v libx264 -c:a aac -strict -2 {new_video_path} >/dev/null 2>&1

print(output_video_path, '->', new_video_path)

!cp {new_video_path} /content/videos/tmp.mp4
mp4 = open('/content/videos/tmp.mp4','rb').read()

decoded_vid = "data:video/mp4;base64," + b64encode(mp4).decode()
HTML(f'<video width=400 controls><source src="{decoded_vid}" type="video/mp4"></video>')
```
