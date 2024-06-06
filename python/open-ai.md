## Python + OpenAI

---
### 调用API的方法

首先当然是需要一个apikey。

```python
import openai

openai.api_key = "{{SECRET_KEY}}"
try:
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt="Write a tagline about artificial intelligence.",
        temperature=0.9
    )
    print(response)
    
except Exception as e:
    print(f"Error: {e}")
```

### 以下是基于Completions Endpoint的玩法
### 分类任务

分类的时候使用关键字Classify提示，然后给他相应标签。最后的Sentiment相当于给他一个答案的开头。

```python
import openai

openai.api_key = "{{SECRET_KEY}}"
prompt_text = ("Classify the Text's label as positive, neutral, or negative.\n" 
                + "Text: I loved the new Spiderman movie!\n"
                + "Sentiment:")

try:
    response = openai.Completion.create(
        model="text-davinci-002",
        prompt=prompt_text
    )

    print("Prompt text:\n" + prompt_text + response.choices[0].text)
    
except Exception as e:
    print(f"Error: {e}")
```

**尝试设置较高的温度**，温度调节为0到1之间，越高越有创意，越大胆。

```python
import openai

openai.api_key = "{{SECRET_KEY}}"
prompt_text = ("Classify the Text.\n" 
                + "Text: I loved the new Spiderman movie!\n")

try:
    response = openai.Completion.create(
        model="text-davinci-002",
        prompt=prompt_text,
        temperature=0.9,
        top_p=1
    )
    print("Prompt text:\n" + prompt_text + response.choices[0].text)
    
except Exception as e:
    print(f"Error: {e}")
```

解释`response.choices[0].text`:

1. `response`: 这是通过调用 OpenAI API 后得到的响应对象。它包含了从模型返回的所有信息，包括生成的文本、模型的置信度等。

2. `response.choices`: 这是响应中的一个属性，表示模型的选择列表。在这里，模型只返回了一个选择（最可能的生成文本），因此我们可以通过 `response.choices[0]` 来访问这个唯一的选择。

3. `response.choices[0].text`: 这是选择中的一个属性，表示模型生成的文本。它包含了模型认为是最适合请求的文本部分。



**尝试进行多条文本分类**

```python
import openai

openai.api_key = "{{SECRET_KEY}}"
try:
    prompt_text = ("Classify the sentiment in these tweets:\n"
                    + "1. \"I enjoy this program\"\n"
                    + "2. \"I don't like him anymore\"\n"
                    + "3. \"I can't wait for Halloween!!!\"\n"
                    + "4. \"I do not hate much jokers\"\n"
                    + "5. \"I love eating chocolates <3!\"\n"
                    + "Tweet sentiment ratings:")

    response = openai.Completion.create(
        model="text-davinci-002",
        prompt=prompt_text,
        max_tokens = 60,
        temperature=0.9
    )
    print("Prompt text:\n" + prompt_text)
    print(response.choices[0].text)
    
except Exception as e:
    print(f"Error: {e}")
```

max_tokens的默认是16，这里肯定会超出限制，所以改为60

**给项目分类**因为他擅长将项目分为离散类别。

```python
import openai

openai.api_key = "{{SECRET_KEY}}"
prompt_text = "The following is a list of companies and the categories they fall into:\n\nApple, Facebook, Educative\n\nApple\nCategory:"

try:
  response = openai.Completion.create(
    model="text-davinci-002",
    prompt=prompt_text,
    temperature=0,
    max_tokens=64,
    top_p=1.0,
  )

  print("Prompt: ", prompt_text, response.choices[0].text)

except Exception as e:
    print(f"Error: {e}")
```

### 生成任务

**提示以生成文本**

```python
import openai

openai.api_key = "{{SECRET_KEY}}"
prompt_text = "Brainstorm some ideas combining deep learning and images:\n"

try:
    response = openai.Completion.create(
        model="text-davinci-002",
        prompt=prompt_text,
        temperature=0.75,
        max_tokens=200,
        top_p=1,
        frequency_penalty=1,
        presence_penalty=1
    )
    print ("Prompt text:\n" + prompt_text)
    print(response.choices[0].text)
    
except Exception as e:
    print(f"Error: {e}")
```

max_tokens的最大长度就是200了。

在 OpenAI GPT 模型的生成请求中，`frequency_penalty` 和 `presence_penalty` 是两个用于调整生成文本行为的参数：

1. **`frequency_penalty`：**
   - **定义：** 控制模型生成重复单词的倾向。值越大，模型越不愿意生成已经出现过的单词。如果希望生成文本更多地包含不同的单词，可以增加这个值。
   - **范围：** 通常在 0 到正无穷大之间，1 表示默认行为。

2. **`presence_penalty`：**
   - **定义：** 控制模型生成新颖短语或思想的倾向。值越大，模型越不愿意生成已经在输入中出现过的内容。如果你希望生成文本更加新颖，可以增加这个值。
   - **范围：** 通常在 0 到正无穷大之间，1 表示默认行为。

### 转换任务

文本翻译啦，总结文本啦，文本转换为emoji啦都是这类任务。

**文本翻译**

```python
import openai

openai.api_key = "{{SECRET_KEY}}"
prompt_text = ("Translate this into 1. French, 2. Spanish and 3. Japanese.\n" +
              "We provide an online learning platform made by developers, created for developers.")

try:
      response = openai.Completion.create(
            model="text-davinci-002",
            prompt=prompt_text,
            temperature=0,
            max_tokens=100,
      )
      print("Prompt: \n" + prompt_text)
      print(response.choices[0].text)
      
except Exception as e:
      print(f"Error: {e}")
```

这里温度设置为0表示不需要翻译的随机性。max_tokens的设置是因为知道会输出多长。

**内容提炼**

```python
import openai
import textwrap

openai.api_key ="{{SECRET_KEY}}"
prompt_text = ("Summarize this for a second-grade student:\nJupiter is the fifth planet from the "+ 
               "Sun and the largest in the Solar System. It is a gas giant with a mass " + 
               "one-thousandth that of the Sun, but two-and-a-half times that of all the " +
               "other planets in the Solar System combined. Jupiter is one of the brightest " + 
               "objects visible to the naked eye in the night sky, and has been known to ancient " +  
               "civilizations since before recorded history. It is named after the Roman god " + 
               "Jupiter. When viewed from Earth, Jupiter can be bright enough for its " + 
               "reflected light to cast visible shadows, and is on average the third-brightest " +
               "natural object in the night sky after the Moon and Venus.")

try:
    response = openai.Completion.create(
        model="text-davinci-002",
        prompt=prompt_text,
        temperature=0.9,
        max_tokens=64,
        frequency_penalty=1,
        presence_penalty=1
    )
    print("Prompt:")
    print (textwrap.fill(prompt_text, width=80))
    print("\nSummary:")
    print (textwrap.fill(response.choices[0].text, width=80))

except Exception as e:
    print(f"Error: {e}")
```

**转换文本为emoji**

```python
import openai

openai.api_key = "{{SECRET_KEY}}"

try:
    prompt_text = ("Convert movie titles into emojis.\n" +
                "Back to the Future: 👨👴🚗🕒\n" +
                "Batman: 🤵🦇\n" +
                "Transformers: 🚗🤖\n" +
                "Antman: ")

    response = openai.Completion.create(
        model="text-davinci-002",
        prompt=prompt_text,
        temperature=0.9,
    )
    print("Prompt: \n" + prompt_text)
    print(response.choices[0].text)

except Exception as e:
    print(f"Error: {e}")
```

### 文本插入

给定前缀和后缀文本，进行文本插入。

```python
import openai
import textwrap

openai.api_key = "{{SECRET_KEY}}"

try:
  response = openai.Completion.create(
    model="text-davinci-002",
    prompt="Educative is a leading online learning platform made by developers, created for developers.",
    suffix="Educative helps software engineers grow their skill sets and reach their full potential.",
    max_tokens=260
  )

  print("Insertion result:")
  print(textwrap.fill(response.choices[0].text, width=80))

except Exception as e:
    print(f"Error: {e}")
```

另一个例子：

```python
import openai
import textwrap

openai.api_key = "{{SECRET_KEY}}"

prompt_prefix = "How to lose weight:\n1. Do not skip breakfast."
prompt_suffix = "12. Plan your meals"
try:
  response = openai.Completion.create(
    model="text-davinci-002",
    prompt=prompt_prefix,
    suffix=prompt_suffix,
    max_tokens=400
  )

  print("\nPrefix:\n", prompt_prefix)
  print("\nInsertion result:")
  print(textwrap.fill(response.choices[0].text, width=80))
  print("\nSuffix:\n", prompt_suffix)

except Exception as e:
    print(f"Error: {e}")
```

### 句子完成

只给一个句子开头，他会帮你完成。

```python
import openai

openai.api_key = "{{SECRET_KEY}}"
prompt_text = "Although the tiger is a solitary beast and "

try:
  response = openai.Completion.create(
    model="text-davinci-002",
    prompt=prompt_text,
    temperature=0.20,
    max_tokens=64,
    top_p=1,
  )

  print("Prompt: " + prompt_text)
  print("\nCompletion: " + response.choices[0].text)

except Exception as e:
  print(f"Error: {e}")
```

如果想告诉他，比如当他不知道该怎么做的时候，给出具体的回答方法。

```python
import openai

openai.api_key = "{{SECRET_KEY}}"
prompt_text=("Q: Who is Batman?\n"+
            "A: Batman is a fictional comic book character.\n"+
            "Q: What is torsalplexity?\n"+
            "A: ?\n"+
            "Q: What is Devz9?\n"+
            "A: ?\n"+
            "Q: Who is George Lucas?\n"+
            "A: George Lucas is American film director and producer famous for creating Star Wars.\n"+
            "Q: What is the capital of California?\n"+
            "A: Sacramento.\n"+
            "Q: What orbits the Earth?\n"+
            "A: The Moon.\n"+
            "Q: Who is Fred Rickerson?\n"+
            "A: ?\n"+
            "Q: What is an atom?\n"+
            "A: An atom is a tiny particle that makes up everything.\n"+
            "Q: Who is Alvan Muntz?\n"+
            "A: ?\n"+
            "Q: What is Kozar-09?\n"+
            "A: ?\n"+
            "Q: How many moons does Mars have?\n"+
            "A: Two, Phobos and Deimos.\n"+
            "Q: what is the meaning of xyzz?\n"+
            "A:")

try:
  response = openai.Completion.create(
    model="text-davinci-002",
    prompt=prompt_text,
    temperature=0,
    max_tokens=64,
  )
  
  print("Prompt: \n" + prompt_text)
  print("Response: \n" + response.choices[0].text)

except Exception as e:
  print(f"Error: {e}")
```

### 文本编辑的Endpoint

```python
response = openai.Edit.create(
  model="<engine_id>",
  input="<string>",
  instruction="<string>",
  ...
)
```

**比如语法修正**

```python
import openai

openai.api_key = "{{SECRET_KEY}}"

try:
  response = openai.Edit.create(
    model="text-davinci-edit-001",
    input="Educative is a leading online learning platform made by developers, create for developers.",
    instruction="Fix the grammar.",
    temperature=0,
    top_p=1
  )

  print(response)

except Exception as e:
  print(f"Error: {e}")
```

**语态修正**

```python
import openai

openai.api_key = "{{SECRET_KEY}}"

try:
  response = openai.Edit.create(
    model="text-davinci-edit-001",
    input="The kangaroo carries her baby in her pouch.",
    instruction="Convert sentence to passive voice",
    temperature=0,
    top_p=1
  )

  print(response)

except Exception as e:
  print(f"Error: {e}")
```

### 嵌入Embedding模型

文本相似度（回归，可视化，聚类，异常检测），文字搜索（上下文检索，搜索相关），代码搜索（搜索相关代码）

相关模块：

```python
response = openai.Embedding.create(
  input="The text whose embeddings are required",
  engine="<engine_id>"
)
```

**文本相似度**

```python
import openai
openai.api_key = "{{SECRET_KEY}}"

try:
  response = openai.Embedding.create(
    input="The burger was fantastic!",
    engine="text-similarity-babbage-001"
  )

  embedding = response.data[0].embedding
  print("Embedding: ", embedding)
  print("Length: ", len(embedding))

except Exception as e:
  print(f"Error: {e}")
```

response.data[0].embedding可以从响应对象中提取嵌入向量。

**计算嵌入向量的余弦相似度**

```python
import openai
from numpy import dot

openai.api_key = "{{SECRET_KEY}}"
try:
    response = openai.Embedding.create(
        input=["The burger was fantastic", "The quick brown fox jumps over the lazy dog"],
        engine="text-similarity-davinci-001")

    emb_1 = response['data'][0]['embedding']
    emb_2 = response['data'][1]['embedding']

    # Formula to find the cosine similarity
    similarity_score = dot(emb_1, emb_2)
    print(similarity_score)

except Exception as e:
    print(f"Error: {e}")
```
