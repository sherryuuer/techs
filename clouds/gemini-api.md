## Gemini API 互动初体验：只能说好喜欢

### 简短的GeminiAPI介绍

（模型是不断变化的，以下内容以今天时间2024年4月14日为准）

首先把最重要的资源贴出来：[GeminiAPI的官方Cookbook](https://github.com/google-gemini/cookbook)提供了完整的代码示例和文本。基本参照cookbook就可以玩了。

我自己玩的笔记本在[我的Github](https://github.com/sherryuuer/machine-learning-lab/tree/main/Gemini-api-pjs)中。

从官网可以免费获取api-key，设置好环境就可以直接上手。我比较懒没有放进os环境变量，直接从本地文本读取。抱着尝试一下的心态拿了key，看到官方给开发者是可以每天免费尝试50条的。执行以下代码可以看到包括以下模型清单。很快乐。

```python
import google.generativeai as genai

genai.configure(api_key=api_key)

# iterate through the list of available models
for m in genai.list_models():
    # check if the 'generateContent' method is supported by the model
    if 'generateContent' in m.supported_generation_methods:
        # if so, print the model name
        print(m.name)

# output：
# models/gemini-1.0-pro
# models/gemini-1.0-pro-001
# models/gemini-1.0-pro-latest
# models/gemini-1.0-pro-vision-latest
# models/gemini-1.5-pro-latest
# models/gemini-pro
# models/gemini-pro-vision
```

### 基本的游玩方式

简单的初始化模型，然后就可以进行提问了。英文自不必说，各种语言也都ok。

```python
# create a model instance
model = genai.GenerativeModel('gemini-pro')
# get response by a text input
response = model.generate_content('What do you think about Tokyo?')
response.text
```

Streaming方式的输出，加上option就可以一点一点的输出内容。PS：我要求他讲了一个故事，结果下面故事里小狼的爸爸是老鹰，哈哈这可能就是创造力的体验，另外，默认的温度是0.9，是一个很高的创意指数了。

```python
prompt = 'Write me a story about a little wolf'
response = model.generate_content(prompt, stream=True)
for chunk in response:
    print(chunk.text)
    print('-' * 100)

# output:
# In the heart of a sprawling forest, hidden amidst towering trees, lived a tiny
# ----------------------------------------------------------------------------------------------------
#  wolf pup named Luna. Her silver-gray fur shone like moonlight, and her bright blue eyes sparkled with mischief.

# Luna was the youngest in her pack
# ----------------------------------------------------------------------------------------------------
# , and the most curious. She loved exploring the forest, discovering its hidden nooks and crannies. One sunny afternoon, as she was wandering through a dense undergrowth, a peculiar scent caught her attention.

# It was the sweet fragrance of honey. Luna followed her nose until she stumbled upon a towering beehive.
# ----------------------------------------------------------------------------------------------------
#  Bees buzzed around the hive, guarding their golden treasure.

# Undeterred, Luna crept closer, her claws scraping against the rough bark of the tree. She extended her long tongue and cautiously dipped it into the honeycomb. To her delight, the honey was sticky and sweet.

# As she feasted on the honey, Luna became oblivious to the danger she was in. The queen bee, furious at her uninvited guest, summoned her loyal subjects to attack.

# Luna squealed in surprise as a swarm of bees descended upon her. She frantically dodged their stingers, but they were relentless. Just when she thought all hope was lost
# ----------------------------------------------------------------------------------------------------
# , a shadow passed overhead.

# With a powerful swoop, a majestic eagle swooped down and scattered the bees. It was Luna's father, the Alpha of the pack. He had been watching over her from afar and had rushed to her aid.

# Luna was overjoyed to be safe. She nuzzled her father's neck, grateful for his protection. The Alpha led her back to the den, where her siblings greeted her with warm licks and playful nips.

# From that day forward, Luna learned a valuable lesson. Curiosity may lead to adventure, but it is always important to be mindful of the dangers that lurk in the shadows. And so, the little wolf continued to explore the forest, but always with her father's watchful gaze nearby.
# ----------------------------------------------------------------------------------------------------
```

### 惊喜的图像内容输出

我输入了一张图片，我并没有用日文提问，但是这是一张在日本六本木拍的图，我使用英文提问，它给了我日文的回答。严重怀疑它读取了我图片的背后信息。在图片描述的提问中，也非常准确，快速返回了，内容，甚至包括模糊不清的背景。

```python
# create a model instance from gemini pro vision
model = genai.GenerativeModel('gemini-pro-vision')
response = model.generate_content(img)
response.text

# output:
# ' 夜景を眺めながら飲むカクテルは最高ですね。'

prompt = 'list all the object in the picture.'
response = model.generate_content([prompt, img])
print(response.text)

# output
# 1. Martini glass
# 2. Cocktail
# 3. Coaster
# 4. Table
# 5. Plate
# 6. Bowl of nuts
# 7. Lamp
# 8. Window
# 9. Cityscape
```

### 参数调节

通过打印`help(genai.types.GenerationConfig)`可以找到官方帮助。

通过打印`genai.get_model('models/gemini-pro')`则可以看到模型的默认参数。

或者可以像下面这样自己来定义。包括输出的回答数量，终止符号，最大令牌数量，温度，以及topk，topp等常见参数。

可以在model初始化的时候就定义参数项目，或者可以针对具体的提问进行定义。

```python
# define a GenerationConfig object (default ⬇)
generation_config = genai.types.GenerationConfig()
# with different config options
# generation_config = genai.types.GenerationConfig(
#     canditate_count=1,
#     stop_sequence=[','], # stop when the response meet a ','
#     max_output_tokens=32000, # output token limit
#     temperature=0.9, # 0 is steady and 1 is creativity
#     top_p=1,
#     top_k=1
# )

# 2 way use the config
# define on the model instance
model = genai.GenerativeModel('gemini-pro', generation_config=generation_config)
# define on the single prompt
prompt = 'Write me a little song about Gemini.'
response = model.generate_content(prompt, generation_config=generation_config)
```

### 项目1:对目标文件夹内的所有图片进行重命名

```python
import os
from pathlib import Path
from PIL import Image

# creata image generator to save the memory
def get_images(dir):
    supported_extensions = ('.png', '.jpg', '.jpeg')
    for root, subdirs, filenames in os.walk(dir):
        for file in filenames:
            if file.lower().endswith(supported_extensions):
                absolute_path = os.path.join(root, file)
                img = Image.open(absolute_path)
                yield img, absolute_path
            
model = genai.GenerativeModel('gemini-pro-vision')
# prompt clearly explain renaming job
prompt = '''
Analyze the image in detail.
Generate a descriptive image filename using only these rules:
* Relevant key words describe the image, seperated by underscores.
* Lowercase letters only.
* No special characters.
* Keep it short and accurate.
* Response only with the image filename, no extensions.

Example: cat_running_in_the_garden
'''
my_directory = 'images'

for img, absolute_path in get_images(my_directory):
    response = model.generate_content([prompt, img])
    root, ext = os.path.splitext(absolute_path)  # split the path into path and '.extension'
    new_filename = response.text.strip() + ext
    base_dir = os.path.dirname(absolute_path)
    new_filepath = base_dir + '/' + new_filename
    
    os.rename(absolute_path, new_filepath)
    print(f'{absolute_path} -> {new_filepath}')
    print('*' * 100)
```

然后他给了我满意的输出：它认出了我的小猫，还认出了很多人都不认识的我最爱的游戏女神异闻录5～（图片可以在我的hub里看到）

```python
# images/AF9BCB7D-4CDC-4599-ACE3-F22EF013A730_1_105_c.jpeg -> images/cat_looking_at_camera.jpeg
# ****************************************************************************************************
# images/827E5A87-C7D9-4151-AF60-591AA419F6E4_1_105_c.jpeg -> images/gaming_setup_persona_5.jpeg
# ****************************************************************************************************
```

### 项目2:聊天机器人

之需要定义`model.start_chat(history=[])`就可以像提示一样的开始对话了。

chat类型的示例，重点在于它把回答存放在了一个 history 的列表中，每次都可以参考 history 进行回答。

```python
import time
model = genai.GenerativeModel('gemini-pro')
chat = model.start_chat(history=[])
while True:
    prompt = input('User: ')
    if prompt.lower() not in ['exit', 'bye', 'quit']:
        response = chat.send_message(prompt)
        print(f'{chat.history[-1].role.capitalize()} : {chat.history[-1].parts[0].text}')
        print('\n' + '*' * 100 + '\n')
    else:
        print('Quitting...')
        time.sleep(2)
        print('I will missing u, see you!')
        break
```

### Recap：

别的不说，但是一天可以让你测试50条就是一个美好的举动。去尝试一下openai的api必须要氪金，连测试的机会都不给。langchain是一个强大的框架，还没有尝试过结合Gemini玩，有空可以试试看。

我还是喜欢谷歌。
