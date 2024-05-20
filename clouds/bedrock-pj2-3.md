## Amazon Bedrock 项目实践2-3: Notes Summarisation & Image Generation & 模型评估 & RAG

---
### 项目介绍

关于 Bedrock 的介绍和讲解以及注意事项，还有参数设置的讲解，都在上一篇第一个项目的篇头。

关于这两个项目，一个是文本总结的项目，代码生成项目是通过 prompt 的短序列生成代码长序列，这里的项目就是一个长序列生成短序列的过程，不仅如此，还可以针对提供的文本进行提问，生成自己的文本库，本质上来说，有点微调的感觉。另一个是图像生成项目，构架大同小异。

### Notes Summarisation 文本总结项目

项目构架来说包括如下service：

- API Getway ：使用 post 方法作为trigger。
- Lambda Function ：使用代码呼出 Bedrock 的API。
- S3 ：从 Bedrock 输出的总结内容会存入 S3。

**Lambda：**

在权限设置部分依然是测试环境，所以暂且不做拘泥，设置 timeout 为5分钟就可以，然后编写使用的代码。

```python
import boto3
import botocore.config
import json
import base64
from datetime import datetime
from email import message_from_bytes


# 提取文本
def extract_text_from_multipart(data):
    msg = message_from_bytes(data)

    text_content = ''

    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                text_content += part.get_payload(decode=True).decode('utf-8') + "\n"

    else:
        if msg.get_content_type() == "text/plain":
            text_content = msg.get_payload(decode=True).decode('utf-8')

    return text_content.strip() if text_content else None

# 调用API生成总结（可以从第一个项目copy和修改
def generate_summary_from_bedrock(content:str) ->str:
    prompt_text = f"""Human: Summarize the following meeting notes: {content}
    Assistant:"""

    body = {
        "prompt":prompt_text,
        "max_tokens_to_sample":5000,
        "temperature":0.1,
        "top_k":250,
        "top_p":0.2,
        "stop_sequences": ["\n\nHuman:"]
    }

    try:
        bedrock = boto3.client("bedrock-runtime",region_name="us-west-2",config = botocore.config.Config(read_timeout=300, retries = {'max_attempts':3}))
        response = bedrock.invoke_model(body=json.dumps(body),modelId="anthropic.claude-v2")
        response_content = response.get('body').read().decode('utf-8')
        response_data = json.loads(response_content)
        summary = response_data["completion"].strip()
        return summary

    except Exception as e:
        print(f"Error generating the summary: {e}")
        return ""

# 存储总结结果到s3桶（可以从第一个项目copy和修改
def save_summary_to_s3_bucket(summary, s3_bucket, s3_key):

    s3 = boto3.client('s3')

    try:
        s3.put_object(Bucket = s3_bucket, Key = s3_key, Body = summary)
        print("Summary saved to s3")

    except Exception as e:
        print("Error when saving the summary to s3")

# 主程序
def lambda_handler(event,context):

    decoded_body = base64.b64decode(event['body'])

    text_content = extract_text_from_multipart(decoded_body)

    if not text_content:
        return {
            'statusCode':400,
            'body':json.dumps("Failed to extract content")
        }


    summary = generate_summary_from_bedrock(text_content)

    if summary:
        current_time = datetime.now().strftime('%H%M%S') #UTC TIME, NOT NECCESSARILY YOUR TIMEZONE
        s3_key = f'summary-output/{current_time}.txt'
        s3_bucket = 'bedrock-bucket'

        save_summary_to_s3_bucket(summary, s3_bucket, s3_key)

    else:
        print("No summary was generated")


    return {
        'statusCode':200,
        'body':json.dumps("Summary generation finished")
    }
```

最后记得点 deploy 发布。

和第一个项目相同，这里也需要最新的boto3的layer，所以点击 *add layer*，将之前的项目创建的 layer 加入到现在的 function 上去。（layer之需要 create 一次，就可以在 add 在之后的各种 function 上了。 

**S3：**如上代码中的 bedrock-bucket 为bucket的名字，和代码统一。

**API Gateway：**

API endpoint的立刻搭建，不需要写任何代码，就可以立刻享有，非常方便。这里使用HTTP API。

1，编辑 Routes：

设置一个 `POST` 方法，输入 `/meeting-summary`。当使用了这个方法，需要内部启动 Lambda Function。

在 details 设置中，Attach authorization 是指将授权设置（authorization settings）应用到 API Gateway 中的特定端点或 API。这个操作允许你将之前设置的授权方式（如 API Key、IAM 角色、AWS Cognito 用户池等）与特定的 API 端点关联起来，从而限制谁可以访问该端点或 API，并授予访问者相应的权限。这里可以先不设置。方便访问。

Integrations 设置中，要进行对 lambda function 的设置。在 target 选项卡中选择 Lambda Function，然后选择对应的 region， 和对应的 lambda 函数名字就可以了。

然后在 APIs 列表中就可以看到刚刚设置的 API，所生成的 URL 链接。

2，发布 Deploy：

最好的方式是创建 stage，在 Deploy 中标签中选择 stage 创建一个 dev，并且不要勾选自动发布（default的是自动发布的）。

然后就可以去 Routes 的右上角点击 Deploy 然后发布对应的 dev stage 的API了。这时候列表中会出现一个针对 dev 环境的新的 URL。

3，post 测试：

可以使用 postman 等API 测试工具，对端点 POST 前半段链接省略/dev/meeting-summary 发送一个文件对象。

### Image Generation 图像生成

这个项目构架差不多一样，主要关注代码，和一些不同点。

**Lambda：**

```python
import json
import boto3
import botocore
from datetime import datetime
import base64

def lambda_handler(event, context):

    event = json.loads(event['body'])
    message = event['message']

    bedrock = boto3.client("bedrock-runtime",region_name="us-west-2",config = botocore.config.Config(read_timeout=300, retries = {'max_attempts':3}))

    s3 = boto3.client('s3')

    payload = {
        "text_prompts":[{f"text":message}],
        "cfg_scale":10,
        "seed":0,
        "steps":100
    }

    response = bedrock.invoke_model(body=json.dumps(payload),modelId = 'stability.stable-diffusion-xl-v0',contentType = "application/json",accept = "application/json")

    response_body = json.loads(response.get("body").read())
    base_64_img_str = response_body["artifacts"][0].get("base64")
    image_content = base64.decodebytes(bytes(base_64_img_str,"utf-8"))

    bucket_name = 'bedrock-bucket'
    current_time = datetime.now().strftime('%H%M%S')
    s3_key = f"output-images/{current_time}.png"

    s3.put_object(Bucket = bucket_name, Key = s3_key, Body = image_content, ContentType = 'image/png')



    return {
        'statusCode': 200,
        'body': json.dumps('Image Saved to s3')
    }
```

**API Gateway：**

POST 端点为 /image-generation。其他的和上面的项目步骤一致。

deploy 的 stage 使用 dev。dev在做第一个项目的时候 create 过了可以复用。

### 项目小结

- 大体上所有的 debug 都可以在 CW 中进行。
- 通过 API 可以呼出 bedrock 模型进行使用。

### 其他一些资料：Bedrock 模型评估功能

在服务的左侧tag，又一个可以进行LLM模型评估的功能。

现在虽然还是preview版本，但是可以测试一些已经有的模型。

设置的时候需要选择的是模型评估标准，模型的使用数据，以及最后的输出位置，S3等。然后就可以进行执行了。

输出的结果是一个文件，其中包含了进行评估进行的prompt，正确答案，以及模型的回答等。可以检测模型等效果如何。

是一个很不错的功能，因为我们自己很难像一般的分类模型一样，去评估大语言模型。

官方文档：

https://docs.aws.amazon.com/zh_cn/bedrock/latest/userguide/what-is-bedrock.html

### Konwledge base & RAG 知识库和RAG项目

RAG（Retrieval-Augmented Generation）是一种用于自然语言处理的模型架构，它结合了检索（retrieval）和生成（generation）两种技术，旨在提高文本生成任务的性能和效果。RAG 的主要组件包括：

1. **Retriever（检索器）：** 检索器负责从大规模的文本语料库中检索与给定输入相关的信息。它可以使用各种技术，如倒排索引、BM25 算法等，来高效地找到与输入相关的文本片段或文档。

在这里，大规模语料库就是我们要创建的 Knowledge base 向量仓库，也是一个网页上点点就可以创建的服务，数据源来自 S3，将要嵌入的文本放进 S3，**选择要用来 embedding 的模型，然后选择向量数据库**，如果使用官方的 OpenSearch 数据库，可能会比较贵，但是可以一键完成设置，氪金玩家请自便。创建知识库的过程一般需要几分钟。

当创建完成后，进入知识库，在下面的 data source 部分点击同步 sync 按钮，这时候开始将S3中的文本和向量数据库同步。不管你删除了还是增加了文本都需要进行同步操作。完成后进入 OpenSearch 查看向量数据库，发现已经存储了嵌入向量数据了。

这时候知识库建立完毕后，在知识库服务的右手边就会有对话框了，可以尽心 Query ，使用大语言模型进行检索。如果关闭（生成答案）的标签，就会返回相关的文档这是一个检索的功能，这样做就需要人自己找到答案，但是相对来说比较便宜。

2. **Reader（阅读器）：** 阅读器负责从检索器返回的文本中理解和提取信息。它可以是一个简单的模型，用于抽取关键信息，也可以是一个更复杂的模型，如预训练的语言模型，用于理解和推理文本。

这里就是要用于生成答案的过程了，你问它答，就需要它去对检索回来的文本进行阅读和提取信息的过程。

3. **Generator（生成器）：** 生成器负责根据检索器返回的文本和阅读器提取的信息生成最终的输出。它可以是一个基于规则的系统，也可以是一个深度学习模型，如循环神经网络（RNN）、变换器（Transformer）等。

最后它也是用一个模型，生成我们要的答案了。

在这个页面中的图，就描述了 Bedrock 知识库的工作过程。很清晰易懂：

https://docs.aws.amazon.com/zh_cn/bedrock/latest/userguide/kb-how-it-works.

另外在页面中进行 RAG 提问的操作，全都可以用 boto3 API进行：

https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock-agent-runtime.html

注意：如果是项目联系，记得删除知识库和opensearch，os非常贵，一个月几百刀。


