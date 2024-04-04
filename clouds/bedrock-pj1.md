## Amazon Bedrock 项目实践1: Code generation

---
### Bedrock介绍

Amazon Bedrock 是一项完全托管的服务，可简化生成式人工智能 (Generative AI) 应用程序的开发。它提供以下功能：

- 来自领先 AI 公司的各种高性能基础模型 (FM)，包括 AI21 Labs、Anthropic、Cohere、Meta、Stability AI 和 Amazon 自己的模型等。
- 构建生成式 AI 应用程序所需的一系列广泛功能，包括以下内容。
- 微调：使用自己的数据自定义模型输出。
- 检索增强生成 (RAG)：将知识和数据整合到模型中。
- 托管代理：将模型部署到生产环境。
- 安全性、隐私性。输入的数据不会离开AWS的加密系统，所以私人数据是安全的。

Bedrock 可用于构建各种生成式 AI 应用程序，包括：聊天机器人，内容生成，代码生成，翻译，问答，创意写作，数据分析，等等。

目前（2024年4月）进入服务页面，会有部分区域可用部分区域不可用，可用区域的模型也不一样。所以可以多探索一下。一开始的模型也不是立刻可用状态，需要在左侧找到model列表，然后点击编辑，全选模型，更新请求使用的按钮后方可使用。

在他的游乐场中有很多可以玩的模型，并且给了详细的示例，可以随便逛逛看看什么样的prompt会生成什么样的数据。

**关于参数调节：**

在游玩游乐场中的模型的时候有很多参数可以调节，比如温度，topK采样，topP采样等。他们在内部是如何实现的，我突然很感兴趣所以查了一下。他们都是通过调节模型输出的 *概率分布* 来控制生成结果的多样性和质量。这些技术通常应用于基于概率的生成模型，比如语言模型，其中模型会为下一个词的生成提供一个概率分布。

温度调节是指通过调节模型输出的概率分布的熵来控制生成结果的多样性。具体来说，温度参数 τ 越大，模型输出的概率分布的熵越大，生成的结果越多样化；反之，温度参数越小，生成的结果越趋于确定性。温度调节的内部原理是在模型输出的概率分布上应用一个 softmax 操作，并且通过除以温度参数 τ 来缩放分布的值，然后再进行 softmax 操作，得到新的概率分布。

Top-k 采样是一种用于调节生成结果质量的方法，它通过保留概率最高的前 k 个词来限制生成结果的数量，以及在这 k 个词上重新归一化概率分布，从而提高生成结果的质量。Top-k 采样的内部原理是在模型输出的概率分布上应用一个截断操作，只保留概率最高的前 k 个词，并在这些词上 *重新归一化* 概率分布。这样可以避免生成一些低概率的、不合理的词语。具体来说，对于每个词的概率分布，只保留概率最高的前 k 个词，其他的概率设为0，然后在这 k 个词上重新归一化概率分布。

Top-P 采样（也称为Nucleus采样）是一种与Top-k 采样类似的方法，它也用于控制生成结果的多样性和质量。在Top-P 采样中，不是保留概率最高的前k个词，而是保留累积概率大于 *阈值P* 的词。

具体来说，给定一个阈值P（通常在0和1之间），Top-P 采样会保留累积概率大于P的词，然后在这些词上重新归一化概率分布。这样可以根据生成任务的需要动态地调整词汇量，以保持一定的多样性。

Top-P 采样相比于固定的Top-k值，更加灵活，能够根据上下文动态地选择词汇量，因此在一些生成任务中被广泛使用。

### Code generation PJ

项目构架来说包括如下service：

- API Getway ：使用 post 方法作为trigger。
- Lambda Function ：使用代码呼出 Bedrock 的API。
- S3 ：从 Bedrock 输出的代码会存入 S3。

**Lambda：**

在设置权限部分，需要Function对Bedrock和S3，以及CW日志输出享有权限。如果是自己的小项目可以赋予更高的权限，让整个过程更加流畅，在实际项目中，需要设置真正需要的一部分权限，才是最佳实践。当然一开始没设置全没关系，在test阶段可以进行补充。

```python
import boto3
import boto3core.config
import json
from datetime import datetime


def generate_code_using_bedrock(message: str, language: str):
    # 这里最好是 Human 立刻跟在引号后面
    prompt_text = f"""Human: Write {language} code for the following instruction: {message}.
    Assistant:
    """
    body = {
        "prompt": prompt_text,
        "max_tokens_to_sample": 2048,
        "temperature": 0.1,
        "top_k": 250,
        "top_p": 0.2,
        "stop_sequences": ["\n\nHuman:"]
    }

    try:
        bedrock = boto3.client(
            "bedrock-runtime",
            region_name="us-west-2",
            config=botocore.config.Config(
                read_timeout=300, retries={"max_attempts": 3})
        )
        response = bedrock.invoke_model(
            body=json.dumps(body),
            modelId="anthropic.claude-v2"
        )
        response_content = response.get("body").read().decode("utf-8")
        response_data = json.loads(response_content)
        code = response_data["completion"].strip()
        return code

    except Exception as e:
        print(f"Error generating the code: {e}")
        return ""


def save_code_to_s3_bucket(code, s3_bucket, s3_key):

    s3 = boto3.client("s3")

    try:
        s3.put_object(Bucket=s3_bucket, Key=s3_key, Body=code)
        print("Code saved to s3")

    except Exception as e:
        print("Error when saving the code to s3")


def lambda_handler(event, context):

    event = json.loads(event["body"])
    message = event["message"]
    language = event["key"]
    print(message, language)

    generated_code = generate_code_using_bedrock(message, language)

    if generated_code:
        # UTC time
        current_time = datetime.now().strftime("%H%M%S")
        s3_key = f"code-output/{current_time}.py"
        s3_bucket = "bedrock-bucket"

        save_code_to_s3_bucket(generated_code, s3_bucket, s3_key)

    else:
        print("No code was generated")

    # 这里在实际生产中，返回的内容要根据实际的运行结果，这里做了简化
    return {
        "statusCode": 200,
        "body": json.dumps("Code generation")
    }
```

最后记得点 deploy 发布。

**S3：**如上代码中的 bedrock-bucket 为bucket的名字，和代码统一就好。

**API Gateway：**

API endpoint的立刻搭建，不需要写任何代码，就可以立刻享有，非常方便。这里使用HTTP API。

1，编辑 Routes：

设置一个 `PUT` 方法，输入 `/code-generation`。当使用了这个方法，需要内部启动 Lambda Function。

在 details 设置中，Attach authorization 是指将授权设置（authorization settings）应用到 API Gateway 中的特定端点或 API。这个操作允许你将之前设置的授权方式（如 API Key、IAM 角色、AWS Cognito 用户池等）与特定的 API 端点关联起来，从而限制谁可以访问该端点或 API，并授予访问者相应的权限。这里可以先不设置。方便访问。

Integrations 设置中，要进行对 lambda function 的设置。在 target 选项卡中选择 Lambda Function，然后选择对应的 region， 和对应的 lambda 函数名字就可以了。

然后在 APIs 列表中就可以看到刚刚设置的 API，所生成的 URL 链接。

2，发布 Deploy：

最好的方式是创建 stage，在 Deploy 中标签中选择 stage 创建一个 dev，并且不要勾选自动发布（default的是自动发布的）。

然后就可以去 Routes 的右上角点击 Deploy 然后发布对应的 dev stage 的API了。这时候列表中会出现一个针对 dev 环境的新的 URL。

3，put 测试：

可以使用 postman 等API 测试工具，发送一个 put 对象。根据代码`lambda handler`，我们发送如下内容：

```json
{
    "message": "implement the binary search",
    "key": "python"
}
```

**debug 可能会出现的错误：**

一种 bug 可能是 timeout，因为 lambda 的初始设定是3秒。所以要在设置中调节运行的 timeout 时间。一种 bug 可能是拼写错误，记得修改了代码需要再次点击 deploy 按钮。

还有一种 bug：确实输出了"Code generation"，但是结果没有被保存进 S3。通过 Lambda 的 moniter 可以进入 CW 查看日志特定错误原因。

结果发现，提示没有 bedrock runtime，这是因为 lambda 中的 boto3 版本太低没发跑最新的这个服务，如果是早期的时候很有可能遇到这个问题。

**解决办法：给Lambda function添加新的layer**

第一步，通过以下内容制作一个 zip 包，用于上传 lambda 作为新的 layer。

```
# create a new directory for the layer
mkdir boto3_layer
cd boto3_layer

# create a python directory.(AWS Lambda expects Python packages in a python directory for python runtimes)
mkdir python

# create a virtual environment
python3 -m venv venv

# activate the virtural environment
source venv/bin/activate

# install the latest boto3 into the python directory
pip install boto3 -t ./python

# deactivate the virtual environment
deactivate

# zip the package
zip -r boto3_layer.zip ./python

# optionally, remove the virtual environment after packaging
rm -r venv
```

解释说明，其实就是执行以下步骤： 创建一个干净的虚拟环境，以防止依赖项冲突，然后将 boto3 的 runtime 安装进对应的文件夹后，将文件夹打包即可。这个 zip 文件就是 layer 本身。

第二步，然后去到 lambda 服务中，点击 *create layer*，将刚刚的 zip 上传，设置需要兼容的 python 版本即可。

第三步，点击 *add layer*，将刚刚创建的 layer 加入到现在的 function 上去。

**什么是 runtime（概念补充）：**

在编程中，“runtime” 通常指的是程序在运行时所需要的环境或平台，它包括了执行程序所需的各种资源、库、依赖项等。简单来说，runtime 就是程序在运行时需要的一切。

在不同的编程语言和开发环境中，runtime 的具体含义和实现方式可能有所不同。例如，在 Python 中，你可能需要一个 Python runtime 环境来执行 Python 脚本或应用程序。这个 Python runtime 包括了 Python 解释器、标准库、第三方库以及其他运行时所需的资源。

通常情况下，你可以通过安装 Python 解释器来获取 Python runtime。Python 解释器可以在各种操作系统上运行，包括 Windows、Mac OS 和 Linux。一旦安装了 Python 解释器，你就可以在其上执行 Python 脚本或应用程序，并利用其提供的各种库和功能来实现程序的功能。

总之，“runtime” 在程序中通常指的是程序在运行时所需的环境或平台，它是程序正确运行的基础。在编程中，了解并配置适当的 runtime 环境对于程序的开发和运行至关重要。

### 总结

1，这是一个很不错的开始，可以自己做很多修改，比如修改prompt，或者让API直接返回答案，而不是进入S3。
2，在API-Gateway的基础上可以改建成很多其他的服务。
3，学习了一下添加layer的方法。重新注意到干净的虚拟环境的重要性。
