## 使用 Github Actions 进行 workflow 自动化的实践（包括template）

---

### Github Actions 极简介绍

当谈到持续集成和持续部署时，GitHub Actions是一个备受推崇的选择。它是GitHub提供的一项内置的自动化工作流服务，可以让开发者轻松地设置、测试和部署他们的代码。GitHub Actions的独特之处在于其简单易用的设计理念，使得任何人都可以迅速上手，无论是新手还是经验丰富的开发者。

通过GitHub Actions，可以利用预定义的工作流程或自定义工作流程来执行各种任务，例如构建、测试、发布还有通知等。之需要使用yml文件，就可以轻松地根据项目的需求来配置工作流，而无需过多的学习成本。通过定义yml文件，GitHub Actions就会自动执行其中定义的任务。

无论是为了自动化构建和测试，还是为了实现持续部署，GitHub Actions都是一个强大而灵活的工具。其集成了丰富的生态系统和第三方服务，如Docker、npm、Maven等开发程序员的工具，以及现在的各种云计算环境的集成，都可以方便的从网上找到资源，进行简单的修改和设置，就可以安全使用。

### 案例介绍

这篇文章将总结以下内容和对应的模板，包括内部使用的代码：

- 使用Pytest进行python代码测试的简单方式
- 自动化部署通过了PR的代码到GCS（谷歌云存储桶）
- 根据执行结果进行消息通知（slack）
- 根据执行结果进行消息通知（teams）（两种方式）

### 使用Pytest进行python代码测试的简单方式

这是一个超级入门的yaml文件，通过这个例子可以立刻入门这个功能。

前提任务：

- 代码托管于GitHub仓库。
- 项目使用Python编写。
- 使用pytest作为测试框架。
- 项目依赖通过root阶层的`requirements.txt`文件定义。主要记入`pytest`即可。

```yaml
name: Python Test

on:
  push:
    branches:
      - main

jobs:
  test:
    runs-on: ubuntu-latest  

    steps:
    - name: Checkout code
      uses: actions/checkout@v2  

    - name: Set up Python
      uses: actions/setup-python@v2 
      with:
        python-version: 3.8 

    - name: Install dependencies
      run: pip install -r requirements.txt 

    - name: Run tests
      run: pytest 

    # this will just run the command but can not prevent the deploy, add other logic please
```

注意事项：

- 该工作流仅在代码推送到`main`分支时触发。
- 使用了GitHub Actions的预定义动作来执行常见任务，如检出代码、设置Python环境和安装依赖。
- 此处仅执行了测试步骤，没有包括部署或其他逻辑。
- 关于pytest请从别处进行学习。

小结：这个案例通过GitHub Actions，可以轻松地设置自动化测试流程，确保代码在每次提交后都能够通过预定义的测试。自动化测试确保了测试过程的一致性和可重复性，消除了手动执行测试所带来的差异性。这之后可以设置拒绝merge，以及通知等（在下面等步骤）。

### 自动化部署通过了PR的代码到GCS（谷歌云存储桶）

自动化部署代码到Google Cloud Storage（GCS）带来了许多好处，可以极大地加速部署过程。一旦设置好自动化流程，部署就可以在几分钟内完成，而不是等待人工手动部署所需的时间。

同时，人为错误是部署过程中的常见问题，可能导致配置错误、文件遗漏等。通过自动化部署，减少了人工干预的机会，降低了出错的可能性。

自动化部署确保每次部署都是一致的，从而消除了手动部署中的差异性。这样还可以在发生错误的时候，更容易地复现问题，和回滚到之前的版本。

在部署到GCS到任务中，前置任务包括以下内容：

- 安全起见，设置Workload Identity（这部分内容涉及GCP中的设置，详见我的其他文章和官方文档，是一种安全，可控的认证管理方式），以及对要使用的服务账号进行GCS储存桶的权限赋予。之需要 ObjectUser 权限即可，方便Actions进行内容的创建。
- 得到了上述设置的 `workload_identity_provider` 和 `service_account`后就可以创建以下的文件，之后只需要部署在repository中就可以了。

```yaml
name: deploy_to_gcs
on:
  workflow_dispatch:
  push:
    branches:
      - main
    paths:
      - the_trigger_path_in_your_repo/**  # src/foldername/**


jobs:
  deploy:
    runs-on: ubuntu-latest

    permissions:
      contents: 'read'
      id-token: 'write'  # will not delete file in gcs

    steps:
      - id: 'checkout'
        uses: 'actions/checkout@v3'

      - id: 'auth'
        uses: 'google-github-actions/auth@v1'
        with:
          # set up by workload identity in gcp
          workload_identity_provider: 'projects/${{ vars.PROJECT_ID }}/locations/global/workloadIdentityPools/${{ vars.WORKLOAD_IDENTITY_POOL }}/providers/github'
          # grant policy to SA to access gcs
          service_account: ${{ vars.SERVICE_ACCOUNT }}

      - id: 'upload-folder'
        uses: 'google-github-actions/upload-cloud-storage@v2'
        with:
          path: 'path_in_your_repo/the_same_folder_name_in_bucket'
          destination: 'bucket_name'
```

注意事项：

- 这里的部署没有包括删除操作，可以自行增加删除操作。（这里的案例是部署到composer的airflow环境，如果加入删除操作，会引起环境的不一致，所以我们选择手动删除）
- 如代码中所说，源path和目的地path的设置上，path要写到和桶中的文件夹一致的地方，比如我要部署的是dags文件夹，那么我的github的path就需要写到dags的部分（后面的斜线不需要写），目的地中的dags则不需要写，之需要写到bucket的名字即可。如果你的测试出现了问题，那么请多测试。
- 上述使用的uses也是从Github中提取的template，最佳实践是使用官网更新的最新的库。 
- 可以看到 `workload_identity_provider` 和 `service_account`的设置，将变量存储为repository的变量，这是一种很好的实践，比较安全，当然你hardcoding也没关系，重要的是你的 repository 不应该是公开的即可。

### 根据执行结果进行消息通知（slack）

Actions和Slack的通知可以简单的实现，通过下面的例子简单部署即可。

前提条件：

- 工作流触发条件为代码推送到`main`分支。
- 需要在GitHub仓库的Secrets中设置一个名为`SLACK_WEBHOOK_URL`的Secret，用于发送Slack通知。

```yaml
name: slack_notification
on:
  push:
    branches:
      - main

env:
  SLACK_USERNAME: DeployBot  # anything you like
  SLACK_ICON: https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png  # anything you like
  SLACK_WEBHOOK: ${{ secrets.SLACK_WEBHOOK_URL }}


jobs:
  deploy:
    runs-on: ubuntu-latest

    steps:
      - id: 'checkout'
        uses: 'actions/checkout@v3'

      # 成功
      - name: 'Slack Notification on Success'
        uses: 'rtCamp/action-slack-notify@v2'
        if: ${{ success() }}
        env:
          SLACK_TITLE: 'Deploy / Success'
          SLACK_COLOR: 'good'
          SLACK_MESSAGE: 'your_message:rocket:'

      # 失敗
      - name: 'Slack Notification on Failure'
        uses: 'rtCamp/action-slack-notify@v2'
        if: ${{ failure() }}
        env:
          SLACK_TITLE: 'Deploy / Failure'
          SLACK_COLOR: 'danger'
          SLACK_MESSAGE: 'your_message:cry:'
```

注意事项：

- 此工作流设置了两个Slack通知步骤，分别用于在部署成功和失败时发送通知。
- 在部署失败时，Slack通知将显示为红色或者绿色，标题和内容都可以进行相应的设置，更多的format的内容，由于工作任务的关系，我还没有完全整理，如果你的项目中需要，请进行自我探索，应该很有趣。

### 根据执行结果进行消息通知（teams）（两种方式）

Teams通知和Slack对通知是同样的功能只是对象不同。这里列举两种方式，是因为在我的任务中，需要对发布内容进行更好的控制，比如添加teams的mention功能，以及加入更多的信息，而且mention功能似乎在这两年才更好的集成了进去。因此这里列出两种方式。

如果你只是想发送一个简单的通知，那么使用`crul`命令就可以，代码如下：

```yaml
name: Notification

on:
  push:
    branches:
      - main

jobs:
  test:
    runs-on: ubuntu-latest  

    steps:
    - name: Checkout code
      uses: actions/checkout@v2  

    #-- Teams通知 --#
    # way 1
    - name: 'Teams Notification on Success'
      if: ${{ success() }}  # ${{ failure() }} change the message in MESSAGE too
      run: |
        MESSAGE="ジョブ詳細:  ${{ github.repository }} ComposerのGCSへのデプロイは成功しました。(コミットメッセージ: ${{ github.event.head_commit.message }} | 作成者: ${{ github.event.head_commit.author.name }} | タイムスタンプ: ${{ github.event.head_commit.timestamp }})"
        curl -X POST -H "Content-Type: application/json" -d "{\"@type\": \"MessageCard\",\"title\": \"デプロイ成功\",\"text\": \"$MESSAGE\"}" ${{ secrets.TEAMS_WEBHOOK_URL }}
      env:
        WEBHOOK_URL: ${{ secrets.TEAMS_WEBHOOK_URL }}
```

通过这种方式就可以将很多信息发送到teams，你可能在网上找到带有mention的命令行，但是在我的测试中我没有成功，多次测试和格式修改都不能很好的mention到成员，所以我转向了执行python脚本的方法。代码如下（仅包括最后一个部分）：

```yaml
#### other steps
    # way 2
    - name: 'Teams Notification on Success'
      if: ${{ success() }}  # ${{ failure() }} change the parameter in run code too
      run: |
        pip install -r requirements.txt
        python notify_teams.py ${{ secrets.TEAMS_WEBHOOK_URL }} 'success' ${{ github.event.repository.url }} ${{ github.event.head_commit.message }} ${{ github.event.head_commit.author.name }} ${{ github.event.head_commit.timestamp }}
      env:
        WEBHOOK_URL: ${{ secrets.TEAMS_WEBHOOK_URL }}
```

过程中使用的requirements文件中只有requests包。而python脚本则集成在文件中。脚本的主要内容如下，我省略了注释和对于理解没有必要的部分，请在生产环境中自行添加。

该脚本使用的外部文件只有mention成员的列表，方便分离管理。

```python
import requests
import sys
from mention_members import MENTION_MEMBERS

def _prep_job_message(info, repository, commit_message, author, timestamp):
    """
    Returns:
        dict: 通知メッセージ
    """
    entries = []
    for m in MENTION_MEMBERS:
        entries.append(
            {
                'type': 'mention',
                'text': f'<at>{m}</at>',
                'mentioned': {
                    'id': m,
                    'name': m.split('@')[0]
                },
            }
        )
    mentions = ''.join([f'<at>{m}</at> ' for m in MENTION_MEMBERS])
    if info == 'failure':
        mentions_text = f'{mentions}\n\nデプロイ失敗しました！\n\n'
    else:
        mentions_text = f'{mentions}\n\nデプロイ成功しました！\n\n'
        # mentions_text = f'デプロイ成功しました！\n\n' for test

    text = f'{mentions_text}リポジトリ：{repository}\n\nコミットメッセージ：{commit_message}\n\n作成者：{author}\n\nタイムスタンプ：{timestamp}'

    message = {
        'type': 'message',
        'attachments': [{
            'contentType': 'application/vnd.microsoft.card.adaptive',
            'content': {
                'type': 'AdaptiveCard',
                'body': [{
                    'type': 'TextBlock',
                    'text': text,
                    'wrap': True
                }],
                '$schema': 'http://adaptivecards.io/schemas/adaptive-card.json',
                'version': '1.0',
                'msteams': {
                    'entities': entries,
                    'width': 'Full',
                },
            },
        }],
    }

    return message


def notify_message(webhook_url, info, repository, commit_message, author, timestamp):
    response = requests.post(
        webhook_url,
        json=_prep_job_message(info, repository, commit_message, author, timestamp),
    )
    print(f'response: {response}')


if __name__ == "__main__":
    webhook_url = sys.argv[1]
    info = sys.argv[2]
    repository = sys.argv[3]
    commit_message = sys.argv[4]
    author = sys.argv[5]
    timestamp = sys.argv[6]

    notify_message(webhook_url, info, repository, commit_message, author, timestamp)
```

注意事项：

这其中用到了一种发送消息的`AdaptiveCard`类型格式，可以在下面链接中找到更多custom的方法，欢迎探索。

https://learn.microsoft.com/ja-jp/microsoftteams/platform/task-modules-and-cards/cards/cards-format?tabs=adaptive-md%2Cdesktop%2Cconnector-html

recap：以上就是全部内容。关于 actions 的功能还有很多，google 是最好的老师。
