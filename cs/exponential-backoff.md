## 网络限流(throttling)，指数退避(ExponentialBackoff)和Python的代码实现

---
### 什么是限流？

限流（throttling）通常是指在网络或系统中限制请求或通信速度。

网络和服务器具有一定的性能和处理能力，但是如果特定用户过度负载，其他用户可能无法正常使用。因此，使用称为限流的机制来限制每个用户可以使用资源的数量，并保持整体使用公平有效。

具体来说，限流会在以下情况下进行：

* **网络带宽限制：** 互联网服务提供商 (ISP) 可能会限制每个用户可用的带宽。这是为了防止特定用户消耗过多的带宽并降低其他用户的通信速度。
* **API 请求限制：** Web 服务和 API 可能会限制每小时或每天的请求数。这是为了防止服务过载并保持服务稳定性。项目中就遇到了API请求到了上限，然后batch失败的情况。
* **CPU 时钟速度限制：** CPU 过热时，可能会自动降低时钟速度。这是为了保护 CPU 并防止损坏。

限流是维护网络和系统平稳运行的必要机制。但是，对于用户来说，它也可能带来一些缺点，例如请求延迟或服务不可用。

**如何避免限流的影响？**

为了避免限流的影响：

* **检查使用条款：** 许多服务在其使用条款中规定了有关限流的规定。在使用服务之前，务必阅读使用条款并了解有哪些限制。读doc是一个开发者必备技能。
* **了解自己的使用情况：** 了解自己使用了多少带宽或 API 请求，以便调整使用量以避免受到限流的影响。
* **考虑替代方案：** 如果受到限流的影响，可以考虑替代方案。例如切换到其他 ISP 或使用其他 API 服务。

### 指数退避（Exponential Backoff）

指数退避（Exponential Backoff）是一种算法，用于在遇到冲突或失败时，以指数级方式增加重试之间的延迟时间。其目的是减少对网络或系统的拥塞，并提高重试成功的可能性。

指数退避算法通常用于网络协议中，例如 TCP/IP 和 HTTP。在这些协议中，数据包可能会在传输过程中丢失或损坏。指数退避算法可确保在数据包丢失的情况下，发送方不会立即重新发送数据包，从而导致网络拥塞。相反，发送方会等待一段随机时间，然后重试。如果重试失败，则等待时间会指数级增加。

指数退避算法的优点包括：

* **减少网络拥塞：** 通过在重试之间增加延迟时间，可以减少发送方同时发送大量数据包的可能性，从而降低网络拥塞的风险。
* **提高重试成功率：** 随着等待时间的增加，网络拥塞的可能性会降低，重试成功的可能性也会提高。
* **公平性：** 指数退避算法可以确保所有发送方都有公平的机会重试失败的数据包。

当然也有一些缺点：

* **增加延迟：** 在重试成功之前，用户可能会经历额外的延迟。
* **复杂性：** 指数退避算法的实现可能比简单的重试算法更复杂。

以下是一个简单的指数退避算法示例：

1. 初始化重试次数为 0 和等待时间为 1 秒。
2. 发送数据包。
3. 如果数据包成功传输，则停止。
4. 如果数据包传输失败，则将重试次数加 1 并将等待时间乘以 2。
5. 等待等待时间，然后重试步骤 2。
6. 重复步骤 3 到 5，直到数据包成功传输或达到最大重试次数。

在这个示例中，如果数据包在第一次尝试后失败，则等待时间将增加到 2 秒。如果第二次尝试失败，则等待时间将增加到 4 秒，依此类推。这有助于确保发送方不会在短时间内发送大量数据包，从而导致网络拥塞。

指数退避是一种有效的算法，可用于减少网络拥塞并提高重试成功的可能性。它已广泛用于各种网络协议和系统中。

### Python的指数退避代码

使用Python进行API请求时候的指数退避大概如下：这是一段伪代码，使用api_func模拟api请求。

```python
import time
import random

def request_with_backoff(api_func, *args, **kwargs):
    """使用指数退避算法请求API"""
    max_tries = 5  # 最大重试次数
    base_delay = 1  # 初始等待时间（秒）

    for _ in range(max_tries):
        try:
            response = api_func(*args, **kwargs)
            if response.status_code == 200:
                return response
            else:
                raise Exception(f"API request failed with status code {response.status_code}")
        except Exception as e:
            print(f"API request failed: {e}")
            delay = random.uniform(base_delay, base_delay * 2)
            time.sleep(delay)
            base_delay *= 2

    raise Exception(f"Maximum retries exceeded ({max_tries})")
```

这个函数进入一个循环，该循环将在最大重试次数内运行。在每个循环中，该函数将尝试调用 API 函数。如果 API 请求成功，则函数将返回响应。如果 API 请求失败，则函数将打印错误消息并计算新的等待时间。新的等待时间是随机值，介于初始等待时间和初始等待时间的两倍之间。然后，函数将休眠指定的时间，然后重试 API 请求。如果所有重试都失败，则函数将引发异常。

以下伪代码使用了上面这个函数：

```python
import requests

def get_user(user_id):
    response = requests.get(f"https://api.example.com/users/{user_id}")
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Failed to get user: {response.status_code}")

try:
    user = request_with_backoff(get_user, 123)
    print(user)
except Exception as e:
    print(f"Failed to get user: {e}")
```

### 通过类实现的伪代码

下面是另一个示例代码，演示了如何使用 `requests` 库发送API请求，并在请求失败时执行指数退避重试：

```python
import requests
import time
import random

class ExponentialBackoff:
    def __init__(self, max_attempts=5):
        self.max_attempts = max_attempts
        self.attempt = 0

    def wait(self):
        wait_time = (2 ** self.attempt) * random.uniform(0.5, 1.5)
        time.sleep(wait_time)
        self.attempt += 1
        print("Retry after:", wait_time)

def make_api_request(url):
    backoff = ExponentialBackoff()
    while backoff.attempt < backoff.max_attempts:
        try:
            response = requests.get(url)
            if response.status_code == 200:
                return response.json()
            else:
                print("Request failed with status code:", response.status_code)
                backoff.wait()
        except Exception as e:
            print("Request failed with exception:", str(e))
            backoff.wait()
    print("Max attempts reached. Failed to fetch data from API.")
    return None

# 示例使用
url = "https://api.example.com/data"
data = make_api_request(url)
if data:
    print("Data fetched successfully:", data)
```

在这个示例中，`ExponentialBackoff` 类用于执行指数退避策略。`make_api_request` 函数发送API请求，并在请求失败时执行指数退避重试，最多尝试5次。当达到最大尝试次数后，函数将返回 `None`。

### 我工作中遇到的代码

项目代码用于实现与 Braze 平台的数据集成和营销活动的自动化执行。主要功能是将来自 BigQuery 的用户数据同步到 Braze 平台，并在过程中实现了对 Braze API 的指数退避重试，以确保数据的稳定传输和处理。

```python
import itertools
import concurrent.futures
import time
import json
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from google.cloud import bigquery

def post_data():

    def splitparN(iterable, N=70):
        # braze rest api 一次只能插入70，因此分割数据，创建生成器
        for i, item in itertools.groupby(enumerate(iterable), lambda x: x[0] // N):
            yield (x[1] for x in item)


    def create_data(query_result):
        attr_list = []
        for rows in query_result:
            row_dict = {}
            for col_name, data in rows.items():
                if data is not None:
                    row_dict[col_name] = data
            attr_list.append(row_dict)
        json_data = {'attributes': attr_list}
        return json_data


    def post_req(result_data):
        url = 'https://BRAZE_LINK/users/track'
        api_token = Variable.get('api_key')
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_token}',
            'X-Braze-Bulk': 'true'
        }
        with requests.Session() as s:
            retries = Retry(
                total=3,
                backoff_factor=10,
                status_forcelist=[429, 500, 502, 503, 504],
                allowed_methods=['POST']
            )

            s.mount('https://', HTTPAdapter(max_retries=retries))
            try:
                # connect timeout 5 sec, read timeout 10 sec
                response = s.post(
                    url=url,
                    headers=headers,
                    json=result_data,
                    timeout=(5.0, 10.0)
                )
                response.raise_for_status()  # 如果请求不成功，抛出异常
            except requests.exceptions.HTTPError as http_err:
                return {
                    'response_code': '',
                    'response_text': '',
                    'error_text': http_err,
                    'member_id_list': [r['external_id'] for r in result_data['attributes']]
                }
            except Exception as e:
                return {
                    'response_code': '',
                    'response_text': '',
                    'error_text': e,
                    'member_id_list': [r['external_id'] for r in result_data['attributes']]
                }

            return {
                'response_code': str(response.status_code),
                'response_text': response.text,
                'error_text': '',
                'member_id_list': [r['external_id'] for r in result_data['attributes']]
            }


    # 从BQ取得数据
    client = bigquery.Client()
    query = """
        get the data id query
    """
    rows = client.query(query).result()

    # API
    post_row_cnt = 60
    start = time.perf_counter()

    with concurrent.futures.ThreadPoolExecutor() as executor:
        response_process = []
        for query_result_list in splitparN(rows, post_row_cnt):
            result_data = create_data(query_result_list)
            response_process.append(executor.submit(post_req, result_data))

        print(f'request_cnt:{len(response_process)}')

        error_rows = []
        error_cnt = 0
        insert_row_cnt = 0
        for res in concurrent.futures.as_completed(response_process):
            res_result = res.result()
            if res_result['response_code'] == '201':
                insert_row_cnt += json.loads(res_result['response_text'])['attributes_processed']
            else:
                error_rows.append(res_result)
                error_cnt += 1

                # 实现指数退避，等待一段时间后重试
                retry_count = 0
                max_retries = 3
                while retry_count < max_retries:
                    wait_time = (2 ** retry_count) * 5  # 指数退避策略
                    time.sleep(wait_time)
                    retry_count += 1
                    retry_response = post_req(result_data)
                    if retry_response['response_code'] == '201':
                        insert_row_cnt += json.loads(retry_response['response_text'])['attributes_processed']
                        break
                    elif retry_count == max_retries:
                        error_rows.append(retry_response)
                        error_cnt += 1

        print(f'insert_row_cnt:{insert_row_cnt}')
        print(f'error_request_cnt：{error_cnt}')
        print(f'Duration: {time.perf_counter() - start}')

post_member_data()
```

在 `post_req` 函数中，当请求不成功时，会捕获 `requests.exceptions.HTTPError` 异常并进行重试，等待时间按照指数退避的方式逐渐增加。
