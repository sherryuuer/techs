## 图算法：拓扑排序-区间合并问题

---
### 相关概念和适用范围

Merge Intervals问题是一种经典的区间合并问题，通常用于解决一系列区间的重叠情况。问题的描述是：给定一组区间，每个区间用[start, end]表示，要求将所有重叠的区间合并成一个或多个不重叠的区间，并返回合并后的区间列表。

在现实世界中它能够帮助解决许多与时间段重叠相关的问题，包括日程安排、会议安排、任务调度等。

- 在日历中显示用户的忙碌时间段，以便其他用户了解其忙碌时段，但又不会透露具体的会议安排细节。这种情况下，可以使用合并区间模式来将连续的忙碌时间段合并成更大的时间段，以便清晰地显示用户的忙碌时间。
- 在用户的临时会议安排中添加新的会议，确保新的会议与已有的会议时间段不重叠。这种情况下，可以使用合并区间模式来检查新会议与现有会议时间段的重叠情况，并根据需要调整会议安排，以确保时间不冲突。
- 在操作系统中，需要根据任务的优先级和计算机处理时间表中的空闲时间段，为任务安排调度。这种情况下，可以使用合并区间模式来将任务的执行时间段合并成较大的时间段，以便更有效地安排任务的执行顺序，提高系统的资源利用率。

例如，给定区间列表intervals = [[1,3],[2,6],[8,10],[15,18]]，其中[1,3]和[2,6]重叠，应该合并成一个区间[1,6]，[8,10]和[15,18]没有重叠，保持原样。所以合并后的区间列表应该是[[1,6],[8,10],[15,18]]。

解决这个问题的常见方法是首先对区间列表按照起始位置进行排序，然后遍历排序后的区间列表，依次检查当前区间与前一个合并后的区间是否重叠，如果重叠则合并，否则将当前区间加入结果列表中。这个问题的解决方法有很多种，可以通过贪心算法、排序算法、栈等不同的方法来实现。

### 问题1:Task Scheduler

任务调度程序题，也是力扣的[621原题](https://leetcode.com/problems/task-scheduler/description/)。它给出一个CPU数组 tasks，用于执行任务，每个CPU都用A-Z的字母表示，另外给出一个冷却时间 n，每个周期，或者间隔只允许完成一项任务。比如A服务器只能在 n 间隔后再次启动。任务的顺序没有限制。返回完成所有任务的最小间隔数。

比如任务列表是["A","A","A","B","B","B"], 间隔 n = 2，输出应该是 8，一种可能的顺序是：A -> B -> 空闲 -> A -> B -> 空闲 -> A -> B。因为完成任务 A 后，必须等待两个周期才能再次执行 A。任务 B 也是如此。在第3个区间，A 和 B 都做不到，所以加入一个空闲间隔。到第 4 个周期，经过 2 个间隔后，可以再次执行 A。

我一开始的解题思路是这样的：

- 统计任务出现的次数，存储在一个字典中。
- 对频次进行排序，从多到少。（在之后动态调整保持顺序，如果是这样的话我可能会选择最大堆）
- 计算最短执行时间。填充任务间隔：从高到低进行填充，如果没有任务了则使用空闲进行填充。
- 更新任务列表，直到所有任务都执行完毕。
- 返回任务列表长度作为最终结果，就是最短执行时间。

看起来似乎可行（总之当事人自己就是一开始觉得ok）但是结果就是只有一部分的case可以通过。最后发现给出的答案思路完全不一样，一万点暴击。思维的漏洞让人难过，一开始就应该先分析清楚。

首先自己的代码是这样的：(这部分代码可以忽略不看了，总之就是用两个堆达到取得最大频率和最小index的目的，但是方向完全不对，所以也得不到正确的结果，这个故事告诉我，**不要受到前一天学习的太大影响，放空思维从问题出发，打开思路。有时候完全抛弃错误的解答重新思考很重要！！！**)

```python
import heapq


def least_time1(tasks, n):
    dict = {}
    for t in tasks:
        dict[t] = dict.get(t, 0) + 1
    tempList1 = [[-1 * v, k, None]
                 for k, v in dict.items()]  # [count, task, index]
    tempList2 = []
    index = 0  # track the index in res
    res = []
    heapq.heapify(tempList1)

    while tempList1:
        print(tempList1, len(res))
        if tempList1[0][-1] is None or not res or len(res) - tempList1[0][-1] >= n:
            task = heapq.heappop(tempList1)
            res.append(task[1])
            index += 1
            if task[0] + 1 != 0:
                tempList2.append([task[0] + 1, task[1], index])
        else:
            res.append(None)
            index += 1

        if not tempList1 and tempList2:
            tempList1 = tempList2[:]
            heapq.heapify(tempList1)
            tempList2 = []

    return len(res)
# 没法使用另一个temp列表的元素，在n上也测试了好几次


def least_time2(tasks, n):
    dict = {}
    for t in tasks:
        dict[t] = dict.get(t, 0) + 1
    tempList = [[-1 * v, k, None]
                for k, v in dict.items()]  # [count, task, index]
    index = 0  # track the index in res
    res = []
    heapq.heapify(tempList)

    while tempList:
        print(tempList, len(res))
        if tempList[0][-1] is None or len(res) - tempList[0][-1] >= n:
            task = heapq.heappop(tempList)
            res.append(task[1])
            index += 1
            if task[0] + 1 != 0:
                tempList.append([task[0] + 1, task[1], index])
        else:
            res.append(None)
            index += 1

    return res, len(res)
# 每次都强制次数最多的元素在前面，这是没必要的


def least_time3(tasks, n):
    dict = {}
    for t in tasks:
        dict[t] = dict.get(t, 0) + 1
    tempList1 = [[-1 * v, k, None]
                 for k, v in dict.items()]  # [count, task, index]
    tempList2 = []
    index = 0  # track the index in res
    res = []
    heapq.heapify(tempList1)

    while tempList1:
        print(tempList1, len(res))
        task = heapq.heappop(tempList1)
        if task[2] is None or len(res) - task[2] >= n:
            res.append(task[1])
            index += 1
            if task[0] + 1 != 0:
                tempList2.append([task[0] + 1, task[1], index])
        else:
            tempList2.append(task)
            res.append(None)
            index += 1

        if not tempList1 and tempList2:
            tempList1 = tempList2[:]
            heapq.heapify(tempList1)
            tempList2 = []

    return res, len(res)


print(least_time3(["A", "K", "X", "M", "W", "D", "X",
      "B", "D", "C", "O", "Z", "D", "E", "Q"], 3))
# (['D', 'X', 'A', 'B', 'C', 'E', 'K', 'M', 'O', 'Q', 'W', 'Z', 'D', 'X', None, None, 'D'], 17)
# 感觉需要调整当count相同的时候index越小越优先


def least_time4(tasks, n):
    dict = {}
    for t in tasks:
        dict[t] = dict.get(t, 0) + 1
    tempList1 = [[-1 * v, -1, k]
                 for k, v in dict.items()]  # [count, index, task]

    index = 0  # track the index in res
    res = []
    heapq.heapify(tempList1)

    while tempList1:

        task = heapq.heappop(tempList1)
        if task[1] == -1 or not res or len(res) - task[1] >= n:
            res.append(task[2])
            index += 1
            if task[0] + 1 != 0:
                heapq.heappush(tempList1, [task[0] + 1, index, task[2]])
        else:
            heapq.heappush(tempList1, task)
            res.append(None)
            index += 1

    return res, len(res)


print(least_time4(["A", "K", "X", "M", "W", "D", "X",
      "B", "D", "C", "O", "Z", "D", "E", "Q"], 3))
# 还是不行，我无法处理index和频率的优先度
```

然后就是参考答案，它尽可能地填充了每个任务之间的间隔，并且保证了没有重叠。代码中的关键部分如下：

- 计算任务频率：首先，代码通过迭代任务列表 `tasks` 并计算每个任务的频率，将结果存储在 `frequencies` 字典中。
- 按频率排序：然后，代码对 `frequencies` 字典按照频率从低到高进行排序，并找到最高频率 `max_freq` 对应的任务。这样做是为了最大化地利用空闲时间。
- 计算空闲时间：计算可能的空闲时间 `idle_time`，它等于 `(max_freq - 1) * n`。这个值代表了在执行最频繁的任务之后可能出现的最长连续空闲时间。
- 遍历频率并减少空闲时间：接下来，代码遍历剩余的任务频率，并尽可能地填充空闲时间。它通过不断减少 `idle_time` 来实现，直到 `idle_time` 减少到零或者所有的任务都已经被处理完毕。
- 返回总时间：最后，代码返回总时间，它等于任务执行的总时间加上可能的空闲时间。

总的来说，这个答案有效地利用了任务的频率信息，并通过减少空闲时间来最大化地优化了任务执行顺序，从而得到了一个合理的执行时间。

代码：

```python
def least_time(tasks, n):
    frequencies = {}

    for t in tasks:
        frequencies[t] = frequencies.get(t, 0) + 1

    frequencies = dict(sorted(frequencies.items(), key=lambda x: x[1]))
    max_freq = frequencies.popitem()[1]
    idle_time = (max_freq - 1) * n

    while frequencies and idle_time > 0:
        idle_time -= min(max_freq - 1, frequencies.popitem()[1])
    idle_time = max(0, idle_time)

    return len(tasks) + idle_time
```

学习笔记：时间复杂度来说一次遍历算频率是O(n)，经过了排序但是因为26个字母有上限，所以为常数，总体来说是O(n)。空间复杂度来说用了一个有限的字典，所以是O(1)。

### 问题2:Insert Interval

给定一个按升序排列的区间列表，以及一个新的区间，将新的区间插入到区间列表中，并确保最终的区间列表也是按升序排列的。是[leetcode57](https://leetcode.com/problems/insert-interval/description/)题，一道中等难度的题。

例如，假设现有区间列表为 [1, 3], [6, 9]，要插入的新区间为 [2, 5]，那么插入后的区间列表为 [1, 5], [6, 9]。结果的返回值是插入新区间后的区间列表。

这个问题通常涉及对区间的合并和插入操作，需要考虑各种情况，如新区间与已有区间的交集情况等。解决方法可以通过遍历区间列表，逐一判断新区间与已有区间的关系，并根据不同情况进行合并或插入操作。

解题步骤：

- 遍历现有区间，将出现在新区间前的区间都append到output列表中。
- 检查刚刚加入output中的最后一个区间是否和新区间有重合。
- 如果重合就将这个区间的end端点更新为 max（该区间end，新区间end）。
- 否则，将新区间加入output区间。
- 继续遍历剩余的区间，如果和output最后一个元素有任何重合，就进行融合操作（更新output中最后一个区间的end为要加入区间的end）。
- 返回output列表。

代码尝试：这道题只debug了一次，就是代码中注释的地方。**深感只要按照清晰的思路去做，总能写出正确的结果，重要的永远是思路和步骤，以及条件的完备。**

```python
def insert_interval(existing_intervals, new_interval):
    output = []
    index = 0
    for ei in existing_intervals:

        if ei[1] <= new_interval[0]:
            output.append(ei)
            index += 1
        else:
            break

    # 这里要防止上面的操作没有加进任何元素而报错的情况
    if output and output[-1][1] >= new_interval[0]:
        output[-1][1] = max(output[-1][1], new_interval[1])
    else:
        output.append(new_interval)

    for i in range(index, len(existing_intervals)):
        ei = existing_intervals[i]

        if ei[0] <= output[-1][1]:
            output[-1][1] = max(output[-1][1], ei[1])
        else:
            output.append(ei)

    return output


print(insert_interval([[1, 4], [5, 6], [7, 8], [9, 10]], [1, 5]))
```

给出的答案只和我稍有不同，它使用的是while进行loop，然后在条件判断上和我的条件顺序相反，其他的基本没什么不同。思路一致。

答案解答如下：

```python
def insert_interval(existing_intervals, new_interval):
    new_start, new_end = new_interval[0], new_interval[1]
    i = 0
    n = len(existing_intervals)
    output = []
    while i < n and existing_intervals[i][0] < new_start:
        output.append(existing_intervals[i])
        i = i + 1
    if not output or output[-1][1] < new_start:
        output.append(new_interval)
    else:
        output[-1][1] = max(output[-1][1], new_end)
    while i < n:
        ei = existing_intervals[i]
        start, end = ei[0], ei[1]
        if output[-1][1] < start:
            output.append(ei)
        else:
            output[-1][1] = max(output[-1][1], end)
        i += 1
    return output
```

学习笔记：时间复杂度上，一共对列表进行了一轮遍历，所以是O(n)，空间复杂度上，没有使用额外的空间进行操作，所以为O(1)。
