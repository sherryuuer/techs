## 排序算法全家桶

---
### 冒泡排序（bubble sort）

暴力破解的泡泡。

冒泡排序是一种简单的比较排序算法，它重复地遍历要排序的列表，一次比较两个元素，并且如果它们的顺序错误就将它们交换过来。这个过程一直持续，直到没有再需要交换的元素，即列表已经排序完成。

基本思想是通过不断地交换相邻的元素，将较大的元素逐渐“浮”到数组的末端，而较小的元素逐渐沉到数组的前端。就像是泡泡不断浮动。


比较相邻元素：从列表的开头开始，比较相邻的两个元素。
交换元素：如果顺序不正确，交换这两个元素。
遍历列表：重复以上两步，直到整个列表排序完成。每一轮遍历都能确定一个未排序部分的最大值或最小值。
时间复杂度：

冒泡排序的时间复杂度为 O(n^2)，其中 n 是数组的长度。在最坏情况下，即列表完全逆序时，需要进行 n*(n-1)/2 次比较和交换。在最好情况下，即列表已经有序时，只需要进行一次遍历，但仍然需要进行 n 次比较。

是稳定的排序算法，即相等元素的相对位置在排序前后不发生改变。

是一种简单但效率较低的排序算法，对于小型数据集或者基本有序的数据集来说可能是合适的选择。

由于其简单的实现方式，冒泡排序通常用于教学和理解排序算法的基本原理。但是其实我觉得在代码实现上不是很好理解，至少我需要打印出来进行观察。

冒泡排序是可以优化的，可以通过设置一个标志来进行优化，如果在一轮遍历中没有进行过交换，说明列表已经有序，可以提前结束排序。

```python
def bubbleSort(arr):  # O(n^2)
    count = 0
    for i in range(len(arr) - 1):
        for j in range(len(arr) - i - 1):
            count += 1
            if array[j] > array[j + 1]:
                array[j], array[j + 1] = array[j + 1], array[j]
    return (f'{array} \nNumber of comparisons = {count}')
```

### 选择排序（Selection sort）

也是一种简单的排序算法，其基本思想是通过不断选择未排序部分的最小（或最大）元素，将其放到已排序部分的末尾。总之就是：**选择和swap**

选择排序的时间复杂度是 O(n^2)，其中 n 是数组的长度。在每一轮选择过程中，需要进行 n 次比较，找到最小元素。总共需要进行 n-1 轮选择操作。

它是一种不稳定的排序算法，即相等元素的相对位置可能发生改变。

选择排序对于小型数据集可能是一种适用的选择，但在大型数据集上其性能相对较差。

总体而言，选择排序是一种简单但不是最优的排序算法，它的主要优点在于实现简单，但在大规模数据集上通常不如其他高效的排序算法。

一种解决方案：

```python
def findSmallest(array):
    smallest = array[0]
    smallest_index = 0
    for i in range(1, len(array)):
        if array[i] < smallest:
            smallest = array[i]
            smallest_index = i 
    return smallest_index

# 这个方法是重新填充一个新的数组
def selectionSort(array):
    newarr = []
    for i in range(len(array)):
        smallest = findSmallest(array)
        newarr.append(array.pop(smallest))
    return newarr
```

另一种(from educative)更加选择排序一些：大O是n的平方级别

```python
def swap(array, firstIndex, secondIndex):
  temp = array[firstIndex]
  array[firstIndex] = array[secondIndex]
  array[secondIndex] = temp

  
def indexOfMinimum(array, startIndex):
  minValue = array[startIndex]
  minIndex = startIndex

  for i in xrange(minIndex + 1,len(array)):
    if array[i] < minValue:
      minIndex = i
      minValue = array[i]
  return minIndex


def selectionSort(array):
  for i in range(len(array)):
    min_index = indexOfMinimum(array, i)
    swap(array, i, min_index)
  return array
```


### 插入排序（Insertion sort）

通过遍历数组，每次都对子数组的最后一个元素进行（和他之前的每个元素）比较操作，然后插入正确位置，最终得到结果。

是一个稳定的排序算法（当出现相同元素的时候，排序后他们的相对位置不发生改变），时间复杂度是O(n)。但是注意n的时间复杂度不是最坏的情况，最坏可能会是n方复杂度。

```python
def insertionSort(arr):
    for i in range(1, len(arr)):
        j = i - 1
        while j >= 0 and arr[j + 1] < arr[j]:
            arr[j + 1], arr[j] = arr[j], arr[j + 1]
            j -= 1
    return arr
```

将一部分代码分开的写法如下，看起来似乎比较好理解一点。

```python
def insert(array, rightIndex, value):
  j = rightIndex
  while j >= 0 and array[j] > value:
    array[j + 1] = array[j]
    j = j - 1
  array[j + 1] = value

def insertionSort(array):
  for i in range(1, len(array)):
    rightIndex = i - 1
    value = array[i]
    insert(array, rightIndex, value)
  
  return array
```

### 合并算法（Merge sort）

是一种最常用的高效算法。Divide&Conquer是很有名的算法思想。

不停地将数组一分为二，直到不能再分为止，然后递归地拿回结果进行合并。在合并的时候，每次都顺序从，两个数组的第一个开始取数并比较。

时间复杂度的计算：

1. 将数组一分为二直到1个元素需要logn的时间，因为n/2^x = 1,x = logn。
2. 然后需要对每个分开的元素进行逐次比较是n的时间复杂度。
3. 对每个细分的组都需要进行比较排序，所以最终的时间复杂度是O(nlogn)。

代码步骤：

1. 如果数组的长度为1或0，那么它已经被视为排序好的。
2. 否则，将数组分为左右两部分。
3. 对左子数组和右子数组分别应用递归排序。
4. 一旦左子数组和右子数组都已经排序，开始合并它们。创建一个新的空数组，并比较左右子数组的元素，按升序将它们合并到新数组中。
5. 当一边的子数组被合并完后，将另一边的子数组中的剩余元素全部追加到新数组中。
6. 返回新数组，它是已排序的。

其实有很多代码的写法，我个人比较喜欢下面这样的写法，对我来说很好理解。

```python
def merge(left, right):  # O(n)
    result = []
    leftindex = 0
    rightindex = 0
    while leftindex < len(left) and rightindex < len(right):
        if left[leftindex] < right[rightindex]:
            result.append(left[leftindex])
            leftindex += 1
        else:
            result.append(right[rightindex])
            rightindex += 1
    return result + left[leftindex:] + right[rightindex:]

def mergeSort(arr): # O(logn)
    if len(arr) == 1:
        return arr
    middle = len(arr) // 2
    left = arr[:middle]
    right = arr[middle:]
    return merge(mergeSort(left), mergeSort(right))
```

### 快速排序（Quick sort）

快速排序也是一种分治思想的算法，和合并算法一样，时间复杂度也是O(nlogn)。但是最坏的情况（选择的基准元素正好是最大，或者最小）出现的话，时间复杂度将增加为O(n^2)。因此其实选择一个随机的元素是最好的选择。

是一种不稳定的排序算法，因为在分区的过程中，相对位置会发生变化。

基本方法是选择一个基准元素pivot，然后根据每个元素和这个基准元素的大小比较，而分成两个部分，一部分比基准小，一部分比基准大。

然后重复以上步骤，对分开的两个部分，*递归的*进行同样的操作。

快速排序是一种高效的算法，很适合大型数据集。

```python
def quickSort(arr, start, end):
    if end - start + 1 <= 1:
        return

    pivot = arr[end]
    left = start

    for i in range(start, end):
        if arr[i] < pivot:
            arr[left], arr[i] = arr[i], arr[left]
            left += 1

    arr[end] = arr[left]
    arr[left] = pivot
    
    quickSort(arr, start, left - 1)
    quickSort(arr, left + 1, end)

    return arr
```

还有一种不考虑原地排序很好理解的写法。

```python
def quickSort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[0]
    less = [x for x in arr[1:] if x <= pivot]
    greater = [x for x in arr[1:] if x > pivot]
    return quickSort(less) + [pivot] + quickSort(greater)
```

### 桶排序（Bucket sort）

桶排序适用于输入数据均匀分布在某个范围内的情况，而且适用于外部排序，即数据量很大，无法一次性加载到内存中进行排序的情况。

假设有一组数据在范围 [0, 1) 内，可以将这个范围划分成若干个桶，每个桶对应一个区间，然后将数据分到各个桶中，对每个桶中的数据进行排序，最后合并所有桶的数据即可得到有序结果。

假设一个数组里面只有0，1，2也就是他们在一个范围内。可以看出代码活用了index和相对元素的各种变换进行排序。

```python
def bucketSort(arr):
    counts = [0, 0, 0]
    for n in arr:
        count[n] += 1

    i = 0
    for n in range(len(counts)):
        for j in range(counts[n]):
            arr[i] = n
            i += 1
    return arr
```

这个排序方法用的挺少的，除了特定场合很少用到。
