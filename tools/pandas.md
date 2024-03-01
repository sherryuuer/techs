## Some pandas tool code

---

### 数据处理 Data preprocessing

- How to create from a python data structure:

pd.Series(list)/pd.DataFrame(dict)

- Get the data:

```python
import pandas as pd
# usecols will set the column name for the dataframe
table = pd.read_csv("table.csv", usecols=["col1","col2","col3"])
```

- Change a DataFrame's columns

```python
array = np.random.randint(1, 10, size=(3,2))
df = pd.DataFrame(array, columns=["A","B"])
```

- Different between len, shape, size:

`len(df)` is the rows of df, shape is m x n of df, size is the number of elements of df.

- Check datatypes: `df.dtypes()`

- Unique elements

```python
# get number of unique
print(df["col"].nunique())
# unique elements well return as a list
print(df["col"].unique())
```

- Value counts method:

```python
pd["col"].value_counts()
```

- Measures of Central Tendency:

```python
df["col"].mean()
df["col"].median()
df["col"].mode()
df["col"].max()
df["col"].min()
```

- Filter data:

loc and iloc: by columns or index

```python
pd.loc[:4, ["col1","col2"]]
pd.iloc[5:9, :2]
```

select a subset of culumns:

```python
selected_cols = ["A", "B"]
df[selected_cols].head()

# or pass the list to the df
# notice the 2 layer of the list []
df[["A", "B"]].head()
```

by condition:

```python
df_filtered = df[df["col"] > 100]
```

by query:

```python
filtered = table.query("condition1 > 100 and condition2 < 400")
print(filtered[["col1","col2","col3"]].head())
```

### Pandas对字符串的处理一般使用str方法

index:

```python
df["col"].str[0:3]
```

split and expand

```python
df["new"] = df["old"].str.split(" ", expand=True)[1]
```

lower,upper,capitalize

```python
df["lower"] = df["name"].str.lower()
df["upper"] = df["name"].str.upper()
df["capital"] = df["name"].str.capitalize()
```

使用pandas的str方法比pyhon内置的更有效率，并且可以作用于**整个列**，而python的内置方法不可以。
python内置的模块只能用于一个文字串比如：

```python
df["single"] = df["name"][0].upper()
```

replace method for part of thes value:

```python
df["name"].str.replace(",", "-")
```

Dataframe.replace() is for replace the entire value.

```python
staff["state"].replace(
    {"TX": "Texas", "CA": "California", "FL": "Florida", "GA": "Georgia"},
    inplace = True
)
```

### Dates and times (single value)

string to timestamp:

```python
my_timestamp = pd.to_datetime("2024-01-27")
```

convert object type to datetime64:

```python
df = df.astype({
    "date_of_birth": "datetime64[ns]",
    "start_date": "datetime64[ns]",
})
# ns means nanosecond precision.
print(staff.dtypes)
```

timedelta is a type to get the days between 2 datetime:

```python
first_date = pd.to_datetime("2021-10-10")
second_date = pd.to_datetime("2021-10-02")

diff = first_date - second_date

print(type(diff))  # timedelta datatype
print("\n")
print(diff.days) # get 8 (days)
```

Timestamp information: by attributes or methods:

Attributes will return us int:

```python
mydate = pd.to_datetime("2021-10-10")

print(f"The year part is {mydate.year}")
print(f"The month part is {mydate.month}")
print(f"The week number part is {mydate.week}")  # get the calendar week
print(f"The day part is {mydate.day}")
print(f"The hour part of mydate is {mydate.hour}")
print(f"The minute part of mydate is {mydate.minute}")
print(f"The second part of mydate is {mydate.second}")
```

Method will return string:

```python
print(f"The date part is {mydate.date()}")
print(f"The day of week is {mydate.weekday()}")
print(f"The name of the month is {mydate.month_name()}")
print(f"The name of the day is {mydate.day_name()}")
```

### datetime on DataFrame (columns)

similar to str accessor: weekday, hour, minite, second

```python
df["month"] = df["datetime"].dt.month
```

column should be dtype of datetime64

isocalendar() will return year, calendar week, day of week from datetime64

```python
df["datetime"].dt.isocalendar()
```

age calculate:

```python
(staff["start_date"] - staff["date_of_birth"]).dt.days / 365
```

### DateOffset to get time interval (and timedelta)

```python
df["new_date"] = df["date"] + pd.DateOffset(years=1)
df["new_date"] = df["date"] + pd.DateOffset(years= -1)
df["new_date"] = df["date"] - pd.DateOffset(years=1)
```

other params: months, weeks, days, hours, minites, seconds, microseconds, nanoseconds

Timedelta:

```python
# add 12 week
df["date"] + pd.Timedelta(value=12, unit="W")
# or with string
df["date"] + pd.Timedelta("12 W")
```

### pd.Int64Dtype() from Pandas 1.0

在列的类型转换中，如果含有NaN，则整个整数列会成为float类型。如果要改变这个情况，使得除了NaN的浮点数字变回整数，就在astype中使用这个参数。`df["col"] = df["col"].astype(pd.Int64Dtype())`。


### Deal with missing value
### isna(),notna()

Find out how many NaN values.In ml world, isna() is useful.

```python
# count NaN value of each column.
df.isna().sum()
```

Drop rows or columns if any value is NaN.

默认的肯定是丢弃一条数据，你不能不应该丢掉整个col那可以一个特征向量。

```python
# drop rows
df.dropna(axis=0, how="any")
# drop cols
df.dropna(axis=1, how="any")
```

Save change to df with inplace parameter.

```python
# Drop rows that have less than 4 non-missing values
df.dropna(thresh=4, inplace=True)
```

fillna by value or by dict

```python
df["col"].fillna(value = df["col"].mean())

# find the replacement values
value_a = df["A"].mean()
value_d = df["D"].mean()

# replace the missing values
print(df.fillna({"A": value_a, "D": value_d}))
```

fillena method bfill, ffill

使用该row前面的或者后面的数据进行填充，比如股票。

```python
print("Filling backward")
print(df["A"].fillna(method="bfill"))

print("\nFilling forward")
print(df["A"].fillna(method="ffill"))

# limit限制只复制几个长度的数据
df.fillna(method="bfill", limit=1)
```

### Data analysis

group by function

```python
print(df[["groupby_col","price"]].groupby("groupby_col").mean())
```

aggregation and sort_values function

```python
print(
  grocery.groupby("product_description").agg(
    avg_price = ("price","mean"),
    total_sales = ("sales_quantity", "sum")
  ).sort_values(
    by="total_sales",
    ascending=False
  )
)
```

当使用多个col进行groupby的时候，输出的是，用于groupby的所有col的排列组合。

```python
import pandas as pd

grocery = pd.read_csv("grocery.csv")

print(
  grocery.groupby(
    ["product_description", "product_group"]
  ).agg(
    avg_price = ("price","mean"),
    total_sales = ("sales_quantity", "sum")
  )
)
```

### pivot table

```python
import pandas as pd

grocery = pd.read_csv("grocery.csv")

# Creating the week column
grocery["sales_date"] = grocery["sales_date"].astype("datetime64[ns]")
grocery["week"] = grocery["sales_date"].dt.week

# Creating the pivot table
print(
  pd.pivot_table(
    data = grocery, 
    values = "sales_quantity", 
    index = "product_group", 
    columns = "week",
    aggfunc = "sum",
    margins = True,
    margins_name = "Total"
  )
)
```

It’s possible to show column and row subtotals and the grand total as well. We can do so using the `margins` and `margins_name` parameters. The following example illustrates the use of these parameters.

### where function

```python
grocery = pd.read_csv("grocery.csv")

grocery["price_updated"] = grocery["price"].where(
  grocery["price"] >= 3,
  other = grocery["price"] * 1.1  # update the value that don't match the condition
)
```

### visualize with pandas plot function

histogram:

```python
import pandas as pd
import matplotlib.pyplot as plt

grocery = pd.read_csv("grocery.csv")

grocery["price"].plot(
    kind = "hist",
    figsize = (10, 6),
    title = "Histogram of grocery prices",
    xticks = [2,3,4,5,6,7,8,9,10,11,12]
)

plt.savefig('output/abc.png')
```

line:

```python
import pandas as pd
import matplotlib.pyplot as plt

grocery = pd.read_csv("grocery.csv")

grocery[grocery["product_description"]=="tomato"].plot(
    x = "sales_date", 
    y = ["sales_quantity", "price"],
    kind = "line",
    figsize = (10,5),
    title = "Daily tomato sales and prices",
    # the price will be show at the right of the plot
    secondary_y = "price"
)

plt.savefig('output/abc.png')
```

scatter:this is very important in data science.

```python
import pandas as pd
import matplotlib.pyplot as plt

sales = pd.read_csv("sales.csv")

sales.plot(
    x = "price",
    y = "cost",
    kind = "scatter",
    figsize = (8, 5),
    title = "Cost vs Price",
    xlim = (0, 1000),
    ylim = (0, 800),
    grid = True
)

plt.savefig('output/abc.png')
```
