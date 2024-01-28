## Some pandas tool code

---

如题总结一些有用的pandas工具代码。

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
