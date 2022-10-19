Make it longer like [this](https://walkenho.github.io/merging-multiple-dataframes-in-pyspark/) 
```python
from functools import reduce
from pyspark.sql import DataFrame

dfs = [df1,df2,df3]
df = reduce(DataFrame.unionAll, dfs)
```