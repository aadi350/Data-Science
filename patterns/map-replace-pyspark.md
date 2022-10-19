How to replace values in a pyspark dataframe using a dictionary mapping

```python
from itertools import chain
from pyspark.sql.functions import create_map, lit

# _map is the dictionary
mapping = create_map([F.lit(x) for x in chain(*_map.items())])


data.withColumn('Mapped', mapping[data['Original']])
```