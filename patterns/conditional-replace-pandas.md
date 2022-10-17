Given a `pandas` dataframe, replacing a column based on a condition is done in the following:

```python
conditions = [
    (df['column']==val1),
    (df['column']==val2),
]

values = [
    'val1replace',
    'val2replace',
]

np.select(conditions, values)
```