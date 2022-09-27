
Given a `pandas` dataframe of the following form:

| | start_date | other_date |  
| --- | --- | --- |  
| 0 | 2022-04-30 15:22:53.008753 |  2022-07-24 15:22:53.008753 |  
| 1 | 2022-04-12 15:22:53.008753 |  2022-05-26 15:22:53.008753 |  
| 2 | 2022-07-30 15:22:53.008753 |  2022-05-22 15:22:53.008753 |  
| 3 | 2022-05-22 15:22:53.008753 |  2022-07-15 15:22:53.008753 |  
| 4 | 2022-09-04 15:22:53.008753 |  2022-06-14 15:22:53.008753 |  

We can get the difference between dates in the following ways:
### Difference in Days
```python
((df.start_date - df.other_date)/np.timedelta64(1, 'D'))
```

### Difference in Months
```python
((df.start_date - df.other_date)/np.timedelta64(1, 'M'))
```  

### Difference in Multiple Units
```python
((df.start_date - df.other_date)/np.timedelta64(3, 'D'))
```