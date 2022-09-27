# How to Get Random Strings

```python
def get_random_string(length):
    # choose from all lowercase letter
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for i in range(length))

    return result_str

```

**Random from fixed set**
```python
import random 

result_str = ''.join((random.choice('abcdxyzpqr') for i in range(5)))
```

**Random without Repeats**  
`choice` can repeat results, `sample` does not
```python
# get random string of length 8 without repeating letters
result_str = ''.join(random.sample(string.ascii_lowercase, 8))
```