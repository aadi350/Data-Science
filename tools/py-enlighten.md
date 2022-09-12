# Progress Bars
```python
import time
import enlighten

OUTER_RANGE=100
INNER_RANGE=50

manager = enlighten.get_manager()

pbar_outer = manager.counter(
    total=OUTER_RANGE,
    desc='Epochs',
    unit='',
    color='green',
    leave=False
)



for outer in range(OUTER_RANGE):
    time.sleep(1)
    pbar_outer.update()
    pbar_inner = manager.counter(
        total=INNER_RANGE,
        desc='Batch',
        unit='',
        color='purple',
        leave=False
    )
    for inner in range(INNER_RANGE):
        pbar_inner.update()
        time.sleep(0.02)
        pbar_valid = manager.counter(
            total=10,
            desc='Validation Steps',
            unit='',
            color='yellow',
            leave=False
        )
        for i in range(10):
            pbar_valid.update()
            time.sleep(0.1)
```