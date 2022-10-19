```python
from plotly import graph_objects as go
import numpy as np

fig.add_trace(
  go.Scatter(
    x=x,
    y=y,
    name='Avg_Num_NSF_Cheques',
    mode='lines+markers'
  ), row=1, col=1
)

polyfit = np.polyfit(x=[i for i in range(len(x))], y=y, deg=1)
fig.add_trace(
  go.Scatter(
    x=x,
    y=[polyfit[0]*i + polyfit[1] for i in range(len(df.year_month))],
    name='Trend',
    mode='lines'
  ), row=1, col=1
)
```