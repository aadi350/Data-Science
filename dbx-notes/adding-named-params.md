In task-file:
```python 
from argparse import ArgumentParser


def entrypoint():  # pragma: no cover
    """Entrypoint for spark wheel jobs."""
    parser = ArgumentParser()
    parser.add_argument("--localmode", dest="localmode", default=False)
    parser.add_argument("--fundingsdatapath", dest="fundingsdatapath", default="tmp/fundings")
    parser.add_argument("--datalakename", dest="datalakename", default="datalakename")

    args = parser.parse_args()
```