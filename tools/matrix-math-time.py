from pathlib import Path

# import cudf
import numpy
import pandas
import polars

DATA_PATH = Path('/storage/data')

df = pandas.read_csv(DATA_PATH / 'airline_2m.csv', encoding='latin-1')

# Aggregations
## .is_unique on 
df.Dest.is_unique

## .is_monotonic_increasing
df.CRSDepTime.is_monotonic_increasing

## .quantile()
df.ArrDelay.quantile(q=[0.00, 0.01, 0.25, 0.50, 0.75, 0.99, 1.00])

## percent meeting a criteria
df.ArrDelay.gt(10).sum().mean()

## Pearson correlation
df.ArrDelay.agg('corr', other=df.DepDelay)

## Kurtosis
df.ArrDelay.agg('kurt')

def max_mean_median_min():
    ## max, mean, median, min
    df.agg(
        mean_dep_delay=('DepDelay','mean'),
        max_dep_delay=('DepDelay','max'),
        min_dep_delay=('DepDelay','min'),
        median_dep_delay=('DepDelay','median'),
        std_dep_delay=('DepDelay','std'),
    )


df.ArrDelay.mean() 