import numba
from datetime import datetime
from multiprocessing import Pool
import pandas as pd
import multiprocessing as mp

df = pd.read_csv('../tools/airline_2m.csv', encoding='latin-1', nrows=100)

pool = mp.Pool(processes=(mp.cpu_count() - 1))

@numba.jit
def f(arg):
    print(datetime.now())
    return arg**2

f(df.Quarter.to_numpy())