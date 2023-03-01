import logging
from collections import UserDict
from typing import Callable

from pandas import read_csv


class AutoLoad(dict):
    """Automatically loads tables when key is set"""

    def __init__(self, reader: Callable, ROOT: str, *args, **kwargs):
        self.reader = reader
        self.ROOT = ROOT
        self.update(*args, **kwargs)

    def __setitem__(self, key: str, item: str) -> None:
        table = self.reader(self.ROOT + item)
        dict.__setitem__(self, key, table)

    def update(self, *args, **kwargs):
        for k, v in dict(*args, **kwargs).items():
            self[k] = v


tables = AutoLoad(read_csv, "./", {"a": "10_test.csv"})
