from pandas import DataFrame, Series

# add processing functions
# funcs = [
#   fix_date,
#   mul_column
# ]
# etc
inputs = {
    "1": DataFrame({"id": [10] * 2}),
    "2": DataFrame({"id": [10] * 2}),
    "3": DataFrame({"id": [10] * 2}),
    "4": DataFrame({"id": [10] * 2}),
}

outputs = inputs

for i in inputs:
    fixture_code = f"""
import pytest
    
@pytest.fixture
def input{i}():
    return inputs[i]

def test{i}(input{i}):
    assert input{i}.equals(outputs[str({i})])
    """
    exec(fixture_code)
