from pandas import DataFrame

df = DataFrame({
    'id': ['a', 'b'],
    'list_column': ['[1, 2, 3]', '[4, 5, 6]']
})

df['list_column'].apply(ast.literal_eval)


'''Output:
0    [1, 2, 3]
1    [4, 5, 6]
Name: list_column, dtype: object
'''
