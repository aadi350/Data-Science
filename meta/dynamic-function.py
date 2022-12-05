def functiongenerator(functionname, operator, a, b):
    from types import FunctionType
    functionstring = []
    for i in operator:
        functionstring.append('''
def arithmetic(a, b):
    op = __import__('operator')
    result = '''+ i + '''(a, b)
    return result
        ''')
        functiontemplate = []
    for i in functionstring:
        functiontemplate.append(compile(i, 'functionstring', 'exec'))
        dynamicfunction = []
    for i, j in zip(functiontemplate, functionname):
        dynamicfunction.append(FunctionType(i.co_consts[0], globals(), j))
