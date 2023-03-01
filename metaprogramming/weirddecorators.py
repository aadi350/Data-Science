def df(func, name):
    def wrapper(*args, **kwargs):

        return func(*args, **kwargs)

    return wrapper

@df(name='cc')
def preprocess():
    return 
