from typing import Literal
from functools import wraps
# def limited_argument_choices(choices: Dict[int or str, Iterable] = None) -> Callable:
#     """decorator factory: force arguments of a func limited in the given choices

#     :param choices: a dict which describes the choices for the value-limited arguments.
#             the key of the dict must be either the index of args or the key_str of kwargs,
#             while the value of the dict must be an iterable."""
#     err_fmt = "value of '{}' is not a valid choice: '{}'"

#     def decorator(func):
#         if not choices:
#             return func

#         @wraps(func)
#         def decorated_func(*args, **kwargs):
#             for i in range(len(args)):
#                 if i in choices and args[i] not in choices[i]:
#                     param_name = func.__code__.co_varnames[i]
#                     valid_choices = list(choices[i])
#                     raise ValueError(err_fmt.format(param_name, valid_choices))
#             for k in kwargs:
#                 if k in choices and kwargs[k] not in choices[k]:
#                     raise ValueError(err_fmt.format(k, list(choices[k])))

#             return func(*args, **kwargs)

#         return decorated_func

#     return decorator

# @limited_argument_choices({1: (0, 1, 2), 'y': ('hi', 'hello')})
# def test(a, b, c, y=1):
#     print(a, b, c, y)


from functools import partial


def func1(x, a, b, c):
    return a*x**2 + b*x + c


func2 = partial(func1, a=3, b=2, c=1)

print(func2(1))