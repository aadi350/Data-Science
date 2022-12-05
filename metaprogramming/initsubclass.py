import os

'''
initsubclass so that we don't need metaclass
'''

class BaseClass:
    def __init_subclass__(cls, **kwargs):
        # does some initialization 
        print(f'{cls} __init_subclass__')
        super().__init_subclass__(**kwargs)

class SubClass(BaseClass):
    pass

import weakref

class WeakAttribute:
    def __init__(self, *args, **kwargs):
        print('WeakAttribute __init__')
        super().__init__(*args, **kwargs)

    def __get__(self, instance, owner):
        return instance.__dict__[self.name]()

    def __set__(self, instance, value):
        instance.__dict__[self.name] = weakref.ref(value)

    def __set_name__(self, owner, name):
        print(self, owner, name)
        self.name = name

'''
The __set_name__ magic method lets you know 
where instances of this class are used and 
what attribute they are assigned to. 
The owner field is the class where it is used. 
The name field is the attribute name it is assigned 
to
'''

class A:
    def __set_name__(self, owner, name):
        print(f'Calling class :{owner}')
        print(f'Calling name:{name}')

class B:
    a = A()
    b = A()
    c = A()

'''
Output:
Calling class :<class '__main__.B'>
Calling name:a
Calling class :<class '__main__.B'>
Calling name:b
Calling class :<class '__main__.B'>
Calling name:c
'''
