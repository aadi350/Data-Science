# Python Properties

Using `getter` and `setter` is non-Pythonic and breaks encapsulation, see [this RealPython article](https://realpython.com/python-property/#managing-attributes-in-your-classes) for more details.

This shows the general pattern

```python
# foo.py
class Foo:
    def __init__(self, prop):
        self._prop = prop

    @property
    def prop(self,):
        return self._prop
    
    @prop.setter
    def prop(self, value):
        self._prop = value

    @prop.deleter
    def prop(self,):
        del self.prop
    
```

How to access/set:
```python
f = Foo(10)

f.prop # getter
f.prop = 12 # setter
```