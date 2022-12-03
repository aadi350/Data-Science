_NA = object()


class OverloadList(list):
    pass


class OverloadDictionary(dict):
    def __setitem__(self, key, value):
        prior_val = self.get(key, _NA)

        # checks whether to actually overload
        overloaded = getattr(value, "__overload__", False)

        if prior_val is _NA:  # if not exist
            # need a list if overloaded
            insert_val = OverloadList([value]) if overloaded else value
            super().__setitem__(key, insert_val)

        elif isinstance(prior_val, OverloadList):
            # if try to set value with another overload list
            #   and not overloaded
            #   raise error
            if not overloaded:
                raise AttributeError("Must mark overloaded classes with @overload")
            prior_val.append(value)
        else:  # need to overload
            if overloaded:
                raise AttributeError("Must mark overloaded classes with @overload")
            super().__setitem__(key, value)


d = OverloadDictionary()

# this sets the overload meta
def overload(fn):
    fn.__overload__ = True
    return fn


@overload
def f(
    self,
):
    pass


d["a"] = 1
d["b"] = 2
d["f"] = f
d["f"] = f


print(d)


class OverloadMeta(type):
    @classmethod
    def __prepare__(mcs, name, bases):
        return OverloadDictionary()

    def __new__(cls, *args):
        print(cls)
        print("\n")
        print(args)
        return type.__new__(cls, *args)


class WithOverload(metaclass=OverloadMeta):
    def test(self, a):
        print("test")

    def test(self, b):
        print("test")


# type(name, bases, dict)
w = WithOverload()

"""
    __prepare__ method customizes the class namespace dict
    when class created, __prepare__ is what does this
    __prepare__ returns a dictionary-like object which is used to store the 
        class member definitions during evaluation of the class body.
        In other words, the class body is evaluated as a function block 
        (just like it is now), except that the local variables dictionary is 
        replaced by the dictionary returned from __prepare__. 
        This dictionary object can be a regular dictionary or a custom mapping type.
"""
