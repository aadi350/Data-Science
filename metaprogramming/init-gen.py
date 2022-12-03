from dataclasses import dataclass


class DataClassMeta(type):
    def __new__(mcs, name, bases, namespace, **kwargs):
        annotations = namespace.get("__annotations__")
        print(annotations)

        return super().__new__(mcs, name, bases, namespace, **kwargs)


class DataClass(metaclass=DataClassMeta):
    pass


class MyClass(DataClass):
    x: int
    y: int


m = MyClass()
