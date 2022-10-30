class BaseMeta(type):
    def __new__(cls, name, bases, body):
        # example
        if 'baz' not in body:
            raise TypeError()
        return super().__new__(cls, name, bases, body)

class Base(metaclass=BaseMeta):
    def bar(self,):
        return self.baz()