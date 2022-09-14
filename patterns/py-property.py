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
    