class Consts:
    _res = None
    _n0 = 1

    @property
    def res(self):
        if self._res is None:
            raise AttributeError("res is uninitialized")
        return self._res

    @property
    def n0(self):
        return self._n0

    @res.setter
    def res(self, value):
        self._res = value


config = Consts()
