from pycuda import gpuarray


def param_check(**kwargs):
    name0 = None
    shape0 = None
    for name in kwargs:
        arr = kwargs[name]
        if arr is None:
            continue
        # type check
        assert isinstance(arr, gpuarray.GPUArray)
        # shape check
        if name0 is None:
            name0 = name
            shape0 = arr.shape
        else:
            if arr.shape != shape0:
                raise ValueError(f"cannot match '{name}' shape {arr.shape} with '{name0}' shape {shape0}")


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
        self._res = tuple(float(res_i) for res_i in value)
        assert len(self._res) == 3

    def __call__(self, **kwargs):
        for key in kwargs:
            if key == "res":
                self.res = kwargs[key]
            else:
                raise TypeError(f"'{key}' is invalid as a configuration item")


config = Consts()
