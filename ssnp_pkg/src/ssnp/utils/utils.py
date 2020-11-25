from functools import lru_cache
from pycuda import gpuarray, driver as cuda


def param_check(**kwargs):
    name0 = None
    shape0 = None
    for name in kwargs:
        arr = kwargs[name]
        if arr is None:
            continue
        # type check
        if not isinstance(arr, gpuarray.GPUArray):
            raise TypeError(f"'{name}' is not a GPUArray")
        # shape check
        if name0 is None:
            name0 = name
            shape0 = arr.shape
        else:
            if arr.shape != shape0:
                raise ValueError(f"cannot match '{name}' shape {arr.shape} with '{name0}' shape {shape0}")


@lru_cache(maxsize=None)
def get_stream(ctx):
    if ctx == cuda.Context.get_current():
        return cuda.Stream()
    else:
        raise NotImplementedError("first time can only get stream in current context. "
                                  f"ctx:{ctx}, current:{cuda.Context.get_current()}")


def get_stream_in_current():
    return get_stream(cuda.Context.get_current())


class Config:
    _res = None
    _n0 = 1.
    _xyz = None
    _lambda = None
    _callbacks = None

    def __init__(self):
        self.clear_updater()

    @property
    def xyz(self):
        if self._xyz is None:
            raise AttributeError("xyz is uninitialized")
        return self._xyz

    @property
    def lambda0(self):
        if self._lambda is None:
            raise AttributeError("wave length is uninitialized")
        return self._lambda

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
        # if self._res is not None:
        #     warn(f"resetting res value from {self._res}")
        value = tuple(float(res_i) for res_i in value)
        if self._res != value:
            assert len(value) == 3
            self._update(attr='res', old=self._res, new=value)
            self._res = value

    @xyz.setter
    def xyz(self, value):
        self._xyz = tuple(float(size_i) for size_i in value)
        assert len(self._xyz) == 3
        self._try_calc_res()

    @lambda0.setter
    def lambda0(self, value):
        self._lambda = float(value)
        self._try_calc_res()

    @n0.setter
    def n0(self, value):
        value = float(value)
        if self._n0 != value:
            self._try_n0fix_res(value)
            self._update(attr='n0', old=self._n0, new=value)
            self._n0 = value

    def _try_calc_res(self):
        try:
            self.res = (size_i / self.lambda0 * self.n0 for size_i in self.xyz)
            # self.res = (res_i * self.n0 for res_i in self.res)
        except AttributeError:
            pass

    def _try_n0fix_res(self, new_value):
        try:
            value = (res_i / self._n0 * new_value for res_i in self.res)
            self._res = tuple(float(res_i) for res_i in value)
        except AttributeError:
            pass

    def register_updater(self, updater):
        if updater is not None:
            assert hasattr(updater, "__call__"), "updater function is not callable"
            self._callbacks.append(updater)

    def clear_updater(self):
        self._callbacks = []

    def _update(self, **kwargs):
        for i in self._callbacks:
            i(**kwargs)

    def set(self, **kwargs):
        for attr in ('n0', 'xyz', 'lambda0', 'res'):  # the order is important
            if value := kwargs.pop(attr, None):
                setattr(self, attr, value)
        # for key in kwargs:
        #     if key == "res":
        #         self.res = kwargs[key]
        #     else:
        #         raise TypeError(f"'{key}' is invalid as a configuration item")

    def __copy__(self):
        cp = type(self)()
        cp._res = self._res
        cp._n0 = self._n0
        return cp


config = Config()
