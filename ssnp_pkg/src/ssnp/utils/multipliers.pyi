from typing import Callable, overload, Literal, Tuple, Optional
from numbers import Real
from numpy import ndarray
from pycuda.gpuarray import GPUArray
from pycuda.driver import Stream


def _cache_array(func: Callable[..., Tuple[tuple, Callable[[], ndarray]]]): ...


class Multipliers:
    _cache: dict
    _gpu_cache: dict
    _xy_size: Tuple[int, int]
    _shape: Tuple[int, int]
    res: Tuple[Real, Real, Real]
    stream: Optional[Stream]

    def __init__(self, shape: Tuple[int, int], res: Tuple[Real, Real, Real], stream: Stream = None): ...

    @overload
    def tilt(self, c_ab, *, trunc, periodic_params=None, c_ab_out=None, gpu: Literal[True]) -> GPUArray: ...

    @overload
    def tilt(self, c_ab, *, trunc, periodic_params=None, c_ab_out=None, gpu: Literal[False] = False) -> ndarray: ...

    @overload
    def binary_pupil(self, na, n0=1, *, gpu: Literal[True]) -> GPUArray: ...

    @overload
    def binary_pupil(self, na, n0=1, *, gpu: Literal[False] = False) -> ndarray: ...

    @overload
    def c_gamma(self, *, gpu: Literal[True]) -> GPUArray: ...

    @overload
    def c_gamma(self, *, gpu: Literal[False] = False) -> ndarray: ...

    @staticmethod
    def _near_0(size, pos_0): ...

    @overload
    def soft_crop(self, width: Real, *, total_slices: int = 1, pos: Real = 0, strength: Real = 1,
                  gpu: Literal[True]) -> GPUArray: ...

    @overload
    def soft_crop(self, width: Real, *, total_slices: int = 1, pos: Real = 0, strength: Real = 1,
                  gpu: Literal[False] = False) -> ndarray: ...

    @overload
    def hard_crop(self, width, gpu: Literal[True]) -> GPUArray: ...

    @overload
    def hard_crop(self, width, gpu: Literal[False] = False) -> ndarray: ...

    @overload
    def gaussian(self, sigma: Real, mu: Tuple[Real, Real] = (0, 0), *, gpu: Literal[True]) -> GPUArray: ...

    @overload
    def gaussian(self, sigma: Real, mu: Tuple[Real, Real] = (0, 0), *, gpu: Literal[False] = False) -> ndarray: ...
