from numbers import Real
from typing import List, Tuple, Callable, Any, overload, Literal

import numpy as np
from pycuda.gpuarray import GPUArray


# def tilt(img: GPUArray, c_ab: tuple, *, trunc: bool = False, copy: bool = False): ...


def param_check(**kwargs): ...


def _cache_array(func: Callable[..., Tuple[tuple, Callable[[], np.ndarray]]]): ...


class Multipliers:
    _cache: dict
    _gpu_cache: dict
    shape: Tuple[int, int]
    res: Tuple[float, float, float]

    def __init__(self, shape: Tuple[int, int], res: Tuple[float, float, float]): ...

    @overload
    def tilt(self, c_ab, *, trunc, gpu: Literal[True]) -> GPUArray: ...

    @overload
    def tilt(self, c_ab, *, trunc, gpu: Literal[False] = False) -> np.ndarray: ...

    @overload
    def binary_pupil(self, na, *, gpu: Literal[True]) -> GPUArray: ...

    @overload
    def binary_pupil(self, na, *, gpu: Literal[False] = False) -> np.ndarray: ...

    @overload
    def c_gamma(self, *, gpu: Literal[True]) -> GPUArray: ...

    @overload
    def c_gamma(self, *, gpu: Literal[False] = False) -> np.ndarray: ...

    @overload
    def soft_crop(self, width: Real, *, total_slices: int = 1, pos: Real = 0, strength: Real = 1,
                  gpu: Literal[True]) -> GPUArray: ...

    @overload
    def soft_crop(self, width: Real, *, total_slices: int = 1, pos: Real = 0, strength: Real = 1,
                  gpu: Literal[False] = False) -> np.ndarray: ...
