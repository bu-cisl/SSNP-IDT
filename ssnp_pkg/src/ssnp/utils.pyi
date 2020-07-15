from typing import List, Tuple, Callable, Any

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

    def tilt(self, c_ab, *, trunc, gpu: bool = False) -> GPUArray: ...

    def binary_pupil(self, na, *, gpu: bool = False) -> GPUArray: ...

    def c_gamma(self, *, gpu: bool = False) -> GPUArray: ...
