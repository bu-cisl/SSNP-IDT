import numpy as np
from pycuda.gpuarray import GPUArray
from pycuda.elementwise import ElementwiseKernel
from reikna.core.computation import ComputationCallable


def _c_gamma(shape, res) -> np.ndarray: ...


class Funcs:
    shape: tuple
    res: tuple
    n0: float
    kz: np.ndarray
    kz_gpu: GPUArray
    eva: np.ndarray
    _fft_callable: ComputationCallable
    __temp_memory_pool: dict
    _prop_cache: dict
    _pupil_cache: dict

    def __init__(self, arr_like: GPUArray, res, n0): ...

    def fft(self, arr: GPUArray, output: GPUArray = None, copy: bool = False, inverse=False): ...

    def ifft(self, arr: GPUArray, output: GPUArray = None, copy: bool = False): ...

    def diffract(self, *args) -> None: ...

    def scatter(self, *args) -> None: ...

    def binary_pupil(self, u: GPUArray, na: float) -> GPUArray: ...

    @staticmethod
    def get_temp_mem(arr_like: GPUArray, index=0): ...


class BPMFuncs(Funcs):
    def _get_prop(self, dz): ...

    def diffract(self, a, dz) -> None: ...

    def scatter(self, u, n, dz) -> None: ...


class SSNPFuncs(Funcs):
    __fused_mam_callable: ElementwiseKernel

    def _get_prop(self, dz): ...

    def diffract(self, a, a_d, dz) -> None: ...

    def scatter(self, u, u_d, n, dz) -> None: ...
