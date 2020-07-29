from typing import Callable

import numpy as np
from pycuda.gpuarray import GPUArray
from pycuda.elementwise import ElementwiseKernel
from reikna.core.computation import ComputationCallable
from .utils import Multipliers


# def _c_gamma(shape, res) -> np.ndarray: ...


class Funcs:
    shape: tuple
    res: tuple
    n0: float
    kz: np.ndarray
    kz_gpu: GPUArray
    eva: np.ndarray
    multiplier: Multipliers
    _fft_callable: ComputationCallable
    __temp_memory_pool: dict
    _prop_cache: dict

    def __init__(self, arr_like: GPUArray, res, n0): ...

    def fft(self, arr: GPUArray, output: GPUArray = None, copy: bool = False, inverse=False): ...

    def ifft(self, arr: GPUArray, output: GPUArray = None, copy: bool = False): ...

    def diffract(self, *args) -> None: ...

    def scatter(self, *args) -> None: ...

    @staticmethod
    def get_temp_mem(arr_like: GPUArray, index=0): ...

    @staticmethod
    def reduce_mse_cr(u: GPUArray, m: GPUArray) -> GPUArray: ...

    @staticmethod
    def reduce_mse_cc(u: GPUArray, m: GPUArray) -> GPUArray: ...

    @staticmethod
    def mse_cr_grad(u: GPUArray, m: GPUArray, out: GPUArray): ...

    @staticmethod
    def mse_cc_grad(u: GPUArray, m: GPUArray, out: GPUArray): ...

    @staticmethod
    def mul_grad_bp(ug: GPUArray, mul: GPUArray): ...


class BPMFuncs(Funcs):
    def _get_prop(self, dz): ...

    def diffract(self, a, dz) -> None: ...

    def diffract_g(self, ag, dz): ...

    def scatter(self, u, n, dz) -> None: ...

    def scatter_g(self, u, n, ug, ng, dz): ...


class SSNPFuncs(Funcs):
    __fused_mam_callable: ElementwiseKernel

    @staticmethod
    def merge_prop_kernel(af, ab, kz): ...

    @staticmethod
    def split_prop_kernel(a, a_d, kz): ...

    @staticmethod
    def merge_grad_kernel(afg, abg, kz): ...

    def _get_prop(self, dz): ...

    def diffract(self, a, a_d, dz) -> None: ...

    def diffract_g(self, ag, a_dg, dz): ...

    def scatter(self, u, u_d, n, dz) -> None: ...

    def scatter_g(self, u, n, ug, u_dg, ng, dz): ...
