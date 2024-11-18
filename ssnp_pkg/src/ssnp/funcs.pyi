from typing import Literal, Dict

import numpy as np
from pycuda.gpuarray import GPUArray
from pycuda.elementwise import ElementwiseKernel
from pycuda.driver import Stream, Function
from pycuda.reduction import ReductionKernel
from .utils import Multipliers


class Funcs:
    _initialized: bool
    shape: tuple
    batch: int
    res: tuple
    n0: float
    stream: Stream
    kz: np.ndarray
    kz_gpu: GPUArray
    eva: np.ndarray
    multiplier: Multipliers
    # __temp_memory_pool: dict
    _prop_cache: dict
    reduce_sse_cr_krn: ReductionKernel
    reduce_sse_cc_krn: ReductionKernel
    mse_cr_grad_krn: ElementwiseKernel
    mse_cc_grad_krn: ElementwiseKernel
    abs_cc_krn: ElementwiseKernel
    mul_conj_krn: ElementwiseKernel
    mul_krn: ElementwiseKernel
    sum_cmplx_batch_krn: ElementwiseKernel
    sum_double_batch_krn: ElementwiseKernel
    _fft_reikna: callable
    _fft_sk: callable

    def __init__(self, arr_like: GPUArray, res, n0, stream: Stream = None,
                 fft_type: Literal["reikna", "skcuda"] = "skcuda"): ...

    @staticmethod
    def _compile_reikna_fft(shape, dtype, stream): ...

    def _fft(self, arr, out, inverse): ...

    def fft(self, arr: GPUArray, output: GPUArray = None, copy: bool = False, inverse=False) -> GPUArray: ...

    def ifft(self, arr: GPUArray, output: GPUArray = None, copy: bool = False) -> GPUArray: ...

    def fourier(self, arr: GPUArray, copy: bool = False): ...

    def diffract(self, *args) -> None: ...

    def scatter(self, *args) -> None: ...

    def diffract_g(self, *args) -> None: ...

    def scatter_g(self, *args) -> None: ...

    def _get_prop(self, dz): ...

    def reduce_sse(self, field: GPUArray, measurement: GPUArray) -> GPUArray: ...

    def mse_grad(self, field: GPUArray, measurement: GPUArray, gradient: GPUArray): ...

    def sum_batch(self, batch: GPUArray, sum_: GPUArray): ...

    @staticmethod
    def _op_krn(batch, xt, yt, zt, operator, name=None, y_func=None) -> Function: ...

    def op(self, x: GPUArray, operator: Literal["+", "-", "*", "/"], y: GPUArray, *,
           out: GPUArray = None, batchwise: bool = True, name: str = None, y_func: str = None): ...

    def conj(self, arr: GPUArray, out: GPUArray = None) -> None:
        """
        copied from GPUArray.conj(self), do conj in-place
        :param arr: the input GPUArray to apply conjugate
        :param out: the output GPUArray. None for in-place operation (Default)
        :return: out
        """


class BPMFuncs(Funcs):
    def _get_prop(self, dz): ...

    def diffract(self, a: GPUArray, dz) -> None: ...

    def diffract_g(self, ag, dz): ...

    def scatter(self, u, n, dz) -> None: ...

    def scatter_g(self, u, n, ug, ng, dz): ...


class SSNPFuncs(Funcs):
    _fused_mam_callable_krn: ElementwiseKernel
    _merge_prop_krn: ElementwiseKernel
    _split_prop_krn: ElementwiseKernel
    _merge_grad_krn: ElementwiseKernel
    _split_grad_krn: ElementwiseKernel

    def merge_prop(self, af, ab): ...

    def split_prop(self, a, a_d): ...

    def merge_grad(self, afg, abg): ...

    def split_grad(self, ag, a_dg): ...

    def _get_prop(self, dz): ...

    def diffract(self, a, a_d, dz) -> None: ...

    def diffract_g(self, ag, a_dg, dz): ...

    def scatter(self, u, u_d, n, dz) -> None: ...

    def scatter_g(self, u, n, ug, u_dg, ng, dz): ...
