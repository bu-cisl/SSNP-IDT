from typing import Tuple, Literal
from pycuda.gpuarray import GPUArray
import numpy as np
from numbers import Real


def ssnp_step(u: GPUArray, u_d: GPUArray, dz: Real, n: GPUArray = None, output: GPUArray = None) -> Tuple[
    GPUArray, GPUArray]: ...


def bpm_step(u: GPUArray, dz: Real, n: GPUArray = None, output: GPUArray = None) -> GPUArray: ...


def bpm_grad_bp(u, ug, dz, n=None, ng=None) -> GPUArray: ...


def reduce_mse(u: GPUArray, measurement: GPUArray) -> np.double: ...


def reduce_mse_grad(u: GPUArray, measurement: GPUArray, output: GPUArray = None) -> GPUArray: ...


def pure_forward_d(u: GPUArray, output: GPUArray = None) -> GPUArray: ...


def binary_pupil(u: GPUArray, na: float) -> GPUArray: ...


def merge_prop(ub: GPUArray, uf: GPUArray, copy: bool = False) -> Tuple[GPUArray, GPUArray]: ...


def split_prop(u: GPUArray, u_d: GPUArray, copy: bool = False) -> Tuple[GPUArray, GPUArray]: ...


def get_funcs(arr_like: GPUArray, res, model: Literal['ssnp', 'bpm', 'any']): ...
