from typing import Tuple, Literal, Union
from .funcs import BPMFuncs, SSNPFuncs
from pycuda.gpuarray import GPUArray
import numpy as np
from numbers import Real
from .utils import Multipliers


def ssnp_step(u: GPUArray, u_d: GPUArray, dz: Real, n: GPUArray = None, output: GPUArray = None) -> Tuple[
    GPUArray, GPUArray]: ...


def bpm_step(u: GPUArray, dz: Real, n: GPUArray = None, output: GPUArray = None) -> GPUArray: ...


def bpm_grad_bp(u, ug, dz, n=None, ng=None) -> GPUArray: ...


def ssnp_grad_bp(u, ug, u_dg, dz, n=None, ng=None) -> GPUArray:


def reduce_mse(u: GPUArray, measurement: GPUArray) -> np.double: ...


def reduce_mse_grad(u: GPUArray, measurement: GPUArray, output: GPUArray = None) -> GPUArray: ...


def pure_forward_d(u: GPUArray, output: GPUArray = None) -> GPUArray: ...


def binary_pupil(u: GPUArray, na: float, multiplier: Multipliers = None) -> GPUArray: ...


def get_multiplier(arr_like): ...


def u_mul_grad_bp(ug, mul): ...


def merge_prop(uf: GPUArray, ub: GPUArray, copy: bool = False) -> Tuple[GPUArray, GPUArray]: ...


def split_prop(u: GPUArray, u_d: GPUArray, copy: bool = False) -> Tuple[GPUArray, GPUArray]: ...


def merge_grad(ufg: GPUArray, ubg: GPUArray, copy: bool = False) -> Tuple[GPUArray, GPUArray]: ...


def get_funcs(arr_like: GPUArray, res, model: Literal['ssnp', 'bpm', 'any']) -> Union[BPMFuncs, SSNPFuncs]: ...
