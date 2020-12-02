from typing import Tuple, Literal, Union
from pycuda.driver import Stream
from ssnp.funcs import BPMFuncs, SSNPFuncs, Funcs
from pycuda.gpuarray import GPUArray
import numpy as np
from numbers import Real
# from .utils import Multipliers
from ssnp.utils import Config


def ssnp_step(u: GPUArray, u_d: GPUArray, dz: Real, n: GPUArray = None, output: GPUArray = None,
              config: Config = None, stream: Stream = None) -> Tuple[GPUArray, GPUArray]: ...


def bpm_step(u: GPUArray, dz: Real, n: GPUArray = None, output: GPUArray = None,
             config: Config = None, stream: Stream = None) -> GPUArray: ...


def bpm_grad_bp(u: GPUArray, ug: GPUArray, dz: Real, n: GPUArray = None, ng=None,
                config: Config = None, stream: Stream = None) -> GPUArray: ...


def ssnp_grad_bp(u, ug, u_dg, dz, n=None, ng=None,
                 config: Config = None, stream: Stream = None) -> GPUArray: ...


def reduce_mse(u: GPUArray, measurement: GPUArray, stream: Stream = None) -> np.double: ...


def reduce_mse_grad(u: GPUArray, measurement: GPUArray, output: GPUArray = None, stream: Stream = None) -> GPUArray: ...


# def pure_forward_d(u: GPUArray, output: GPUArray = None) -> GPUArray: ...


# def binary_pupil(u: GPUArray, na: float, multiplier: Multipliers = None) -> GPUArray: ...

def sum_batch(u: GPUArray, output: GPUArray = None, stream: Stream = None): ...


def get_multiplier(shape, res=None, stream: Stream = None): ...


def u_mul(u: GPUArray, mul, copy: bool = False, stream: Stream = None, conj: bool = False) -> GPUArray: ...


def merge_prop(uf: GPUArray, ub: GPUArray, config: Config = None,
               copy: bool = False, stream: Stream = None) -> Tuple[GPUArray, GPUArray]: ...


def split_prop(u: GPUArray, u_d: GPUArray, config: Config = None,
               copy: bool = False, stream: Stream = None) -> Tuple[GPUArray, GPUArray]: ...


def merge_grad(ufg: GPUArray, ubg: GPUArray, config: Config = None,
               copy: bool = False, stream: Stream = None) -> Tuple[GPUArray, GPUArray]: ...


def get_funcs(arr_like: GPUArray, config: Config = None, *,
              model: Literal['ssnp', 'bpm', 'any', 'BPM', 'SSNP', 'Any', 'ANY'] = 'any',
              stream: Stream = None,
              fft_type: Literal["reikna", "skcuda"] = "skcuda") -> Union[BPMFuncs, SSNPFuncs, Funcs]: ...
