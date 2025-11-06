from typing import Literal, Union, Optional, Sequence
from pycuda.driver import Stream
from ssnp.funcs import BPMFuncs, SSNPFuncs, Funcs, MLBFuncs
from pycuda.gpuarray import GPUArray
import numpy as np
from numbers import Real, Complex
# from .utils import Multipliers
from ssnp.utils import Config


def ssnp_step(u: GPUArray, u_d: GPUArray, dz: Real, n: GPUArray = None,
              output: Sequence[Optional[GPUArray], Optional[GPUArray]] = None,
              config: Config = None, stream: Stream = None) -> tuple[GPUArray, GPUArray]: ...


def bpm_step(u: GPUArray, dz: Real, n: GPUArray = None, output: GPUArray = None,
             config: Config = None, stream: Stream = None) -> GPUArray: ...


def bpm_grad_bp(u: Optional[GPUArray], ug: GPUArray, dz: Real, n: GPUArray = None, ng=None,
                config: Config = None, stream: Stream = None) -> GPUArray: ...


def ssnp_grad_bp(u: Optional[GPUArray], ug: GPUArray, u_dg: Optional[GPUArray], dz: Real, n=None, ng=None,
                 config: Config = None, stream: Stream = None) -> GPUArray: ...


def mlb_step(u: GPUArray, temp_like_u: Optional[GPUArray], dz: Real, n: GPUArray = None,
             config: Config = None, stream: Stream = None): ...


def reduce_mse(u: GPUArray, measurement: GPUArray, stream: Stream = None) -> np.double: ...


def reduce_mse_grad(u: GPUArray, measurement: GPUArray, output: GPUArray = None, stream: Stream = None) -> GPUArray: ...


# def pure_forward_d(u: GPUArray, output: GPUArray = None) -> GPUArray: ...


# def binary_pupil(u: GPUArray, na: float, multiplier: Multipliers = None) -> GPUArray: ...

def sum_batch(u: GPUArray, output: GPUArray = None, stream: Stream = None): ...


def copy_batch(u: GPUArray, output: GPUArray, stream=None): ...


def abs_c2c(u: GPUArray, output: GPUArray = None, stream=None): ...


def get_multiplier(shape, res=None, stream: Stream = None): ...


def u_mul(u: GPUArray, mul: Union[Complex, GPUArray], *,
          copy: bool = False, out: GPUArray = None, stream: Stream = None, conj: bool = False) -> GPUArray: ...


def merge_prop(uf: GPUArray, ub: GPUArray, config: Config = None,
               copy: bool = False, stream: Stream = None) -> tuple[GPUArray, GPUArray]: ...


def split_prop(u: GPUArray, u_d: GPUArray, config: Config = None,
               copy: bool = False, stream: Stream = None) -> tuple[GPUArray, GPUArray]: ...


def merge_grad(ufg: GPUArray, ubg: GPUArray, config: Config = None,
               copy: bool = False, stream: Stream = None) -> tuple[GPUArray, GPUArray]: ...


def split_grad(ug: GPUArray, u_dg: GPUArray, config: Config = None,
               copy: bool = False, stream: Stream = None) -> tuple[GPUArray, GPUArray]: ...


def get_funcs(arr_like: GPUArray, config: Config = None, *,
              model: Literal['ssnp', 'bpm', 'any', 'BPM', 'SSNP', 'mlb', 'Any', 'ANY'] = 'any',
              stream: Stream = None,
              fft_type: Literal["reikna", "skcuda"] = "skcuda") -> Union[BPMFuncs, SSNPFuncs, MLBFuncs, Funcs]: ...
