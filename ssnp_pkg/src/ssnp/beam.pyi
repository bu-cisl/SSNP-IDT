from pycuda.gpuarray import GPUArray
from pycuda.driver import Stream
from typing import Literal, Optional, List, Union
import numpy as np
from ssnp.utils import Multipliers
from ssnp.funcs import Funcs

G_PRO = Union[GPUArray, property]
ARR = Union[GPUArray, np.ndarray]

class TrackStack:
    def __init__(self, u_num, shape): ...

    def clear(self): ...


class BeamArray:
    dtype = np.complex128
    DERIVATIVE = 0
    BACKWARD = 1
    relation: Literal[0, 1]
    _get_array_times: int
    _u1: GPUArray
    _u2: Optional[GPUArray]
    forward: G_PRO
    backward: Optional[G_PRO]
    field: G_PRO
    derivative: Optional[G_PRO]
    multiplier: Multipliers
    _array_pool: List[GPUArray, ...]
    _tape: list
    ops_number: dict
    _fft_funcs: Funcs
    batch: int
    stream: Stream

    def __init__(self, u1: ARR, u2: ARR = None, relation: Literal[0, 1] = DERIVATIVE, total_ops: int = 0): ...

    def _parse(self, info, dz, n, track: bool): ...

    def ssnp(self, dz, n: GPUArray = None, *, track: bool = False): ...

    def bpm(self, dz, n: GPUArray = None, *, track: bool = False): ...

    def forward_mse_loss(self, measurement: GPUArray, *, track: bool = False): ...

    def n_grad(self, output: GPUArray = None) -> GPUArray: ...

    def binary_pupil(self, na: float): ...

    def mul(self, arr: GPUArray, *, hold: BeamArray = None, track=False): ...

    def a_mul(self, arr: GPUArray, *, hold: BeamArray = None, track=False): ...

    def __imul__(self, other): ...

    @staticmethod
    def _iadd_isub(self: BeamArray, other: BeamArray, add: bool): ...

    def __iadd__(self, other: BeamArray): ...

    def __isub__(self, other: BeamArray): ...

    def split_prop(self): ...

    def merge_prop(self): ...

    def _get_array(self) -> GPUArray: ...

    def recycle_array(self, arr: GPUArray): ...

    @staticmethod
    def _u_setter(u: GPUArray, value, dtype=dtype): ...
