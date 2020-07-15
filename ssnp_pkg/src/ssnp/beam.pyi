from pycuda.gpuarray import GPUArray
from typing import Literal, Optional, List, Union
import numpy as np
from .utils import Multipliers


class TrackStack:
    def __init__(self, u_num, shape): ...

    def clear(self): ...


class BeamArray:
    dtype = np.complex128
    DERIVATIVE = 0
    BACKWARD = 1
    _u1: GPUArray
    _u2: Optional[GPUArray]
    forward: Union[GPUArray, property]
    backward: Union[Optional[GPUArray], property]
    field: Union[GPUArray, property]
    derivative: Union[Optional[GPUArray], property]
    multiplier: Multipliers
    _array_pool: List[GPUArray, ...]

    _history: list

    def __init__(self, u1: GPUArray, u2: GPUArray = None, relation: Literal[0, 1] = DERIVATIVE):
        ...

    def _parse(self, info, dz, n, track: bool): ...

    def ssnp(self, dz, n: GPUArray = None, *, track: bool = False): ...

    def bpm(self, dz, n: GPUArray = None, *, track: bool = False): ...

    def forward_mse_loss(self, measurement: GPUArray, *, track: bool = False): ...

    def n_grad(self, output: GPUArray = None) -> GPUArray: ...

    def binary_pupil(self, na: float): ...

    def mul(self, arr: GPUArray, *, track=False): ...

    def split_prop(self): ...

    def merge_prop(self): ...

    # def set_pure_forward(self): ...

    def _get_array(self) -> GPUArray: ...

    def recycle_array(self, arr: GPUArray): ...

    @staticmethod
    def _u_setter(u: GPUArray, value, dtype=dtype): ...
