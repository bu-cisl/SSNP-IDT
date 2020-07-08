from pycuda.gpuarray import GPUArray
from typing import Literal, Optional


class BeamArray:
    dtype: type
    DERIVATIVE = 0
    BACKWARD = 1
    _u1: GPUArray
    _u2: Optional[GPUArray]
    forward: GPUArray
    backward: Optional[GPUArray]
    field: GPUArray
    derivative: Optional[GPUArray]


    def __init__(self, u1: GPUArray, u2: GPUArray = None, relation: Literal[0, 1] = DERIVATIVE): ...

    @staticmethod
    def _parse(func, dz, n): ...

    def ssnp(self, dz, n: GPUArray=None): ...

    def bpm(self, dz, n: GPUArray=None): ...

    def binary_pupil(self, na: float): ...

    def split_prop(self): ...

    def merge_prop(self): ...

    def set_pure_forward(self): ...