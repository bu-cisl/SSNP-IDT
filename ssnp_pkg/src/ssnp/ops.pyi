from pycuda.driver import Stream
from pycuda.gpuarray import GPUArray

from ssnp import BeamArray
from ssnp.utils.auto_gradient import Operation, Variable as Var


class MulOp(Operation):
    _beam: BeamArray
    _bi_dir: bool

    def __init__(self, other: GPUArray, beam: BeamArray): ...


class FourierMulOp(MulOp):
    pass


class MSELossOp(Operation):
    _grads: list[GPUArray]
    _beam: BeamArray
    def __init__(self, beam: BeamArray, other_forward: GPUArray | None, other_backward: GPUArray | None): ...

    def gradient(self, out=None): ...
