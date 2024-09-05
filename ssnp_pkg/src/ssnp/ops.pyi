from pycuda.gpuarray import GPUArray
from ssnp.utils.auto_gradient import Operation, Variable as Var
from ssnp import BeamArray


class MulOp(Operation):
    _beam: BeamArray
    _bi_dir: bool

    def __init__(self, other: GPUArray, beam: BeamArray): ...


class FourierMulOp(MulOp):
    pass
