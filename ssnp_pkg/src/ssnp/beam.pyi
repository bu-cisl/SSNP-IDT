from contextlib import contextmanager

from pycuda.gpuarray import GPUArray
from pycuda.driver import Stream
from typing import Literal, Optional, List, Union, Tuple, Iterable
import numpy as np
from ssnp.utils import Multipliers, Config
from ssnp.funcs import Funcs
from ssnp.utils.auto_gradient import OperationTape, Variable

G_PRO = Union[GPUArray, property]
ARR = Union[GPUArray, np.ndarray]


class BeamArray:
    dtype = np.complex128
    DERIVATIVE = 0
    BACKWARD = 1
    relation: Literal[0, 1]
    _config: Optional[Config]
    _track: bool
    config: Union[Config, property]
    _get_array_times: int
    _u1: GPUArray
    _u2: Optional[GPUArray]
    forward: G_PRO
    backward: Optional[G_PRO]
    field: G_PRO
    derivative: Optional[G_PRO]
    multiplier: Multipliers
    _array_pool: List[GPUArray, ...]
    tape: OperationTape
    ops_number: dict
    _fft_funcs: Funcs
    batch: int
    stream: Stream

    def __init__(self, u1: ARR, u2: ARR = None, relation: Literal[0, 1] = DERIVATIVE, total_ops: int = 0): ...

    def _parse(self, info, dz, n, track: bool): ...

    def ssnp(self, dz, n: GPUArray = None, *, track: bool = False): ...

    def bpm(self, dz, n: GPUArray = None, *, track: bool = False): ...

    def forward_mse_loss(self, measurement: GPUArray): ...

    def midt_batch_mse_loss(self, measurement): ...

    def mse_loss(self, forward: GPUArray = None, *, backward: GPUArray = None): ...

    @contextmanager
    def track(self): ...

    def n_grad(self, output: Iterable = None) -> GPUArray: ...

    def binary_pupil(self, na: float): ...

    def mul(self, arr: GPUArray, *, hold: BeamArray = None, track=False): ...

    def a_mul(self, arr: GPUArray, *, hold: BeamArray = None, track=False): ...

    def __imul__(self, other): ...

    def __iadd__(self, other: BeamArray, sign=1): ...

    def __isub__(self, other: BeamArray): ...

    def conj(self):
        """
        Conjugate the field. Should only apply to forward-only beams
        """

    def split_prop(self): ...

    def merge_prop(self): ...

    def _get_array(self) -> GPUArray: ...

    def recycle_array(self, arr: GPUArray): ...

    def apply_grad(self, grad1: GPUArray, grad2: GPUArray = None): ...

    @staticmethod
    def _u_setter(u: GPUArray, value, dtype=dtype): ...

    def _a_mul_op(self, other): ...

    def _bpm_op(self, u_out: Variable, n_data, dz): ...

    def _ssnp_op(self, u_out: Tuple[Variable, Variable], n_data, dz): ...

    def _change_op(self, vars_in, vars_out): ...
