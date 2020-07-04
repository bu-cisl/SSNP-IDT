import numpy as np
from pycuda import gpuarray
from pycuda.gpuarray import GPUArray
from warnings import warn
from .calc import split_prop, merge_prop, pure_forward_d


class BeamArray:
    dtype = np.complex128
    DERIVATIVE = 0
    BACKWARD = 1

    def __init__(self, u1: GPUArray, u2: GPUArray = None, relation=DERIVATIVE):
        def to_gpu(u, name):
            if isinstance(u, GPUArray):
                if u.dtype != self.dtype:
                    raise ValueError(f"GPUArray {name} must be {self.dtype} but not {u.dtype}")
                return u
            elif isinstance(u, np.ndarray):
                if u.dtype != self.dtype:
                    warn(f"force casting {name} to {self.dtype}", stacklevel=3)
                u = u.astype(self.dtype)
                return gpuarray.to_gpu(u)
            else:
                u = np.array(u, dtype=np.complex128)
                warn(f"converting {name} to numpy array as fallback", stacklevel=3)
                to_gpu(u, name)

        self._u1 = to_gpu(u1, "u1")
        if u2 is not None:
            if u1.shape != u2.shape:
                raise ValueError(f"u1 shape {u1.shape} cannot match u2 shape {u2.shape}")
            if relation not in {BeamArray.DERIVATIVE, BeamArray.BACKWARD}:
                raise ValueError
            self.relation = relation
            self._u2 = to_gpu(u2, "u2")
        else:
            self._u2 = None

    def split_prop(self):
        ...

    def set_pure_forward(self):
        if self._u2 is not None:
            warn("resetting backward part to 0")
            if self.relation == BeamArray.BACKWARD:
                self.relation = BeamArray.DERIVATIVE
            elif self.relation == BeamArray.DERIVATIVE:
                split_prop(self._u1, self._u2)
            else:
                raise AttributeError
        else:
            self.relation = BeamArray.DERIVATIVE
        # now relation is D, u1 is forward
        self._u2 = pure_forward_d(self._u1)
