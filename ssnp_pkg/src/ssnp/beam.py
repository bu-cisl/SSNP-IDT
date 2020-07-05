from collections import Iterable

import numpy as np
from pycuda import gpuarray
from pycuda.gpuarray import GPUArray
from warnings import warn
from .calc import split_prop as calc_split, merge_prop as calc_merge, pure_forward_d
from .calc import ssnp_step, bpm_step, binary_pupil


class BeamArray:
    dtype = np.complex128
    DERIVATIVE = 0
    BACKWARD = 1

    def __init__(self, u1, u2=None, relation=DERIVATIVE):
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
        if self.relation == BeamArray.DERIVATIVE:
            calc_split(self._u1, self._u2)
            self.relation = BeamArray.BACKWARD

    def merge_prop(self):
        if self.relation == BeamArray.BACKWARD:
            calc_merge(self._u1, self._u2)
            self.relation = BeamArray.DERIVATIVE

    def set_pure_forward(self):
        if self._u2 is not None:
            warn("overwrite existing backward part")
            if self.relation == BeamArray.BACKWARD:
                self.relation = BeamArray.DERIVATIVE
            elif self.relation == BeamArray.DERIVATIVE:
                calc_split(self._u1, self._u2)
            else:
                raise AttributeError
        else:
            self.relation = BeamArray.DERIVATIVE
        # now relation is D, u1 is forward
        self._u2 = pure_forward_d(self._u1)

    @property
    def forward(self):
        if self._u2 is not None and self.relation == BeamArray.DERIVATIVE:
            self.split_prop()
        return self._u1

    @property
    def backward(self):
        if self._u2 is not None and self.relation == BeamArray.DERIVATIVE:
            self.split_prop()
        return self._u2

    @property
    def field(self):
        if self._u2 is not None and self.relation == BeamArray.BACKWARD:
            self.merge_prop()
        return self._u1

    @property
    def derivative(self):
        if self._u2 is not None and self.relation == BeamArray.BACKWARD:
            self.merge_prop()
        return self._u2

    def ssnp(self, dz, n=None):
        if self._u2 is None:
            warn("incomplete field information, assuming u is pure forward")
            self.set_pure_forward()
        self.merge_prop()
        if n is None:
            if isinstance(dz, Iterable):
                for dzi in dz:
                    ssnp_step(self._u1, self._u2, dzi)
            else:
                ssnp_step(self._u1, self._u2, dz)
        else:
            if isinstance(n, GPUArray):
                if len(n.shape) == 2:
                    ssnp_step(self._u1, self._u2, dz, n)
                    return
                elif len(n.shape) != 3:
                    raise ValueError(f"invalid n shape {n.shape}")
            if isinstance(dz, Iterable):
                for dz_n in zip(dz, n):
                    ssnp_step(self._u1, self._u2, *dz_n)
            else:
                for ni in n:
                    ssnp_step(self._u1, self._u2, dz, ni)

    def bpm(self, dz, n=None):
        if self._u2 is not None:
            warn("discarding backward propagation part of bidirectional field", stacklevel=2)
            self.split_prop()
            self._u2 = None
        self._parse(lambda var_dz, var_n: bpm_step(self._u1, var_dz, var_n),
                    dz, n)

    @staticmethod
    def _parse(func, dz, n):
        if n is None:
            if isinstance(dz, Iterable):
                for dzi in dz:
                    func(dzi, None)
            else:
                func(dz, None)
        else:
            if isinstance(n, GPUArray):
                if len(n.shape) == 2:
                    func(dz, n)
                    return
                elif len(n.shape) != 3:
                    raise ValueError(f"invalid n shape {n.shape}")
            if isinstance(dz, Iterable):
                for dz_n in zip(dz, n):
                    func(*dz_n)
            else:
                for ni in n:
                    func(dz, ni)
