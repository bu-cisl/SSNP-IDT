from collections import Iterable
from numbers import Number
import numpy as np
from pycuda import gpuarray
from pycuda.gpuarray import GPUArray
from warnings import warn
from .calc import split_prop as calc_split, merge_prop as calc_merge, pure_forward_d
from .calc import ssnp_step, bpm_step, binary_pupil as calc_pupil, reduce_mse
from .calc import bpm_grad_bp, reduce_mse_grad


# class TrackStack:
#     def __init__(self, u_num, arr_like):
#         self.track = []
#         self._len = 0
#         self._num = u_num
#
#
#     def __len__(self):
#         return self._len
#
#     def __getitem__(self, item):
#         if item < self._len:
#             return self.track[item]
#         else:
#             raise IndexError("index out of range")
#
#     def clear(self):
#         self._len = 0
#
#     def get_push(self):
#         try:
#             return self.track[self._len]
#         except IndexError:
#             self.track.append({'u': [gpuarray.empty_like(self.), gpuarray.e]})
#         finally:
#             self._len += 1


class BeamArray:
    dtype = np.complex128
    DERIVATIVE = 0
    BACKWARD = 1

    def __init__(self, u1, u2=None, relation=BACKWARD):
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
                raise ValueError("unknown relation type")
            self.relation = relation
            self._u2 = to_gpu(u2, "u2")
        else:
            self._u2 = None
        self._history = []
        self._array_pool = []

    def _get_array(self):
        if len(self._array_pool) == 0:
            return gpuarray.empty_like(self._u1)
        else:
            return self._array_pool.pop()

    def recycle_array(self, arr):
        self._array_pool.append(arr)

    def split_prop(self):
        if self.relation == BeamArray.DERIVATIVE:
            calc_split(self._u1, self._u2)
            self.relation = BeamArray.BACKWARD

    def merge_prop(self):
        if self.relation == BeamArray.BACKWARD:
            calc_merge(self._u1, self._u2)
            self.relation = BeamArray.DERIVATIVE

    # def set_pure_forward(self):
    #     if self._u2 is not None:
    #         warn("overwrite existing backward part")
    #         if self.relation == BeamArray.BACKWARD:
    #             self.relation = BeamArray.DERIVATIVE
    #         elif self.relation == BeamArray.DERIVATIVE:
    #             calc_split(self._u1, self._u2)
    #         else:
    #             raise AttributeError
    #     else:
    #         self.relation = BeamArray.DERIVATIVE
    #     # now relation is D, u1 is forward
    #     self._u2 = pure_forward_d(self._u1)

    @property
    def forward(self):
        if self._u2 is not None:
            self.split_prop()
        return self._u1

    @property
    def backward(self):
        if self._u2 is not None:
            self.split_prop()
        return self._u2

    @property
    def field(self):
        if self._u2 is not None:
            self.merge_prop()
        return self._u1

    @property
    def derivative(self):
        if self._u2 is not None:
            self.merge_prop()
        return self._u2

    @staticmethod
    def _u_setter(u, value, dtype=dtype):
        if isinstance(value, (GPUArray, np.ndarray)):
            u.set(value)
        elif isinstance(value, Number):
            u.fill(np.array(value, dtype=dtype))
        else:
            raise AttributeError(f"cannot set value type {type(value)} to BeamArray")

    @forward.setter
    def forward(self, value):
        if self._u2 is not None:
            self.split_prop()
        self._u_setter(self._u1, value)

    @backward.setter
    def backward(self, value):
        if self._u2 is not None:
            self.split_prop()
        if value is None:
            self._u2 = None
        else:
            if self._u2 is None:
                self._u2 = self._get_array()
                self.relation = BeamArray.BACKWARD
            self._u_setter(self._u2, value)

    @field.setter
    def field(self, value):
        if self._u2 is not None:
            self.merge_prop()
        self._u_setter(self._u1, value)

    @derivative.setter
    def derivative(self, value):
        if self._u2 is not None:
            self.merge_prop()
        if value is None:
            self._u2 = None
        else:
            if self._u2 is None:
                self._u2 = self._get_array()
                self.relation = BeamArray.DERIVATIVE
            self._u_setter(self._u2, value)

    def ssnp(self, dz, n=None, track=False):
        if self._u2 is None:
            warn("incomplete field information, assuming u is pure forward")
            self.set_pure_forward()
        self.merge_prop()
        self._parse(('ssnp', self._u1, self._u2), dz, n, track)

    def bpm(self, dz, n=None, track=False):
        if self._u2 is not None:
            warn("discarding backward propagation part of bidirectional field", stacklevel=2)
            self.split_prop()
            self._u2 = None
        self._parse(('bpm', self._u1), dz, n, track)

    def n_grad(self, output=None):
        scatter_num = 0
        ug = None
        for op in self._history:
            if op[0] in {'ssnp', 'bpm'} and op[-1] is not None:
                scatter_num += 1
        if output is None:
            # scatter_num times of empty_like and dtype double
            output = GPUArray((scatter_num, *self._u1.shape), np.double)
        elif len(output) != scatter_num:
            raise ValueError(f"output length {len(output)} is different from scatter operation number {scatter_num}")
        while len(self._history) > 0:
            op = self._history.pop()
            if op[0] == "mse":
                if ug is not None:
                    raise ValueError("multiple reduce operation")
                ug = self._get_array()
                reduce_mse_grad(self.forward, op[1], output=ug)

            elif op[0] == "bpm":
                _, u, dz, n = op
                if ug is None:
                    raise ValueError("no reduce operation")
                if n is None:
                    bpm_grad_bp(u, ug, dz)
                else:
                    scatter_num -= 1
                    bpm_grad_bp(u, ug, dz, n, output[scatter_num])
                    self.recycle_array(u)
            else:
                raise NotImplementedError(f"unknown operation {op[0]}")
        assert scatter_num == 0
        return output

    def binary_pupil(self, na):
        calc_pupil(self._u1, na)
        if self._u2 is not None:
            calc_pupil(self._u2, na)

    def forward_mse_loss(self, measurement, *, track=False):
        if track:
            self._history.append(("mse", measurement))
        return reduce_mse(self.forward, measurement)

    def _parse(self, info, dz, n, track):
        def step(var_dz, var_n):
            if info[0] == 'bpm':
                bpm_step(info[1], var_dz, var_n)
                if track:
                    if n is None:
                        u_save = None
                    else:
                        u_save = self._get_array()
                        u_save.set(info[1])
                    self._history.append(('bpm', u_save, var_dz, var_n))
            elif info[0] == 'ssnp':
                ssnp_step(info[1], info[2], var_dz, var_n)
                if track:
                    u_save = self._get_array()
                    u_save.set(info[1])
                    ud_save = self._get_array()
                    ud_save.set(info[2])
                    self._history.append(('ssnp', u_save, ud_save, var_dz, var_n))

        if n is None:
            if isinstance(dz, Iterable):
                for dzi in dz:
                    step(dzi, None)
            else:
                step(dz, None)
        else:
            if isinstance(n, GPUArray):
                if len(n.shape) == 2:
                    step(dz, n)
                    return
                elif len(n.shape) != 3:
                    raise ValueError(f"invalid n shape {n.shape}")
            if isinstance(dz, Iterable):
                for dz_n in zip(dz, n):
                    step(*dz_n)
            else:
                for ni in n:
                    step(dz, ni)
