"""
Module for BeamArray class
"""
from collections import Iterable
from numbers import Number
import numpy as np
from pycuda import gpuarray
from pycuda.gpuarray import GPUArray
from warnings import warn
from .calc import split_prop as calc_split, merge_prop as calc_merge, get_multiplier
from .calc import ssnp_step, bpm_step, binary_pupil as calc_pupil, reduce_mse
from .calc import bpm_grad_bp, reduce_mse_grad, u_mul_grad_bp
from .utils import param_check


class BeamArray:
    """
    BeamArray provide convenient ways to perform operations for GPUArray (or GPUArrays pair):

    1. For SSNP, forward/backward and field/derivation representation can convert transparently when needed.

    2. Operation can be tracked for ``n_grad`` to get n gradient easily and fast
    """
    dtype = np.complex128
    DERIVATIVE = 0
    BACKWARD = 1

    def __init__(self, u1, u2=None, relation=BACKWARD):
        def to_gpu(u, name):
            if isinstance(u, GPUArray):
                if u.dtype != self.dtype:
                    raise ValueError(f"GPUArray {name} must be {self.dtype} but not {u.dtype}")
                return u.copy()
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
        self.multiplier = get_multiplier(self._u1)

    def _get_array(self):
        if len(self._array_pool) == 0:
            return gpuarray.empty_like(self._u1)
        else:
            return self._array_pool.pop()

    def recycle_array(self, arr):
        param_check(beam=self._u1, recycle=arr)
        assert arr.dtype == self.dtype
        self._array_pool.append(arr)

    def split_prop(self):
        if self.relation == BeamArray.DERIVATIVE:
            calc_split(self._u1, self._u2)
            self.relation = BeamArray.BACKWARD

    def merge_prop(self):
        if self.relation == BeamArray.BACKWARD:
            calc_merge(self._u1, self._u2)
            self.relation = BeamArray.DERIVATIVE

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
            value: float
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
            self.recycle_array(self._u2)
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
            self.recycle_array(self._u2)
            self._u2 = None
        else:
            if self._u2 is None:
                self._u2 = self._get_array()
                self.relation = BeamArray.DERIVATIVE
            self._u_setter(self._u2, value)

    def ssnp(self, dz, n=None, *, track=False):
        """
        Perform an SSNP operation for this beam array

        :param dz: z slice thickness, unit: z pixel size
        :param n: refractive index, default is background n0
        :param track: track this operation for gradient calculation
        """
        if self._u2 is None:
            warn("incomplete field information, assuming u is pure forward")
            self.backward = 0
        self.merge_prop()
        self._parse(('ssnp', self._u1, self._u2), dz, n, track)

    def bpm(self, dz, n=None, *, track=False):
        """
        Perform a BPM operation for this beam array

        :param dz: z slice thickness, unit: z pixel size
        :param n: refractive index, default is background n0
        :param track: track this operation for gradient calculation
        """
        if self._u2 is not None:
            warn("discarding backward propagation part of bidirectional field", stacklevel=2)
            self.backward = None
        self._parse(('bpm', self._u1), dz, n, track)

    def n_grad(self, output=None):
        # output shape checking/setting
        scatter_num = 0
        ug = None
        for op in self._history:
            if op[0] in {'ssnp', 'bpm'} and op[-1] is not None:
                scatter_num += 1
        if output is None:
            # new is scatter_num times of empty_like with double dtype
            output = GPUArray((scatter_num, *self._u1.shape), np.double)
        elif len(output) != scatter_num:
            raise ValueError(f"output length {len(output)} is different from scatter operation number {scatter_num}")

        # processing
        while len(self._history) > 0:
            op = self._history.pop()
            if op[0] == "mse":
                if ug is not None:
                    raise ValueError("multiple reduce operation")
                ug = self._get_array()
                reduce_mse_grad(self.forward, op[1], output=ug)

            elif op[0] in {"bpm", "u*"}:
                if ug is None:
                    raise ValueError("no reduce operation")

                if op[0] == "bpm":
                    _, u, dz, n = op
                    if n is None:
                        bpm_grad_bp(u, ug, dz)
                    else:
                        scatter_num -= 1
                        bpm_grad_bp(u, ug, dz, n, output[scatter_num])
                        self.recycle_array(u)

                elif op[0] == "u*":
                    _, mul = op
                    u_mul_grad_bp(ug, mul)
            else:
                raise NotImplementedError(f"unknown operation {op[0]}")
        assert scatter_num == 0
        return output

    def binary_pupil(self, na):
        calc_pupil(self._u1, na)
        if self._u2 is not None:
            calc_pupil(self._u2, na)

    def mul(self, arr, *, hold=None, track=False):
        """
        calculate ``beam *= arr``

        If hold is given, ``beam = (beam - hold) * arr + hold``

        :param arr: other array to multiply
        :param hold: some part not perform multiply
        :param track: track this operation for gradient calculation
        """
        param_check(field=self._u1, arr=arr, hold=None if hold is None else hold._u1)
        if hold is not None:
            self.__isub__(hold)
            self.mul(arr, track=track)
            self.__iadd__(hold)
        else:
            self.__imul__(arr)
            if track:
                self._history.append(("u*", arr))

    def __imul__(self, other):
        self._u1 *= other
        if self._u2 is not None:
            self._u2 *= other
        return self

    @staticmethod
    def _iadd_isub(self, other, add):
        assert isinstance(other, BeamArray)
        param_check(self=self._u1, add=other._u1)
        if self._u2 is None:
            if other._u2 is None:
                if add:
                    self._u1 += other._u1
                else:
                    self._u1 -= other._u1
            else:
                raise ValueError("incompatible BeamArray type")
        else:
            if other._u2 is None:
                raise ValueError("incompatible BeamArray type")
            else:
                if self.relation == BeamArray.DERIVATIVE:
                    other.merge_prop()
                else:
                    other.split_prop()
                if add:
                    self._u1 += other._u1
                    self._u2 += other._u2
                else:
                    self._u1 -= other._u1
                    self._u2 -= other._u2
        return self

    def __iadd__(self, other):
        return BeamArray._iadd_isub(self, other, add=True)

    def __isub__(self, other):
        return BeamArray._iadd_isub(self, other, add=False)

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
