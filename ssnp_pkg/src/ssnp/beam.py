"""
Module for BeamArray class
"""
from functools import partial
from warnings import warn
import copy
from collections import Iterable
from numbers import Number
import numpy as np
from contextlib import contextmanager
from pycuda import gpuarray
from pycuda.gpuarray import GPUArray
from ssnp import calc
from ssnp.utils import param_check, Config
from ssnp.utils.auto_gradient import Variable as Var, Operation, OperationTape, DataMissing
import logging


class BeamArray:
    """
    BeamArray provide convenient ways to perform operations for GPUArray (or GPUArrays pair):

    1. For SSNP, forward/backward and field/derivation representation can convert transparently when needed.

    2. Operation can be tracked for ``n_grad`` to get n gradient easily and fast
    """
    dtype = np.complex128
    DERIVATIVE = 0
    BACKWARD = 1
    _config = None
    _track = False

    def __init__(self, u1, u2=None, relation=BACKWARD, total_ops=0, stream=None):
        def to_gpu(u, name):
            if isinstance(u, (np.ndarray, GPUArray)):
                if u.dtype != self.dtype:
                    warn(f"force casting {name} to {self.dtype}", stacklevel=3)
                    u = u.astype(self.dtype)
            else:
                u = np.array(u, dtype=self.dtype)
                warn(f"converting {name} to numpy array as fallback", stacklevel=3)
            return gpuarray.to_gpu_async(u, stream=stream)  # async version of copy()

        self._u1 = to_gpu(u1, "u1")
        shape = self._u1.shape
        if len(shape) == 2:
            self.batch = 1
        elif len(shape) == 3:
            self.batch = shape[0]
        else:
            raise ValueError(f"cannot process {len(shape)}-D data with shape {shape}")

        if u2 is not None:
            self._u2 = to_gpu(u2, "u2")
            if self._u2.shape != shape:
                raise ValueError(f"u1 shape {shape} cannot match u2 shape {self._u2.shape}")
            if relation not in {BeamArray.DERIVATIVE, BeamArray.BACKWARD}:
                raise ValueError("unknown relation type")
            self.relation = relation
        else:
            self._u2 = None
        self._tape = []
        self._tape_new = OperationTape(total_ops)
        self._array_pool = []
        self.shape = shape[-2:]
        # self.multiplier = calc.get_multiplier(shape, stream=stream)
        self._get_array_times = 0

        max_ops = int(np.sqrt(2 * total_ops)) + 1 if total_ops > 0 else 0
        self.ops_number = {"max": max_ops, "remainder": max_ops, "current": max_ops}
        self._fft_funcs = calc.get_funcs(self._u1, stream=stream)
        self.stream = stream

    @property
    def multiplier(self):
        res = None if self._config is None else self._config.res
        return calc.get_multiplier(self.shape, res, stream=self.stream)

    @property
    def config(self):
        if self._config is None:
            self._config = Config()
            self.register_config_updater()

        return self._config

    @config.setter
    def config(self, value):
        assert isinstance(value, Config)
        self._config = copy.deepcopy(value)
        self._config.clear_updater()
        self.register_config_updater()

    def register_config_updater(self):
        def update(attr, **_):
            if self._u2 is not None:
                if attr == 'res':
                    self.split_prop()
                elif attr == 'n0':
                    self.merge_prop()

        self._config.register_updater(update)

    def _get_array(self):
        if self._array_pool:
            return self._array_pool.pop()
        else:
            self._get_array_times += 1
            logging.info(f"get array times: {self._get_array_times}")
            return gpuarray.empty_like(self._u1)

    def recycle_array(self, arr):
        param_check(beam=self._u1, recycle=arr)
        assert arr.dtype == self.dtype
        self._array_pool.append(arr)

    def split_prop(self):
        if self._u2 is None:
            warn("split_prop for forward-only beam is useless")
            return
        if self.relation == BeamArray.DERIVATIVE:
            calc.split_prop(self._u1, self._u2, self._config, stream=self.stream)
            self.relation = BeamArray.BACKWARD
        if self._track:
            # def forward(u, u_d):
            #     uf, ub = Var(), Var()
            #     uf.data, ub.data = calc.split_prop(u.data, u_d.data, self._config,
            #                                        copy=True, stream=self.stream)
            #     return uf, ub
            op = Operation([Var(), Var()], [Var(), Var()])
            op.gradient = partial(calc.merge_grad, config=self._config, stream=self.stream)
            self._tape_new.append(op)

    def merge_prop(self):
        if self._u2 is None:
            warn("merge_prop for forward-only beam is useless")
            return
        if self.relation == BeamArray.BACKWARD:
            calc.merge_prop(self._u1, self._u2, self._config, stream=self.stream)
            self.relation = BeamArray.DERIVATIVE
        if self._track:
            warn("cannot track 'merge_prop' operation (not implemented)")

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
            if self._u2 is not None:
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
            if self._u2 is not None:
                self.recycle_array(self._u2)
                self._u2 = None
        else:
            if self._u2 is None:
                self._u2 = self._get_array()
                self.relation = BeamArray.DERIVATIVE
            self._u_setter(self._u2, value)

    def ssnp(self, dz, n=None, *, track=None):
        """
        Perform an SSNP operation for this beam array

        :param dz: z slice thickness, unit: z pixel size
        :param n: refractive index, default is background n0
        :param track: track this operation for gradient calculation
        """
        if track is None:
            track = self._track
        if self._u2 is None:
            warn("incomplete field information, assuming u is pure forward")
            self.backward = 0
        self.merge_prop()
        self._parse(('ssnp', self._u1, self._u2), dz, n, track)

    def bpm(self, dz, n=None, *, track=None):
        """
        Perform a BPM operation for this beam array

        :param dz: z slice thickness, unit: z pixel size
        :param n: refractive index, default is background n0
        :param track: track this operation for gradient calculation
        """
        if track is None:
            track = self._track

        if self._u2 is not None:
            warn("discarding backward propagation part of bidirectional field", stacklevel=2)
            self.backward = None
        self._parse(('bpm', self._u1), dz, n, track)

    def n_grad(self, output=None):  # todo: correct config tracking
        # output shape checking/setting
        scatter_num = 0
        ug = None
        u_dg = None
        ng_batch = gpuarray.empty_like(self._u1, dtype=np.float64)
        for op in self._tape:
            if op[0] in {'ssnp', 'bpm'} and op[-1] is not None:
                scatter_num += 1
        if output is None:
            # new is scatter_num times of empty_like with double dtype
            output = GPUArray((scatter_num, *self._u1.shape[-2:]), np.double)
        elif len(output) != scatter_num:
            raise ValueError(f"output length {len(output)} is different from scatter operation number {scatter_num}")

        # processing
        while len(self._tape) > 0:
            op = self._tape.pop()
            if op[0] == "mse":
                if ug is not None:
                    raise ValueError("multiple reduce operation")
                ug = self._get_array()
                calc.reduce_mse_grad(self.forward, op[1], output=ug)

            elif op[0] == "bpm":
                if ug is None:
                    raise ValueError("no reduce operation")
                _, u, dz, n = op
                if n is None:
                    calc.bpm_grad_bp(u, ug, dz)
                else:
                    if u is None:
                        index = len(self._tape) - 1
                        while True:
                            if self._tape[index][1] is None:
                                index = index - 1
                                assert index >= 0
                            else:
                                if self._tape[index][0] == "bpm":
                                    break
                        index_u = self._get_array()
                        index_u.set(self._tape[index][1])
                        index += 1
                        # recalculate previous u for n!=None but u==None ops
                        while index < len(self._tape):
                            if self._tape[index][0] == "bpm":
                                calc.bpm_step(index_u, *self._tape[index][2:])
                                if self._tape[index][3] is not None:
                                    assert self._tape[index][1] is None
                                    self._tape[index][1] = index_u
                                    index_u = self._get_array()
                                    index_u.set(self._tape[index][1])
                            index += 1
                        calc.bpm_step(index_u, dz, n)  # recalculate current step u
                        u = index_u
                    # u, ug, dz, n is all checked and not None
                    scatter_num -= 1
                    if self.batch > 1:
                        calc.bpm_grad_bp(u, ug, dz, n, ng_batch)
                        calc.sum_batch(ng_batch, output[scatter_num])
                    else:
                        calc.bpm_grad_bp(u, ug, dz, n, output[scatter_num])
                    self.recycle_array(u)

            elif op[0] == "ssnp":
                if ug is None:
                    raise ValueError("no reduce operation")
                _, u, _, dz, n = op
                if u_dg is None and ug is not None:
                    ufg = ug
                    ubg = self._get_array()
                    ubg.fill(np.array(0, self.dtype))
                    u, u_dg = calc.merge_grad(ufg, ubg, self._config, stream=self.stream)
                if n is None:
                    calc.ssnp_grad_bp(u, ug, u_dg, dz, config=self._config, stream=self.stream)
                else:
                    scatter_num -= 1
                    if self.batch > 1:
                        calc.ssnp_grad_bp(u, ug, u_dg, dz, n, ng_batch, self._config, self.stream)
                        calc.sum_batch(ng_batch, output[scatter_num], self.stream)
                    else:
                        calc.ssnp_grad_bp(u, ug, u_dg, dz, n, output[scatter_num], self._config, self.stream)
                    self.recycle_array(u)

            elif op[0] == "u*":
                _, mul = op
                calc.u_mul(ug, mul, conj=True)
                if u_dg is not None:
                    calc.u_mul(u_dg, mul, conj=True)

            elif op[0] == "a*":
                _, mul = op
                with self._fft_funcs.fourier(ug):
                    calc.u_mul(ug, mul, conj=True)
                if u_dg is not None:
                    with self._fft_funcs.fourier(u_dg):
                        calc.u_mul(u_dg, mul, conj=True)
            else:
                raise NotImplementedError(f"unknown operation {op[0]}")
        assert scatter_num == 0
        if ug is not None:
            self.recycle_array(ug)
        if u_dg is not None:
            self.recycle_array(u_dg)
        self.ops_number["remainder"] = self.ops_number["current"] = self.ops_number["max"]
        return output

    def n_grad2(self, output=None):
        pass

    def binary_pupil(self, na):
        self.a_mul(self.multiplier.binary_pupil(na, gpu=True))

    def mul(self, arr, *, hold=None, track=None):
        """
        calculate ``beam *= arr``

        If hold is given, ``beam = (beam - hold) * arr + hold``

        :param arr: other array to multiply
        :param hold: some part not perform multiply
        :param track: track this operation for gradient calculation
        """
        if track is None:
            track = self._track
        param_check(field=self._u1, multiplier=arr, hold=hold and hold._u1)
        if hold is not None:
            self.__isub__(hold)
            self.mul(arr, track=track)
            self.__iadd__(hold)
        else:
            self.__imul__(arr)
            if track:
                self._tape.append(("u*", arr))

    def a_mul(self, arr, hold=None, track=None):
        if track is None:
            track = self._track
        fourier = self._fft_funcs.fourier
        if self.batch == 1:
            param_check(angular_spectrum=self._u1, multiplier=arr, hold=hold and hold._u1)
        else:
            param_check(angular_spectrum=self._u1[0], multiplier=arr)
        if hold is not None:
            self.__isub__(hold)
            self.a_mul(arr, track=track)
            self.__iadd__(hold)
        else:
            with fourier(self._u1):
                calc.u_mul(self._u1, arr, stream=self.stream)
            if self._u2 is not None:
                with fourier(self._u2):
                    calc.u_mul(self._u2, arr, stream=self.stream)
            if track:
                self._tape.append(("a*", arr))
                self._tape_new.append(self._a_mul_op(arr))

    def __imul__(self, other):
        calc.u_mul(self._u1, other, stream=self.stream)
        if self._u2 is not None:
            calc.u_mul(self._u2, other, stream=self.stream)
        if self._track:
            if self._u2 is None:
                op = Operation(Var(), Var(), "mul")
            else:
                op = Operation((Var(), Var()), (Var(), Var()), "mul2")
            forward = lambda *u_vars: [
                Var(data=calc.u_mul(var.data, other,
                                    out=self._get_array() if var.bound else None,
                                    stream=self.stream))
                for var in u_vars]
            gradient = lambda *ug: [calc.u_mul(i, other, stream=self.stream, conj=True) for i in ug]
            op.set_funcs(forward, gradient)
            self._tape_new.append(op)
        return self

    def __iadd__(self, other, sign=1):
        assert isinstance(other, BeamArray)
        param_check(self=self._u1, add=other._u1)
        u_self = (self._u1,) if self._u2 is None else (self._u1, self._u2)
        u_other = (other._u1,) if other._u2 is None else (other._u1, other._u2)
        if len(u_self) != len(u_other):
            raise ValueError("incompatible BeamArray type")
        if self._u2 is not None:  # if u1 & u2, relation must be the same
            if self.relation == BeamArray.DERIVATIVE:
                other.merge_prop()
            else:
                other.split_prop()
        # Main part: u_s = u_s * 1 + u_o * (sub?-1:1)
        for u_s, u_o in zip(u_self, u_other):
            u_s._axpbyz(1, u_o, sign, u_s, stream=self.stream)
        return self

    def __isub__(self, other):
        return type(self).__iadd__(self, other, -1)

    def forward_mse_loss(self, measurement):
        loss = calc.reduce_mse(self.forward, measurement)
        if self._track:
            self._tape.append(("mse", measurement))
            ufg = calc.reduce_mse_grad(self._u1, measurement, self._get_array(), self.stream)
            if self._u2 is None:
                op = Operation(Var(), [], "mse")
                op.gradient = lambda: ufg
            else:
                op = Operation([Var(), Var()], [], "mse")
                ubg = self._get_array()
                ubg.fill(0, self.stream)
                op.gradient = lambda: ufg, ubg
            self._tape_new.append(op)
        return loss

    def _parse(self, info, dz, n, track):
        def step(var_dz, var_n):
            if info[0] == 'bpm':
                calc.bpm_step(info[1], var_dz, var_n, config=self._config)
                if track:
                    if var_n is None:
                        u_save = None
                    else:
                        if self.ops_number["current"] >= self.ops_number["remainder"]:
                            u_save = self._get_array()
                            u_save.set(info[1])
                            if self.ops_number["remainder"] > 0:
                                self.ops_number["remainder"] -= 1
                                self.ops_number["current"] = 0
                        else:
                            self.ops_number["current"] += 1
                            u_save = None
                    self._tape.append(['bpm', u_save, var_dz, var_n])
                    # new tape
                    if var_n is not None and next(self._tape_new.save_hint):
                        u_save = self._get_array()
                        u_save.set(info[1])
                    else:
                        u_save = None
                    self._tape_new.append(self._bpm_op(Var('u', u_save), var_n, dz))

            elif info[0] == 'ssnp':
                calc.ssnp_step(info[1], info[2], var_dz, var_n, config=self._config)
                if track:
                    if var_n is None:
                        u_save = None
                    else:
                        u_save = self._get_array()
                        u_save.set(info[1])
                    # ud_save = self._get_array()
                    # ud_save.set(info[2])
                    ud_save = None
                    self._tape.append(('ssnp', u_save, ud_save, var_dz, var_n))
                    # new tape
                    if var_n is not None and next(self._tape_new.save_hint):
                        u_save = self._get_array()
                        u_save.set(info[1])
                    else:
                        u_save = None
                    self._tape_new.append(self._bpm_op(Var('u', u_save), var_n, dz))

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

    def __del__(self):
        for arr in self._array_pool:
            arr.gpudata.free()
        self._u1.gpudata.free()
        if isinstance(self._u2, GPUArray):
            self._u2.gpudata.free()

    def __repr__(self):
        if self._u2:
            if self.relation == BeamArray.DERIVATIVE:
                beam_type = "derivative"
                data = ("total complex field U", "diff[U(z), z]")
            elif self.relation == BeamArray.BACKWARD:
                beam_type = "backward"
                data = ("forward component Uf", "backward component Ub")
            else:
                raise TypeError("unknown beam type")

            beam_type = "bi-directional beam, additional data is " + beam_type
            data = f"{{{data[0]}:\n{self._u1},\n{data[1]}:\n{self._u2}}}"
        else:
            beam_type = "forward-only beam"
            data = f"{{complex field U:\n{self._u1}}}"

        return f"{{size: {self.shape},\nbeam type: {beam_type},\ndata: {data}\n}}"

    @contextmanager
    def track(self):
        if self._track:
            warn("this BeamArray is already tracked")
        else:
            self._track = True
        yield
        self._track = False

    def _a_mul_op(self, other):
        if self._u2 is None:
            op = Operation(Var(), Var(), "a_mul")
        else:
            op = Operation((Var(), Var()), (Var(), Var()), "a_mul2")

        def gradient(*ug_list):
            for ug in ug_list:
                with self._fft_funcs.fourier(ug) as ag:
                    calc.u_mul(ag, other, stream=self.stream, conj=True)
            return ug_list

        op.set_funcs(None, gradient)
        return op

    def _bpm_op(self, u_out, n_data, dz):
        vars_in = Var('u_in') if n_data is None else (Var('u_in'), Var('n', n_data, external=True))

        def gradient(ug_in_data, out: dict = None):
            if n_data:
                if not u_out:
                    raise DataMissing
                ng_data = out and out.get('n', None) or gpuarray.empty_like(n_data)
            else:
                ng_data = None
            calc.bpm_grad_bp(u_out.data, ug_in_data, dz, n_data, ng_data, self._config, self.stream)
            return ug_in_data if ng_data is None else (ug_in_data, ng_data)

        def forward(u_in: Var):
            if n_data:
                if u_in.bound:
                    u_out.data = self._get_array()
                else:
                    u_out.data = u_in.data
                u_return = u_out
            else:
                if u_in.bound:
                    u_return = Var(u_in.tag, self._get_array())
                else:
                    u_return = u_in
            calc.bpm_step(u_in.data, dz, n_data, u_return.data, self._config, self.stream)
            return u_return

        def clear():
            if u_out:
                self.recycle_array(u_out.data)

        op = Operation(vars_in, u_out, "bpm")
        op.set_funcs(forward, gradient, clear)
        return op

    def _ssnp_op(self, u_out, n_data, dz):
        def gradient(ug_in_data, u_dg_in_data, out: dict = None):
            if n_data:
                if not u_out[0]:
                    raise DataMissing
                ng_data = out and out.get('n', None) or gpuarray.empty_like(n_data)
            else:
                ng_data = None
            calc.ssnp_grad_bp(u_out[0].data, ug_in_data, u_dg_in_data, dz, n_data, ng_data,
                              config=self._config, stream=self.stream)
            return ug_in_data if ng_data is None else (ug_in_data, ng_data)

        def forward(*u_in):
            if n_data:  # scatter: save in u_out, can reuse mem if not bound
                for i, ui in enumerate(u_in):
                    u_out[i].data = self._get_array() if ui.bound else ui.data
                u_return = u_out
            else:  # no scatter: not save in u_out, create new unbound if ui is bound
                u_return = [None, None]
                for i, ui in enumerate(u_in):
                    u_return[i] = Var(ui.tag, self._get_array()) if ui.bound else ui
            calc.ssnp_step(u_in[0].data, u_in[1].data, dz, n_data,
                           output=[u_return[0].data, u_return[1].data],
                           config=self._config, stream=self.stream)
            return u_return

        def clear():
            for v in u_out:
                if v:
                    self.recycle_array(v.data)

        vars_in = Var('u_in') if n_data is None else (Var('u_in'), Var('n', n_data, external=True))
        op = Operation(vars_in, u_out, "ssnp")
        op.set_funcs(forward, gradient, clear)
        return op
