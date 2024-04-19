"""
Module for BeamArray class
"""
from functools import partial
from warnings import warn
import copy
from collections.abc import Iterable
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
        self.tape = OperationTape(total_ops)
        self._array_pool = []
        self.shape = shape[-2:]
        self._get_array_times = 0
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
        try:
            param_check(beam=self._u1, recycle=arr)
            assert arr.dtype == self.dtype
        except Exception as e:
            warn("given array is incompatible and not recycled" + str(e), stacklevel=2)
        else:
            self._array_pool.append(arr)

    def apply_grad(self, grad1, grad2=None):
        if self._track:
            param_check(beam=self._u1, grad1=grad1)
            param_check(beam=self._u2, grad1=grad2)
            u1g = self._get_array()
            u1g.set_async(grad1, stream=self.stream)
            if self._u2 is None:
                op = Operation(Var(), [], "apply_1grad")
                op.gradient = lambda: (u1g,)
            else:
                op = Operation([Var(), Var()], [], "apply_2grad")
                u2g = self._get_array()
                u2g.set_async(grad2, stream=self.stream)
                op.gradient = lambda: (u1g, u2g)
            self.tape.append(op)
        else:
            raise ValueError(f"applying grad without tracking is useless")

    def split_prop(self):
        if self._u2 is None:
            warn("split_prop for forward-only beam is useless")
        elif self.relation == BeamArray.DERIVATIVE:
            calc.split_prop(self._u1, self._u2, self._config, stream=self.stream)
            self.relation = BeamArray.BACKWARD
            if self._track:
                op = Operation([Var(), Var()], [Var(), Var()], "split")
                op.gradient = partial(calc.merge_grad, config=self._config, stream=self.stream)
                self.tape.append(op)

    def merge_prop(self):
        if self._u2 is None:
            warn("merge_prop for forward-only beam is useless")
        elif self.relation == BeamArray.BACKWARD:
            calc.merge_prop(self._u1, self._u2, self._config, stream=self.stream)
            self.relation = BeamArray.DERIVATIVE
            if self._track:
                op = Operation([Var(), Var()], [Var(), Var()], "merge")
                op.gradient = partial(calc.split_grad, config=self._config, stream=self.stream)
                self.tape.append(op)

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

    def n_grad(self, output=None):
        self.tape.collect_gradient({'n': output})
        return output

    def binary_pupil(self, na):
        self.a_mul(self.multiplier.binary_pupil(na, self._config.n0 if self._config else 1, gpu=True))

    def mul(self, arr, *, hold=None):
        """
        calculate ``beam *= arr``

        If hold is given, ``beam = (beam - hold) * arr + hold``

        :param arr: other array to multiply
        :param hold: some part not perform multiply
        """
        param_check(field=self._u1, multiplier=arr, hold=None if hold is None else hold._u1)
        if hold is not None:
            self.__isub__(hold)
            self.mul(arr)
            self.__iadd__(hold)
        else:
            self.__imul__(arr)

    def a_mul(self, arr, hold=None, track=None):
        if track is None:
            track = self._track
        fourier = self._fft_funcs.fourier
        # TODO: have bug and not necessary, removed but can be fixed in future
        # if self.batch == 1:
        #     param_check(angular_spectrum=self._u1, multiplier=arr, hold=hold and hold._u1)
        # else:
        #     param_check(angular_spectrum=self._u1[0], multiplier=arr)
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
                self.tape.append(self._a_mul_op(arr))

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
            self.tape.append(op)
        return self

    def __iadd__(self, other, sign=1):  # TODO: check if this is correct for recalculation in gradient computation
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

    def conj(self):
        if self._u2 is not None:
            raise NotImplementedError("does not support conj op for bi-dir beam (is it meaningful?)")
        self._fft_funcs.conj(self._u1)
        if self._track:
            op = Operation(Var(), Var(), "conj")
            op.set_funcs(
                forward=lambda var: Var(
                    data=self._fft_funcs.conj(var.data, out=self._get_array() if var.bound else None)
                ),
                gradient=lambda ug: [self._fft_funcs.conj(ug)]
            )
            self.tape.append(op)
        return self

    def mse_loss(self, forward=None, *, backward=None):
        # parameter check
        if self._u2 is None and backward is not None:
            warn("computing mse loss for forward only beam, backward part is ignored")
            backward = None
        if forward is None and backward is None:
            raise TypeError(f"mse_loss needs at least 1 argument")
        # mse (and grad) computation
        loss = 0
        ufg = ubg = None
        if forward is not None:
            loss += calc.reduce_mse(self.forward, forward, self.stream)
            if self._track:
                ufg = calc.reduce_mse_grad(self.forward, forward, self._get_array(), self.stream)
        if backward is not None:
            loss += calc.reduce_mse(self.backward, backward, self.stream)
            if self._track:
                ubg = calc.reduce_mse_grad(self.backward, backward, self._get_array(), self.stream)
        # append mse op to tape
        if self._track:
            if self._u2 is None:
                op = Operation(Var('uf'), [], "mse_f")
                op.gradient = lambda: (ufg,)
            else:
                op = Operation([Var('uf'), Var('ub')], [], "mse_fb")
                if ufg is None:
                    ufg = self._get_array().fill(0, self.stream)
                if ubg is None:
                    ubg = self._get_array().fill(0, self.stream)
                op.gradient = lambda: (ufg, ubg)
            self.tape.append(op)
        return loss

    def forward_mse_loss(self, measurement):
        loss = calc.reduce_mse(self.forward, measurement)
        if self._track:
            ufg = calc.reduce_mse_grad(self._u1, measurement, self._get_array(), self.stream)
            if self._u2 is None:
                op = Operation(Var(), [], "mse")
                op.gradient = lambda: (ufg,)
            else:
                op = Operation([Var(), Var()], [], "mse")
                ubg = self._get_array()
                ubg.fill(0, self.stream)
                op.gradient = lambda: (ufg, ubg)
            self.tape.append(op)
        return loss

    def midt_batch_mse_loss(self, measurement):
        # one direction only
        assert self._u2 is None
        assert measurement.dtype == np.float64
        batch_abs = calc.abs_c2c(self._u1, output=self._get_array(), stream=self.stream)
        batch_abs **= 2  # TODO: change to sqr/sqrt to improve performance
        summed_abs = self._get_array()
        calc.sum_batch(batch_abs, summed_abs[0], stream=self.stream)
        summed_abs[0] **= 0.5
        loss = calc.reduce_mse(summed_abs[0], measurement)
        if self._track:
            # 2 * (sqrt(sum_(u^2)) - m) / sqrt(sum_(u^2)) * u
            u1g = self._get_array()
            temp = u1g
            temp[0].set_async(summed_abs[0], self.stream)
            temp[0] += 1.e-12
            summed_abs[0] -= measurement
            summed_abs[0] /= temp[0]
            summed_abs[0] *= 2
            # del temp
            u1g.set_async(self._u1, self.stream)
            calc.u_mul(u1g, summed_abs[0])  # broadcast to whole batch
            op = Operation(Var(), [], "midt_batch_mse")
            op.gradient = lambda: (u1g,)
            self.tape.append(op)
        self.recycle_array(batch_abs)
        self.recycle_array(summed_abs)
        return loss

    def _parse(self, info, dz, n, track):
        def step(_dz, _n):
            if info[0] == 'bpm':
                calc.bpm_step(info[1], _dz, _n, config=self._config)
                if track:
                    if _n is not None and next(self.tape.save_hint):
                        u_save = self._get_array()
                        u_save.set(info[1])
                    else:
                        u_save = None
                    self.tape.append(self._bpm_op(Var('u', u_save), _n, _dz))

            elif info[0] == 'ssnp':
                calc.ssnp_step(info[1], info[2], _dz, _n, config=self._config)
                if track:
                    if _n is not None and next(self.tape.save_hint):
                        u_save = self._get_array()
                        ud_save = self._get_array()
                        u_save.set_async(info[1], self.stream)
                        ud_save.set_async(info[2], self.stream)
                    else:
                        u_save = ud_save = None
                    self.tape.append(self._ssnp_op((Var('u', u_save), Var('ud', ud_save)), _n, _dz))

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
        if old_track := self._track:
            warn("this BeamArray is already tracked")
        else:
            self._track = True
        if not self.tape:
            v_in = [Var("u1_in", external=True)]
            v_out = [Var()]
            if self._u2 is not None:
                v_in.append(Var("u2_in", external=True))
                v_out.append(Var())
            self.tape.append(self._change_op(v_in, v_out))
        yield
        self._track = old_track

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
            if n_data is not None:
                if not u_out.has_data():
                    raise DataMissing
                if out:
                    if out.get('n', None) is not None:
                        ng_data = out['n']
                    else:
                        logging.info("allocate memory for ng")
                        ng_data = gpuarray.empty_like(n_data)
                else:
                    ng_data = None  # TODO: will raise error if get here. Should fix calc.bpm_grad_bp
            else:
                ng_data = None
            calc.bpm_grad_bp(u_out.data, ug_in_data, dz, n_data, ng_data, self._config, self.stream)
            return (ug_in_data,) if ng_data is None else (ug_in_data, ng_data)

        def forward(u_in: Var):
            if n_data is not None:
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
            calc.bpm_step(u_in.data, dz, n_data, output=u_return.data, config=self._config, stream=self.stream)
            return u_return

        def clear():
            if u_out.has_data():
                self.recycle_array(u_out.data)

        op = Operation(vars_in, u_out, "bpm")
        op.set_funcs(forward, gradient, clear)
        return op

    def _ssnp_op(self, u_out, n_data, dz):
        def gradient(ug_in_data, u_dg_in_data, out: dict = None):
            if n_data is not None:
                if not u_out[0].has_data():
                    raise DataMissing
                if out:
                    if out.get('n', None) is not None:
                        ng_data = out['n']
                    else:
                        logging.info("allocate memory for ng")
                        ng_data = gpuarray.empty_like(n_data)
                else:
                    ng_data = None  # TODO: will raise error if get here. Should fix calc.ssnp_grad_bp
            else:
                ng_data = None
            calc.ssnp_grad_bp(u_out[0].data, ug_in_data, u_dg_in_data, dz, n_data, ng_data,
                              config=self._config, stream=self.stream)
            return (ug_in_data, u_dg_in_data) if ng_data is None else (ug_in_data, u_dg_in_data, ng_data)

        def forward(*u_in):
            if n_data is not None:  # scatter: save in u_out, can reuse mem if not bound
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
                if v.has_data():
                    self.recycle_array(v.data)

        vars_in = [Var('u_in'), Var('ud_in')]
        if n_data is not None:
            vars_in.append(Var('n', n_data, external=True))
        op = Operation(vars_in, u_out, "ssnp")
        op.set_funcs(forward, gradient, clear)
        return op

    def _change_op(self, vars_in, vars_out):
        """
        Add or delete self._u(1/2)

        :param vars_in: Previous variables list. If add, must have same length as vars_out.
        Use `external=True` for new places.

        :param vars_out: New variables list.
        :return: an `Operation(name="change")`
        """

        def gradient(*arr_in, out=None):
            arr_out = []
            for vi, _, go in zip(vars_in, vars_out, arr_in):
                if vi.external:
                    if out and vi.tag in out:
                        assert out[vi.tag] is None  # not support out container
                        arr_out.append(go)
                    else:
                        self.recycle_array(go)
                        arr_out.append(None)  # only a placeholder
                else:
                    arr_out.append(go)
            return arr_out + [self._get_array().fill(0) for _ in range(li - lo)]

        # TODO: fix forward & clear
        # def forward(*v_in):
        #     for v in v_in[lo:]:
        #         if not v.bound:
        #             self.recycle_array(v.data)
        #     return v_in[:lo] + tuple(vars_out[li:])

        # def clear():
        #     for v in vars_out:
        #         if v:
        #             self.recycle_array(v)

        li, lo = len(vars_in), len(vars_out)
        op = Operation(vars_in, vars_out, "change")
        op.gradient = gradient
        # op.set_funcs(forward, gradient, clear)
        return op
