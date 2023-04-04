from pycuda import elementwise, gpuarray, reduction
from pycuda.gpuarray import GPUArray
from pycuda.tools import dtype_to_ctype
import numpy as np
from ssnp.utils import Multipliers, get_stream_in_current
from contextlib import contextmanager
from functools import partial, lru_cache
from skcuda import fft as skfft
import logging


class Funcs:
    # __temp_memory_pool = {}
    _funcs_cache = {}
    reduce_sse_cr_krn = None

    def __new__(cls, arr_like, res, n0, stream=None, fft_type="reikna"):
        if cls.reduce_sse_cr_krn is None:
            Funcs.reduce_sse_cr_krn = reduction.ReductionKernel(
                dtype_out=np.double, neutral=0,
                reduce_expr="a+b",
                map_expr="(cuCabs(x[i]) - y[i]) * (cuCabs(x[i]) - y[i])",
                arguments="double2 *x, double *y",
                preamble='#include "cuComplex.h"'
            )
            Funcs.reduce_sse_cc_krn = reduction.ReductionKernel(
                dtype_out=np.double, neutral=0,
                reduce_expr="a+b",
                map_expr="cuCabs(cuCsub(x[i], y[i])) * cuCabs(cuCsub(x[i], y[i]))",
                arguments="double2 *x, double2 *y",
                preamble='#include "cuComplex.h"'
            )
            Funcs.mse_cc_grad_krn = elementwise.ElementwiseKernel(
                "double2 *u, double2 *m, double2 *out",
                """
                out[i] = cuCsub(u[i], m[i]);
                out[i].x *= 2. / (double)n; out[i].y *= 2. / (double)n;
                """,
                preamble='#include "cuComplex.h"'
            )
            Funcs.mse_cr_grad_krn = elementwise.ElementwiseKernel(
                "double2 *u, double *m, double2 *out",
                """
                temp = 2 * (1 - m[i] / cuCabs(u[i]));
                if (!isfinite(temp))
                    temp = (double)0;
                out[i].x = temp * u[i].x / (double)n; out[i].y = temp * u[i].y / (double)n;
                """,
                loop_prep="double temp",
                preamble='#include "cuComplex.h"'
            )
            Funcs.abs_cc_krn = elementwise.ElementwiseKernel(
                "double2 *x, double2 *out",
                "out[i] = make_cuDoubleComplex(cuCabs(x[i]), 0)",
                preamble='#include "cuComplex.h"'
            )

        if res is not None:
            res = tuple(round(res_i * 1e12) for res_i in res)
        if n0 is not None:
            n0 = round(n0 * 1e12)
        key = (tuple(arr_like.shape), arr_like.dtype, res, n0, stream, fft_type)
        try:
            return cls._funcs_cache[key]
        except KeyError:
            cls._funcs_cache[key] = super().__new__(cls)
            return cls._funcs_cache[key]

    def __init__(self, arr_like, res, n0, stream=None, fft_type="skcuda"):
        if self._initialized():
            return
        self._funcs_cache = None  # only used for mark `self` as initialized

        if stream is None:
            stream = get_stream_in_current()
        self.stream = stream

        shape = tuple(arr_like.shape)
        if len(shape) == 3:
            batch = shape[0]
            shape = shape[1:]
        elif len(shape) == 2:
            batch = 1
        else:
            raise NotImplementedError(f"cannot process {len(shape)}-D data")
        self.shape = shape
        self.batch = batch

        if fft_type == "reikna":
            def _fft(arr, o, inv):
                nonlocal fft_callable
                if fft_callable is None:
                    logging.debug(f"assigning reikna fft_callable in an instance of {type(self)}")
                    fft_callable = self._compile_reikna_fft(arr_like.shape, arr_like.dtype, stream)
                fft_callable(o, arr, inverse=inv)

            fft_callable = None
            self._fft = _fft
        elif fft_type == "skcuda":
            self._skfft_plan = skfft.Plan(shape, np.complex128, np.complex128, batch, stream)
            self._fft = self._fft_sk
        else:
            raise ValueError(f"Unknown FFT type {fft_type}")
        self.ifft = partial(self.fft, inverse=True)

        # kz and related multipliers
        if type(self) is not Funcs:
            self._prop_cache = {}
            self._pupil_cache = {}
            self.res = res
            self.n0 = n0
            self.multiplier = Multipliers(shape, res, stream)
            c_gamma = self.multiplier.c_gamma()
            c_gamma = np.broadcast_to(c_gamma, arr_like.shape)
            kz = c_gamma * (2 * np.pi * res[2])
            self.kz = kz.astype(np.double)
            self.kz_gpu = gpuarray.to_gpu(kz)
            self.eva = np.exp(np.minimum((c_gamma - 0.2) * 5, 0))

        # function interface
        self.sum_cmplx_batch_krn = elementwise.ElementwiseKernel(
            "double2 *out, double2 *u",
            f"""
            out[i] = u[i];
            for (j = 1; j < {self.batch}; j++) {{
                out[i].x += u[i + j * n].x;
                out[i].y += u[i + j * n].y;
            }}
            """,
            loop_prep="unsigned j",
        )
        self.sum_double_batch_krn = elementwise.ElementwiseKernel(
            "double *out, double *u",
            f"""
            out[i] = u[i];
            for (j = 1; j < {self.batch}; j++)
                out[i] += u[i + j * n];
            """,
            loop_prep="unsigned j",
        )

    def _initialized(self):
        return self._funcs_cache is None

    @staticmethod
    @lru_cache
    def op_krn(batch, xt, yt, zt, operator, name=None, y_func=None):
        y_i = f"y[i % (n / {batch})]"
        if y_func is not None:
            y_i = f"{y_func}({y_i})"
        return elementwise.get_elwise_kernel(
            f"{xt} *x, {yt} *y, {zt} *z",
            f"z[i] = x[i] {operator} {y_i}",
            name or "op")

    def op(self, x, operator, y, out=None, **kwargs):
        if out is None:
            out = x
        xt = dtype_to_ctype(x.dtype)
        yt = dtype_to_ctype(y.dtype)
        zt = dtype_to_ctype(out.dtype)
        func = self.op_krn(self.batch, xt, yt, zt, operator, **kwargs)
        func.prepared_async_call(x._grid, x._block, self.stream,
                                 x.gpudata, y.gpudata,
                                 out.gpudata, x.mem_size)
        return out

    def reduce_sse(self, field, measurement):
        if field.dtype == np.complex128:
            if measurement.dtype == np.float64:
                return Funcs.reduce_sse_cr_krn(field, measurement, stream=self.stream)
            elif measurement.dtype == np.complex128:
                return Funcs.reduce_sse_cc_krn(field, measurement, stream=self.stream)
        raise TypeError(f"incompatible dtype: field {field.dtype}, measurement {measurement.dtype}")

    def mse_grad(self, field, measurement, gradient):
        if field.dtype == np.complex128 and gradient.dtype == np.complex128:
            if measurement.dtype == np.float64:
                return Funcs.mse_cr_grad_krn(field, measurement, gradient, stream=self.stream)
            elif measurement.dtype == np.complex128:
                return Funcs.mse_cc_grad_krn(field, measurement, gradient, stream=self.stream)
        raise TypeError(f"incompatible dtype: field {field.dtype}, measurement {measurement.dtype}, "
                        f"gradient {gradient.dtype}")

    def sum_batch(self, batch, sum_):
        if batch.dtype == np.complex128 and sum_.dtype == np.complex128:
            return self.sum_cmplx_batch_krn(sum_, batch, stream=self.stream)
        elif batch.dtype == np.float64 and sum_.dtype == np.float64:
            return self.sum_double_batch_krn(sum_, batch, stream=self.stream)
        raise TypeError(f"incompatible dtype: field {batch.dtype}, measurement {sum_.dtype}")

    @staticmethod
    @lru_cache
    def _compile_reikna_fft(shape, dtype, stream):
        from reikna.fft import FFT
        import reikna.cluda as cluda
        logging.info(f"compiling reikna fft: {shape=}, {dtype=}")
        thr = cluda.cuda_api().Thread(stream)
        arr_like = type("", (), {})()
        arr_like.shape = shape
        arr_like.dtype = dtype
        axes = None
        if len(arr_like.shape) == 3:
            axes = (1, 2)
        return FFT(arr_like, axes=axes).compile(thr)

    def fft(self, arr, output=None, copy=False, inverse=False):
        if output is not None:
            o = output
        elif copy:
            o = gpuarray.empty_like(arr)
        else:
            o = arr
        self._fft(arr, o, inverse)
        return o

    def _fft_sk(self, arr, out, inverse):
        fft = skfft.ifft if inverse else skfft.fft
        fft(arr, out, self._skfft_plan)
        if inverse:
            scale = 1 / self.shape[-1] / self.shape[-2]
            out._axpbz(scale, 0, out, stream=self.stream)

    @contextmanager
    def fourier(self, arr, copy=False):
        yield self.fft(arr, copy=copy)
        self.ifft(arr)

    def diffract(self, *args):
        raise NotImplementedError

    def scatter(self, *args):
        raise NotImplementedError

    def diffract_g(self, ag, dz):
        raise NotImplementedError

    def scatter_g(self, u, n, ug, ng, dz):
        raise NotImplementedError

    def _get_prop(self, dz):
        """
            P: 2D angular spectrum transmittance numpy array(s). Evanescent wave removed:
            For 0.98<N.A.<1, decrease to 1~1/e per step;
            For N.A.>1 decrease to 1/e per step

            Q: scattering operation, different between SSNP & BPM

            Pg/Qg: transposed array/operation of P/Q

            :return: Dictionary of P, Q, Pg, Qg
        """
        raise NotImplementedError

    def conj(self, arr, out=None):
        if out is None:
            out = arr
        dtype = arr.dtype
        if not arr.flags.forc:
            raise RuntimeError("only contiguous arrays may "
                               "be used as arguments to this operation")
        func = elementwise.get_conj_kernel(dtype)
        func.prepared_async_call(arr._grid, arr._block, self.stream,
                                 arr.gpudata, out.gpudata, arr.mem_size)
        return out

    # @staticmethod
    # def get_temp_mem(arr_like: GPUArray, index=0):
    #     key = (arr_like.shape, arr_like.dtype, index)
    #     try:
    #         return Funcs.__temp_memory_pool[key]
    #     except KeyError:
    #         mem = gpuarray.empty_like(arr_like)
    #         Funcs.__temp_memory_pool[key] = mem
    #         return mem


class SSNPFuncs(Funcs):
    _funcs_cache = {}

    def __init__(self, *args, **kwargs):
        if self._initialized():
            return
        super(SSNPFuncs, self).__init__(*args, **kwargs)
        self._fused_mam_callable_krn = elementwise.ElementwiseKernel(
            "double2 *a, double2 *b, "
            "double2 *p00, double2 *p01, double2 *p10, double2 *p11",
            f"""
            p_i = i % (n / {self.batch});
            temp = cuCfma(a[i], p00[p_i], cuCmul(b[i], p01[p_i]));
            b[i] = cuCfma(a[i], p10[p_i], cuCmul(b[i], p11[p_i]));
            a[i] = temp;
            """,
            loop_prep="cuDoubleComplex temp; unsigned p_i",
            preamble='#include "cuComplex.h"',
            name=f'fused_mam_{self.batch}'
        )

        # share these definition to avoid mistakes
        kzi_init = f"kzi = kz[i % (n / {self.batch})];"
        local_vars = "cuDoubleComplex temp; double kzi"
        self._split_prop_krn = elementwise.ElementwiseKernel(
            # ab = (a + I a_d / kz) / 2
            # af = a - ab = (a - I a_d / kz) / 2
            "double2 *a, double2 *a_d, double *kz",
            f"""
            {kzi_init}
            temp.x = (cuCreal(a[i]) - cuCimag(a_d[i])/kzi)*0.5;
            temp.y = (cuCimag(a[i]) + cuCreal(a_d[i])/kzi)*0.5;
            a_d[i] = temp;
            a[i] = cuCsub(a[i], a_d[i]);
            """,
            loop_prep=local_vars,
            preamble='#include "cuComplex.h"'
        )
        self._merge_prop_krn = elementwise.ElementwiseKernel(
            # a = af + ab
            # a_d = (af - ab) * I kz
            "double2 *af, double2 *ab, double *kz",
            f"""
            {kzi_init}
            temp = cuCsub(af[i], ab[i]);
            af[i] = cuCadd(af[i], ab[i]);
            ab[i] = make_cuDoubleComplex(-cuCimag(temp)*kzi, cuCreal(temp)*kzi);
            """,
            loop_prep=local_vars,
            preamble='#include "cuComplex.h"'
        )
        self._merge_grad_krn = elementwise.ElementwiseKernel(
            # ag = (afg + abg) / 2
            # a_dg = (afg - abg) / 2 * I / kz
            "double2 *afg, double2 *abg, double *kz",
            f"""
            {kzi_init}
            afg[i].x *= 0.5; afg[i].y *= 0.5;
            abg[i].x *= 0.5; abg[i].y *= 0.5;
            temp = cuCsub(afg[i], abg[i]);
            afg[i] = cuCadd(afg[i], abg[i]);
            abg[i] = make_cuDoubleComplex(-cuCimag(temp)/kzi, cuCreal(temp)/kzi);
            """,
            loop_prep=local_vars,
            preamble='#include "cuComplex.h"'
        )
        self._split_grad_krn = elementwise.ElementwiseKernel(
            # afg = ag - a_dg * I kz = 2 * ag - abg
            # abg = ag + a_dg * I kz
            "double2 *ag, double2 *a_dg, double *kz",
            f"""
            {kzi_init}
            temp_double = ag[i].x - kzi * a_dg[i].y;
            a_dg[i].y = ag[i].y + kzi * a_dg[i].x;
            a_dg[i].x = temp_double;
            ag[i].x = 2 * ag[i].x - a_dg[i].x;
            ag[i].y = 2 * ag[i].y - a_dg[i].y;
            """,
            loop_prep="double temp_double; double kzi",
            preamble='#include "cuComplex.h"'
        )

    def _get_prop(self, dz):
        res = self.res
        n0 = self.n0
        key = round(dz * 1000)
        try:
            return self._prop_cache[key]
        except KeyError:
            kz = self.kz
            eva = self.eva
            p_mat = [np.cos(kz * dz), np.sin(kz * dz) / kz,
                     -np.sin(kz * dz) * kz, np.cos(kz * dz)]
            p_mat = [gpuarray.to_gpu((i * eva).astype(np.complex128)) for i in p_mat]
            phase_factor = (2 * np.pi * res[2] / n0) ** 2 * dz
            q_op = elementwise.ElementwiseKernel(
                "double2 *u, double2 *ud, double *n_",
                f"""
                    temp = n_[i % (n / {self.batch})];
                    temp = {phase_factor} * temp * ({2 * n0} + temp);
                    ud[i].x -= temp * u[i].x;
                    ud[i].y -= temp * u[i].y;
                """,
                loop_prep="double temp",
                name="ssnp_q"
            )
            q_op_g = elementwise.ElementwiseKernel(
                # note: put n as 1st param to get un-batched thread number
                "double *n_, double2 *u, double2 *ug, double2 *udg, double *n_g",
                # Forward: ud = ud - temp * u
                # ng = Re{-udg * conj(u) * (d_temp(n) / d_n)}
                f"""
                    temp = {phase_factor} * n_[i] * ({2 * n0} + n_[i]);
                    for (j = 0; j < {self.batch}; j++) {{
                        ug[i + j * n].x -= temp * udg[i + j * n].x;
                        ug[i + j * n].y -= temp * udg[i + j * n].y;
                        n_g[i + j * n] =- (udg[i + j * n].x * u[i + j * n].x + udg[i + j * n].y * u[i + j * n].y) * 
                            {phase_factor} * ({2 * n0} + 2 * n_[i]);
                    }}
                """,
                loop_prep="double temp; unsigned j",
                preamble='#include "cuComplex.h"',
                name="ssnp_qg"
            )
            new_prop = {"P": p_mat, "Pg": [p_mat[i].conj() for i in (0, 2, 1, 3)], "Q": q_op, "Qg": q_op_g}
            self._prop_cache[key] = new_prop
            return new_prop

    def diffract(self, a, a_d, dz):
        self._fused_mam_callable_krn(a, a_d, *self._get_prop(dz)["P"], stream=self.stream)

    def scatter(self, u, u_d, n, dz):
        self._get_prop(dz)["Q"](u, u_d, n, stream=self.stream)

    def diffract_g(self, ag, a_dg, dz):
        self._fused_mam_callable_krn(ag, a_dg, *self._get_prop(dz)["Pg"], stream=self.stream)

    def scatter_g(self, u, n, ug, u_dg, ng, dz):
        self._get_prop(dz)["Qg"](n, u, ug, u_dg, ng, stream=self.stream)  # put n as 1st param (match CUDA code)

    def merge_prop(self, af, ab):
        self._merge_prop_krn(af, ab, self.kz_gpu, stream=self.stream)

    def split_prop(self, a, a_d):
        self._split_prop_krn(a, a_d, self.kz_gpu, stream=self.stream)

    def merge_grad(self, afg, abg):
        self._merge_grad_krn(afg, abg, self.kz_gpu, stream=self.stream)

    def split_grad(self, ag, a_dg):
        self._split_grad_krn(ag, a_dg, self.kz_gpu, stream=self.stream)


class BPMFuncs(Funcs):
    _funcs_cache = {}

    def _get_prop(self, dz):
        res = self.res
        n0 = self.n0
        key = (round(dz * 1000), "bpm")
        try:
            return self._prop_cache[key]
        except KeyError:
            kz = self.kz.astype(np.complex128)
            eva = self.eva
            p_mat = np.exp(kz * (1j * dz))
            p_mat = gpuarray.to_gpu(p_mat * eva)
            phase_factor = 2 * np.pi * res[2] / n0 * dz
            q_op = elementwise.ElementwiseKernel(
                "double2 *u, double *n_",
                f"""
                    ni = n_[i % (n / {self.batch})];
                    temp = make_cuDoubleComplex(cos(ni * {phase_factor}), sin(ni * {phase_factor}));
                    u[i] = cuCmul(u[i], temp);
                """,
                loop_prep="double2 temp; double ni",
                preamble='#include "cuComplex.h"'
            )
            q_op_g = elementwise.ElementwiseKernel(
                "double2 *u, double *n_, double2 *ug, double *n_g",
                # Forward: u=u*temp
                f"""
                    ni = n_[i % (n / {self.batch})];
                    temp_conj = make_cuDoubleComplex(cos(ni * {phase_factor}), -sin(ni * {phase_factor}));
                    n_g[i] = {phase_factor} * 
                        (cuCimag(ug[i]) * cuCreal(u[i]) - cuCimag(u[i]) * cuCreal(ug[i]));
                    ug[i] = cuCmul(ug[i], temp_conj);
                """,
                loop_prep="double2 temp_conj; double ni",
                preamble='#include "cuComplex.h"'
            )
            new_prop = {"P": p_mat, "Pg": p_mat.conj(), "Q": q_op, "Qg": q_op_g}
            self._prop_cache[key] = new_prop
            return new_prop

    def diffract(self, a, dz):  # todo: test it
        self.op(a, "*", self._get_prop(dz)["P"])
        # a._elwise_multiply(self._get_prop(dz)["P"], a, stream=self.stream)
        # a *= self._get_prop(dz)["P"]

    def diffract_g(self, ag, dz):
        self.op(ag, "*", self._get_prop(dz)["Pg"])
        # ag._elwise_multiply(self._get_prop(dz)["Pg"], ag, stream=self.stream)
        # ag *= self._get_prop(dz)["Pg"]

    def scatter(self, u, n, dz):
        self._get_prop(dz)["Q"](u, n, stream=self.stream)

    def scatter_g(self, u, n, ug, ng, dz):
        self._get_prop(dz)["Qg"](u, n, ug, ng, stream=self.stream)
