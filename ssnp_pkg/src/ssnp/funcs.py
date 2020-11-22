from pycuda import elementwise, gpuarray, reduction
from pycuda.gpuarray import GPUArray
import numpy as np
from ssnp.utils import Multipliers, get_stream_in_current
from contextlib import contextmanager
from functools import partial, lru_cache
from skcuda import fft as skfft


class Funcs:
    # __temp_memory_pool = {}
    _funcs_cache = {}
    reduce_mse_cr_krn = None

    def __new__(cls, arr_like, res, n0, stream=None, fft_type="reikna"):
        if cls.reduce_mse_cr_krn is None:
            Funcs.reduce_mse_cr_krn = reduction.ReductionKernel(
                dtype_out=np.double, neutral=0,
                reduce_expr="a+b",
                map_expr="(cuCabs(x[i]) - y[i]) * (cuCabs(x[i]) - y[i])",
                arguments="double2 *x, double *y",
                preamble='#include "cuComplex.h"'
            )
            Funcs.reduce_mse_cc_krn = reduction.ReductionKernel(
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
                out[i].x *= 2; out[i].y *= 2;
                """,
                preamble='#include "cuComplex.h"'
            )
            Funcs.mse_cr_grad_krn = elementwise.ElementwiseKernel(
                "double2 *u, double *m, double2 *out",
                """
                temp = 2 * (1 - m[i] / cuCabs(u[i]));
                out[i].x = temp * u[i].x; out[i].y = temp * u[i].y;
                """,
                loop_prep="double temp",
                preamble='#include "cuComplex.h"'
            )

        shape = tuple(arr_like.shape)
        key = (shape, res, stream, fft_type)
        try:
            return cls._funcs_cache[key]
        except KeyError:
            cls._funcs_cache[key] = super().__new__(cls)
            return cls._funcs_cache[key]

    def __init__(self, arr_like, res, n0, stream=None, fft_type="reikna"):
        if self._funcs_cache is None:
            return
        self._funcs_cache = None  # only for mark as initialized

        if stream is None:
            stream = get_stream_in_current()
        self.stream = stream

        # FFTs
        shape = tuple(arr_like.shape)
        if fft_type == "reikna":
            if len(shape) != 2:
                raise NotImplementedError(f"cannot process {len(shape)}-D data with reikna")
            self._fft_callable = self._compile_fft(arr_like.shape, arr_like.dtype, stream)
            self.fft = self._fft_reikna
            batch = 1
        elif fft_type == "skcuda":
            if len(shape) == 3:
                batch = shape[0]
                shape = shape[1:]
            else:
                batch = 1
            if len(shape) != 2:
                raise NotImplementedError(f"cannot process {len(shape)}-D data with skcuda")
            self._skfft_plan = skfft.Plan(shape, np.complex128, np.complex128, batch, stream)
            self.fft = self._fft_sk
        else:
            raise ValueError(f"Unknown FFT type {fft_type}")
        self.ifft = partial(self.fft, inverse=True)
        self.shape = shape
        self.batch = batch

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

        # function interface todo add non-complex mul
        self.mul_conj_krn = elementwise.ElementwiseKernel(
            "double2 *ug, double2 *mul",
            f"""
            ug[i] = cuCmul(ug[i], cuConj(mul[i % (n / {self.batch})]));
            """,
            preamble='#include "cuComplex.h"'
        )
        self.mul_krn = elementwise.ElementwiseKernel(
            "double2 *u, double2 *mul",
            f"""
            u[i] = cuCmul(u[i], mul[i % (n / {self.batch})]);
            """,
            preamble='#include "cuComplex.h"'
        )
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
        # self.reduce_mse_cr = partial(self.reduce_mse_cr_krn, stream=stream)
        # self.reduce_mse_cc = partial(self.reduce_mse_cc_krn, stream=stream)
        # self.mse_cc_grad = partial(self.mse_cc_grad_krn, stream=stream)
        # self.mse_cr_grad = partial(self.mse_cr_grad_krn, stream=stream)
        self.mul = partial(self.mul_krn, stream=stream)
        self.mul_conj = partial(self.mul_conj_krn, stream=stream)

    def reduce_mse(self, field, measurement):
        if field.dtype == np.complex128:
            if measurement.dtype == np.float64:
                return Funcs.reduce_mse_cr_krn(field, measurement, stream=self.stream)
            elif measurement.dtype == np.complex128:
                return Funcs.reduce_mse_cc_krn(field, measurement, stream=self.stream)
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
    def _compile_fft(shape, dtype, stream):
        from reikna.fft import FFT
        import reikna.cluda as cluda
        thr = cluda.cuda_api().Thread(stream)
        arr_like = type("", (), {})()
        arr_like.shape = shape
        arr_like.dtype = dtype
        return FFT(arr_like).compile(thr)

    def _fft_reikna(self, arr, output=None, copy=False, inverse=False):
        if output is not None:
            o = output
        elif copy:
            o = gpuarray.empty_like(arr)
        else:
            o = arr
        self._fft_callable(o, arr, inverse=inverse)
        return o

    def _fft_sk(self, arr, *, output=None, copy=False, inverse=False):
        if output is not None:
            o = output
        elif copy:
            o = gpuarray.empty_like(arr)
        else:
            o = arr
        fft = skfft.ifft if inverse else skfft.fft
        fft(arr, o, self._skfft_plan)
        if inverse:
            scale = 1 / self.shape[-1] / self.shape[-2]
            o._axpbz(scale, 0, o, stream=self.stream)
        return o

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

    def conj(self, arr: GPUArray):
        """copy from GPUArray.conj(self), do conj in-place"""
        dtype = arr.dtype
        if not arr.flags.forc:
            raise RuntimeError("only contiguous arrays may "
                               "be used as arguments to this operation")
        func = elementwise.get_conj_kernel(dtype)
        func.prepared_async_call(arr._grid, arr._block, self.stream,
                                 arr.gpudata, arr.gpudata, arr.mem_size)

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
    _fused_mam_callable_krn = elementwise.ElementwiseKernel(
        "double2 *a, double2 *b, "
        "double2 *p00, double2 *p01, double2 *p10, double2 *p11",
        """
        temp = cuCfma(a[i], p00[i], cuCmul(b[i], p01[i]));
        b[i] = cuCfma(a[i], p10[i], cuCmul(b[i], p11[i]));
        a[i] = temp;
        """,
        loop_prep="cuDoubleComplex temp",
        preamble='#include "cuComplex.h"',
        name='fused_mam'
    )
    _split_prop_krn = elementwise.ElementwiseKernel(
        # ab = (a + I a_d / kz) / 2
        # af = a - ab = (a - I a_d / kz) / 2
        "double2 *a, double2 *a_d, double *kz",
        """
        temp.x = (cuCreal(a[i]) - cuCimag(a_d[i])/kz[i])*0.5;
        temp.y = (cuCimag(a[i]) + cuCreal(a_d[i])/kz[i])*0.5;
        a_d[i] = temp;
        a[i] = cuCsub(a[i], a_d[i]);
        """,
        loop_prep="cuDoubleComplex temp",
        preamble='#include "cuComplex.h"'
    )
    _merge_prop_krn = elementwise.ElementwiseKernel(
        # a = af + ab
        # a_d = (af - ab) * I kz
        "double2 *af, double2 *ab, double *kz",
        """
        temp = cuCsub(af[i], ab[i]);
        af[i] = cuCadd(af[i], ab[i]);
        ab[i] = make_cuDoubleComplex(-cuCimag(temp)*kz[i], cuCreal(temp)*kz[i]);
        """,
        loop_prep="cuDoubleComplex temp",
        preamble='#include "cuComplex.h"'
    )
    _merge_grad_krn = elementwise.ElementwiseKernel(
        # ag = (afg + abg) / 2
        # a_dg = (afg - abg) / 2 * I / kz
        "double2 *afg, double2 *abg, double *kz",
        """
        afg[i].x *= 0.5; afg[i].y *= 0.5;
        abg[i].x *= 0.5; abg[i].y *= 0.5;
        temp = cuCsub(afg[i], abg[i]);
        afg[i] = cuCadd(afg[i], abg[i]);
        abg[i] = make_cuDoubleComplex(-cuCimag(temp)/kz[i], cuCreal(temp)/kz[i]);
        """,
        loop_prep="cuDoubleComplex temp",
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
            phase_factor = (2 * np.pi * res[2]) ** 2 * dz
            q_op = elementwise.ElementwiseKernel(
                "double2 *u, double2 *ud, double *n_",
                f"""
                    temp = n_[i % (n / {self.batch})];
                    temp = {phase_factor / n0 ** 2} * temp * ({2 * n0} + temp);
                    ud[i].x -= temp * u[i].x;
                    ud[i].y -= temp * u[i].y;
                """,
                loop_prep="double temp",
                name="ssnp_q"
            )
            q_op_g = elementwise.ElementwiseKernel(
                "double2 *u, double *n_, double2 *ug, double2 *udg, double *n_g",
                # Forward: ud = ud - temp * u
                # ng = Re{-udg * conj(u) * (d_temp(n) / d_n)}
                f"""
                    ni = n_[i % (n / {self.batch})];
                    temp = {phase_factor} * ni * ({2 * n0} + ni);
                    ug[i].x -= temp * udg[i].x;
                    ug[i].y -= temp * udg[i].y;
                    n_g[i] = -(udg[i].x * u[i].x + udg[i].y * u[i].y) * {phase_factor} * ({2 * n0} + 2 * ni)
                """,
                loop_prep="double temp, ni",
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
        self._get_prop(dz)["Qg"](u, n, ug, u_dg, ng, stream=self.stream)

    def merge_prop(self, af, ab):
        self._merge_prop_krn(af, ab, self.kz_gpu, stream=self.stream)

    def split_prop(self, a, a_d):
        self._split_prop_krn(a, a_d, self.kz_gpu, stream=self.stream)

    def merge_grad(self, afg, abg):
        self._merge_grad_krn(afg, abg, self.kz_gpu, stream=self.stream)

    def split_grad(self, ag, a_dg):
        raise NotImplementedError


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
            phase_factor = 2 * np.pi * res[2] * n0 * dz
            q_op = elementwise.ElementwiseKernel(
                "double2 *u, double *n_",
                f"""
                    temp = n_[i % (n / {self.batch})];
                    temp = make_cuDoubleComplex(cos(temp * {phase_factor}), sin(temp * {phase_factor}));
                    u[i] = cuCmul(u[i], temp);
                """,
                loop_prep="double2 temp",
                preamble='#include "cuComplex.h"'
            )
            q_op_g = elementwise.ElementwiseKernel(
                "double2 *u, double *n_, double2 *ug, double *n_g",
                # Forward: u=u*temp
                f"""
                    temp = n_[i % (n / {self.batch})];
                    temp_conj = make_cuDoubleComplex(cos(temp * {phase_factor}), -sin(temp * {phase_factor}));
                    n_g[i] = {phase_factor} * 
                        (cuCimag(ug[i]) * cuCreal(u[i]) - cuCimag(u[i]) * cuCreal(ug[i]));
                    ug[i] = cuCmul(ug[i], temp_conj);
                """,
                loop_prep="double2 temp_conj",
                preamble='#include "cuComplex.h"'
            )
            new_prop = {"P": p_mat, "Pg": p_mat.conj(), "Q": q_op, "Qg": q_op_g}
            self._prop_cache[key] = new_prop
            return new_prop

    def diffract(self, a, dz):
        a._elwise_multiply(self._get_prop(dz)["P"], a, stream=self.stream)
        # a *= self._get_prop(dz)["P"]

    def diffract_g(self, ag, dz):
        ag._elwise_multiply(self._get_prop(dz)["Pg"], ag, stream=self.stream)
        # ag *= self._get_prop(dz)["Pg"]

    def scatter(self, u, n, dz):
        self._get_prop(dz)["Q"](u, n, stream=self.stream)

    def scatter_g(self, u, n, ug, ng, dz):
        self._get_prop(dz)["Qg"](u, n, ug, ng, stream=self.stream)
