from pycuda import elementwise, gpuarray, driver, reduction
from pycuda.gpuarray import GPUArray
from reikna.fft import FFT
import numpy as np
import reikna.cluda as cluda
from ssnp.utils import Multipliers
from contextlib import contextmanager

s = driver.Stream()
api = cluda.cuda_api()
thr = api.Thread(s)


class Funcs:
    __temp_memory_pool = {}
    reduce_mse_cr = reduction.ReductionKernel(
        dtype_out=np.double, neutral=0,
        reduce_expr="a+b",
        map_expr="(cuCabs(x[i]) - y[i]) * (cuCabs(x[i]) - y[i])",
        arguments="double2 *x, double *y",
        preamble='#include "cuComplex.h"'
    )
    reduce_mse_cc = reduction.ReductionKernel(
        dtype_out=np.double, neutral=0,
        reduce_expr="a+b",
        map_expr="cuCabs(cuCsub(x[i], y[i])) * cuCabs(cuCsub(x[i], y[i]))",
        arguments="double2 *x, double2 *y",
        preamble='#include "cuComplex.h"'
    )
    mse_cc_grad = elementwise.ElementwiseKernel(
        "double2 *u, double2 *m, double2 *out",
        """
        out[i] = cuCsub(u[i], m[i]);
        out[i].x *= 2; out[i].y *= 2;
        """,
        preamble='#include "cuComplex.h"'
    )
    mse_cr_grad = elementwise.ElementwiseKernel(
        "double2 *u, double *m, double2 *out",
        """
        temp = 2 * (1 - m[i] / cuCabs(u[i]));
        out[i].x = temp * u[i].x; out[i].y = temp * u[i].y;
        """,
        loop_prep="double temp",
        preamble='#include "cuComplex.h"'
    )
    mul_grad_bp = elementwise.ElementwiseKernel(
        "double2 *ug, double2 *mul",
        """
        ug[i] = cuCmul(ug[i], cuConj(mul[i]))
        """,
        preamble='#include "cuComplex.h"'
    )

    def __init__(self, arr_like, res, n0):
        shape = tuple(arr_like.shape)
        self._fft_callable = FFT(arr_like).compile(thr)
        self.shape = shape
        self._prop_cache = {}
        self._pupil_cache = {}
        self.res = res
        self.n0 = n0
        self.multiplier = Multipliers(shape, res)
        c_gamma = self.multiplier.c_gamma()
        kz = c_gamma * (2 * np.pi * res[2] * n0)
        self.kz = kz.astype(np.double)
        self.kz_gpu = gpuarray.to_gpu(kz)
        self.eva = np.exp(np.minimum((c_gamma - 0.2) * 5, 0))

    def fft(self, arr, output=None, copy=False, inverse=False):
        if output is not None:
            o = output
        elif copy:
            o = gpuarray.empty_like(arr)
        else:
            o = arr
        self._fft_callable(o, arr, inverse=inverse)
        return o

    def ifft(self, *args, **kwargs):
        return self.fft(*args, **kwargs, inverse=True)

    @contextmanager
    def fourier(self, arr, copy=False):
        yield self.fft(arr, copy=copy)
        self.ifft(arr)

    def diffract(self, *args):
        raise NotImplementedError

    def scatter(self, *args):
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

    @staticmethod
    def conj(arr: GPUArray):
        """copy from GPUArray.conj(self), do conj in-place"""
        dtype = arr.dtype
        if not arr.flags.forc:
            raise RuntimeError("only contiguous arrays may "
                               "be used as arguments to this operation")
        func = elementwise.get_conj_kernel(dtype)
        func.prepared_call(arr._grid, arr._block,
                           arr.gpudata, arr.gpudata,
                           arr.mem_size)

    @staticmethod
    def get_temp_mem(arr_like: GPUArray, index=0):
        key = (arr_like.shape, arr_like.dtype, index)
        try:
            return Funcs.__temp_memory_pool[key]
        except KeyError:
            mem = gpuarray.empty_like(arr_like)
            Funcs.__temp_memory_pool[key] = mem
            return mem


class SSNPFuncs(Funcs):
    __fused_mam_callable = elementwise.ElementwiseKernel(
        "double2 *a, double2 *b, "
        "double2 *p00, double2 *p01, double2 *p10, double2 *p11",
        """
        temp = cuCfma(a[i], p00[i], cuCmul(b[i], p01[i]));
        b[i] = cuCfma(a[i], p10[i], cuCmul(b[i], p11[i]));
        a[i] = temp;
        """,
        loop_prep="cuDoubleComplex temp",
        preamble='#include "cuComplex.h"'
    )
    split_prop_kernel = elementwise.ElementwiseKernel(
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
    merge_prop_kernel = elementwise.ElementwiseKernel(
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
    merge_grad_kernel = elementwise.ElementwiseKernel(
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
                    temp = {phase_factor} * n_[i] * ({2 * n0} + n_[i]);
                    ud[i].x -= temp * u[i].x;
                    ud[i].y -= temp * u[i].y;
                """,
                loop_prep="double temp",
            )
            q_op_g = elementwise.ElementwiseKernel(
                "double2 *u, double *n_, double2 *ug, double2 *udg, double *n_g",
                # Forward: ud = ud - temp * u
                # ng = Re{-udg * conj(u) * (d_temp(n) / d_n)}
                f"""
                    temp = {phase_factor} * n_[i] * ({2 * n0} + n_[i]);
                    ug[i].x -= temp * udg[i].x;
                    ug[i].y -= temp * udg[i].y;
                    n_g[i] = -(udg[i].x * u[i].x + udg[i].y * u[i].y) * {phase_factor} * ({2 * n0} + 2 * n_[i])
                """,
                loop_prep="double temp",
                preamble='#include "cuComplex.h"'
            )
            new_prop = {"P": p_mat, "Pg": [p_mat[0].conj(), p_mat[2].conj(), p_mat[1].conj(), p_mat[3].conj()],
                        "Q": q_op, "Qg": q_op_g}
            self._prop_cache[key] = new_prop
            return new_prop

    def diffract(self, a, a_d, dz):
        self.__fused_mam_callable(a, a_d, *self._get_prop(dz)["P"])

    def scatter(self, u, u_d, n, dz):
        self._get_prop(dz)["Q"](u, u_d, n)

    def diffract_g(self, ag, a_dg, dz):
        self.__fused_mam_callable(ag, a_dg, *self._get_prop(dz)["Pg"])

    def scatter_g(self, u, n, ug, u_dg, ng, dz):
        self._get_prop(dz)["Qg"](u, n, ug, u_dg, ng)


class BPMFuncs(Funcs):
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
                    temp = make_cuDoubleComplex(cos(n_[i] * {phase_factor}), sin(n_[i] * {phase_factor}));
                    u[i] = cuCmul(u[i], temp);
                """,
                loop_prep="double2 temp",
                preamble='#include "cuComplex.h"'
            )
            q_op_g = elementwise.ElementwiseKernel(
                "double2 *u, double *n_, double2 *ug, double *n_g",
                # Forward: u=u*temp
                f"""
                    temp_conj = make_cuDoubleComplex(cos(n_[i] * {phase_factor}), -sin(n_[i] * {phase_factor}));
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
        a *= self._get_prop(dz)["P"]

    def diffract_g(self, ag, dz):
        ag *= self._get_prop(dz)["Pg"]

    def scatter(self, u, n, dz):
        self._get_prop(dz)["Q"](u, n)

    def scatter_g(self, u, n, ug, ng, dz):
        self._get_prop(dz)["Qg"](u, n, ug, ng)
