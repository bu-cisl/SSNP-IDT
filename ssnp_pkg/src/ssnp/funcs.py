from pycuda import elementwise, gpuarray, driver
from pycuda.gpuarray import GPUArray
from reikna.fft import FFT
import numpy as np

import reikna.cluda as cluda

s = driver.Stream()
api = cluda.cuda_api()
thr = api.Thread(s)


def _c_gamma(shape, res):
    """
    Calculate cos(gamma) as a constant array at frequency domain. Gamma is the angle
    between the wave vector and z-axis. Note: N.A.=sin(gamma)

    The array is pre-shifted for later FFT operation.

    :return: cos(gamma) numpy array
    """
    eps = 1E-8
    c_alpha, c_beta = [
        np.fft.ifftshift(np.arange(-shape[i] / 2, shape[i] / 2).astype(np.double)) / shape[i] / res[i]
        for i in (0, 1)
    ]
    c_gamma = np.sqrt(np.maximum(1 - (np.square(c_alpha) + np.square(c_beta[:, None])), eps))
    return c_gamma


class Funcs:
    __temp_memory_pool = {}

    def __init__(self, arr_like, res, n0):
        shape = tuple(arr_like.shape)
        self._fft_callable = FFT(arr_like).compile(thr)
        self.shape = shape
        self._prop_cache = {}
        self._pupil_cache = {}
        self.res = res
        self.n0 = n0
        kz = _c_gamma(shape, res) * (2 * np.pi * res[2] * n0)
        self.kz = kz.astype(np.double)
        self.kz_gpu = gpuarray.to_gpu(kz)
        self.eva = np.exp(np.minimum((_c_gamma(shape, res) - 0.2) * 5, 0))

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

    # def ifft_old(self, arr: GPUArray, copy=False):
    #     w, d = arr.shape
    #     if copy:
    #         o = arr.copy()
    #     else:
    #         o = arr
    #     Funcs.conj(o)
    #     o /= w * d
    #     self._fft_callable(o, o)
    #     Funcs.conj(o)
    #     return o

    def diffract(self, *args):
        raise NotImplementedError

    def scatter(self, *args):
        raise NotImplementedError

    def binary_pupil(self, u, na):
        key = round(na * 1000)
        try:
            mask = self._pupil_cache[key]
        except KeyError:
            mask = np.greater(_c_gamma(u.shape, self.res), np.sqrt(1 - na ** 2))
            mask = mask.astype(np.complex128)
            mask = gpuarray.to_gpu(mask)
            self._pupil_cache[key] = mask
        self.fft(u)
        u *= mask
        self.ifft(u)
        return u

    @staticmethod
    def conj(arr: GPUArray):
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

#     """
#         For 0.98<N.A.<1, decrease to 1~1/e per step
#
#         For N.A.>1 decrease to 1/e per step
#
#         :return: a 2D angular spectrum transmittance numpy array
#     """


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

    def __init__(self, arr_like: GPUArray, res, n0):
        super().__init__(arr_like, res, n0)
        self.split_prop_kernel = elementwise.ElementwiseKernel(
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
        self.merge_prop_kernel = elementwise.ElementwiseKernel(
            "double2 *af, double2 *ab, double *kz",
            """
            temp = cuCsub(af[i], ab[i]);
            af[i] = cuCadd(af[i], ab[i]);
            ab[i] = make_cuDoubleComplex(-cuCimag(temp)*kz[i], cuCreal(temp)*kz[i]);
            """,
            loop_prep="cuDoubleComplex temp",
            preamble='#include "cuComplex.h"'
        )

    def _get_prop(self, dz):
        shape = self.shape
        res = self.res
        n0 = self.n0
        key = round(dz * 1000)
        try:
            return self._prop_cache[key]
        except KeyError:
            kz = self.kz
            eva = np.exp(np.minimum((_c_gamma(shape, res) - 0.2) * 5, 0))
            p_mat = [np.cos(kz * dz), np.sin(kz * dz) / kz,
                     -np.sin(kz * dz) * kz, np.cos(kz * dz)]
            p_mat = [gpuarray.to_gpu((i * eva).astype(np.complex128)) for i in p_mat]
            q_op = elementwise.ElementwiseKernel(
                "double2 *u, double2 *ud, double *n_",
                f"""
                    temp = {(2 * np.pi * res[2]) ** 2 * dz} * n_[i] * ({2 * n0} + n_[i]);
                    ud[i].x -= temp*u[i].x;
                    ud[i].y -= temp*u[i].y;
                """,
                loop_prep="double temp",
            )
            new_prop = {"P": p_mat, "Q": q_op}
            self._prop_cache[key] = new_prop
            return new_prop

    def diffract(self, a, a_d, dz):
        self.__fused_mam_callable(a, a_d, *self._get_prop(dz)["P"])

    def scatter(self, u, u_d, n, dz):
        self._get_prop(dz)["Q"](u, u_d, n)


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
            p_mat = np.exp(kz * (1j * dz)) * eva
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
            new_prop = {"P": p_mat, "Q": q_op}
            self._prop_cache[key] = new_prop
            return new_prop

    def diffract(self, a, dz):
        a *= self._get_prop(dz)["P"]

    def scatter(self, u, n, dz):
        self._get_prop(dz)["Q"](u, n)
