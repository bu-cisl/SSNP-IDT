# import tensorflow as tf
import numpy as np
from pycuda import elementwise, gpuarray
from .const import EPS
from reikna.fft import FFT
import reikna.cluda as cluda

_res_deprecated = (0.1, 0.1, 0.1)
_N0_deprecated = 1
size = None
__funcs_cache = {}
api = cluda.cuda_api()
thr = api.Thread.create()


def ssnp_step(u, u_d, dz, n=None):
    """
    SSNP main operation of one step

    :param u: x-y complex amplitude
    :param u_d: z partial derivative of u
    :param dz: step size along z axis
    :param n: refractive index along x-y distribution in this slice. Use background N0 if not provided.
    :return: new (u, u_d) after a step towards +z direction
    """
    shape = u.shape
    try:
        assert u_d.shape == shape, "u_d"
        if n is not None:
            assert n.shape == shape
    except AssertionError as name:
        raise ValueError(f"cannot match {name} shape {n.shape} with u shape {shape}")
    # p = _get_ssnp_prop(shape, _res_deprecated, dz)
    funcs = get_funcs(u, _res_deprecated)

    a = funcs.fft(u)
    a_d = funcs.fft(u_d)
    funcs.diffract(a, a_d, dz)
    u = funcs.ifft(a)
    u_d = funcs.ifft(a_d)
    # n = n / N0
    if n is not None:
        funcs.scatter(u, u_d, n, dz)
    #    u_d -= ((2 * np.pi * _res_deprecated[2]) ** 2 * dz) * (n * (2 * _N0_deprecated + n) * u)
    # if not PERIODIC_BOUNDARY:
    #     absorb = tf.constant(_outflow_absorb(), DATA_TYPE)
    #     u *= absorb
    #     u_d *= absorb
    return u, u_d


def _c_gamma(shape, res):
    """
    Calculate cos(gamma) as a constant array at frequency domain. Gamma is the angle
    between the wave vector and z-axis. Note: N.A.=sin(gamma)

    The array is pre-shifted for later FFT operation.

    :return: cos(gamma) numpy array
    """
    c_alpha, c_beta = [
        np.fft.ifftshift(np.arange(-shape[i] / 2, shape[i] / 2).astype(np.double)) / shape[i] / res[i]
        for i in (0, 1)
    ]
    c_gamma = np.sqrt(np.maximum(1 - (np.square(c_alpha) + np.square(c_beta[:, None])), EPS))
    return c_gamma


def get_funcs(p: gpuarray, res):
    global __funcs_cache
    key = (tuple(p.shape), tuple(res))
    try:
        return __funcs_cache[key]
    except KeyError:
        funcs = _Funcs(p, res, _N0_deprecated)
        __funcs_cache[key] = funcs
    return funcs


class _Funcs:
    fused_mam_callable = None

    def __init__(self, arr_like: gpuarray, res, n0):
        self.fft_callable = FFT(arr_like).compile(thr)
        self.shape = arr_like.shape
        self.__prop_cache = {}
        self.res = res
        self.n0 = n0

    def fft(self, arr: gpuarray, output=False):
        if output:
            o = gpuarray.empty_like(arr)
        else:
            o = arr
        self.fft_callable(o, arr)
        return o

    def ifft(self, arr: gpuarray, output=False):
        w, d = arr.shape
        if output:
            o = arr.copy()
        else:
            o = arr
        _Funcs.conj(o)
        o /= w * d
        self.fft_callable(o, o)
        _Funcs.conj(o)
        return o

    @classmethod
    def _fused_mam(cls, *args):
        if cls.fused_mam_callable is None:
            kernel = elementwise.ElementwiseKernel(
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
            cls.fused_mam_callable = kernel
        cls.fused_mam_callable(*args)

    def diffract(self, a, a_d, dz):
        self._fused_mam(a, a_d, *self._get_prop(dz)["P"])

    def scatter(self, u, u_d, n, dz):
        self._get_prop(dz)["Q"](u, u_d, n)

    @staticmethod
    def conj(arr: gpuarray):
        dtype = arr.dtype
        if not arr.flags.forc:
            raise RuntimeError("only contiguous arrays may "
                               "be used as arguments to this operation")
        func = elementwise.get_conj_kernel(dtype)
        func.prepared_call(arr._grid, arr._block,
                           arr.gpudata, arr.gpudata,
                           arr.mem_size)

    def _get_prop(self, dz):
        shape = self.shape
        res = self.res
        n0 = self.n0
        key = round(dz * 1000)

        try:
            return self.__prop_cache[key]
        except KeyError:
            kz = _c_gamma(shape, res) * (2 * np.pi * res[2] * n0)
            kz = kz.astype(np.complex128)
            eva = np.exp(np.minimum((_c_gamma(shape, res) - 0.2) * 5, 0))
            p_mat = [np.cos(kz * dz), np.sin(kz * dz) / kz,
                     -np.sin(kz * dz) * kz, np.cos(kz * dz)]
            p_mat = [thr.to_device(i * eva) for i in p_mat]
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
            self.__prop_cache[key] = new_prop
            return new_prop

# def _evanescent_absorb(shape, res):
#     """
#         For 0.98<N.A.<1, decrease to 1~1/e per step
#
#         For N.A.>1 decrease to 1/e per step
#
#         :return: a 2D angular spectrum transmittance numpy array
#     """
#     return np.exp(np.minimum((_c_gamma(shape, res) - 0.2) * 5, 0))
