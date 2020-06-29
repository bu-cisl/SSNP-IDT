# import tensorflow as tf
import numpy as np
from pycuda import elementwise, gpuarray
from reikna.fft import FFT
import reikna.cluda as cluda

_res_deprecated = (0.1, 0.1, 0.1)
_N0_deprecated = 1
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


def pure_forward_d(u):
    """
    Calculate z partial derivative for a initial x-y complex amplitude in free
    (or homogeneous) space due to pure forward propagation.

    :param u: x-y complex amplitude
    :return: z partial derivative of u
    """
    funcs = get_funcs(u, _res_deprecated)
    i_kz = gpuarray.to_gpu(funcs.kz * 1j)
    a = funcs.fft(u, copy=True)
    a *= i_kz
    u_d = funcs.ifft(a)
    return u_d


def pure_forward_d2(u):
    funcs = get_funcs(u, _res_deprecated)
    af = funcs.fft(u, copy=True)
    ab = gpuarray.zeros_like(af)
    funcs.merge_prop_kernel(af, ab, funcs.kz_gpu)
    u_d = funcs.ifft(ab)
    return u_d


def merge_prop(uf, ub, copy=False):
    assert uf.shape == ub.shape
    funcs = get_funcs(uf, _res_deprecated)
    af = funcs.fft(uf, copy=copy)
    ab = funcs.fft(ub, copy=copy)
    funcs.merge_prop_kernel(af, ab, funcs.kz_gpu)
    u = funcs.ifft(af)
    u_d = funcs.ifft(ab)
    return u, u_d


def split_prop(u, u_d, copy=False):
    assert u.shape == u_d.shape
    funcs = get_funcs(u, _res_deprecated)
    a = funcs.fft(u, copy=copy)
    a_d = funcs.fft(u_d, copy=copy)
    funcs.split_prop_kernel(a, a_d, funcs.kz_gpu)
    uf = funcs.ifft(a)
    ud = funcs.ifft(a_d)
    return uf, ud


def binary_pupil(u, na):
    funcs = get_funcs(u, _res_deprecated)
    mask = np.greater(_c_gamma(u.shape, _res_deprecated), np.sqrt(1 - na ** 2))
    mask = mask.astype(np.complex128)
    mask = gpuarray.to_gpu(mask)
    funcs.fft(u)
    u *= mask
    funcs.ifft(u)
    return u


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


def tilt(img, c_ab, *, trunc=False, copy=False):
    """
    Tilt an image as illumination

    :param copy:
    :param img: Amplitude graph
    :param c_ab: (cos(alpha), cos(beta))
    :param trunc: whether trunc to a grid point in Fourier plane
    :return: complex tf Tensor of input field
    """
    size = img.shape[::-1]
    if len(size) != 2:
        raise ValueError(f"Illumination should be a 2-D tensor rather than shape '{img.shape}'.")
    norm = [size[i] * _res_deprecated[i] * _N0_deprecated for i in (0, 1)]
    if trunc:
        c_ab = [np.trunc(c_ab[i] * norm[i]) / norm[i] for i in (0, 1)]
    xr, yr = [np.arange(size[i]) / size[i] * c_ab[i] * norm[i] for i in (0, 1)]
    phase = np.mod(xr + yr[:, None], 1).astype(np.double) * 2 * np.pi
    phase = gpuarray.to_gpu(np.exp(1j * phase))
    if copy:
        img = img * phase
    else:
        img *= phase
    return img, c_ab


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
    kz: np.ndarray
    fused_mam_callable = None

    def __init__(self, arr_like: gpuarray, res, n0):
        self.fft_callable = FFT(arr_like).compile(thr)
        self.shape = arr_like.shape
        self.__prop_cache = {}
        self.res = res
        self.n0 = n0
        kz = _c_gamma(self.shape, res) * (2 * np.pi * res[2] * n0)
        self.kz = kz.astype(np.double)
        self.kz_gpu = gpuarray.to_gpu(kz)
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

    def fft(self, arr: gpuarray, copy=False):
        if copy:
            o = gpuarray.empty_like(arr)
        else:
            o = arr
        self.fft_callable(o, arr)
        return o

    def ifft(self, arr: gpuarray, copy=False):
        w, d = arr.shape
        if copy:
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
    def conj(arr: gpuarray.GPUArray):
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
            self.__prop_cache[key] = new_prop
            return new_prop

#     """
#         For 0.98<N.A.<1, decrease to 1~1/e per step
#
#         For N.A.>1 decrease to 1/e per step
#
#         :return: a 2D angular spectrum transmittance numpy array
#     """
