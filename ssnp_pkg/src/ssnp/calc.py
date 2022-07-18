# import tensorflow as tf
import numpy as np
from pycuda import elementwise
from pycuda.compiler import SourceModule

from .const import EPS, COMPLEX_TYPE
from pycuda import gpuarray
# from pycuda import
from reikna.fft import FFT
import reikna.cluda as cluda

_res_deprecated = (0.1, 0.1, 0.1)
_N0_deprecated = 1
size = None
__prop_cache = {}
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
    for t_in in (u_d, n):
        if t_in is not None:
            if t_in.shape != shape:
                raise ValueError(f"the x,y shape {t_in.shape} is not {shape}")
    p = _get_ssnp_prop(shape, _res_deprecated, dz)
    fft = _get_funcs(u)

    a = fft.fft(u)
    a_d = fft.fft(u_d)
    fft.fused_mam(a, a_d, p[0][0], p[0][1], p[1][0], p[1][1])
    u = fft.ifft(a)
    u_d = fft.ifft(a_d)
    # n = n / N0
    if n is not None:
        u_d -= ((2 * np.pi * _res_deprecated[2]) ** 2 * dz) * (n * (2 * _N0_deprecated + n) * u)
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


def _get_ssnp_prop(shape, res, dz):
    global __prop_cache
    try:
        return __prop_cache[(shape, res, dz)]
    except KeyError:
        kz = _c_gamma(shape, res) * (2 * np.pi * res[2] * _N0_deprecated)
        kz = kz.astype(np.complex128)
        eva = np.exp(np.minimum((_c_gamma(shape, res) - 0.2) * 5, 0))
        new_prop = [[], []]
        new_prop[0].append(thr.to_device(np.cos(kz * dz) * eva))
        new_prop[0].append(thr.to_device(np.sin(kz * dz) / kz * eva))
        new_prop[1].append(thr.to_device(-np.sin(kz * dz) * kz * eva))
        new_prop[1].append(thr.to_device(np.cos(kz * dz) * eva))
        __prop_cache[(shape, res, dz)] = new_prop
    return new_prop


def _get_funcs(p: gpuarray):
    global __funcs_cache
    try:
        return __funcs_cache[p.shape]
    except KeyError:
        funcs = _FFTFuncs(p)
        __funcs_cache[p.shape] = funcs
    return funcs


class _FFTFuncs:
    def __init__(self, arr_like: gpuarray):
        self.fft_callable = FFT(arr_like).compile(thr)
        self.fused_mam_callable = _FFTFuncs.fused_mam_kernel(arr_like.shape)

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
        _FFTFuncs.conj(o)
        o /= w * d
        self.fft_callable(o, o)
        _FFTFuncs.conj(o)
        return o

    def fused_mam(self, *args):
        block = args[0]._block
        grid = args[0]._grid
        assert block[1] * block[2] * grid[1] == 1
        self.fused_mam_callable(*args, block=block, grid=grid)

    # @staticmethod
    # def fused_mam_kernel2(shape):
    #     mam = elementwise.ElementwiseKernel(get)

    @staticmethod
    def fused_mam_kernel(shape):
        h, w = shape
        len_ = h * w
        mam = SourceModule(f"""
        #include "cuComplex.h"
        #define LEN {len_}
        __global__ void fmam(cuDoubleComplex *a, cuDoubleComplex *b,
                             cuDoubleComplex *p00, cuDoubleComplex *p01,
                             cuDoubleComplex *p10, cuDoubleComplex *p11)
        {{
            int index = threadIdx.x + blockIdx.x*blockDim.x;
            cuDoubleComplex temp;
            if (index < LEN)
            {{
                temp = cuCfma(a[index], p00[index], cuCmul(b[index], p01[index]));
                b[index] = cuCfma(a[index], p10[index], cuCmul(b[index], p11[index]));
                a[index] = temp;
            }}
        }}
        """)
        # mam = elementwise.ElementwiseKernel()
        return mam.get_function('fmam')

    @staticmethod
    def ud_modify_kernel():
        elementwise.ElementwiseKernel(
            "double2 *u, double2 *ud, double n0, double mul",
            "ud[i] = cuCsub(ud[i], make_cuDoubleComplex(u[i].x*mul,u[i].y*mul))",
            preamble='#include "cuComplex.h"'
        )

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

# def _evanescent_absorb(shape, res) -> np.ndarray:
#     """
#         For 0.98<N.A.<1, decrease to 1~1/e per step
#
#         For N.A.>1 decrease to 1/e per step
#
#         :return: a 2D angular spectrum transmittance numpy array
#     """
#     return np.exp(np.minimum((_c_gamma(shape, res) - 0.2) * 5, 0))
