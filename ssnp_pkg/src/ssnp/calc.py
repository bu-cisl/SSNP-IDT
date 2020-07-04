import numpy as np
from pycuda import gpuarray
from pycuda.gpuarray import GPUArray
from .funcs import Funcs, BPMFuncs, SSNPFuncs, _c_gamma

_res_deprecated = (0.1, 0.1, 0.1)
_N0_deprecated = 1
__funcs_cache = {}


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
        assert u_d.shape == shape, f"u_d shape {u_d.shape}"
        if n is not None:
            assert n.shape == shape, f"n shape {n.shape}"
    except AssertionError as name:
        raise ValueError(f"cannot match {name} with u shape {shape}")
    funcs: SSNPFuncs = get_funcs(u, _res_deprecated, model="ssnp")

    a = funcs.fft(u)
    a_d = funcs.fft(u_d)
    funcs.diffract(a, a_d, dz)
    u = funcs.ifft(a)
    u_d = funcs.ifft(a_d)
    # n = n / N0
    if n is not None:
        funcs.scatter(u, u_d, n, dz)
    # if not PERIODIC_BOUNDARY:
    #     absorb = tf.constant(_outflow_absorb(), DATA_TYPE)
    #     u *= absorb
    #     u_d *= absorb
    return u, u_d


def bpm_step(u, dz, n=None):
    """
    BPM main operation of one step

    :param u: x-y complex amplitude
    :param dz: step size along z axis
    :param n: refractive index along x-y distribution in this slice. Use background N0 if not provided.
    :return: new (u, u_d) after a step towards +z direction
    """
    shape = u.shape
    if n is not None:
        if n.shape != shape:
            raise ValueError(f"cannot match n shape {n.shape} with u shape {shape}")
    funcs: BPMFuncs = get_funcs(u, _res_deprecated, model="bpm")
    a = funcs.fft(u)
    funcs.diffract(a, dz)
    u = funcs.ifft(a)
    if n is not None:
        funcs.scatter(u, n, dz)
    # if not PERIODIC_BOUNDARY:
    #     absorb = tf.constant(_outflow_absorb(), DATA_TYPE)
    #     u *= absorb
    return u


# def pure_forward_d_old(u):
#     """
#     Calculate z partial derivative for a initial x-y complex amplitude in free
#     (or homogeneous) space due to pure forward propagation.
#
#     :param u: x-y complex amplitude
#     :return: z partial derivative of u
#     """
#     funcs = get_funcs(u, _res_deprecated)
#     i_kz = gpuarray.to_gpu(funcs.kz * 1j)
#     a = funcs.fft(u, copy=True)
#     a *= i_kz
#     u_d = funcs.ifft(a)
#     return u_d


def pure_forward_d(u):
    """
    Calculate z partial derivative for a initial x-y complex amplitude in free
    (or homogeneous) space due to pure forward propagation.

    :param u: x-y complex amplitude
    :return: z partial derivative of u
    """
    funcs = get_funcs(u, _res_deprecated, model="ssnp")
    af = funcs.fft(u, copy=True)
    ab = gpuarray.zeros_like(af)
    funcs.merge_prop_kernel(af, ab, funcs.kz_gpu)
    u_d = funcs.ifft(ab)
    return u_d


def merge_prop(uf, ub, copy=False):
    assert uf.shape == ub.shape
    funcs = get_funcs(uf, _res_deprecated, model="ssnp")
    af = funcs.fft(uf, copy=copy)
    ab = funcs.fft(ub, copy=copy)
    funcs.merge_prop_kernel(af, ab, funcs.kz_gpu)
    u = funcs.ifft(af)
    u_d = funcs.ifft(ab)
    return u, u_d


def split_prop(u, u_d, copy=False):
    assert u.shape == u_d.shape
    funcs = get_funcs(u, _res_deprecated, model="ssnp")
    a = funcs.fft(u, copy=copy)
    a_d = funcs.fft(u_d, copy=copy)
    funcs.split_prop_kernel(a, a_d, funcs.kz_gpu)
    uf = funcs.ifft(a)
    ud = funcs.ifft(a_d)
    return uf, ud





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


def get_funcs(arr_like: GPUArray, res, model="ssnp"):
    global __funcs_cache
    model = model.lower()
    key = (tuple(arr_like.shape), tuple(res), model)
    try:
        return __funcs_cache[key]
    except KeyError:
        try:
            model = {"ssnp": SSNPFuncs, "bpm": BPMFuncs, "general": Funcs}[model]
            funcs = model(arr_like, res, _N0_deprecated)
        except KeyError:
            raise ValueError(f"unknown model {model}") from None
    __funcs_cache[key] = funcs
    return funcs
