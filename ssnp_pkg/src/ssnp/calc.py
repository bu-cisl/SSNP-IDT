import numpy as np
from pycuda import gpuarray
from .funcs import BPMFuncs, SSNPFuncs
# from warnings import warn

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
    assert isinstance(u, gpuarray.GPUArray)
    assert isinstance(u_d, gpuarray.GPUArray)
    if n is not None:
        assert isinstance(n, gpuarray.GPUArray)
    shape = u.shape
    try:
        assert u_d.shape == shape, f"u_d shape {u_d.shape}"
        if n is not None:
            assert n.shape == shape, f"n shape {n.shape}"
    except AssertionError as name:
        raise ValueError(f"cannot match {name} with u shape {shape}") from None
    funcs: SSNPFuncs = get_funcs(u, _res_deprecated, model="ssnp")

    a = funcs.fft(u)
    a_d = funcs.fft(u_d)
    funcs.diffract(a, a_d, dz)
    u = funcs.ifft(a)
    u_d = funcs.ifft(a_d)
    # n = n / N0
    if n is not None:
        funcs.scatter(u, u_d, n, dz)
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


def binary_pupil(u, na):
    funcs = get_funcs(u, _res_deprecated, model="any")
    funcs.binary_pupil(u, na)


def pure_forward_d(u, out=None):
    """
    Calculate z partial derivative for a initial x-y complex amplitude in free
    (or homogeneous) space due to pure forward propagation.

    :param u: x-y complex amplitude
    :param out: (optional) output memory, must have same shape and dtype
    :return: z partial derivative of u
    """
    funcs = get_funcs(u, _res_deprecated, model="ssnp")
    af = funcs.fft(u, output=funcs.get_temp_mem(u))
    if out is None:
        ab = gpuarray.zeros_like(af)
    else:
        if out.shape != u.shape or out.dtype != u.dtype:
            raise ValueError("incompatible output memory")
        ab = out
        zero = np.zeros((), u.dtype)
        ab.fill(zero)
    # Variables: af = fft(u), ab = 0, u is not changed
    funcs.merge_prop_kernel(af, ab, funcs.kz_gpu)
    ud = funcs.ifft(ab)
    return ud


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


def get_funcs(arr_like, res, model):
    global __funcs_cache
    model = model.lower()
    shape = tuple(arr_like.shape)
    res = tuple(res)
    if model == "any":
        try:
            key = (shape, res, "ssnp")
            return __funcs_cache[key]
        except KeyError:
            return get_funcs(arr_like, res, "bpm")

    key = (shape, res, model)
    try:
        return __funcs_cache[key]
    except KeyError:
        try:
            model = {"ssnp": SSNPFuncs, "bpm": BPMFuncs}[model]
            funcs = model(arr_like, res, _N0_deprecated)
        except KeyError:
            raise ValueError(f"unknown model {model}") from None
    __funcs_cache[key] = funcs
    return funcs
