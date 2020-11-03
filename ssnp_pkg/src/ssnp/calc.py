__funcs_cache = {}

import numpy as np
from pycuda import gpuarray
from ssnp.funcs import BPMFuncs, SSNPFuncs, Funcs
from ssnp.utils import param_check, config as global_config, Multipliers


def ssnp_step(u, u_d, dz, n=None, output=None, stream=None):
    """
    SSNP main operation of one step

    :param u: x-y complex amplitude
    :param u_d: z partial derivative of u
    :param dz: step size along z axis
    :param n: refractive index along x-y distribution in this slice. Use background N0 if not provided.
    :param output:
    :param stream:
    :return: new (u, u_d) after a step towards +z direction
    """
    param_check(u=u, u_d=u_d, n=n)
    funcs: SSNPFuncs = get_funcs(u, model="ssnp", stream=stream)
    a = funcs.fft(u, output=output)
    a_d = funcs.fft(u_d, output=output)
    funcs.diffract(a, a_d, dz)
    u = funcs.ifft(a)
    u_d = funcs.ifft(a_d)
    # n = n / N0
    if n is not None:
        funcs.scatter(u, u_d, n, dz)
    return u, u_d


def bpm_step(u, dz, n=None, output=None, stream=None):
    """
    BPM main operation of one step

    :param u: x-y complex amplitude
    :param dz: step size along z axis
    :param n: refractive index along x-y distribution in this slice. Use background N0 if not provided.
    :param output:
    :param stream:
    :return: new (u, u_d) after a step towards +z direction
    """
    param_check(u=u, n=n, output=output)
    funcs: BPMFuncs = get_funcs(u, model="bpm", stream=stream)
    a = funcs.fft(u, output=output)
    funcs.diffract(a, dz)
    u = funcs.ifft(a)
    if n is not None:
        funcs.scatter(u, n, dz)
    return u


def bpm_grad_bp(u, ug, dz, n=None, ng=None, stream=None):
    param_check(u_1=u, u_grad=ug, n=n, n_grad=ng)
    funcs: BPMFuncs = get_funcs(ug, model="bpm", stream=stream)
    if n is not None:
        funcs.scatter_g(u, n, ug, ng, dz)
    with funcs.fourier(ug) as ag:
        funcs.diffract_g(ag, dz)
    return ng


def ssnp_grad_bp(u, ug, u_dg, dz, n=None, ng=None, stream=None):
    param_check(u_1=u, u_grad=ug, u_d_grad=u_dg, n=n, n_grad=ng)
    funcs: SSNPFuncs = get_funcs(ug, model="ssnp", stream=stream)
    if n is not None:
        funcs.scatter_g(u, n, ug, u_dg, ng, dz)
    with funcs.fourier(ug) as ag, funcs.fourier(u_dg) as a_dg:
        funcs.diffract_g(ag, a_dg, dz)
    return ng


def reduce_mse(u, measurement, stream=None):
    param_check(u=u, measurement=measurement)
    if u.dtype != np.complex128:
        raise ValueError(f"u dtype {u.dtype} is incompatible")
    funcs = get_funcs(u, model="any", stream=stream)
    if measurement.dtype == np.complex128:
        result = funcs.reduce_mse_cc(u, measurement)
    elif measurement.dtype == np.double:
        result = funcs.reduce_mse_cr(u, measurement)
    else:
        raise ValueError(f"measurement dtype {measurement.dtype} is incompatible")
    if stream is None:
        return result.get()
    else:
        return result.get(pagelocked=True, async_=True, stream=stream)


def reduce_mse_grad(u, measurement, output=None, stream=None):
    param_check(u=u, measurement=measurement, output=output)
    if u.dtype != np.complex128:
        raise ValueError(f"u dtype {u.dtype} is incompatible")
    funcs = get_funcs(u, model="any", stream=stream)
    if output is None:
        output = gpuarray.empty_like(u)
    if measurement.dtype == np.complex128:
        funcs.mse_cc_grad(u, measurement, output)
    elif measurement.dtype == np.double:
        funcs.mse_cr_grad(u, measurement, output)
    else:
        raise ValueError(f"measurement dtype {measurement.dtype} is incompatible")
    return output


def get_multiplier(arr_like, res=None, stream=None):
    # funcs = get_funcs(arr_like, model="any", stream=stream)
    if res is None:
        res = global_config.res
    return Multipliers(arr_like.shape, res, stream)


def u_mul_grad_bp(ug, mul, copy=False, stream=None):
    funcs = get_funcs(ug, model="any", stream=stream)
    if copy:
        out = ug.copy()
    else:
        out = ug
    funcs.mul_grad_bp(out, mul)
    return out


# def pure_forward_d(u, output=None):
#     """
#     Calculate z partial derivative for a initial x-y complex amplitude in free
#     (or homogeneous) space due to pure forward propagation.
#
#     :param u: x-y complex amplitude
#     :param output: (optional) output memory, must have same shape and dtype
#     :return: z partial derivative of u
#     """
#     warn("pure_forward_d(u) is deprecated, use merge_prop(u, zero) instead",
#          DeprecationWarning, stacklevel=2)
#     funcs = get_funcs(u, model="ssnp")
#     af = funcs.fft(u, output=funcs.get_temp_mem(u))
#     if output is None:
#         ab = gpuarray.zeros_like(af)
#     else:
#         param_check(u=u, output=output)
#         if output.dtype != u.dtype:
#             raise ValueError("incompatible output memory")
#         ab = output
#         zero = np.zeros((), u.dtype)
#         ab.fill(zero)
#     # Variables: af = fft(u), ab = 0, u is not changed
#     funcs.merge_prop_kernel(af, ab, funcs.kz_gpu)
#     ud = funcs.ifft(ab)
#     return ud


def merge_prop(uf, ub, copy=False, stream=None):
    param_check(uf=uf, ub=ub)
    funcs = get_funcs(uf, model="ssnp", stream=stream)
    af = funcs.fft(uf, copy=copy)
    ab = funcs.fft(ub, copy=copy)
    funcs.merge_prop(af, ab)
    u = funcs.ifft(af)
    u_d = funcs.ifft(ab)
    return u, u_d


def split_prop(u, u_d, copy=False, stream=None):
    param_check(u=u, u_d=u_d)
    funcs = get_funcs(u, model="ssnp", stream=stream)
    a = funcs.fft(u, copy=copy)
    a_d = funcs.fft(u_d, copy=copy)
    funcs.split_prop(a, a_d)
    uf = funcs.ifft(a)
    ud = funcs.ifft(a_d)
    return uf, ud


def merge_grad(ufg, ubg, copy=False, stream=None):
    param_check(uf_grad=ufg, ub_grad=ubg)
    funcs = get_funcs(ufg, model="ssnp", stream=stream)
    afg = funcs.fft(ufg, copy=copy)
    abg = funcs.fft(ubg, copy=copy)
    funcs.merge_grad(afg, abg)
    ug = funcs.ifft(afg)
    u_dg = funcs.ifft(abg)
    return ug, u_dg


def get_funcs(arr_like, config=None, *, model="any", stream=None):
    global __funcs_cache
    model = model.lower()
    shape = tuple(arr_like.shape)
    if config is None:
        config = global_config

    res = config.res
    # if model == "any":
    #     try:
    #         key = (shape, res, "ssnp", stream)
    #         return __funcs_cache[key]
    #     except KeyError:
    #         return get_funcs(arr_like, config, model="bpm", stream=stream)

    key = (shape, res, model, stream)
    try:
        return __funcs_cache[key]
    except KeyError:
        try:
            model = {"ssnp": SSNPFuncs, "bpm": BPMFuncs, "any": Funcs}[model]
            funcs = model(arr_like, res, config.n0, stream)
        except KeyError:
            raise ValueError(f"unknown model {model}") from None
    __funcs_cache[key] = funcs
    return funcs
