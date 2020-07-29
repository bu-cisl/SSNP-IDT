__funcs_cache = {}

import numpy as np
from pycuda import gpuarray
from .funcs import BPMFuncs, SSNPFuncs, Funcs
from .utils import param_check
from .const import config


def ssnp_step(u, u_d, dz, n=None, output=None):
    """
    SSNP main operation of one step

    :param u: x-y complex amplitude
    :param u_d: z partial derivative of u
    :param dz: step size along z axis
    :param n: refractive index along x-y distribution in this slice. Use background N0 if not provided.
    :param output:
    :return: new (u, u_d) after a step towards +z direction
    """
    param_check(u=u, u_d=u_d, n=n)
    funcs: SSNPFuncs = get_funcs(u, config.res, model="ssnp")
    a = funcs.fft(u, output=output)
    a_d = funcs.fft(u_d, output=output)
    funcs.diffract(a, a_d, dz)
    u = funcs.ifft(a)
    u_d = funcs.ifft(a_d)
    # n = n / N0
    if n is not None:
        funcs.scatter(u, u_d, n, dz)
    return u, u_d


def bpm_step(u, dz, n=None, output=None):
    """
    BPM main operation of one step

    :param u: x-y complex amplitude
    :param dz: step size along z axis
    :param n: refractive index along x-y distribution in this slice. Use background N0 if not provided.
    :param output:
    :return: new (u, u_d) after a step towards +z direction
    """
    param_check(u=u, n=n, output=output)
    funcs: BPMFuncs = get_funcs(u, config.res, model="bpm")
    a = funcs.fft(u, output=output)
    funcs.diffract(a, dz)
    u = funcs.ifft(a)
    if n is not None:
        funcs.scatter(u, n, dz)
    # if not PERIODIC_BOUNDARY:
    #     absorb = tf.constant(_outflow_absorb(), DATA_TYPE)
    #     u *= absorb
    return u


def bpm_grad_bp(u, ug, dz, n=None, ng=None):
    param_check(u_1=u, u_grad=ug, n=n, n_grad=ng)
    funcs: BPMFuncs = get_funcs(ug, config.res, model="bpm")
    if n is not None:
        funcs.scatter_g(u, n, ug, ng, dz)
    ag = funcs.fft(ug)
    funcs.diffract_g(ag, dz)
    funcs.ifft(ag)
    return ng


def ssnp_grad_bp(u, ug, u_dg, dz, n=None, ng=None):
    param_check(u_1=u, u_grad=ug, u_d_grad=u_dg, n=n, n_grad=ng)
    funcs: SSNPFuncs = get_funcs(ug, config.res, model="ssnp")
    if n is not None:
        funcs.scatter_g(u, n, ug, u_dg, ng, dz)
    ag = funcs.fft(ug)
    a_dg = funcs.fft(u_dg)
    funcs.diffract_g(ag, a_dg, dz)
    funcs.ifft(ag)
    funcs.ifft(a_dg)
    return ng


def reduce_mse(u, measurement):
    param_check(u=u, measurement=measurement)
    if u.dtype != np.complex128:
        raise ValueError(f"u dtype {u.dtype} is incompatible")
    if measurement.dtype == np.complex128:
        result = Funcs.reduce_mse_cc(u, measurement)
    elif measurement.dtype == np.double:
        result = Funcs.reduce_mse_cr(u, measurement)
    else:
        raise ValueError(f"measurement dtype {measurement.dtype} is incompatible")
    return result.get()


def reduce_mse_grad(u, measurement, output=None):
    param_check(u=u, measurement=measurement, output=output)
    if u.dtype != np.complex128:
        raise ValueError(f"u dtype {u.dtype} is incompatible")
    if output is None:
        output = gpuarray.empty_like(u)
    if measurement.dtype == np.complex128:
        Funcs.mse_cc_grad(u, measurement, output)
    elif measurement.dtype == np.double:
        Funcs.mse_cr_grad(u, measurement, output)
    else:
        raise ValueError(f"measurement dtype {measurement.dtype} is incompatible")
    return output


def binary_pupil(u, na, multiplier=None):
    funcs = get_funcs(u, config.res, model="any")
    a = funcs.fft(u)
    if multiplier is None:
        a *= funcs.multiplier.binary_pupil(na, gpu=True)
    else:
        a *= multiplier.binary_pupil(na, gpu=True)
    return funcs.ifft(a)


def get_multiplier(arr_like):
    funcs = get_funcs(arr_like, config.res, model="any")
    return funcs.multiplier


def u_mul_grad_bp(ug, mul):
    funcs = get_funcs(ug, config.res, model="any")
    funcs.mul_grad_bp(ug, mul)
    return ug


def pure_forward_d(u, output=None):
    """
    Calculate z partial derivative for a initial x-y complex amplitude in free
    (or homogeneous) space due to pure forward propagation.

    :param u: x-y complex amplitude
    :param output: (optional) output memory, must have same shape and dtype
    :return: z partial derivative of u
    """
    funcs = get_funcs(u, config.res, model="ssnp")
    af = funcs.fft(u, output=funcs.get_temp_mem(u))
    if output is None:
        ab = gpuarray.zeros_like(af)
    else:
        param_check(u=u, output=output)
        if output.dtype != u.dtype:
            raise ValueError("incompatible output memory")
        ab = output
        zero = np.zeros((), u.dtype)
        ab.fill(zero)
    # Variables: af = fft(u), ab = 0, u is not changed
    funcs.merge_prop_kernel(af, ab, funcs.kz_gpu)
    ud = funcs.ifft(ab)
    return ud


def merge_prop(uf, ub, copy=False):
    param_check(uf=uf, ub=ub)
    funcs = get_funcs(uf, config.res, model="ssnp")
    af = funcs.fft(uf, copy=copy)
    ab = funcs.fft(ub, copy=copy)
    funcs.merge_prop_kernel(af, ab, funcs.kz_gpu)
    u = funcs.ifft(af)
    u_d = funcs.ifft(ab)
    return u, u_d


def split_prop(u, u_d, copy=False):
    param_check(u=u, u_d=u_d)
    funcs = get_funcs(u, config.res, model="ssnp")
    a = funcs.fft(u, copy=copy)
    a_d = funcs.fft(u_d, copy=copy)
    funcs.split_prop_kernel(a, a_d, funcs.kz_gpu)
    uf = funcs.ifft(a)
    ud = funcs.ifft(a_d)
    return uf, ud


def merge_grad(ufg, ubg, copy=False):
    param_check(uf_grad=ufg, ub_grad=ubg)
    funcs = get_funcs(ufg, config.res, model="ssnp")
    afg = funcs.fft(ufg, copy=copy)
    abg = funcs.fft(ubg, copy=copy)
    funcs.merge_grad_kernel(afg, abg, funcs.kz_gpu)
    ug = funcs.ifft(afg)
    u_dg = funcs.ifft(abg)
    return ug, u_dg


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
            funcs = model(arr_like, res, config.n0)
        except KeyError:
            raise ValueError(f"unknown model {model}") from None
    __funcs_cache[key] = funcs
    return funcs
