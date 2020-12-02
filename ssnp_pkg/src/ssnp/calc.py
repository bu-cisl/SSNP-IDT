import numpy as np
from pycuda import gpuarray
from ssnp.funcs import BPMFuncs, SSNPFuncs, Funcs
from ssnp.utils import param_check, config as global_config, Multipliers
from numbers import Number


def ssnp_step(u, u_d, dz, n=None, output=None, config=None, stream=None):
    """
    SSNP main operation of one step

    :param u: x-y complex amplitude
    :param u_d: z partial derivative of u
    :param dz: step size along z axis
    :param n: refractive index along x-y distribution in this slice. Use background N0 if not provided.
    :param output:
    :param config:
    :param stream:
    :return: new (u, u_d) after a step towards +z direction
    """
    if len(u.shape) == 3:
        param_check(u_batch=u, u_d_batch=u_d, output_batch=output)
        param_check(u=u[0], n=n)
    else:
        param_check(u=u, u_d=u_d, n=n, output=output)
    funcs: SSNPFuncs = get_funcs(u, config, model="ssnp", stream=stream)
    a = funcs.fft(u, output=output)
    a_d = funcs.fft(u_d, output=output)
    funcs.diffract(a, a_d, dz)
    u = funcs.ifft(a)
    u_d = funcs.ifft(a_d)
    # n = n / N0
    if n is not None:
        funcs.scatter(u, u_d, n, dz)
    return u, u_d


def bpm_step(u, dz, n=None, output=None, config=None, stream=None):
    """
    BPM main operation of one step

    :param u: x-y complex amplitude
    :param dz: step size along z axis
    :param n: refractive index along x-y distribution in this slice. Use background N0 if not provided.
    :param output:
    :param config:
    :param stream:
    :return: new (u, u_d) after a step towards +z direction
    """
    if len(u.shape) == 3:
        param_check(u_batch=u, output_batch=output)
        param_check(u=u[0], n=n)
    else:
        param_check(u=u, n=n, output=output)
    funcs: BPMFuncs = get_funcs(u, config, model="bpm", stream=stream)
    a = funcs.fft(u, output=output)
    funcs.diffract(a, dz)
    u = funcs.ifft(a)
    if n is not None:
        funcs.scatter(u, n, dz)
    return u


def bpm_grad_bp(u, ug, dz, n=None, ng=None, config=None, stream=None):
    param_check(u_1=u, u_grad=ug, n_grad=ng)
    param_check(u_1=u[0] if len(u.shape) == 3 else u, n=n)
    funcs: BPMFuncs = get_funcs(ug, config, model="bpm", stream=stream)
    if n is not None:
        funcs.scatter_g(u, n, ug, ng, dz)
    with funcs.fourier(ug) as ag:
        funcs.diffract_g(ag, dz)
    return ng


def ssnp_grad_bp(u, ug, u_dg, dz, n=None, ng=None, config=None, stream=None):
    param_check(u_1=u, u_grad=ug, u_d_grad=u_dg, n_grad=ng)
    param_check(u_1=u[0] if len(u.shape) == 3 else u, n=n)
    funcs: SSNPFuncs = get_funcs(ug, config, model="ssnp", stream=stream)
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
    # if measurement.dtype == np.complex128:
    #     result = funcs.reduce_mse_cc(u, measurement)
    # elif measurement.dtype == np.double:
    #     result = funcs.reduce_mse_cr(u, measurement)
    # else:
    #     raise ValueError(f"measurement dtype {measurement.dtype} is incompatible")
    result = funcs.reduce_mse(u, measurement)
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
    funcs.mse_grad(u, measurement, output)
    # if measurement.dtype == np.complex128:
    #     funcs.mse_cc_grad(u, measurement, output)
    # elif measurement.dtype == np.double:
    #     funcs.mse_cr_grad(u, measurement, output)
    # else:
    #     raise ValueError(f"measurement dtype {measurement.dtype} is incompatible")
    return output


def sum_batch(u, output=None, stream=None):
    param_check(u=u[0], output=output)
    funcs = get_funcs(u, model="any", stream=stream)
    if output is None:
        funcs.sum_batch(u, u[0])
    else:
        funcs.sum_batch(u, output)


def get_multiplier(shape, res=None, stream=None):
    # funcs = get_funcs(arr_like, model="any", stream=stream)
    if res is None:
        res = global_config.res
    return Multipliers(shape, res, stream)


def u_mul(u, mul, copy=False, stream=None, conj=False):
    funcs = get_funcs(u, model="any", stream=stream)
    if copy:
        out = gpuarray.empty_like(u)
    else:
        out = u

    if isinstance(mul, Number):
        if conj:
            mul = mul.conjugate()
        u._axpbz(mul, 0, out, funcs.stream)
        return out

    # GPUArray * GPUArray
    if len(u.shape) == 3:
        if len(mul.shape) == 3:  # rare case: batch * batch, conj will uses temp memory
            param_check(u=u, mul=mul)
            if conj:
                mul = mul.conj()
            u._elwise_multiply(mul, out, funcs.stream)
            return out
        else:
            param_check(u=u[0], mul=mul)
    else:
        param_check(u=u, mul=mul)
    funcs.op(u, "*", mul, out, name="mul")
    return out


def merge_prop(uf, ub, config=None, copy=False, stream=None):
    param_check(uf=uf, ub=ub)
    funcs = get_funcs(uf, config, model="ssnp", stream=stream)
    af = funcs.fft(uf, copy=copy)
    ab = funcs.fft(ub, copy=copy)
    funcs.merge_prop(af, ab)
    u = funcs.ifft(af)
    u_d = funcs.ifft(ab)
    return u, u_d


def split_prop(u, u_d, config=None, copy=False, stream=None):
    param_check(u=u, u_d=u_d)
    funcs = get_funcs(u, config, model="ssnp", stream=stream)
    a = funcs.fft(u, copy=copy)
    a_d = funcs.fft(u_d, copy=copy)
    funcs.split_prop(a, a_d)
    uf = funcs.ifft(a)
    ud = funcs.ifft(a_d)
    return uf, ud


def merge_grad(ufg, ubg, config=None, copy=False, stream=None):
    param_check(uf_grad=ufg, ub_grad=ubg)
    funcs = get_funcs(ufg, config, model="ssnp", stream=stream)
    afg = funcs.fft(ufg, copy=copy)
    abg = funcs.fft(ubg, copy=copy)
    funcs.merge_grad(afg, abg)
    ug = funcs.ifft(afg)
    u_dg = funcs.ifft(abg)
    return ug, u_dg


def get_funcs(arr_like, config=None, *, model="any", stream=None, fft_type="skcuda"):
    name = model.lower()
    if config is None:
        config = global_config
    if name == "any":
        res = n0 = None
    else:
        res = config.res
        n0 = config.n0
    try:
        model_init = {"ssnp": SSNPFuncs, "bpm": BPMFuncs, "any": Funcs}[name]
    except KeyError:
        raise ValueError(f"unknown model {model}") from None
    return model_init(arr_like, res, n0, stream, fft_type)
