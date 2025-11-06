import numpy as np
from pycuda import gpuarray
from ssnp.funcs import BPMFuncs, SSNPFuncs, Funcs, MLBFuncs
from ssnp.utils import param_check, config as global_config, Multipliers
from numbers import Complex


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
    if output is None:
        output = (None, None)
    if len(u.shape) == 3:
        param_check(u_batch=u, u_d_batch=u_d, u_output_batch=output[0], ud_output_batch=output[1])
        param_check(u=u[0], n=n)
    else:
        param_check(u=u, u_d=u_d, n=n, u_output=output[0], ud_output=output[1])
    funcs: SSNPFuncs = get_funcs(u, config, model="ssnp", stream=stream)
    a = funcs.fft(u, output=output[0])
    a_d = funcs.fft(u_d, output=output[1])
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
    funcs: BPMFuncs = get_funcs(ug, config, model="bpm", stream=stream)
    if n is not None:
        param_check(u_grad=ug[0] if len(ug.shape) == 3 else ug, n=n)
        funcs.scatter_g(u, n, ug, ng, dz)
    with funcs.fourier(ug) as ag:
        funcs.diffract_g(ag, dz)
    return ng


def ssnp_grad_bp(u, ug, u_dg, dz, n=None, ng=None, config=None, stream=None):
    """
    SSNP gradient back propagation

    When n is not provided (prop in homogenous media), u and ng are never used;
    When n is provided, ng is optional, and if ng is not provided, u is never used.
    """
    # TODO: why I change ng to full batch size in commit 1b040803 ? Should change back (same for BPM)
    param_check(u_1=u, u_grad=ug, u_d_grad=u_dg, n_grad=ng)
    funcs: SSNPFuncs = get_funcs(ug, config, model="ssnp", stream=stream)
    if n is not None:
        param_check(u_grad=ug[0] if len(ug.shape) == 3 else ug, n=n)
        funcs.scatter_g(u, n, ug, u_dg, ng, dz)
    with funcs.fourier(ug) as ag, funcs.fourier(u_dg) as a_dg:
        funcs.diffract_g(ag, a_dg, dz)
    return ng

def mlb_step(u, temp_like_u, dz, n=None, config=None, stream=None):
    """
    MLB main operation of one step

    :param u: x-y complex amplitude
    :param temp_like_u: temporary memory, array with the same attribute as u, only for scattering
    :param dz: step size along z axis
    :param n: refractive index along x-y distribution in this slice. Use background N0 if not provided.
    :param config:
    :param stream:
    :return: new (u, u_d) after a step towards +z direction
    """
    if len(u.shape) == 3:
        param_check(u=u[0], n=n)
    else:
        param_check(u=u, n=n)
    funcs: MLBFuncs = get_funcs(u, config, model="mlb", stream=stream)
    # Step 1 - calculate u_scatter = u * ((n0 + n) ** 2 - n0 ** 2)
    if n is not None:
        funcs.pure_scatter(u, n, dz, temp_like_u)
    a = funcs.fft(u)
    # Step 2 - a_with_scatter = a_without_scatter + (... / kz) * a_scatter
    if n is not None:
        a_s = funcs.fft(temp_like_u)
        funcs.merge_scatter(a, a_s, dz)
    # Step 3 - angular spectrum propagation as the final step, no matter scattered or not
    funcs.diffract(a, dz)
    u = funcs.ifft(a)
    return u

def reduce_mse(u, measurement, stream=None):
    param_check(u=u, measurement=measurement)
    if u.dtype != np.complex128:
        raise ValueError(f"u dtype {u.dtype} is incompatible")
    funcs = get_funcs(u, model="any", stream=stream)
    result = funcs.reduce_sse(u, measurement)
    if stream is None:
        return result.get() / u.size
    else:
        return result.get(pagelocked=True, async_=True, stream=stream) / u.size


def reduce_mse_grad(u, measurement, output=None, stream=None):
    param_check(u=u, measurement=measurement, output=output)
    if u.dtype != np.complex128:
        raise ValueError(f"u dtype {u.dtype} is incompatible")
    funcs = get_funcs(u, model="any", stream=stream)
    if output is None:
        output = gpuarray.empty_like(u)
    funcs.mse_grad(u, measurement, output)
    return output


def sum_batch(u, output=None, stream=None):
    param_check(u=u[0], output=output)
    funcs = get_funcs(u, model="any", stream=stream)
    if output is None:
        funcs.sum_batch(u, u[0])
    else:
        funcs.sum_batch(u, output)


def copy_batch(u, output, stream=None):
    param_check(u=u, output=output[0])
    for out_i in output:
        out_i.set_async(u, stream)


def abs_c2c(u, output, stream=None):
    param_check(u=u, output=output)
    funcs = get_funcs(u, model="any", stream=stream)
    if output is None:
        output = u
    funcs.abs_cc_krn(u, output)
    return output


def get_multiplier(shape, res=None, stream=None):
    # funcs = get_funcs(arr_like, model="any", stream=stream)
    if res is None:
        res = global_config.res
    return Multipliers(shape, res, stream)


def u_mul(u, mul, *, copy=False, out=None, stream=None, conj=False):
    funcs = get_funcs(u, model="any", stream=stream)
    if out is None:
        if copy:
            out = gpuarray.empty_like(u)
        else:
            out = u
    else:
        param_check(u=u, out=out)

    if isinstance(mul, Complex):
        if conj:
            mul = mul.conjugate()
        u._axpbz(mul, 0, out, funcs.stream)
    else:
        if batchwise := (len(u.shape) != len(mul.shape)):
            param_check(u_single=u[0], mul=mul)
        else:
            param_check(u=u, mul=mul)
        name = f"mul{'_conj' if conj else ''}{'_batch' if batchwise else ''}"
        funcs.op(u, '*', mul, out=out,
                 batchwise=batchwise, name=name, y_func='pycuda::conj' if conj else None)
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


def split_grad(ug, u_dg, config=None, copy=False, stream=None):
    param_check(u_grad=ug, u_d_grad=u_dg)
    funcs = get_funcs(ug, config, model="ssnp", stream=stream)
    ag = funcs.fft(ug, copy=copy)
    a_dg = funcs.fft(u_dg, copy=copy)
    funcs.split_grad(ag, a_dg)
    ufg = funcs.ifft(ag)
    ubg = funcs.ifft(a_dg)
    return ufg, ubg


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
        model_init = {"ssnp": SSNPFuncs, "bpm": BPMFuncs, "mlb": MLBFuncs, "any": Funcs}[name]
    except KeyError:
        raise ValueError(f"unknown model {model}") from None
    return model_init(arr_like, res, n0, stream, fft_type)
