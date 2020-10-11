import numpy as np
import pycuda.cumath
import pycuda.reduction
import pycuda.compiler
import pycuda.elementwise
import pycuda.reduction
from pycuda.gpuarray import GPUArray
from functools import lru_cache

import ssnp
from ssnp.calc import get_funcs

try:
    ssnp.config.res
except AttributeError:
    ssnp.config.res = (1, 1, 1)

discontig_sub_kernel = pycuda.elementwise.ElementwiseKernel(
    # mem range safe, since i>=step, i-step>=0
    "double *arr, double *out, int step, int length, int transpose",
    """
    if (i % length >= step) {
        if (transpose)
            out[i-step] -= arr[i];
        else
            out[i] = arr[i] - arr[i-step];
    }
    else if (!transpose)
        out[i] = 0;
    """)


def discontig_sub(arr, out, *, axis, transpose=False):
    shape = np.array(arr.shape)
    step = np.prod(shape[axis + 1:])
    length = step * shape[axis]
    discontig_sub_kernel(arr, out, step, length, 1 if transpose else 0)
    return out


@lru_cache
def freq_response_dtd(shape, dtype):
    wx = np.linspace(0, 2 * np.pi, shape[1], endpoint=False)
    wy = np.linspace(0, 2 * np.pi, shape[0], endpoint=False)
    wy = wy[:, None]
    out = 4 - 2 * np.cos(wx) - 2 * np.cos(wy)
    out = out.astype(dtype)
    out = pycuda.gpuarray.to_gpu(out)
    assert out.shape == shape and out.dtype == dtype
    return out


def mult(arr: GPUArray, transpose=False) -> GPUArray:
    if transpose:
        out = arr[0] + arr[1]
        discontig_sub(arr[0], out, axis=0, transpose=True)
        discontig_sub(arr[1], out, axis=1, transpose=True)
    else:
        shape = (2, *arr.shape)
        out = GPUArray(shape, arr.dtype, arr.allocator)
        discontig_sub(arr, out[0], axis=0)
        discontig_sub(arr, out[1], axis=1)
    return out


norm = pycuda.reduction.ReductionKernel(
    dtype_out=np.double, neutral=0,
    reduce_expr="a+b",
    map_expr="x[i] * x[i]",
    arguments="double *x",
)


def tv_cost(x):
    z = mult(x)
    z[0] *= z[0]
    z[1] *= z[1]
    z[0] += z[1]
    pycuda.cumath.sqrt(z[0], out=z[0])
    return pycuda.gpuarray.sum(z[0]).get()


def prox_tv(y: GPUArray, lam):
    funcs = get_funcs(y.astype(np.complex128))
    fft, ifft = funcs.fft, funcs.ifft

    def shrink3d_real(yy, tau, _):
        out = yy.copy()
        norm_y = out[0]
        temp = out[1]
        out[0] *= out[0]
        out[1] *= out[1]
        norm_y += out[1]

        pycuda.cumath.sqrt(norm_y, out=norm_y)
        norm_y += norm_y <= 0  # extra arr
        temp = pycuda.gpuarray.maximum(norm_y - tau, 0, out=temp)
        temp /= norm_y
        out[0].set(out[1])
        out[0] *= yy[0]
        out[1] *= yy[1]
        return out

    num_iter = 20
    tol = 1e-6
    verbose = False
    rho = 1
    xhat0 = y.copy()
    xhat = xhat0.copy()
    d = mult(xhat)
    xhat_mult = d.copy()
    s = pycuda.gpuarray.zeros_like(d)
    i = 0
    for i in range(num_iter):
        if verbose:
            print(f"[proxTV: {i + 1}/{num_iter}]")
        xhat_prev = xhat.copy()
        s /= rho
        xhat_mult -= s
        d = shrink3d_real(xhat_mult, lam / rho, 0)
        dat = y + rho * mult(d + s, transpose=True)
        dat = fft(dat.astype(np.complex128))
        dat /= freq_response_dtd(dat.shape, dat.dtype) * rho + 1
        xhat = ifft(dat).real
        xhat_mult = mult(xhat)
        s = s + rho * (d - xhat_mult)
        if norm(xhat - xhat_prev).get() / norm(xhat_prev).get() < tol:
            break
    if i != num_iter - 1:
        print(f"actual prox iterations: {i}")
    return xhat


@lru_cache
def get_q(t: int):
    from math import sqrt
    assert isinstance(t, int)
    if t <= 0:
        return 1.
    else:
        return (1 + sqrt(1 + 4 * get_q(t - 1) ** 2)) / 2


if __name__ == '__main__':
    n0 = np.array([[1, 2, 2], [3, 4, 6], [9, 8, 2]], dtype=np.double) / 100
    n0 = pycuda.gpuarray.to_gpu(n0)
    z0 = prox_tv(n0, 0.00000)
    print(z0)
