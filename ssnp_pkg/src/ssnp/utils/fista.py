from collections.abc import Sequence
import numpy as np
import pycuda.cumath
import pycuda.reduction
import pycuda.compiler
import pycuda.elementwise
import pycuda.reduction
from pycuda.gpuarray import GPUArray
from functools import lru_cache

discontig_sub_kernel = pycuda.elementwise.ElementwiseKernel(
    # mem range safe, since i>=step, i-step>=0
    "double *arr, double *out, unsigned step, unsigned length, unsigned transpose",
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


TVNorm_kernel = pycuda.elementwise.ElementwiseKernel(
    "double *out, double *tv",
    """
    out[i] = tv[i]*tv[i];
    out[i] += tv[i+n]*tv[i+n];
    out[i] += tv[i+2*n]*tv[i+2*n];
    out[i] = sqrt(out[i]);
    if (out[i] < 1.)
        out[i] = 1.;
    """
)


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


def prox_fgp(z, tv_parameter=1.0, *, num_iter=10, out=None):
    q_t1 = 1.0
    shape = (3, *z.shape)
    # g_t = GPUArray(shape, z.dtype, z.allocator).fill(0)
    # g_t1, d_t: xyz 3 channel of 3D
    # z, out/proj_x, tmp_out: 1 channel 3D
    g_t1 = GPUArray(shape, z.dtype, z.allocator).fill(0)
    d_t = GPUArray(shape, z.dtype, z.allocator).fill(0)

    tmp_out = GPUArray(z.shape, z.dtype, z.allocator)
    proj_x = GPUArray(z.shape, z.dtype, z.allocator) if out is None else out
    for iteration in range(num_iter):
        tmp_out.fill(0)
        for i in range(3):
            discontig_sub(d_t[i], tmp_out, axis=i, transpose=True)
        tmp_out *= -tv_parameter
        tmp_out += z
        pycuda.gpuarray.maximum(tmp_out, 0, out=proj_x)
        for i in range(3):
            discontig_sub(proj_x, tmp_out, axis=i)
            tmp_out *= 1 / (12 * tv_parameter)
            d_t[i] += tmp_out
        TVNorm_kernel(tmp_out, d_t)
        for i in range(3):
            d_t[i] /= tmp_out
        # finish 1st line of FGP, now d_t is g_t

        q_t = 0.5 * (1.0 + (1.0 + 4.0 * q_t1 ** 2) ** 0.5)
        beta = (q_t1 - 1.0) / q_t
        for i in range(3):
            # save g_t value
            tmp_out.set(d_t[i])
            # modify g_t in place and store in d_t
            d_t[i]._axpbyz(beta + 1, g_t1[i], -beta, out=d_t[i])
            # g_t -> g_t1, prepare for next iter
            g_t1[i].set(tmp_out)
    tmp_out.fill(0)
    for i in range(3):
        discontig_sub(g_t1[i], tmp_out, axis=i, transpose=True)
    tmp_out *= -tv_parameter
    tmp_out += z
    pycuda.gpuarray.maximum(tmp_out, 0, out=proj_x)
    return proj_x


def prox_tv_Michael(x, tv_parameter=1.0, out=None):
    t_k = 1.0
    num_iter = 20
    if not isinstance(tv_parameter, Sequence):
        tv_parameter = (tv_parameter,) * 3
    shape = (3, *x.shape)
    u_k = GPUArray(shape, x.dtype, x.allocator).fill(0)
    u_k1 = GPUArray(shape, x.dtype, x.allocator).fill(0)

    tmp_out = GPUArray(x.shape, x.dtype, x.allocator)
    grad_u_hat = GPUArray(x.shape, x.dtype, x.allocator) if out is None else out
    grad_u_hat.set(x)

    for iteration in range(num_iter):
        # grad_u_hat: GPUArray = projector(grad_u_hat)
        for i in range(3):
            discontig_sub(grad_u_hat, tmp_out, axis=i)
            tmp_out *= 1 / 12
            u_k1[i] += tmp_out
            u_k1[i] *= 1 / tv_parameter[i]
        # Previously like this (0,1,2)
        # u_k1[:, :, :, 1] = self._indexLastAxis(u_k1, 1) + (
        #         1.0 / 12 / self.parameter) * self._filterD(grad_u_hat, axis=1)

        TVNorm_kernel(tmp_out, u_k1)
        for i in range(3):
            u_k1[i] /= tmp_out

        t_k1 = 0.5 * (1.0 + (1.0 + 4.0 * t_k ** 2) ** 0.5)
        beta = (t_k - 1.0) / t_k1

        for i in range(3):
            tmp_out.set(u_k[i])
            if iteration < num_iter - 1:
                u_k[i].set(u_k1[i])
            u_k1[i] *= 1. + beta
            tmp_out *= beta
            u_k1[i] -= tmp_out  # now u_hat

        # previous code: 2 of (0,1,2)
        # temp = u_k[:, :, :, 2].copy()
        # if iteration < self.maxitr - 1:
        #     u_k[:, :, :, 2] = u_k1[:, :, :, 2]
        # u_k1[:, :, :, 2] = (1.0 + beta) * u_k1[:, :, :, 2] - beta * temp

        grad_u_hat.set(x)
        for i in range(3):
            u_k1[i] *= tv_parameter[i]
        u_k1[0]._axpbyz(1, u_k1[1], 1, tmp_out)  # tmp_out = u_k1[0] + u_k1[1]
        tmp_out += u_k1[2]
        for i in range(3):
            discontig_sub(u_k1[i], tmp_out, axis=i, transpose=True)
        # tmp_out *= tv_parameter
        grad_u_hat -= tmp_out
        # previous code: (at beginning, not at 0, one more at end)
        # grad_u_hat = x - tv_parameter * self._filterDT(u_k1)

    # grad_u_hat = projector(grad_u_hat)
    return grad_u_hat


def prox_tv(y: GPUArray, lam):
    from ssnp.calc import get_funcs
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
