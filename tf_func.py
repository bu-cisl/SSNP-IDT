from const import *
import tensorflow as tf
from tensorflow_core import Tensor
import numpy as np


def _c_gamma():
    c_alpha, c_beta = [
        np.fft.ifftshift(np.arange(-SIZE[i] / 2, SIZE[i] / 2).astype(np.double)) / SIZE[i] / RES[i]
        for i in (0, 1)
    ]
    c_gamma = np.sqrt(np.maximum(1 - (np.square(c_alpha) + np.square(c_beta[:, None])), EPS))
    return c_gamma


def _outflow_absorb():
    x, y = [np.minimum(np.arange(SIZE[i]), SIZE[i] - np.arange(SIZE[i])).astype(np.double) / SIZE[i]
            for i in (0, 1)]
    x, y = map(lambda a: np.power(0.5 + 0.5 * np.sin(np.minimum(a, 0.1) * 5 * np.pi), 0.1), (x, y))
    return x * y[:, None]


@tf.function
def _kz():
    kz = tf.constant(_c_gamma() * (2 * np.pi * RES[2] * N0), DATA_TYPE)
    return kz


@tf.function
def nature_d(u: Tensor):
    a = tf.signal.fft2d(u)
    a_d = _kz() * a * 1j
    u_d = tf.signal.ifft2d(a_d)
    return u_d


@tf.function
def ssnp_step(u: Tensor, u_d: Tensor, dz, n: Tensor = None):
    for t_in in (u, u_d, n):
        if t_in is not None:
            if t_in.shape != SIZE[:2]:
                raise ValueError(f"the x,y shape {t_in.shape} is not {SIZE[:2]}")

    a = tf.signal.fft2d(u)
    a_d = tf.signal.fft2d(u_d)
    evan_rm = tf.constant(np.exp(np.minimum((_c_gamma() - 0.2) * 5, 0)), DATA_TYPE)
    a *= evan_rm
    a_d *= evan_rm
    # pr = a_d[0,0] / a[0,0]
    # tf.print(tf.math.real(_kz()[0,0]))
    # tf.print(tf.math.real(pr), tf.math.imag(pr))
    u = tf.signal.ifft2d(tf.cos(_kz() * dz) * a + (tf.sin(_kz() * dz) / _kz()) * a_d)
    u_d = tf.signal.ifft2d(-(tf.sin(_kz() * dz) * _kz()) * a + tf.cos(_kz() * dz) * a_d)
    # n = n / N0
    if n is not None:
        u_d -= (4 * np.pi * dz * RES[2] ** 2) * (n * (2 * N0 + n) * u)
    absorb = tf.constant(_outflow_absorb(), DATA_TYPE)
    u *= absorb
    u_d *= absorb
    return u, u_d


@tf.function
def n_ball(n, r, *, z_empty: tuple) -> Tensor:
    """
    :param n: refractive index
    :param r: ball radius (times of wavelength)
    :param z_empty: empty space before and after the ball
    :return: tf constant of the 3d refractive index field
    """
    assert type(z_empty) == tuple and len(z_empty) == 2
    arr = tf.square(tf.cast(tf.range(-SIZE[1] / 2, SIZE[1] / 2), tf.float64) * RES[1])[None, None, :]
    arr += tf.square(tf.cast(tf.range(-SIZE[0] / 2, SIZE[0] / 2), tf.float64) * RES[0])[None, :, None]
    arr += tf.square(tf.cast(tf.range(-z_empty[0] - r / RES[2], z_empty[1] + r / RES[2]), tf.float64) * RES[2])[
           :, None, None]
    # out = tf.zeros_like(arr, dtype=DATA_TYPE)
    # n /= 4
    # for i in range(4):
    #     out += tf.cast(arr < r * r - 0.96 + 0.64 * i, dtype=tf.complex64) * n
    out = tf.cast(tf.minimum(tf.exp(-arr / (r * r) * 9.21034), 0.0001) * 10000 * n, DATA_TYPE)
    return out
