from const import *
import tensorflow as tf
from tensorflow_core import Tensor
import numpy as np


@tf.function
def nature_d(u: Tensor):
    c_alpha, c_beta = [
        tf.signal.ifftshift(tf.cast(tf.range(-SIZE[i] / 2, SIZE[i] / 2), DATA_TYPE)) / SIZE[i] / RES[i]
        for i in (0, 1)
    ]

    c_gamma = tf.sqrt(1 - (tf.square(c_alpha) + tf.square(c_beta[:, None])))
    c_gamma = tf.maximum(tf.math.real(c_gamma), EPS)
    c_gamma = tf.cast(c_gamma, dtype=DATA_TYPE)
    kz = c_gamma * (2 * np.pi * RES[2] * N0 * 1j)

    a = tf.signal.fft2d(u)
    a_d = -kz * a
    u_d = tf.signal.ifft2d(a_d)
    return u_d


@tf.function
def step(u: Tensor, u_d: Tensor, n: Tensor, dz):
    for tensor_in in (u, u_d, n):
        print(tensor_in.shape)
        assert tensor_in.shape == SIZE[:2]

    c_alpha, c_beta = [
        tf.signal.ifftshift(tf.cast(tf.range(-SIZE[i] / 2, SIZE[i] / 2), DATA_TYPE)) / SIZE[i] / RES[i]
        for i in (0, 1)
    ]

    c_gamma = tf.sqrt(1 - (tf.square(c_alpha) + tf.square(c_beta[:, None])))
    c_gamma = tf.maximum(tf.math.real(c_gamma), EPS)
    c_gamma = tf.cast(c_gamma, dtype=DATA_TYPE)
    kz = c_gamma * (2 * np.pi * RES[2] * N0)

    a = tf.signal.fft2d(u)
    a_d = tf.signal.fft2d(u_d)
    u = tf.signal.ifft2d(tf.cos(kz * dz) * a + (tf.sin(kz * dz) / kz) * a_d)
    u_d = tf.signal.ifft2d(-(tf.sin(kz * dz) * kz) * a + tf.cos(kz * dz) * a_d)
    # n = n / N0
    u_d -= (4 * np.pi * dz * RES[2] ** 2) * (n * (2 * N0 + n) * u)
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
    arr = tf.square(tf.cast(tf.range(-SIZE[1] / 2, SIZE[1] / 2), tf.float32) * RES[1])[None, None, :]
    arr += tf.square(tf.cast(tf.range(-SIZE[0] / 2, SIZE[0] / 2), tf.float32) * RES[0])[None, :, None]
    arr += tf.square(tf.cast(tf.range(-z_empty[0] - r / RES[2], z_empty[1] + r / RES[2]), tf.float32) * RES[2])[
           :, None, None]
    # out = tf.zeros_like(arr, dtype=DATA_TYPE)
    # n /= 4
    # for i in range(4):
    #     out += tf.cast(arr < r * r - 0.96 + 0.64 * i, dtype=tf.complex64) * n
    out = tf.cast(tf.minimum(tf.exp(-arr / (r * r) * 9.21034), 0.0001) * 10000 * n, tf.complex64)
    return out
