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
