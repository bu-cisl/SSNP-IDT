from const import *
import numpy as np
from time import time
import tensorflow as tf
from tensorflow_core import Tensor

from typing import Tuple, Any, Union


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
    # n = n / N0
    # u_d -= (4 * np.pi * dz * RES[2] ** 2) * (tf.multiply(n, 2 * N0) + tf.square(n))
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
    # u_d -= (4 * np.pi * dz * RES[2] ** 2) * (tf.multiply(n, 2 * N0) + tf.square(n)) * u
    return u, u_d


def main():
    import tifffile
    # u: Tensor = tf.constant(np.random.rand(1024, 1024), dtype=tf.complex64)
    # u_d = tf.constant(np.zeros((1024, 1024)), dtype=tf.complex64)
    # n = tf.constant(np.zeros((26, 512, 512)), dtype=tf.complex64)
    n = [[[0.05 if i * i + j * j + k * k < 100 else 0
           for i in range(-256, 256)] for j in range(-256, 256)] for k in range(-20, 80)]
    # n = np.array(n)
    n = tf.constant(n, dtype=tf.complex64)
    img_in = tifffile.imread("a.tiff")
    img_in = tf.constant(img_in, dtype=tf.complex64)
    # img_in = tf.constant(np.ones((512, 512)), dtype=tf.complex64)
    u_d = nature_d(img_in)
    ooo = [img_in]
    t = time()
    for n_zi in n:
        img_in, u_d = step(img_in, u_d, n_zi, 1)
        ooo.append(img_in)
    print(time() - t)

    ooo = tf.cast(tf.abs(tf.stack(ooo)[..., None]) * 0.7, tf.uint64).numpy()
    ooo[ooo > 65535] = 65535
    # ooo = tf.stack(ooo)[..., None].numpy()
    # for i in ooo:
    #     print(np.sum(i))
    tifffile.imsave("c.tiff", ooo.astype(np.uint16))
    return ooo


if __name__ == '__main__':
    img = main()
