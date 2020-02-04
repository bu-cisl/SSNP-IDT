from const import *
import tensorflow as tf
from tensorflow_core import Tensor


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
