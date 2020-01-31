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
    n /= 4
    arr = tf.square(tf.cast(tf.range(-SIZE[1] / 2, SIZE[1] / 2), tf.float32) * RES[1])[None, None, :]
    arr += tf.square(tf.cast(tf.range(-SIZE[0] / 2, SIZE[0] / 2), tf.float32) * RES[0])[None, :, None]
    arr += tf.square(tf.cast(tf.range(-z_empty[0] - r, z_empty[1] + r), tf.float32) * RES[2])[:, None, None]
    out = tf.zeros_like(arr, dtype=DATA_TYPE)
    for i in range(4):
        out += tf.cast(arr < r - 0.48 + 0.32 * i, dtype=tf.complex64) * n
    return out
