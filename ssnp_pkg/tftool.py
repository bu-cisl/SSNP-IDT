import tensorflow as tf
from warnings import warn


def real_to_complex(real: tf.Tensor):
    if not real.dtype.is_floating:
        warn("Input is not a real floating number. Casting to double", DeprecationWarning)
        real = real.cast(tf.float64)
    return tf.complex(real, tf.zeros_like(real))
