import tensorflow as tf
import numpy as np
from warnings import warn

N0 = 1.
# RES = (0.315534, 0.315534, 0.236686)
RES = (0.1, 0.1, 0.1)


def real_to_complex(real):
    if not real.dtype.is_floating:
        warn("Input is not a real floating number. Casting to double", DeprecationWarning)
        real = real.cast(tf.float64)
    return tf.complex(real, tf.zeros_like(real))


def tilt(img, c_ab, *, trunc=False):
    """
    Tilt an image as illumination

    :param img: Amplitude graph
    :param c_ab: (cos(alpha), cos(beta))
    :param trunc: whether trunc to a grid point in Fourier plane
    :return: complex tf Tensor of input field
    """
    if not img.dtype.is_complex:
        img = real_to_complex(img)
    size = img.shape[::-1]
    if len(size) != 2:
        raise ValueError(f"Illumination should be a 2-D tensor rather than shape '{img.shape}'.")
    norm = [size[i] * RES[i] * N0 for i in (0, 1)]
    if trunc:
        c_ab = [np.trunc(c_ab[i] * norm[i]) / norm[i] for i in (0, 1)]
    xr, yr = [np.arange(size[i]) / size[i] * c_ab[i] * norm[i] for i in (0, 1)]
    phase = np.mod(xr + yr[:, None], 1).astype(np.double) * 2 * np.pi
    phase = tf.constant(phase, img.dtype)
    img = tf.exp(1j * phase) * img
    return img, c_ab
