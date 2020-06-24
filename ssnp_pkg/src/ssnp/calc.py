import tensorflow as tf
import numpy as np
from .const import EPS, COMPLEX_TYPE

_res_deprecated = (0.1, 0.1, 0.1)
_N0_deprecated = 1


@tf.function
def ssnp_step(u, u_d, dz, n=None):
    """
    SSNP main operation of one step

    :param u: x-y complex amplitude
    :param u_d: z partial derivative of u
    :param dz: step size along z axis
    :param n: refractive index along x-y distribution in this slice. Use background N0 if not provided.
    :return: new (u, u_d) after a step towards +z direction
    """
    shape = u.shape
    for t_in in (u_d, n):
        if t_in is not None:
            if t_in.shape != shape:
                raise ValueError(f"the x,y shape {t_in.shape} is not {shape}")

    a = tf.signal.fft2d(u)
    a_d = tf.signal.fft2d(u_d)
    evan_rm = tf.constant(_evanescent_absorb(shape, _res_deprecated), COMPLEX_TYPE)
    a *= evan_rm
    a_d *= evan_rm
    u = tf.signal.ifft2d(tf.cos(_kz() * dz) * a + (tf.sin(_kz() * dz) / _kz()) * a_d, name='U')
    u_d = tf.signal.ifft2d(-(tf.sin(_kz() * dz) * _kz()) * a + tf.cos(_kz() * dz) * a_d, name='Ud')
    # n = n / N0
    if n is not None:
        u_d -= ((2 * np.pi * _res_deprecated[2]) ** 2 * dz) * (n * (2 * _N0_deprecated + n) * u)
    # if not PERIODIC_BOUNDARY:
    #     absorb = tf.constant(_outflow_absorb(), DATA_TYPE)
    #     u *= absorb
    #     u_d *= absorb
    return u, u_d


def _c_gamma(shape, res):
    """
    Calculate cos(gamma) as a constant array at frequency domain. Gamma is the angle
    between the wave vector and z-axis. Note: N.A.=sin(gamma)

    The array is pre-shifted for later FFT operation.

    :return: cos(gamma) numpy array
    """
    c_alpha, c_beta = [
        np.fft.ifftshift(np.arange(-shape[i] / 2, shape[i] / 2).astype(np.double)) / shape[i] / res[i]
        for i in (0, 1)
    ]
    c_gamma = np.sqrt(np.maximum(1 - (np.square(c_alpha) + np.square(c_beta[:, None])), EPS))
    return c_gamma


def _kz(shape, res):
    kz = tf.constant(_c_gamma(shape, res) * (2 * np.pi * res[2] * _N0_deprecated), COMPLEX_TYPE)
    return kz


def _evanescent_absorb(shape, res) -> np.ndarray:
    """
        For 0.98<N.A.<1, decrease to 1~1/e per step

        For N.A.>1 decrease to 1/e per step

        :return: a 2D angular spectrum transmittance numpy array
    """
    print("retracing eva")
    return np.exp(np.minimum((_c_gamma(shape, res) - 0.2) * 5, 0))
