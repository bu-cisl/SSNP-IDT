import numpy as np
from pycuda import gpuarray
from .calc import _res_deprecated, _N0_deprecated


def tilt(img, c_ab, *, trunc=False, copy=False):
    """
    Tilt an image as illumination

    :param copy:
    :param img: Amplitude graph
    :param c_ab: (cos(alpha), cos(beta))
    :param trunc: whether trunc to a grid point in Fourier plane
    :return: complex tf Tensor of input field
    """
    size = img.shape[::-1]
    if len(size) != 2:
        raise ValueError(f"Illumination should be a 2-D tensor rather than shape '{img.shape}'.")
    norm = [size[i] * _res_deprecated[i] * _N0_deprecated for i in (0, 1)]
    if trunc:
        c_ab = [np.trunc(c_ab[i] * norm[i]) / norm[i] for i in (0, 1)]
    xr, yr = [np.arange(size[i]) / size[i] * c_ab[i] * norm[i] for i in (0, 1)]
    phase = np.mod(xr + yr[:, None], 1).astype(np.double) * 2 * np.pi
    phase = gpuarray.to_gpu(np.exp(1j * phase))
    if copy:
        img = img * phase
    else:
        img *= phase
    return img, c_ab
