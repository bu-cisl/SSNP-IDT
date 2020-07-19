import math

import numpy as np
from pycuda import gpuarray
import functools
from warnings import warn


def param_check(**kwargs):
    name0 = None
    shape0 = None
    for name in kwargs:
        arr = kwargs[name]
        if arr is None:
            continue
        # type check
        assert isinstance(arr, gpuarray.GPUArray)
        # shape check
        if name0 is None:
            name0 = name
            shape0 = arr.shape
        else:
            if arr.shape != shape0:
                raise ValueError(f"cannot match '{name}' shape {arr.shape} with '{name0}' shape {shape0}")


def _cache_array(func):
    @functools.wraps(func)
    def get_cache(self, *args, gpu=False, **kwargs):
        key, calc = func(self, *args, **kwargs)
        if gpu:
            try:
                return self._gpu_cache[key]
            except KeyError:
                arr = gpuarray.to_gpu(get_cache(self, *args, gpu=False, **kwargs))
                self._gpu_cache[key] = arr
                return arr
        try:
            return self._cache[key]
        except KeyError:
            arr = calc()
            self._cache[key] = arr
            return arr

    return get_cache


class Multipliers:
    def __init__(self, shape, res):
        self.shape = shape
        self.res = res
        self._cache = {}
        self._gpu_cache = {}

    @_cache_array
    def tilt(self, c_ab, *, trunc):
        res = self.res
        shape = self.shape
        norm = tuple(shape[i] * res[i] for i in (0, 1))  # to be confirmed: * config.n0
        if trunc:
            c_ab = [math.trunc(c_ab[i] * norm[i]) / norm[i] for i in (0, 1)]
        c_ab = tuple(float(i) for i in c_ab)
        key = ("t", c_ab, res)

        def calc():
            xr, yr = [np.arange(shape[i]) / shape[i] * c_ab[i] * norm[i] for i in (0, 1)]
            phase = np.mod(xr + yr[:, None], 1).astype(np.double) * 2 * np.pi
            phase = gpuarray.to_gpu(np.exp(1j * phase))
            return phase

        return key, calc

    @_cache_array
    def binary_pupil(self, na):
        res = self.res
        c_gamma = self.c_gamma()
        key = ("bp", round(na * 1000), res)

        def calc():
            mask = np.greater(c_gamma, np.sqrt(1 - na ** 2))
            mask = mask.astype(np.complex128)
            mask = gpuarray.to_gpu(mask)
            return mask

        return key, calc

    @_cache_array
    def c_gamma(self):
        """
        Calculate cos(gamma) as a constant array at frequency domain. Gamma is the angle
        between the wave vector and z-axis. Note: N.A.=sin(gamma)

        The array is pre-shifted for later FFT operation.

        :return: cos(gamma) numpy array
        """
        shape = self.shape
        res = self.res
        key = ("cg", res)

        def calc():
            eps = 1E-8
            c_alpha, c_beta = [
                np.fft.ifftshift(np.arange(-shape[i] / 2, shape[i] / 2).astype(np.double)) / shape[i] / res[i]
                for i in (0, 1)
            ]
            c_gamma = np.sqrt(np.maximum(1 - (np.square(c_alpha) + np.square(c_beta[:, None])), eps))
            return c_gamma

        return key, calc

    @_cache_array
    def soft_crop(self, width, *, total_slices=1, pos=0, strength=1):
        shape = self.shape
        if width >= 1 or width <= 0:
            raise ValueError("width should be a relative value in 0-1")
        key = ("cr", round(width * 100), round(pos * 100), round(total_slices), round(strength * 100))
        if strength < 0.1:
            warn("strength is too weak, change to 0.1", stacklevel=3)
            strength = 0.1

        def calc():
            x, y = [
                np.exp(
                    -np.exp(
                        -(((np.mod((np.arange(shape[i]).astype(np.double) + 0.5) / shape[i] + 0.5 - pos, 1)
                            * 2 - 1) / width) ** 2
                          ) * (np.log(10 * strength) + 0.8))
                    * 10 * strength / total_slices
                )
                for i in (0, 1)
            ]
            # x, y = np.exp(-mask * 10 * strength / total_slices)
            mask = x[:, None] * y
            return mask

        return key, calc
