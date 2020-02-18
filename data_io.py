from typing import Union

from const import *
import tensorflow as tf
import numpy as np
from tifffile import TiffWriter, imread
from warnings import warn


def tiff_export(path: str, img, *, scale=None, pre_operator: callable = None):
    with TiffWriter(path) as out_file:
        for i in img:
            i = i.numpy()
            if pre_operator is not None:
                i = pre_operator(i)
            if scale is not None:
                i *= scale * 65535
            i = i.astype(np.int64)
            np.clip(i, 0, 65535, out=i)
            i = i[...].astype(np.uint16)
            out_file.save(i, compress=9, predictor=True)


def _phase_init(xy_theta, gamma):
    if gamma > EPS:
        lz = 1 / np.sin(gamma)
        xr, yr = [(np.cos(xy_theta), np.sin(xy_theta))[i] * np.arange(SIZE[i]) * RES[i]
                  for i in (0, 1)]
        phase = (np.mod(xr - yr[:, None], lz) / lz).astype(np.double)
    else:
        phase = np.zeros(SIZE[:2], np.double)
    return phase


def tiff_import(path: str, phase_info: Union[str, tuple]):
    """
    :param path: Amplitude graph path
    :param phase_info: Phase graph path, or (xy_theta, gamma)
    :return: complex tf Tensor of input field
    """
    def convert_01(p: str):
        img = imread(p)
        if img.shape != SIZE[:2]:
            raise ValueError("Input image size incompatible!")
        if img.dtype.type == np.uint16:
            img = img.astype(np.double) / 65535
        elif img.dtype.type == np.uint8:
            warn("Importing uint8 image, please use uint16 if possible")
            img = img.astype(np.double) / 255
        else:
            raise TypeError("Unknown data type of input image")
        return img

    img_in = convert_01(path)
    if type(phase_info) == str:
        phase = convert_01(phase_info)
    else:
        phase = _phase_init(*phase_info)
    assert phase.dtype.type == img_in.dtype.type == np.double
    phase *= 2 * np.pi
    img_in = np.cos(phase) * img_in + 1j * np.sin(phase) * img_in
    return tf.constant(img_in, DATA_TYPE)
