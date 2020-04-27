from const import *
import tensorflow as tf
import numpy as np
from tifffile import TiffWriter, imread
from warnings import warn
from typing import Union


def tiff_export(path: str, images, *, scale=1, pre_operator: callable = None, dtype=np.uint16):
    """
    Export a list of Tensors to a multi-page tiff

    :param path: Target file path
    :param images: List (or other iterable) of Tensors
    :param scale: Multiplier to make value range 0~1. If not provided or is 'None',
        it will write the raw data. Please note that this is **different** with 'scale=1'
    :param pre_operator: Preprocess function for numpy data before scaling
    :param dtype: Color depth of the exported image. Must be 16bit-np.uint16(default) or 8bit-np.uint8
    """
    try:
        with TiffWriter(path) as out_file:
            for i in images:
                try:
                    i = i.numpy()
                except AttributeError:
                    if type(i) == np.ndarray:
                        warn("Export numpy array is not preferred. Use Tensor instead.")
                    else:
                        raise TypeError(f"Must export a list of 2-D Tensors but got {type(i)}")
                if len(i.shape) != 2:
                    raise ValueError(f"Must export a list of 2-D Tensors but got {len(i.shape)}-D data "
                                     f"with shape as {i.shape}.")
                if pre_operator is not None:
                    i = pre_operator(i)
                if scale is not None:
                    i *= scale * {np.uint16: 65535, np.uint8: 255}[dtype]
                i = i.astype(np.int64)
                np.clip(i, 0, {np.uint16: 65535, np.uint8: 255}[dtype], out=i)
                out_file.save(i.astype(dtype), compress=9, predictor=True)
    except KeyError:
        raise ValueError(f"dtype should be either np.uint8 or np.uint16, but not {dtype}")


def _phase_init(c_ab):
    """
    :param c_ab: (cos(alpha), cos(beta))
    :return:
    """
    norm = [SIZE[i] * RES[i] * N0 for i in (0, 1)]
    c_ab = [np.trunc(c_ab[i] * norm[i]) / norm[i] for i in (0, 1)]
    xr, yr = [np.arange(SIZE[i]) / SIZE[i] * c_ab[i] * norm[i] for i in (0, 1)]
    phase = np.mod(xr + yr[:, None], 1).astype(np.double)
    return phase, c_ab


def tiff_illumination(path: str, c_ab: tuple):
    """
    :param path: Amplitude graph path
    :param c_ab: (cos(alpha), cos(beta))
    :return: complex tf Tensor of input field
    """

    def convert_01(p: str):
        raw_img = imread(p)
        if raw_img.shape != SIZE[:2]:
            raise ValueError(f"Input image size {raw_img.shape} is incompatible"
                             "with x-y size {SIZE[:2]}")
        if raw_img.dtype.type == np.uint16:
            img = raw_img.astype(np.double) / 65535
        elif raw_img.dtype.type == np.uint8:
            warn("Importing uint8 image, please use uint16 if possible")
            img = raw_img.astype(np.double) / 255
        else:
            raise TypeError("Unknown data type of input image")
        return img

    img_in = convert_01(path)
    phase, c_ab_trunc = _phase_init(c_ab)
    phase *= 2 * np.pi
    img_in = np.exp(1j * phase) * img_in
    return tf.constant(img_in, DATA_TYPE), c_ab_trunc


def tiff_n(path: str):
    img = imread(path)
    if img.dtype.type == np.uint16:
        img = img.astype(np.double) / 65535
    elif img.dtype.type == np.uint8:
        warn("Importing uint8 image, please use uint16 if possible")
        img = img.astype(np.double) / 255
    return tf.constant(img, DATA_TYPE)
