from const import *
import tensorflow as tf
import numpy as np
from tifffile import TiffWriter, TiffFile
from warnings import warn
import os
import csv

DEFAULT_TYPE = tf.float32

def _phase_init(c_ab, trunc=False):
    """
    :param c_ab: (cos(alpha), cos(beta))
    :return:
    """
    norm = [SIZE[i] * RES[i] * N0 for i in (0, 1)]
    if trunc:
        c_ab = [np.trunc(c_ab[i] * norm[i]) / norm[i] for i in (0, 1)]
    xr, yr = [np.arange(SIZE[i]) / SIZE[i] * c_ab[i] * norm[i] for i in (0, 1)]
    phase = np.mod(xr + yr[:, None], 1).astype(np.double)
    return 2 * np.pi * phase, c_ab


def tiff_illumination(path: str, c_ab: tuple):
    """
    :param path: Amplitude graph path
    :param c_ab: (cos(alpha), cos(beta))
    :return: complex tf Tensor of input field
    """

    def convert_01(p: str):
        with TiffFile(p) as file:
            raw_img = np.squeeze(np.stack([page.asarray() for page in file.pages]))
        if raw_img.shape != SIZE[:2]:
            raise ValueError(f"Input image size {raw_img.shape} is incompatible"
                             f"with x-y size {SIZE[:2]}")
        if raw_img.dtype.type == np.uint16:
            img = raw_img.astype(np.double) / 65535
        elif raw_img.dtype.type == np.uint8:
            warn("Importing uint8 image, please use uint16 if possible")
            img = raw_img.astype(np.double) / 255
        else:
            raise TypeError(f"Unknown data type {raw_img.dtype.type} of input image")
        return img

    if path == "plane":
        img_in = np.ones(SIZE[:2], np.double)
    elif os.access(path, os.R_OK):
        img_in = convert_01(path)
    else:
        raise FileNotFoundError(f"'{path}' is not a known illumination type nor a valid file path")
    phase, c_ab_final = _phase_init(c_ab, trunc=PERIODIC_BOUNDARY)
    img_in = np.exp(1j * phase) * img_in
    return tf.constant(img_in, DATA_TYPE), c_ab_final


def csv_import(path: str):
    table = []
    with open(path, 'r', newline='') as file:
        reader = csv.reader(file, quoting=csv.QUOTE_NONNUMERIC)
        for row in reader:
            table.append(row)
    return table


def csv_export(path: str, table):
    with open(path, 'w', newline='') as file:
        writer = csv.writer(file)
        for row in table:
            writer.writerow(row)

np.ndarray
class Image:
    def __init__(self, path, dtype=DEFAULT_TYPE, shape=None):

        self.tensor = tf.constant(img, dtype, shape)


        self.rank = tf.rank(self.tensor)
        if self.rank not in {2, 3}:
            warn(f"Importing {self.rank}-D image with shape {self.shape}")

    def write(self, path, *, scale=1, pre_operator: callable = None, dtype=np.uint16):
        """
            Export a list of Tensors to a multi-page tiff

            :param path: Target file path
            :param scale: Multiplier to make value range 0~1. If not provided or is 'None',
                it will write the raw data. Please note that this is **different** with 'scale=1'
            :param pre_operator: Preprocess function for numpy data before scaling
            :param dtype: Color depth of the exported image. Must be 16bit-np.uint16(default) or 8bit-np.uint8
            """
        try:
            with TiffWriter(path) as out_file:
                for i in self.tensor:
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

    def __getattr__(self, item):
        return self.tensor.item

class Field

    def (self, path_type, dtype=DEFAULT_TYPE, ):
        if path_type == "plane":
            img_in = np.ones(SIZE[:2], np.double)
            self.periodic = True
        elif os.access(path_type, os.R_OK):
            super().__init__(path_type, dtype)
        else:
            raise FileNotFoundError(f"'{path_type}' is not a known illumination type nor a valid file path")

        with TiffFile(path) as file:
            img = np.squeeze(np.stack([page.asarray() for page in file.pages]))
        if img.dtype.type == np.uint16:
            img = img.astype(np.double) / 65535
        elif img.dtype.type == np.uint8:
            warn("Importing uint8 image, please use uint16 if possible")
            img = img.astype(np.double) / 255
        else:
            raise TypeError(f"Unknown data type {img.dtype.type} of input image")