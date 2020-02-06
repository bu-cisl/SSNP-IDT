from const import *
import tensorflow as tf
import numpy as np
from tifffile import TiffWriter


def tiff_write(path: str, img, *, scale=None, pre_operator: callable = None):
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
