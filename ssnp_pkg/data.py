import tensorflow as tf
import numpy as np
from tifffile import TiffWriter, TiffFile
from warnings import warn
import os
import csv

DEFAULT_TYPE = tf.float32


# if raw_img.shape != SIZE[:2]:
#     raise ValueError(f"Input image size {raw_img.shape} is incompatible"
#                      f"with x-y size {SIZE[:2]}")

def predefined_read(name, shape, dtype=DEFAULT_TYPE):
    """
    Get a predefined tensor
    ``shape`` can be a ``list`` of integers, a ``tuple`` of integers,
    or a 1-D ``Tensor`` of type ``int32``

    :param name: predefined image name
    :param shape: Dimensions of resulting tensor
    :param dtype: DType of the elements of the resulting tensor
    """
    if shape is None:
        raise ValueError("indeterminate shape")
    if name == "plane":
        img = tf.ones(shape, dtype)
    else:
        raise ValueError(f"unknown image name {name}")
    return img


def tiff_read(path, dtype=DEFAULT_TYPE, shape=None):
    """
    Import a TIFF file to ``tf.Tensor``

    :param path: Target file path
    :param dtype: Data type of the elements of the resulting tensor
    :param shape: Dimensions of resulting tensor
    :return: A tensor constant
    """
    with TiffFile(path) as file:
        img = np.squeeze(np.stack([page.asarray() for page in file.pages]))
    if img.dtype.type == np.uint16:
        img = img.astype(np.double) / 65535
    elif img.dtype.type == np.uint8:
        warn("Importing uint8 image, please use uint16 if possible")
        img = img.astype(np.double) / 255
    else:
        raise TypeError(f"Unknown data type {img.dtype.type} of input image")
    img = tf.constant(img, dtype, shape)
    return img


def np_read(path, dtype=DEFAULT_TYPE, shape=None, *, key=None):
    ext = os.path.splitext(path)[-1]
    if key is None:
        key = 'arr_0'
    if ext == '.npy':
        img = np.load(path)
    elif ext == '.npz':
        with np.load(path) as f:
            try:
                img = f[key]
            except KeyError:
                warn(f"'{key}' is not a file in archive {path}. Reading first file instead.", stacklevel=2)
                img = f[f.files[0]]
    else:
        raise ValueError(f"unknown filename extension '{ext}'")
    return tf.constant(img, dtype, shape)


def csv_read(path: str, dtype=DEFAULT_TYPE, shape=None):
    table = []
    with open(path, 'r', newline='') as file:
        reader = csv.reader(file, quoting=csv.QUOTE_NONNUMERIC)
        for row in reader:
            table.append(row)
    return tf.constant(table, dtype, shape)


def read(source: str, dtype=DEFAULT_TYPE, shape=None, **kwargs):
    """
    Read a ``tf.Tensor`` from source. Method is auto-detected corresponding to the file extension.

    :param source:
    :param dtype:
    :param shape:
    :return:
    """
    if source in {'plane'}:
        img = predefined_read(source, shape, dtype)
    elif os.access(source, os.R_OK):
        ext = os.path.splitext(source)[-1]
        if ext in {'.tiff', '.tif'}:
            img = tiff_read(source, dtype, shape)
        elif ext in {'.npy', '.npz'}:
            img = np_read(source, dtype, shape, **kwargs)
        elif ext == '.csv':
            img = csv_read(source, dtype, shape)
        else:
            raise ValueError(f"unknown filename extension '{ext}'")
    else:
        raise FileNotFoundError(f"'{source}' is not a known illumination type nor a valid file path")
    rank = len(img.shape)
    if rank not in {2, 3}:
        warn(f"Importing {rank}-D image with shape {img.shape}", stacklevel=2)
    return img


def tiff_write(path, tensor, *, scale=1, pre_operator: callable = None, dtype=np.uint16, compress: bool = True):
    """
    Export a list of Tensors to a multi-page tiff

    ``pre_operator`` can apply some numpy functions before the data to be exported, such as brightness
    and contrast adjustment. Normally it is used to avoid saturation or too dark pictures

    Argument ``scale`` is a simpler way than `pre_operator` to adjust data range.
    If ``scale`` is ``None``, it will write the raw data. Please note that this is
    **different** with ``scale=1``

    Argument ``dtype`` means the color depth of the exported image. It must be 16bit - ``np.uint16``
    (default) or 8bit - ``np.uint8``

    :param path: Target file path
    :param tensor: Tensor data to be written
    :param scale: Multiplier to adjust value range
    :param pre_operator: Preprocess function before scaling
    :param dtype: Color depth of the exported image
    :param compress: Using lossless compression
    """
    try:
        with TiffWriter(path) as out_file:
            for i in tensor:
                try:
                    i = i.numpy()
                except AttributeError as e:
                    if type(i) == np.ndarray:
                        warn("Export numpy array is not preferred. Use Tensor instead.", DeprecationWarning)
                    else:
                        raise TypeError(f"Must export a list of 2-D Tensors but got {type(i)}") from e
                if len(i.shape) != 2:
                    raise ValueError(f"Must export a list of 2-D Tensors but got {len(i.shape)}-D data "
                                     f"with shape as {i.shape}.")
                if pre_operator is not None:
                    i = pre_operator(i)
                if scale is not None:
                    i *= scale * {np.uint16: 65535, np.uint8: 255}[dtype]
                i = i.astype(np.int64)
                np.clip(i, 0, {np.uint16: 65535, np.uint8: 255}[dtype], out=i)
                i = i.astype(dtype)
                if compress:
                    out_file.save(i, compress=9, predictor=True)
                else:
                    out_file.save(i)
    except KeyError as e:
        raise TypeError(f"dtype should be either np.uint8 or np.uint16, but not {dtype}") from e


def np_write(path, tensor, *, scale=1., pre_operator: callable = None, dtype=None, compress: bool = True):
    try:
        i = tensor.numpy()
    except AttributeError as e:
        try:
            i = np.array(tensor)
        except Exception as ee:
            raise TypeError(f"Must export numpy compatible Tensors but got {type(tensor)}") from ee
    if pre_operator is not None:
        i = pre_operator(i)
    i *= scale
    if dtype is not None:
        i = i.astype(dtype)
    ext = os.path.splitext(path)[-1]
    if ext == '.npy':
        np.save(path, i)
    elif ext == '.npz':
        np.savez_compressed(path, i) if compress else np.savez(path, i)


def csv_write(path: str, table):
    with open(path, 'w', newline='') as file:
        writer = csv.writer(file)
        for row in table:
            writer.writerow(row)


def write(dest, tensor, **kwargs):
    ext = os.path.splitext(dest)[-1]
    if ext in {'.tiff', '.tif'}:
        tiff_write(dest, tensor, **kwargs)
    elif ext in {'.npy', '.npz'}:
        np_write(dest, tensor, **kwargs)
    elif ext == '.csv':
        csv_write(dest, tensor)
    else:
        raise ValueError(f"unknown filename extension '{ext}'")
