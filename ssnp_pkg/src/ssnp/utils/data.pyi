from typing import Any, Tuple, Union, Iterable
import numpy as np
from pycuda.gpuarray import GPUArray


def predefined_read(name: str, shape: Tuple[int, ...], dtype: Any = np.float64) -> np.ndarray: ...


def tiff_read(path: str) -> np.ndarray: ...


def np_read(path: str, *, key: str = None) -> np.ndarray: ...


def mat_read(path: str, *, key: str = None) -> np.ndarray: ...


def np_write(path, arr: np.ndarray, *, scale: float = 1., pre_operator: callable = None, dtype: type = None,
             compress: bool = True): ...


def binary_write(path, arr: np.ndarray, *, scale=1., pre_operator: callable = None, dtype: type = None,
                 add_hint: bool = False): ...


def tiff_write(path, arr, *, scale=1, pre_operator: callable = None, dtype=np.uint16,
               compression='zlib', photometric=None):
    """
    Export a list of Tensors to a multipage tiff

    ``pre_operator`` can apply some numpy functions before the data to be exported, such as brightness
    and contrast adjustment. Normally it is used to avoid saturation or too dark pictures

    Argument ``scale`` is a simpler way than `pre_operator` to adjust data range.
    If ``scale`` is ``None``, it will write the raw data. Please note that this is
    **different** with ``scale=1``

    Argument ``dtype`` means the color depth of the exported image. It must be 16bit - ``np.uint16``
    (default) or 8bit - ``np.uint8``

    :param path: Target file path
    :param arr: Tensor data to be written
    :param scale: Multiplier to adjust value range
    :param pre_operator: Preprocess function before scaling
    :param dtype: Color depth of the exported image
    :param compression: Compression for tifffile (default is ZLIB)
    :param photometric: color information, e.g.: 'rgb'
    """

def read(source: str, dtype: type = np.float64, shape: Tuple[int, ...] = None, *, scale: float = 1.,
         gpu: bool = False, pagelocked=False, **kwargs) -> Union[np.ndarray, GPUArray]: ...


def write(dest: str, array: Union[GPUArray, Iterable], **kwargs): ...
