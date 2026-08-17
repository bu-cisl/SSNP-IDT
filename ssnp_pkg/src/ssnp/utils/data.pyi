from typing import Tuple, Union, Iterable, Optional, overload, Literal
import numpy as np
from pycuda.gpuarray import GPUArray


def predefined_read(name: str, shape: Tuple[int, ...], dtype: Optional[type] = np.float64) -> np.ndarray: ...


def tiff_read(path: str, scale: Optional[float] = 1., dtype: Optional[type] = np.float64) -> np.ndarray:
    """
    Import a TIFF file as numpy array

    :param path: target file path.
    :param scale: scaling factor for floating number output.
    :param dtype: output type.
    :return: An np.ndarray. shape = [(pages), rows, columns, (channels)],
      pages and channels are optional and will be squeezed if equal to 1
    """


def np_read(path: str, *, key: str = None) -> np.ndarray: ...


def mat_read(path: str, *, key: str = None) -> np.ndarray: ...


def np_write(path, arr: np.ndarray, *, scale: float = 1., pre_operator: callable = None, dtype: type = None,
             compress: bool = True): ...


def binary_write(path, arr: np.ndarray, *, scale: float=1., pre_operator: callable = None, dtype: type = None,
                 add_hint: bool = False): ...


@overload
def read(source: str, dtype: type = None, shape: Tuple[int, ...] = None, *, scale: float = 1.,
         gpu: Literal[True], pagelocked=False, **kwargs) -> GPUArray: ...

@overload
def read(source: str, dtype: type = None, shape: Tuple[int, ...] = None, *, scale: float = 1.,
         gpu: Literal[False] = False, pagelocked=False, **kwargs) -> np.ndarray: ...

def write(dest: str, array: Union[GPUArray, Iterable], **kwargs): ...
