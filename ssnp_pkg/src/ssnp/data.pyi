from typing import Any, Tuple, Union, overload, Iterable
import numpy as np
from pycuda import gpuarray

DEFAULT_TYPE: type


def predefined_read(name: str, shape: Tuple[int], dtype: Any = np.float64) -> np.ndarray: ...


def tiff_read(path: str, dtype: Any = np.float64) -> np.ndarray: ...


def np_read(path: str, dtype: Any = np.float64, *, key: str = None) -> np.ndarray: ...


def np_write(path, arr: np.ndarray, *, scale: float = 1., pre_operator: callable = None, dtype: type = None,
             compress: bool = True): ...


def read(source: str, dtype=np.float64, shape: Tuple[int] = None, *, key: str = None) -> np.ndarray: ...


def write(dest: str, tensor: Union[gpuarray, Iterable], **kwargs): ...


