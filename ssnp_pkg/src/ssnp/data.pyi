import tensorflow as tf
from tensorflow import Tensor, DType
from typing import Any, Tuple, Union, overload, Iterable
import numpy as np

DEFAULT_TYPE: DType


def predefined_read(name: str, shape: Tuple[int], dtype: Any = tf.float32) -> Tensor: ...


def tiff_read(path: str, dtype: Any = tf.float32, shape: Tuple[int] = None) -> Tensor: ...


def np_read(path: str, dtype: Any = tf.float32, shape: Tuple[int] = None, *, key: str = None) -> Tensor: ...


def np_write(path, tensor: Tensor, *, scale: float = 1., pre_operator: callable = None, dtype: type = None,
             compress: bool = True): ...


def read(source: str, dtype=tf.float32, shape: Tuple[int] = None, *, key: str = None) -> Tensor: ...


def write(dest: str, tensor: Union[Tensor, Iterable], **kwargs): ...


