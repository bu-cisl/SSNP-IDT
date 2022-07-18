import tensorflow as tf
from tensorflow import Tensor, DType
from typing import Any, Tuple, Union, overload

DEFAULT_TYPE: DType


def predefined_read(name: str, shape: Tuple[int], dtype: Any = tf.float32) -> Tensor: ...


def tiff_read(path: str, dtype: Any = tf.float32, shape: Tuple[int] = None) -> Tensor: ...


def np_read(path: str, dtype: Any = tf.float32, shape: Tuple[int] = None, *, key: str = None) -> Tensor: ...


def read(source: str, dtype=tf.float32, shape: Tuple[int] = None, *, key: str = None) -> Tensor: ...


def tilt_illumination(img: Tensor, c_ab: tuple, *, trunc: bool = False) -> Tuple[Tensor, tuple]: ...
