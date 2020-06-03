from tensorflow import Tensor
from typing import Any, Tuple


def real_to_complex(real: Tensor): ...


def tilt(img: Tensor, c_ab: tuple, *, trunc: bool = False) -> Tuple[Tensor, tuple]: ...
