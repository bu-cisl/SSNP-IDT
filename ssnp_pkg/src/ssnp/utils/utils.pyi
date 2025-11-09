import contextlib
import pycuda.driver as cuda
from typing import Tuple, Union, List, Optional

FFF_P = Union[Tuple[float, float, float], property]
F_P = Union[float, property]


def param_check(**kwargs): ...


def get_stream(ctx: cuda.Context) -> cuda.Stream: ...


def get_stream_in_current() -> cuda.Stream: ...


def pop_pycuda_context() -> Optional[contextlib.AbstractContextManager[cuda.Context]]: ...


class Config:
    res: FFF_P
    xyz: FFF_P
    lambda0: F_P
    n0: F_P
    _callbacks: List[callable, ...]

    def register_updater(self, updater): ...

    def clear_updater(self): ...

    def set(self, **kwargs): ...


config: Config
