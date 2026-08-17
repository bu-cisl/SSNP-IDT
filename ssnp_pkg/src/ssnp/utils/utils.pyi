import contextlib
import pycuda.driver as cuda
from pycuda import gpuarray
from typing import Tuple, Union, List, Optional, Callable


def param_check(*, _expected_type=gpuarray.GPUArray, **kwargs): ...


def get_stream(ctx: cuda.Context) -> cuda.Stream: ...


def get_stream_in_current() -> cuda.Stream: ...


def pop_pycuda_context() -> Optional[contextlib.AbstractContextManager[cuda.Context]]: ...


type float3 = tuple[float, float, float]


class Config:
    _res: float3 | None

    @property
    def res(self) -> float3: ...

    @res.setter
    def res(self, value: float3) -> None: ...

    _xyz: float3 | None

    @property
    def xyz(self) -> float3: ...

    @xyz.setter
    def xyz(self, value: float3) -> None: ...

    _lambda: float | None

    @property
    def lambda0(self) -> float: ...

    @lambda0.setter
    def lambda0(self, value: float) -> None: ...

    _n0: float

    @property
    def n0(self) -> float: ...

    @n0.setter
    def n0(self, value: float) -> None: ...

    _callbacks: list[Callable]

    def register_updater(self, updater): ...

    def clear_updater(self): ...

    def set(self, **kwargs): ...


config: Config
