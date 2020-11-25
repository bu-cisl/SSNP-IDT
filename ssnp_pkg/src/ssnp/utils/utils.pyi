from typing import Tuple, Union, List

FFF_P = Union[Tuple[float, float, float], property]
F_P = Union[float, property]


def param_check(**kwargs): ...


def get_stream(ctx): ...


def get_stream_in_current(): ...


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
