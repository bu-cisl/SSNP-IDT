from typing import Tuple, Union

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


config: Config
