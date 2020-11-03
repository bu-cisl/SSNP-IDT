from typing import Tuple, Union

def param_check(**kwargs): ...


def get_stream(ctx): ...

def get_stream_in_current(): ...

class Config:
    res: Union[Tuple[float, float, float], property]
    n0 = 1


config: Config
