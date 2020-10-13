from typing import Tuple, Union

def param_check(**kwargs): ...


class Config:
    res: Union[Tuple[float, float, float], property]
    n0 = 1


config: Config
