from typing import Any, Sequence, Union, Optional, Generator, Iterator, Iterable
from dataclasses import dataclass
import re

import numpy as np
from warnings import warn


@dataclass
class Variable:
    """
    A dataclass of one variable. All fields are optional (e.g. a simple placeholder if no arguments provided)

    ``tag``: Name for this variable;

    ``data``: Reference of the data of this variable for future use;

    ``external``: Marking this Variable as input/output data (which means it is not an intermediate result)

    ``bound``: Set when constructing an ``Operation``. If bound is not set,
    it is a temporary free variable and the memory is managed by itself (can be recycled after used).
    Otherwise, the memory of its data field should be managed by the operation which it was bound to
    """
    tag: str = None
    data: Any = None
    external: bool = False
    bound: bool = False

    def has_data(self) -> bool: ...


class DataMissing(Exception):
    pass


type Vars = Sequence[Variable] | Variable


class Operation:
    vars_in: Sequence[Variable]
    vars_out: Sequence[Variable]
    name: str = None
    taped_len: tuple[int, int]
    _taped_out: Optional[list[Variable]]
    _taped_in_all_saved: bool

    def __init__(self, vars_in: Vars, vars_out: Vars, name: str = None): ...

    def backprop(self, *grad_out_data, **kwargs) -> Sequence: ...

    def recalculate(self, taped_in: Vars | None = None): ...

    def can_self_recalculate(self) -> bool: ...

    def update_saved(self): ...

    def set_funcs(self, forward, gradient, clear=None): ...

    def forward(self, *args: Variable) -> Sequence[Variable]: ...

    def gradient(self, *args, **kwargs) -> Any: ...

    def clear(self): ...


class OperationTape(list):
    save_hint: Generator[bool, None, None]

    class Restart(Exception): ...

    def __init__(self, size: int | None = None): ...

    def append(self, op: Operation): ...

    def collect_gradient(self, tags: dict[str, Iterable | None] | Sequence[str],
                         clear: bool = True, reverse: bool = False): ...


def arithmetic_sequence_save(total: int) -> Generator[bool, None, None]: ...
