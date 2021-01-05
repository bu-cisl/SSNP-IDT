from typing import overload, Tuple, Union, List


@overload
def broadcast(expected_output: Union[tuple, list] = None, **shapes) -> Tuple[list, Tuple[str, str]]: ...


@overload
def broadcast(shapes: dict, /, expected_output: Union[tuple, list] = None) -> Tuple[list, Tuple[str, str]]: ...


def replace_i(code: str, repl_vars: List[str, ...]): ...
