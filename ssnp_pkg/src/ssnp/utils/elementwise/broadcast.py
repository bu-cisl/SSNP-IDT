from itertools import cycle, islice, chain
from math import prod
from warnings import warn
import re


def roundrobin(iter_in):
    """roundrobin('ABC', 'D', 'EF') --> A D E B F C"""
    # Recipe credited to George Sakkis
    num_active = len(iter_in)
    nexts = cycle(iter(it).__next__ for it in iter_in)
    while num_active:
        try:
            for next_ in nexts:
                yield next_()
        except StopIteration:
            # Remove the iterator we just exhausted from the cycle.
            num_active -= 1
            nexts = cycle(islice(nexts, num_active))


def broadcast(_arg=None, /, expected_output=None, **kwargs):
    """
    Generate code for modified index variables from argument shapes.

    Returns 2 objects:
        - argument names which need broadcasting
        - C code segments for index variables:
        declaration (in loop_prep), calculation (in operation)

    :param shapes: argument shapes, input as keyword variables or a dict
    :return: (broadcasting argument, code)
    """
    if _arg is not None:
        kwargs = _arg
    max_dims = max(len(i) for i in kwargs.values())
    casted = []
    # align shapes: [4 34 234] -> [114 134 234]
    new_shape = {n: [1] * (max_dims - len(l)) + list(l) for n, l in kwargs.items()}
    # squeeze dims
    for i in reversed(range(max_dims)):
        shape_dim = 0
        for name, shape in new_shape.items():
            if shape[i] > 1:
                if shape_dim:
                    assert shape_dim == shape[i]
                else:
                    shape_dim = shape[i]
        if shape_dim:
            casted.append(shape_dim)
        else:  # delete dim which is all 1
            for shape in new_shape.values():
                del shape[i]
    casted = casted[::-1]
    if expected_output is not None:  # it may be an empty tuple if a scalar is wrongly used
        if prod(expected_output) != prod(casted):
            warn("output size is different from expected broadcast size.\n"
                 f"output shape: {tuple(expected_output)}, expected broadcast shape: {tuple(casted)}",
                 stacklevel=2)

    # construct C code segments
    operation = []
    bc_vars = []
    for name, shape in new_shape.items():
        containing_size = prod(casted)
        last_full_dim = -1
        c_name = f'__{name}_i'
        code = []
        for i, ci in enumerate(casted):
            if shape[i] > 1:
                if (cast_from := last_full_dim + 1) < i:
                    read_name = c_name if code else "i"
                    code.append(f'{c_name} = {read_name} % {containing_size}')
                    if cast_from:  # cast from 0 is redundant
                        divided = prod(casted[cast_from:i]) * containing_size
                        code[-1] += f' + {read_name} / {divided} * {containing_size}'
                last_full_dim = i
            containing_size //= ci
        if (cast_from := last_full_dim + 1) != len(casted):
            read_name = c_name if code else "i"
            code.append(f'{c_name} = {read_name} / {prod(casted[cast_from:])}')
        if code:
            bc_vars.append(name)
            operation.append(code)
    # loop_prep = roundrobin(loop_prep)
    end_line = ';\n'
    if bc_vars:
        operation = end_line.join(chain(*operation)) + end_line
        loop_prep = 'unsigned ' + ', '.join([f'__{n}_i' for n in bc_vars]) + end_line
        return bc_vars, (loop_prep, operation)
    else:
        return None, None


def replace_i(code, repl_vars):
    def repl(match):
        arg_name = match[1]
        if match[2].strip() == 'i':
            return f"{arg_name}[__{arg_name}_i]"
        elif match[2].strip()[0] == '_':
            return match[0]
        else:
            raise ValueError(f"cannot process abnormal index: '{match.group()}'")

    names = "|".join(repl_vars)
    return re.sub(rf"\b({names})\[(.*?)]", repl, code, flags=re.DOTALL)
