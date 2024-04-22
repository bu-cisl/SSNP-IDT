from typing import Any, Sequence, List, Union, Optional
from dataclasses import dataclass, field
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
    data: Any = field(default=None, repr=False)
    external: bool = False
    bound: bool = False

    # Deprecated: confusing to test if data is saved
    def __bool__(self):
        warn("Variable.__bool__ is deprecated, use Variable.has_data() instead", DeprecationWarning, stacklevel=2)
        return self.has_data()

    def has_data(self):
        return self.data is not None


class DataMissing(Exception):
    pass


@dataclass
class Operation:
    vars_in: Union[Variable, Sequence[Variable]]
    vars_out: Union[Variable, Sequence[Variable]]
    name: str = None
    taped_len = None
    _taped_out: Optional[List] = field(default=None, init=False, repr=False)
    _taped_in_all_saved: bool = field(default=False, init=False, repr=False)

    # vars_out_saved = True
    # tag_pos = {}

    def __post_init__(self):
        tl = [0, 0]
        if isinstance(self.vars_in, Variable):
            self.vars_in = (self.vars_in,)
        if isinstance(self.vars_out, Variable):
            self.vars_out = (self.vars_out,)

        for v in self.vars_in:
            v.bound = True
            tl[0] += not v.external
        for v in self.vars_out:
            v.bound = True
            tl[1] += not v.external
        self.update_saved()
        self.taped_len = tl

    def backprop(self, *grad_out_data, **kwargs):
        # assert len(grad_out_data) == self.taped_len[1], "grad_out length error"
        grad_in_data = self.gradient(*grad_out_data, **kwargs)
        if len(grad_in_data) != len(self.vars_in):
            raise ValueError(f"cannot match gradient in Operation('{self.name}') with vars_in numbers")
        return grad_in_data

    def recalculate(self, taped_in=None):
        # input param checks
        taped_in_len = self.taped_len[0]
        if self.can_self_recalculate():
            if taped_in is not None:
                warn(f"op {self} can self recalculate. taped_in is ignored", stacklevel=2)
            if self._taped_out is not None:
                return self._taped_out
            taped_in = [v for v in self.vars_in if not v.external]
        if taped_in is None:
            raise ValueError(f"op {self} cannot self recalculate without taped_in")
        if isinstance(taped_in, Variable):
            taped_in = [taped_in]
        assert len(taped_in) == taped_in_len, f"op {self} needs {taped_in_len} taped_in vars but got {len(taped_in)}"
        # calculation delegated to forward method implementation
        taped_out = self.forward(*taped_in)
        self.update_saved()
        # Note: it is NOT require to store the recalculated data in self.vars_out,
        # which allows for multiple times recalculation and further reuse of memory.
        return taped_out

    def can_self_recalculate(self):
        return self._taped_out is not None or self._taped_in_all_saved

    def update_saved(self):
        """
        Rebuild the taped_out list of this operation and check if input is all saved.
        1. If any taped vars_out do not have data stored, set taped_out to None.
        2. If all taped vars_in are saved, set _is_taped_in_saved to True.
        :return: None
        """
        self._taped_out = []
        for v in self.vars_out:
            if v.external:
                continue
            if v.has_data():
                self._taped_out.append(v)
            else:
                self._taped_out = None
                break
        self._taped_in_all_saved = all([(v.external or v.has_data()) for v in self.vars_in])

    def set_funcs(self, forward, gradient, clear=None):
        args = forward, gradient, clear
        for func, name in zip(args, ("forward", "gradient", "clear")):
            if callable(func):
                setattr(self, name, func)

    def forward(self, *args):
        raise NotImplementedError(f"'{self.name}' operation cannot recalculate")

    def gradient(self, *args, **kwargs):
        raise NotImplementedError(f"'{self.name}' operation cannot backprop")

    def clear(self):
        pass


# class CombinedOperation(Operation):
#     forward: Callable
#     gradient: Callable
#
#     def __init__(self, ops, ):
#         self.vars_in = []
#         for op in ops:
#             self.vars_in += op.vars_in


class OperationTape(list):
    class Restart(Exception):
        pass

    def __init__(self, size=None):
        super().__init__()
        if size is not None:
            self.save_hint = arithmetic_sequence_save(size)

    def append(self, op):
        if not isinstance(op, Operation):
            raise TypeError(f"can only append Operation (not '{type(op).__name__}') to OperationTape")
        if self and self[-1].taped_len[1] != op.taped_len[0]:
            raise ValueError("cannot append operation with incompatible taped variable number "
                             f"(last out: {self[-1].taped_len[1]})")
        super().append(op)

    def collect_gradient(self, tags, clear=True, reverse=False):
        if clear:  # reinitialize counter
            next(self.save_hint)  # otherwise the Restart may not be caught
            self.save_hint.throw(OperationTape.Restart)
        if not tags:  # nothing to collect, just clear self if needed
            if clear:
                for op in self:
                    op.clear()
            self.clear()
            return
        if len(self[-1].vars_out):
            raise ValueError("cannot collect gradient with no loss operation")
        if not isinstance(tags, dict):  # able to provide a tag iterable when no container
            tags = {t: None for t in tags}
        tags_container_iter = {}
        tags_build_list = {}
        for tag, container in tags.items():
            if container is not None:  # tag & container: dict values are iterator (store iter state of container)
                tags_container_iter[tag] = iter(container) if reverse else reversed(container)
            else:  # tag only: list to store output gradients
                tags_build_list[tag] = []
        taped_grad = []
        for op_idx in reversed(range(len(self))):
            op: Operation = self[op_idx]
            grad_out_tags = {v.tag for v in op.vars_in if v.tag is not None}
            grad_out_tags |= {f"{op.name}:{t}" for t in grad_out_tags}
            out = {re.sub(rf'^{op.name}:', '', tag): next(tags_container_iter[tag])
                   for tag in grad_out_tags & tags_container_iter.keys()}
            out.update({re.sub(rf'^{op.name}:', '', tag): None
                        for tag in grad_out_tags & tags_build_list.keys()})
            kwargs = {"out": out} if out else {}
            try:
                grad_in_data = op.backprop(*taped_grad, **kwargs)
            except DataMissing:
                for prev_idx in reversed(range(op_idx)):
                    if self[prev_idx].can_self_recalculate():
                        taped_out = None
                        break  # find the nearest operation that can self recalculate
                else:
                    raise ValueError("No previous Operations can recalculate")
                for prev_op in self[prev_idx:op_idx + 1]:
                    taped_out = prev_op.recalculate(taped_out)
                try:
                    grad_in_data = op.backprop(*taped_grad, **kwargs)
                except DataMissing:
                    raise ValueError(f"Recalculation of {op} does not provide enough data for its backpropagation")

            taped_grad = []
            for data, v in zip(grad_in_data, op.vars_in):
                if not v.external:
                    taped_grad.append(data)
                if v.tag in tags_build_list:
                    tags_build_list[v.tag].append(data)
                if (op_var := f"{op.name}:{v.tag}") in tags_build_list:
                    tags_build_list[op_var].append(data)
            if clear:
                op.clear()
        if clear:
            self.clear()
        if tags_build_list:
            if not reverse:
                for tag in tags_build_list:
                    tags_build_list[tag] = tags_build_list[tag][::-1]
            return tags_build_list


def arithmetic_sequence_save(total):
    if total <= 0:
        while True:
            try:
                yield True
            except OperationTape.Restart:
                pass
    remainder = current = int(np.sqrt(2 * total + 0.25) + 0.5)
    while True:
        try:
            if current >= remainder:
                if remainder > 0:
                    remainder -= 1
                    current = 0
                yield True
            else:
                current += 1
                yield False
        except OperationTape.Restart:
            remainder = current = int(np.sqrt(2 * total + 0.25) + 0.5)
            yield

# another possible solution: use generator.send(...) to restart
# def arithmetic_sequence_save(total):
#     # always save if total <= 0
#     if total <= 0:
#         while True:
#             yield True
#
#     remainder = current = int(np.sqrt(2 * total + 0.25) + 0.5)
#     # remainder = current = None  # will be initialized later
#     restart = None
#     while True:
#         if restart is not None:
#             remainder = current = int(np.sqrt(2 * total + 0.25) + 0.5)
#             yield
#         if current >= remainder:
#             if remainder > 0:
#                 remainder -= 1
#                 current = 0
#             restart = yield True
#         else:
#             current += 1
#             restart = yield False
