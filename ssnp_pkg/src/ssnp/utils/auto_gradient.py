from typing import Any, Sequence, List, Iterable, Union
from dataclasses import dataclass, field
import numpy as np


@dataclass
class Variable:
    tag: str = None
    data: Any = field(default=None, repr=False)
    external: bool = False
    bound: bool = False

    def __bool__(self):
        return self.data is not None


class DataMissing(Exception):
    pass


@dataclass
class Operation:
    vars_in: Union[Variable, Sequence[Variable]]
    vars_out: Union[Variable, Sequence[Variable]]
    name: str = None
    taped_len = None
    taped_out = None

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
            # if v.tag:
            #     self.tag_pos[v.tag] = pos
        for v in self.vars_out:
            v.bound = True
            tl[1] += not v.external
        self.update_taped_out()
        self.taped_len = tl

    def backprop(self, *grad_out_data, **kwargs):
        # assert len(grad_in_data) == self.taped_len[0], "grad_in length error"
        grad_in_data = self.gradient(*grad_out_data, **kwargs)
        # assert len(grad_out_data) == self.taped_len[1], "grad_out length error"
        return grad_in_data

    def recalculate(self, taped_in):
        # assert len(vars_in_data) == len(self.vars_in), "vars_in length error"
        if isinstance(taped_in, Variable):
            taped_out = self.forward(taped_in)
        else:
            taped_out = self.forward(*taped_in)
        self.update_taped_out()
        # assert len(taped_data) == len(self.vars_out), "vars_out length error"
        return taped_out

    def update_taped_out(self):
        self.taped_out = []
        for v in self.vars_out:
            if not v.external:
                if v.data:
                    self.taped_out.append(v)
                else:
                    self.taped_out = None
                    return

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


class OperationTape:
    tape: List[Operation] = None

    @staticmethod
    def arithmetic_sequence_save(total):
        if total <= 0:
            while True:
                yield True
        remainder = current = int(np.sqrt(2 * total + 0.25) - 0.5)
        while True:
            if current >= remainder:
                if remainder > 0:
                    remainder -= 1
                    current = 0
                yield True
            else:
                current += 1
                yield False

    def __init__(self, size=None):
        self.tape = []
        if size:
            self.save_hint = self.arithmetic_sequence_save(size)

    def append(self, op):
        if self.tape and self.tape[-1].taped_len[1] != op.taped_len[0]:
            raise ValueError("cannot append operation with incompatible taped variable number "
                             f"(last out: {self.tape[-1].taped_len[1]})")
        self.tape.append(op)

    def collect_gradient(self, tags, clear=True, reverse=False):
        if not tags:  # nothing to collect, just clear self if needed
            if clear:
                for op in self.tape:
                    op.clear()
            self.tape = []
            return
        if len(self.tape[-1].vars_out):
            raise ValueError("cannot collect gradient with no loss operation")
        if not isinstance(tags, dict):  # able to provide a tag iterable when no container
            tags = {t: None for t in tags}
        tags_container_iter = {}
        tags_build_list = {}
        for tag, container in tags.items():
            if container:  # tag & container: dict values are iterator (store iter state of container)
                tags_container_iter[tag] = iter(container) if reverse else reversed(container)
            else:  # tag only: list to store output gradients
                tags_build_list[tag] = []
        taped_grad = []
        for op_i in reversed(range(len(self.tape))):
            op = self.tape[op_i]
            grad_out_tags = [v.tag for v in op.vars_in]
            out = {tag: next(it) for tag, it in tags_container_iter.items() if tag in grad_out_tags}
            kwargs = {"out": out} if out else {}
            try:
                grad_in_data = op.backprop(*taped_grad, **kwargs)
            except DataMissing as e:
                last_saved = op_i - 1
                while not (taped_out := self.tape[last_saved].taped_out):
                    last_saved -= 1
                for i in range(last_saved, op_i):
                    taped_out = self.tape[i + 1].recalculate(taped_out)
                try:
                    grad_in_data = op.backprop(*taped_grad, **kwargs)
                except DataMissing:
                    raise ValueError("recalculation failed in partial-saved backpropagation") from e

            taped_grad = []
            if not isinstance(grad_in_data, Iterable) and len(op.vars_in) == 1:
                grad_in_data = [grad_in_data]
            for data, v in zip(grad_in_data, op.vars_in):
                if not v.external:
                    taped_grad.append(data)
                try:
                    tags_build_list[v.tag].append(data)
                except KeyError:
                    pass
            if clear:
                op.clear()
        if clear:
            self.tape = []
        if tags_build_list:
            if not reverse:
                for tag in tags_build_list:
                    tags_build_list[tag] = tags_build_list[tag][::-1]
            return tags_build_list

    def __repr__(self):
        content = ',\n  '.join(repr(o) for o in self.tape)
        return f"{self.__class__.__name__}([{content}])"
