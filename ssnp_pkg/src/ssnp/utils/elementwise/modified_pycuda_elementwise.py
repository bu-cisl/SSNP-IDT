# Modified from PyCUDA (version 2020.1)
from functools import lru_cache
from pycuda import elementwise
import pycuda.tools
import numpy as np
from . import broadcast


# enable Argument to be cached correctly
class Argument(pycuda.tools.Argument):
    def __eq__(self, other):
        return type(self) == type(other) and (self.dtype, self.name) == (other.dtype, other.name)

    def __hash__(self):
        return hash(self.name)


class VectorArg(pycuda.tools.VectorArg, Argument):
    pass


class ScalarArg(pycuda.tools.ScalarArg, Argument):
    pass


def parse_c_arg(c_arg):
    from pycuda.compyte.dtypes import parse_c_arg_backend
    return parse_c_arg_backend(c_arg, ScalarArg, VectorArg)


@lru_cache
def module_builder(_use_range, *args, **kwargs):
    if _use_range:
        return elementwise.get_elwise_range_module(*args, **kwargs)
    else:
        return elementwise.get_elwise_module(*args, **kwargs)


def get_elwise_kernel_and_types(arguments, operation, call_args,
                                name="kernel", keep=False, options=None, use_range=False,
                                broadcasting=False, **kwargs):
    if isinstance(arguments, str):
        arguments = [parse_c_arg(arg) for arg in arguments.split(",")]

    if use_range:
        arguments.extend([
            ScalarArg(np.intp, "start"),
            ScalarArg(np.intp, "stop"),
            ScalarArg(np.intp, "step"),
        ])
    else:
        arguments.append(ScalarArg(np.uintp, "n"))

    vectors = []
    invocation_args = []
    vectors_descr = []
    for arg, arg_descr in zip(call_args, arguments):
        if isinstance(arg_descr, VectorArg):
            if not arg.flags.forc:
                raise RuntimeError("elementwise kernel cannot "
                                   "deal with non-contiguous arrays")
            vectors.append(arg)
            vectors_descr.append(arg_descr)
            invocation_args.append(arg.gpudata)
        else:
            invocation_args.append(arg)

    if broadcasting:
        bc_names, bc_codes = broadcast.broadcast(
            {descr.name: v.shape for v, descr in zip(vectors, vectors_descr)},
            expected_output=vectors[0].shape
        )
        if bc_names:
            loop_prep = kwargs.get("loop_prep", "")
            kwargs["loop_prep"] = bc_codes[0] + loop_prep
            operation = broadcast.replace_i(operation, bc_names)
            operation = bc_codes[1] + operation
    arguments = tuple(arguments)

    mod = module_builder(use_range, arguments, operation, name,
                         keep, options, **kwargs)

    func = mod.get_function(name)
    func.prepare("".join(arg.struct_char for arg in arguments))

    return mod, func, arguments, (vectors, invocation_args)


class ElementwiseKernel(elementwise.ElementwiseKernel):
    # def __init__(self, *args, broadcasting=False, **kwargs):
    #     super().__init__(*args, **kwargs)
    #     self.gen_kwargs["broadcasting"] = broadcasting

    def generate_stride_kernel_and_types(self, use_range, call_args=None):
        mod, knl, arguments, call_info = get_elwise_kernel_and_types(
            use_range=use_range, call_args=call_args, **self.gen_kwargs)

        assert [i for i, arg in enumerate(arguments)
                if isinstance(arg, VectorArg)], \
            "ElementwiseKernel can only be used with functions that " \
            "have at least one vector argument"

        return mod, knl, arguments, call_info

    def __call__(self, *args, **kwargs):
        range_ = kwargs.pop("range", None)
        slice_ = kwargs.pop("slice", None)
        stream = kwargs.pop("stream", None)

        if kwargs:
            raise TypeError("invalid keyword arguments specified: "
                            + ", ".join(elementwise.six.iterkeys(kwargs)))

        mod, func, arguments, call_info = self.generate_stride_kernel_and_types(
            range_ is not None or slice_ is not None, args)

        vectors, invocation_args = call_info
        repr_vec = vectors[0]

        if slice_ is not None:
            if range_ is not None:
                raise TypeError("may not specify both range and slice "
                                "keyword arguments")

            range_ = slice(*slice_.indices(repr_vec.size))

        if range_ is not None:
            invocation_args.append(range_.start)
            invocation_args.append(range_.stop)
            if range_.step is None:
                invocation_args.append(1)
            else:
                invocation_args.append(range_.step)

            from pycuda.gpuarray import splay
            grid, block = splay(abs(range_.stop - range_.start) // range_.step)
        else:
            block = repr_vec._block
            grid = repr_vec._grid
            invocation_args.append(repr_vec.mem_size)

        func.prepared_async_call(grid, block, stream, *invocation_args)
