import logging
from numbers import Complex
from contextlib import nullcontext

from pycuda import gpuarray

from ssnp.utils import param_check
from ssnp.utils.auto_gradient import Operation, Variable as Var, DataMissing
from ssnp import calc


class MulOp(Operation):
    def __init__(self, other, beam):
        self._beam = beam
        self._other = other
        self._bi_dir = beam._u2 is not None
        var_other = Var("multiplier", external=True)
        u1_save = beam.array_pool.get()
        u1_save.set(beam._u1)
        if not self._bi_dir:
            super().__init__((Var(data=u1_save), var_other), Var(), name="mul")
        else:
            u2_save = beam.array_pool.get()
            u2_save.set(beam._u2)
            super().__init__((Var(data=u1_save), Var(data=u2_save), var_other), (Var(), Var()), name="mul2")

    def forward(self, *vars_in):
        return [
            Var(data=calc.u_mul(
                vin.data, self._other, stream=self._beam.stream,
                out=self._beam.array_pool.get() if vin.bound else vin.data
            ))
            for vin in vars_in
        ]

    def gradient(self, *ug, out=None):
        if out and 'multiplier' in out:
            out_container = out['multiplier']
            # ug[0] * conj(u1_save) -> u1_save, rename as grad_other
            # Caution: ug[0/1] cannot be changed in place, since it will be used in next calculation
            u1_var = self.vars_in[0]
            if u1_var.data is None:
                raise DataMissing
            grad_other = calc.u_mul(ug[0], u1_var.data, out=u1_var.data, conj=True, stream=self._beam.stream)
            u1_var.data = None  # memory is moved
            if self._bi_dir:
                # ug[1] * conj(u2_save) -> u2_save, then added to grad_other
                u2_var = self.vars_in[1]
                grad_other += calc.u_mul(ug[1], u2_var.data, out=u2_var.data, conj=True, stream=self._beam.stream)
                u2_var.data = None
            self._taped_in_all_saved = False
            # sum axis for grad_other if doing broadcast in forward
            if isinstance(self._other, Complex):  # number: return number in cpu memory
                grad_other_arr = grad_other
                grad_other = gpuarray.sum(grad_other, stream=self._beam.stream).get()
                grad_other_arr.dispose()
                if out_container is not None:
                    raise ValueError("cannot assign to multiplier gradient container when multiplier is a number")
            elif self._beam.batch is None:  # elementwise: return array or copy to out_container
                if out_container is not None:
                    param_check(multiplier_gradient=grad_other, multiplier_output=out_container)
                    assert out_container.dtype == grad_other.dtype
                    out_container.set(grad_other)
                    grad_other.dispose()
                    grad_other = out_container
            else:  # batch: sum to allocated array or out_container
                if out_container is None:
                    logging.info("allocate new array for multiplier gradient")
                    out_arr = gpuarray.empty_like(grad_other[0])
                else:
                    param_check(multiplier_gradient=grad_other[0], multiplier_output=out_container)
                    assert out_container.dtype == grad_other.dtype
                    out_arr = out_container
                batched_grad = grad_other
                grad_other = calc.sum_batch(grad_other, output=out_arr, stream=self._beam.stream)
                batched_grad.dispose()
        else:
            grad_other = None

        grad = [calc.u_mul(ug_i, self._other, conj=True, stream=self._beam.stream) for ug_i in ug]
        grad.append(grad_other)
        return grad


class FourierMulOp(MulOp):
    def __init__(self, other, beam):
        super().__init__(other, beam)
        self.name = "a_mul" if not self._bi_dir else "a_mul2"
        self._fourier = beam._fft_funcs.fourier

    def forward(self, *vars_in):
        with self._fourier(vars_in[0].data), self._fourier(vars_in[1].data) if self._bi_dir else nullcontext():
            return super().forward(*vars_in)

    def gradient(self, *ug_list, out=None):
        # Always makes sure that the GPU memory is reused in the overridden super class method
        # otherwise iFFT is not applied to the result
        with self._fourier(ug_list[0]), self._fourier(ug_list[1]) if self._bi_dir else nullcontext():
            return super().gradient(*ug_list, out=out)


class MSELossOp(Operation):
    def __init__(self, beam, other_forward, other_backward):
        self._beam = beam
        # When other_forward / backward is missing, the gradient of the corresponding beam component is 0.
        # For bi-direction, len(grads) = 2, otherwise 1 for forward only
        self._grads = [
            calc.reduce_mse_grad(b_arr, o_arr, output=beam.array_pool.get(), stream=beam.stream)
            if o_arr is not None else beam.array_pool.get().fill(0, stream=beam.stream)
            for b_arr, o_arr in [[beam.forward, other_forward], [beam.backward, other_backward]]
            if b_arr is not None
        ]

        # Note that other_forward / backward is not marked as external args since getting the
        # gradient for measurements seems meaningless
        vars_in = [Var('uf')]
        op_name = "mse_f"
        if beam.backward is not None:
            vars_in.append(Var('ub'))
            op_name = "mse_fb"
        super().__init__(vars_in, [], op_name)

    def gradient(self, out=None):
        if out:  # copy and replace the array
            if 'uf' in out:
                if out['uf'] is None:
                    out['uf'] = self._beam.array_pool.get()
                out['uf'].set_async(self._grads[0], stream=self._beam.stream)
            if 'ub' in out:
                if out['ub'] is None:
                    out['ub'] = self._beam.array_pool.get()
                out['ub'].set_async(self._grads[1], stream=self._beam.stream)
        return self._grads
