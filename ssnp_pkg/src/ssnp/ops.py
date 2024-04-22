import logging
from numbers import Complex

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
        u1_save = beam._get_array()
        u1_save.set(beam._u1)
        if not self._bi_dir:
            super().__init__((Var(data=u1_save), var_other), Var(), name="mul")
        else:
            u2_save = beam._get_array()
            u2_save.set(beam._u2)
            super().__init__((Var(data=u1_save), Var(data=u2_save), var_other), (Var(), Var()), name="mul2")

    def forward(self, *vars_in):
        return [
            Var(data=calc.u_mul(vin.data, self._other, out=self._beam._get_array(), stream=self._beam.stream))
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
                # ug[1] * conj(u2_save) -> u2_save, then added to grad_other and recycled
                u2_var = self.vars_in[1]
                grad_other += calc.u_mul(ug[1], u2_var.data, out=u2_var.data, conj=True, stream=self._beam.stream)
                self._beam.recycle_array(u2_var.data)
                u2_var.data = None
            self._taped_in_all_saved = False
            # sum axis for grad_other if doing broadcast in forward
            if isinstance(self._other, Complex):  # number: return number in cpu memory
                self._beam.recycle_array(grad_other)
                grad_other = gpuarray.sum(grad_other, stream=self._beam.stream).get()
                if out_container is not None:
                    raise ValueError("cannot assign to multiplier gradient container when multiplier is a number")
            elif self._beam.batch is None:  # elementwise: return array or copy to out_container
                if out_container is not None:
                    param_check(multiplier_gradient=grad_other, multiplier_output=out_container)
                    assert out_container.dtype == grad_other.dtype
                    out_container.set(grad_other)
                    self._beam.recycle_array(grad_other)
                    grad_other = out_container
            else:  # batch: sum to allocated array or out_container
                if out_container is None:
                    logging.info("allocate new array for multiplier gradient")
                    out_arr = gpuarray.empty_like(grad_other[0])
                else:
                    param_check(multiplier_gradient=grad_other[0], multiplier_output=out_container)
                    assert out_container.dtype == grad_other.dtype
                    out_arr = out_container
                self._beam.recycle_array(grad_other)
                grad_other = calc.sum_batch(grad_other, output=out_arr, stream=self._beam.stream)
        else:
            grad_other = None

        grad = [calc.u_mul(ug_i, self._other, conj=True, stream=self._beam.stream) for ug_i in ug]
        grad.append(grad_other)
        return grad

    def clear(self):
        if (u1_save := self.vars_in[0].data) is not None:
            self._beam.recycle_array(u1_save)
        if self._bi_dir:
            if (u2_save := self.vars_in[1].data) is not None:
                self._beam.recycle_array(u2_save)
