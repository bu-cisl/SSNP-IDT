from unittest import TestCase
import os
import platform
import warnings
import numpy as np
import pycuda.compiler
from pycuda import gpuarray

if platform.system() == 'Windows':
    # eliminate "non-UTF8 char" warnings
    pycuda.compiler.DEFAULT_NVCC_FLAGS = ['-Xcompiler', '/wd 4819']


from ssnp import BeamArray
import ssnp

ssnp.config.set(xyz=(0.1, 0.2, 0.3), lambda0=0.632)


class TestBeamSetUp(TestCase):
    def setUp(self):
        self.ones_cpu = np.ones(dtype=np.complex128, shape=(128, 128))
        self.ones_gpu = gpuarray.to_gpu(self.ones_cpu)

    def test_defaults(self):
        beam = BeamArray(self.ones_gpu)
        self.assertEqual(beam.dtype, np.complex128)


class TestBeamArraySingle(TestCase):
    def setUp(self) -> None:
        self.shape = (128, 128)
        u = np.ones(self.shape, np.complex128)
        self.beam = BeamArray(u)
        self.rng = np.random.default_rng()

    def assertArrayEqual(self, a: np.ndarray, b: np.ndarray, delta=0.):
        # don't directly calculate GPUArray with numpy function
        self.assertIsInstance(a, np.ndarray)
        self.assertIsInstance(b, np.ndarray)
        self.assertEqual(a.shape, b.shape)
        self.assertLessEqual(np.linalg.norm(a - b), delta)

    def random_like(self, arr, dtype=None):
        if dtype is None:
            dtype = arr.dtype
        match dtype:
            case np.float32 | np.float64:
                return self.rng.random(arr.shape, dtype)
            case np.complex64 | np.complex128:
                dtype_f = np.finfo(dtype).dtype
                return self.rng.random(arr.shape, dtype_f) + 1j * self.rng.random(arr.shape, dtype_f)
            case _:
                raise NotImplementedError(f"random_floating_like does not support {arr.dtype}")

    def test_config(self):
        beam = self.beam
        # test auto create empty config
        self.assertIsNone(beam._config)
        self.assertIsInstance(beam.config, ssnp.utils.Config)
        another_conf = ssnp.utils.Config()
        another_conf.set(xyz=(0.1, 0.2, 0.3), lambda0=0.632, n0=1.33)
        # beam should not copy updater
        another_conf.register_updater(lambda **kwargs: self.fail("beam triggers external updater unexpectedly"))
        beam.config = another_conf
        # beam should make a proper copy
        self.assertIsNot(beam.config, another_conf)
        self.assertEqual(beam.config.res, another_conf.res)
        self.assertEqual(beam.config.n0, another_conf.n0)
        # test update
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # update beam mono
            beam.config.n0 = 1.47
            for i in zip(beam.config.res, another_conf.res):
                self.assertNotAlmostEqual(*i)
            self.assertNotAlmostEqual(beam.config.n0, another_conf.n0)
            # update beam bi-dir
            beam.backward = 0
            beam.config.n0 = 1.33
            self.assertEqual(beam.relation, BeamArray.DERIVATIVE)
            for i in zip(beam.config.res, another_conf.res):
                self.assertAlmostEqual(*i)
            self.assertAlmostEqual(beam.config.n0, another_conf.n0)
            beam.config.lambda0 = 0.5
            self.assertEqual(beam.relation, BeamArray.BACKWARD)
            # # should not cause warnings
            # self.assertListEqual([i.message for i in w], [])

    # def test_forward(self):
    #     print(id(self.beam))
    #
    # def test_backward(self):
    #     self.fail()
    #
    # def test_field(self):
    #     self.fail()
    #
    # def test_derivative(self):
    #     self.fail()
    #
    # def test_ssnp(self):
    #     self.fail()
    #
    # def test_bpm(self):
    #     self.fail()
    #
    # def test_n_grad(self):
    #     self.fail()
    #

    def test_mse_loss(self):
        DELTA = 1e-12

        def _gradient_cpu(arr_cmplx_field, arr_meas):
            self.assertEqual(arr_cmplx_field.dtype, np.complex128)
            if arr_meas.dtype == np.float64:
                grad_amp = 2 * (np.abs(arr_cmplx_field) - arr_meas) / arr_cmplx_field.size
                # circumvent divided by 0 problem
                grad_ang = arr_cmplx_field.copy()
                grad_ang[arr_cmplx_field == 0] = 1
                grad_ang /= np.abs(grad_ang)
                grad_ang[arr_cmplx_field == 0] = 0
                return grad_amp * grad_ang
            if arr_meas.dtype == np.complex128:
                return 2 * (arr_cmplx_field - arr_meas) / arr_cmplx_field.size
            self.fail(f"bad meas type {arr_meas.dtype}")

        def _do_test_1_dir(meas):
            cmplx_field = self.beam.forward.get()
            meas_gpu = gpuarray.to_gpu(meas)
            with self.beam.track():
                loss = self.beam.mse_loss(meas_gpu)
            if meas.dtype == np.complex128:
                diff = np.abs(cmplx_field - meas)
            elif meas.dtype == np.float64:
                diff = np.abs(cmplx_field) - meas
            else:
                self.fail(f"bad meas type {meas.dtype}")
            self.assertAlmostEqual(loss, np.sum(diff ** 2) / meas.size)
            grad_ = self.beam.tape.collect_gradient(['change:u1_in'])['change:u1_in']
            self.assertEqual(len(grad_), 1)
            ufg_ = grad_[0].get()
            self.assertArrayEqual(ufg_, _gradient_cpu(cmplx_field, meas), DELTA)

        def _do_test_2_dir(meas):
            meas_gpu = gpuarray.to_gpu(meas)
            with self.beam.track():
                self.beam.merge_prop()
                self.beam *= 2
                loss = self.beam.mse_loss(meas_gpu)
            cmplx_field = self.beam.forward.get()
            if meas.dtype == np.complex128:
                diff = np.abs(cmplx_field - meas)
            elif meas.dtype == np.float64:
                diff = np.abs(cmplx_field) - meas
            else:
                self.fail(f"bad meas type {meas.dtype}")
            self.assertAlmostEqual(loss, np.sum(diff ** 2) / meas.size)
            grad = self.beam.tape.collect_gradient(['mse_fb:uf', 'mse_fb:ub'])
            self.assertEqual(len(grad['mse_fb:uf']), 1)
            ufg = grad['mse_fb:uf'][0].get()
            self.assertArrayEqual(ufg, _gradient_cpu(cmplx_field, meas), DELTA)
            self.assertEqual(len(grad['mse_fb:ub']), 1)
            ubg = grad['mse_fb:ub'][0].get()
            self.assertArrayEqual(ubg, np.zeros_like(ubg), DELTA)

        # make forward only beam
        arr_beam_f = self.random_like(self.beam.forward)
        arr_beam_f[1, 2:4] = 0
        self.beam.forward = arr_beam_f

        # test1: real forward measurement
        arr_meas_f_re = self.random_like(self.beam.forward, np.float64)
        arr_meas_f_re[1, 3:5] = 0
        _do_test_1_dir(arr_meas_f_re)

        # test2: complex forward measurement
        arr_meas_f_cmp = self.random_like(self.beam.forward)
        arr_meas_f_cmp[1, 3:5] = 0
        _do_test_1_dir(arr_meas_f_cmp)

        # test3: bi-dir beam with single dir measurement
        arr_beam_b = self.random_like(self.beam.forward)
        self.beam.backward = arr_beam_b

        # forward-only measurement: real & complex
        _do_test_2_dir(arr_meas_f_re)
        _do_test_2_dir(arr_meas_f_cmp)

        # TODO: test4: bi-dir measurement

    def test___imul__(self):
        DELTA = 1e-12
        beam = self.beam
        arr = self.random_like(beam.forward)
        # number
        self.beam.forward = arr
        mul_number = np.complex128(1.23 + 4.56j)
        beam *= mul_number
        self.assertArrayEqual(beam.forward.get(), arr * mul_number, DELTA)
        # real array
        self.beam.forward = arr
        mul_real = self.random_like(beam.forward, np.float64)
        mul_gpu_real = gpuarray.to_gpu(mul_real)
        self.beam *= mul_gpu_real
        self.assertArrayEqual(beam.forward.get(), arr * mul_real, DELTA)
        # complex array & memory not change
        self.beam.forward = arr
        fwd = self.beam.forward
        self.beam *= fwd
        self.assertArrayEqual(beam.forward.get(), arr ** 2, DELTA)
        self.assertIs(fwd, self.beam.forward)
        # reject numpy array
        with self.assertRaisesRegex(TypeError, "is not a .*GPUArray"):
            self.beam *= arr

    #
    # def test_a_mul(self):
    #     self.fail()

    def test_conj(self):
        beam = self.beam
        arr = self.random_like(beam.forward)
        beam.forward = arr

        # Basic
        beam.conj()
        self.assertArrayEqual(beam.forward.get(), arr.conj())
        beam.conj()
        self.assertArrayEqual(beam.forward.get(), arr)

        # Reject bi-dir arr and do nothing
        arr2 = self.random_like(beam.forward)
        beam.backward = arr2
        with self.assertRaises(NotImplementedError):
            beam.conj()
        self.assertArrayEqual(beam.forward.get(), arr)
        self.assertArrayEqual(beam.backward.get(), arr2)
        beam.backward = None

        # Test track
        beam.forward = arr
        with beam.track():
            beam.conj()
            self.assertArrayEqual(beam.forward.get(), arr.conj())
            other = beam.array_pool.get()
            other.set(arr2)
            beam.mse_loss(other)
        fwd_grad = beam.tape.collect_gradient(["u1_in"])["u1_in"]
        self.assertEqual(len(fwd_grad), 1)
        grad_cpu = fwd_grad[0].get()
        self.assertArrayEqual(grad_cpu, (arr - arr2.conj()) / arr.size * 2, 1e-12)
        # note: f(z) = abs(conj(z) - z2) ** 2 / size = abs(z - conj(z2)) ** 2 / size
        #   G(f, z) = 2 (z - conj(z2)) / size k
