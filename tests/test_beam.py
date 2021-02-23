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
    # remove code below if you have valid C compiler in `PATH` already
    import glob

    CL_PATH = max(glob.glob(r"C:\Program Files (x86)\Microsoft Visual Studio"
                            r"\*\*\VC\Tools\MSVC\*\bin\Hostx64\x64\cl.exe"))
    os.environ['PATH'] += ";" + CL_PATH[:-7]

from ssnp import BeamArray
import ssnp

ssnp.config.set(xyz=(0.1, 0.2, 0.3), lambda0=0.632)


class TestBeamArraySingle(TestCase):
    def setUp(self) -> None:
        u = ssnp.read("plane", dtype=np.complex128, shape=(128, 128))
        self.beam = BeamArray(u)
        self.rng = np.random.default_rng()

    def assertArrayEqual(self, a, b, delta=0.):
        # currently don't want to directly calculate GPUArray with numpy function
        self.assertIsInstance(a, np.ndarray)
        self.assertIsInstance(b, np.ndarray)
        self.assertEqual(a.shape, b.shape)
        self.assertLessEqual(np.linalg.norm(a - b), delta)

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
        rand = lambda: self.rng.random(self.beam.forward.shape, np.float64)

        # forward only beam
        arr_beam_f = rand() + 1j * rand()
        self.beam.forward = arr_beam_f
        # real forward
        arr_meas_f = rand()
        forward = gpuarray.to_gpu(arr_meas_f)
        with self.beam.track():
            loss = self.beam.mse_loss(forward)
        self.assertAlmostEqual(loss, np.linalg.norm(np.abs(arr_beam_f) - arr_meas_f) ** 2)
        grad = self.beam.tape.collect_gradient(['uf', 'ub'])
        ufg = np.squeeze(np.stack([uf.get() for uf in grad['uf']]))
        self.assertArrayEqual(ufg, 2 * (np.abs(arr_beam_f) - arr_meas_f) * arr_beam_f / np.abs(arr_beam_f), 1e-7)
        self.assertEqual(len(grad['ub']), 0)

        # complex forward
        arr_meas_f = rand() + 1j * rand()
        forward = gpuarray.to_gpu(arr_meas_f)
        with self.beam.track():
            loss = self.beam.mse_loss(forward)
        self.assertAlmostEqual(loss, np.linalg.norm(arr_beam_f - arr_meas_f) ** 2)
        grad = self.beam.tape.collect_gradient(['uf'])
        ufg = np.squeeze(np.stack([uf.get() for uf in grad['uf']]))
        self.assertArrayEqual(ufg, 2 * (arr_beam_f - arr_meas_f), 1e-7)

        # bi-dir beam
        arr_beam_b = rand() + 1j * rand()
        self.beam.backward = arr_beam_b
        self.beam.merge_prop()
        # complex forward
        arr_meas_f = rand() + 1j * rand()
        forward = gpuarray.to_gpu(arr_meas_f)
        with self.beam.track():
            loss = self.beam.mse_loss(forward)
            self.assertAlmostEqual(loss, np.linalg.norm(arr_beam_f - arr_meas_f) ** 2)
        grad = self.beam.tape.collect_gradient(['uf'])
        ufg = np.squeeze(np.stack([uf.get() for uf in grad['uf']]))
        self.assertArrayEqual(ufg, 2 * (arr_beam_f - arr_meas_f), 1e-7)

    def test___imul__(self):
        def assert_minmax(arr, x):
            self.assertAlmostEqual(np.max(arr), x)
            self.assertAlmostEqual(np.min(arr), x)

        fwd = self.beam.forward
        # number
        self.beam *= 1.5j
        assert_minmax(fwd.get(), 1.5j)
        # real array
        multiplier = gpuarray.empty_like(fwd, dtype=np.float64)
        multiplier.fill(2)
        self.beam *= multiplier
        assert_minmax(fwd.get(), 3j)
        # complex array & memory not change
        self.beam *= fwd
        assert_minmax(fwd.get(), -9)
        self.assertIs(fwd, self.beam.forward)
        # reject numpy array
        np_arr = np.empty_like(fwd.get())
        with self.assertRaisesRegex(TypeError, "'.*' is not a GPUArray"):
            self.beam *= np_arr

    #
    # def test_a_mul(self):
    #     self.fail()
