from unittest import TestCase
import warnings

import numpy as np

from ssnp.utils import Config


class TestConfig(TestCase):
    def test_default(self):
        config = Config()
        self.assertListEqual(config._callbacks, [])
        with self.assertRaisesRegex(AttributeError, "res is uninitialized"):
            print(config.res)
        with self.assertRaisesRegex(AttributeError, "xyz pixel sizes are uninitialized"):
            print(config.xyz)
        with self.assertRaisesRegex(AttributeError, "wavelength lambda0 is uninitialized"):
            print(config.lambda0)
        self.assertEqual(config.n0, 1.0)

    def test_setter(self):
        config = Config()
        # normal assignment
        config.xyz = (3, 4, 5)
        self.assertTupleEqual(config.xyz, (3, 4, 5))
        for pix in config.xyz:
            self.assertIsInstance(pix, float)
        # bad assignment
        with self.assertRaises(AssertionError):
            config.xyz = (1, 2)
        with self.assertRaisesRegex(ValueError, 'could not convert string to float'):
            config.xyz = 'abc'
        with self.assertRaisesRegex(TypeError, "not iterable"):
            config.xyz = 1
        # do not change value if assignment fails
        self.assertTupleEqual(config.xyz, (3, 4, 5))
        # accept iterable
        for new_value in (range(4, 7), np.arange(4, 7), iter([4, 5, 6])):
            config.xyz = new_value
            self.assertTupleEqual(config.xyz, (4, 5, 6))
            config.xyz = (3, 4, 5)
