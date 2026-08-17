from unittest import TestCase
import warnings

import numpy as np

from ssnp.utils.utils import Config
from ssnp.utils.array_pool import ManagedObj, ManagedArrayPool


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
        for attr in ['xyz', 'res']:
            for bad_val in [[1,2], (3, 4, 5, 6)]:
                with self.assertRaisesRegex(ValueError, "can only be assigned with an iterable of 3 floats"):
                    setattr(config, attr, bad_val)
        with self.assertRaisesRegex(ValueError, "could not convert string to float"):
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


class MockArray:
    id_counter = 0
    freed_obj = 0
    data_used = False

    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype
        self.unique_id = self.get_unique_id()
        self.data_used = False

    def use(self):
        self.data_used = True

    @classmethod
    def get_unique_id(cls):
        unique_id = cls.id_counter
        cls.id_counter += 1
        return unique_id

    @classmethod
    def bump_freed(cls):
        cls.freed_obj += 1

    @classmethod
    def clear_counters(cls):
        cls.id_counter = cls.freed_obj = 0

    @classmethod
    def empty_like(cls, other):
        return cls(other.shape, other.dtype)

    def method_return_self(self):
        return self

    def __iadd__(self, _):
        return self

    def __del__(self):
        self.bump_freed()

    @property
    def self(self):
        return self


class TestManagedArrayPool(TestCase):
    def test_proxy(self):
        def helper(wrapper, expected):
            def type_assertion(i):
                self.assertIsInstance(a, MockArray)
                if expected[i]:
                    self.assertIsInstance(a, ManagedObj)
                else:
                    self.assertNotIsInstance(a, ManagedObj)

            array_pool = ManagedArrayPool(
                arr_proto, MockArray.empty_like, unique_id=lambda arr: arr.unique_id,
                method_wrapper=wrapper
            )
            # 0 - basic
            a = array_pool.get()
            type_assertion(0)
            # 1 - i-operator
            a += 1
            type_assertion(1)
            # 2 - explicit i-operator
            a = array_pool.get().__iadd__(2)
            type_assertion(2)
            # 3 - dynamic callable attr
            a = array_pool.get()
            a.dynamic_return_self = lambda x: x
            a = a.dynamic_return_self(a)
            type_assertion(3)
            # 4 - bound method
            a = array_pool.get().method_return_self().method_return_self().method_return_self()
            type_assertion(4)
            # 5 - getter property
            a = array_pool.get().self
            type_assertion(5)
            # 6 - class method (should NOT be proxied)
            a = array_pool.get()
            a = a.empty_like(a)
            type_assertion(6)

        arr_proto = MockArray([1024, 1024], float)
        helper(None, [True, True, True, True, False, False, False])
        helper('chainable', [True, True, True, True, True, False, False])
        helper('descriptor', [True, True, True, True, True, True, False])

    def test_pool(self):
        def clear_arr(arr: MockArray):
            arr.data_used = False

        arr_proto = MockArray([1024, 1024], float)
        MockArray.clear_counters()
        array_pool = ManagedArrayPool(arr_proto, MockArray.empty_like, unique_id=lambda arr: arr.unique_id,
                                      method_wrapper='chainable', recycle_hook=clear_arr)
        # Test basic get
        self.assertEqual(array_pool.allocated_count, 0)
        a = array_pool.get()
        a.use()
        self.assertEqual(array_pool.allocated_count, 1)
        self.assertIsInstance(a, MockArray)
        self.assertIsInstance(a, ManagedObj)

        # Test basic recycle
        arr_addr = id(a.__wrapped__)
        self.assertEqual(len(array_pool._pool), 0)
        a.dispose()
        self.assertEqual(len(array_pool._pool), 1)
        self.assertEqual(array_pool.allocated_count, 1)
        a = array_pool.get()
        self.assertFalse(a.data_used)
        for i in range(10):
            del a
            a = array_pool.get()
        self.assertEqual(len(array_pool._pool), 0)
        self.assertEqual(array_pool.allocated_count, 1)
        self.assertEqual(id(a.__wrapped__), arr_addr)
        self.assertIsInstance(a, MockArray)
        self.assertIsInstance(a, ManagedObj)
        del a
        a = [array_pool.get()]
        self.assertEqual(id(a[0].__wrapped__), arr_addr)
        del a
        self.assertEqual(len(array_pool._pool), 1)
        self.assertEqual(array_pool.allocated_count, 1)
        self.assertEqual(MockArray.id_counter, 1)
        self.assertEqual(MockArray.freed_obj, 0)
        a = [array_pool.get()]
        b = a[0]
        arr = a[0].extract()
        self.assertEqual(id(arr), arr_addr)
        self.assertIs(b.__wrapped__, ManagedObj.DISPOSED)
        self.assertEqual(MockArray.freed_obj, 0)
        del arr
        self.assertEqual(MockArray.freed_obj, 1)

        # More recycle
        a = array_pool.get()
        del a
        self.assertEqual(len(array_pool._pool), 1)
        self.assertEqual(array_pool.allocated_count, 2)
        self.assertEqual(MockArray.id_counter, 2)
        self.assertEqual(MockArray.freed_obj, 1)
        del array_pool
        self.assertEqual(MockArray.freed_obj, 2)
        array_pool = ManagedArrayPool(arr_proto, MockArray.empty_like, unique_id=lambda arr: arr.unique_id)
        a = array_pool.get()
        del array_pool
        self.assertEqual(MockArray.id_counter, 3)
        self.assertEqual(MockArray.freed_obj, 2)
        del a
        self.assertEqual(MockArray.freed_obj, 3)

        # Test manage
        raw_arr = MockArray([1024, 1024], float)
        array_pool = ManagedArrayPool(arr_proto, MockArray.empty_like, unique_id=lambda arr: arr.unique_id)
        a = array_pool.manage(raw_arr)
        self.assertIs(a.__wrapped__, raw_arr)
        del a
        a = array_pool.get()
        self.assertEqual(array_pool.allocated_count, 0)
        self.assertIs(a.__wrapped__, raw_arr)
        del a
        with self.assertRaisesRegex(ValueError, "duplicated"):
            array_pool.manage(raw_arr)
