import logging
import weakref
import wrapt
from collections.abc import Callable
from typing import TypeVar, Any, Union, Literal
from wrapt import BaseObjectProxy

from .utils import param_check


class ArrayPool[T]:
    _pool: list[T]
    allocated_count: int

    def __init__(
        self, prototype: T, like_allocator: Callable[[T], T], unique_id: Callable[[T], Any] = id,
        recycle_hook = None
    ):
        self._pool = []
        self._id_record = set()
        self._get_id = unique_id
        self.prototype = prototype
        self.allocator = like_allocator
        self.allocated_count = 0
        self.recycle_hook = recycle_hook

    def get(self):
        if self._pool:
            out = self._pool.pop()
            self._id_record.remove(self._get_id(out))
            return out
        else:
            self.allocated_count += 1
            logging.info(f"get array times: {self.allocated_count}")
            return self.allocator(self.prototype)

    def _check(self, arr):
        param_check(prototype=self.prototype, recycled=arr, _expected_type=type(self.prototype))
        if self.prototype.dtype != arr.dtype:
            raise ValueError(f"recycled array dtype {arr.dtype} is different from "
                             f"the prototype dtype {self.prototype.dtype}")
        if self._get_id(arr) in self._id_record:
            raise ValueError(f"duplicated array found in pool")

    def recycle(self, arr):
        self._check(arr)
        if self.recycle_hook is not None:
            self.recycle_hook(arr)
        self._pool.append(arr)
        self._id_record.add(self._get_id(arr))


class ProxyReferenceError(RuntimeError):
    pass


class ManagedObj(BaseObjectProxy):
    class Disposed:
        def __raise_err(*_, **__):
            raise ProxyReferenceError(f"The underlying object of ManagedObj no longer exists")

        def __wrapped_setattr_fixups__(self):
            pass

        def __repr__(self):
            return "<disposed>"

        __getattr__ = __raise_err

    DISPOSED = Disposed()
    _self_finalizer: weakref.finalize

    def __init__(self, obj, pool: Union[ArrayPool, weakref.ReferenceType[ArrayPool]],
                 method_wrapper: Literal['chainable', 'descriptor', None] = None):
        def recycle_obj():
            new_obj = None
            try:
                # If pool is alive, re-wrap obj as a new proxy and let pool recycle it
                if (p := pool_ref()) is not None:
                    new_obj = cls(obj, p, method_wrapper)
                    p.recycle(new_obj)
            except Exception as e:
                # Leak the obj at failure to prevent infinite loop
                logging.error(f"recycling ManagedObj failed with error {e!r}")
                if new_obj is not None:
                    new_obj._self_finalizer.detach()
                raise

        super().__init__(obj)
        cls = type(self)
        pool_ref = pool if isinstance(pool, weakref.ReferenceType) else weakref.ref(pool)
        self._self_finalizer = weakref.finalize(self, recycle_obj)
        self._self_finalizer.atexit = False
        self._self_method_wrapper = method_wrapper

    def __getattr__(self, item):
        attr = super().__getattr__(item)
        match self._self_method_wrapper:
            case 'chainable':
                if callable(attr):
                    @wrapt.decorator
                    def intercept_return(wrapped, _, args, kwargs):
                        result = wrapped(*args, **kwargs)
                        if result is self.__wrapped__:
                            return self
                        return result

                    return intercept_return(attr)
            case 'descriptor':
                if (getter := getattr(getattr(type(self.__wrapped__), item, None), '__get__', None)) is not None:
                    return getter(self, type(self.__wrapped__))
            case None:
                pass
            case unknown:
                raise ValueError(f"Unknown method wrapper {unknown}")
        # fallback for all other cases
        return attr

    def dispose(self):
        self._self_finalizer()
        self.__wrapped__ = self.DISPOSED

    def extract(self):
        """
        Take the wrapped object out and invalidate the proxy instance
        :return: the wrapped object
        """
        obj = self.__wrapped__
        self.__wrapped__ = self.DISPOSED
        self._self_finalizer.detach()
        return obj

    def __repr__(self):
        return f"{type(self).__qualname__}({self.__wrapped__!r})"


class ManagedArrayPool(ArrayPool):
    def __init__(self, *args, method_wrapper: Literal['chainable', 'descriptor', None] = None, **kwargs):
        super().__init__(*args, **kwargs)
        raw_allocator = self.allocator
        self_wr = weakref.ref(self)
        self.method_wrapper = method_wrapper
        self.allocator = lambda x: ManagedObj(raw_allocator(x), self_wr, method_wrapper)

    def manage(self, raw_arr):
        if isinstance(raw_arr, ManagedObj):
            raise ValueError(f"object is already managed")
        self._check(raw_arr)
        return ManagedObj(raw_arr, self, self.method_wrapper)
