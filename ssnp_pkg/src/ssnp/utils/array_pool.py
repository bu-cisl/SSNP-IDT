import logging
from collections.abc import Callable
from typing import TypeVar, Any

from .utils import param_check

T = TypeVar('T')

class ArrayPool:
    def __init__(self, prototype, like_allocator: Callable[[T], T], unique_id: Callable[[T], Any] = id):
        self._pool = []
        self._id_record = set()
        self._get_id = unique_id
        self.prototype = prototype
        self.allocator = like_allocator
        self.allocated_count = 0

    def get(self):
        if self._pool:
            out = self._pool.pop()
            self._id_record.remove(self._get_id(out))
            return out
        else:
            self.allocated_count += 1
            logging.info(f"get array times: {self.allocated_count}")
            return self.allocator(self.prototype)

    def recycle(self, arr):
        # sanity check
        param_check(prototype=self.prototype, recycled=arr)
        if self.prototype.dtype != arr.dtype:
            raise ValueError(f"recycled array dtype {arr.dtype} is different from "
                             f"the prototype dtype {self.prototype.dtype}")
        if self._get_id(arr) in self._id_record:
            raise ValueError(f"duplicate array found in pool")
        # check passed
        self._pool.append(arr)
        self._id_record.add(self._get_id(arr))
