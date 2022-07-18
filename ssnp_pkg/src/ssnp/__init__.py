"""
SSNP

=====

Features

* Forward model calculation based on TensorFlow 2.x
* Common type of file read/write
* Gradient calculation
* Image reconstruction (working)
"""

import sys

if sys.version_info.major < 3:
    raise ImportError("SSNP package only support python3")

import warnings

if sys.version_info < (3, 6):
    warnings.warn("Untested python version. Please use python>=3.6")

import pycuda
import pycuda.autoinit

from .data import read, write
# from .tftool import real_to_complex, tilt
from .const import VERSION
from .calc import ssnp_step
