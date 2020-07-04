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

VERSION = '0.0.1rc3'
import pycuda
import pycuda.autoinit
from .data import read, write
from .calc import ssnp_step, bpm_step, tilt, pure_forward_d, merge_prop, split_prop
