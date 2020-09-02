"""
SSNP

=====

Features

* Forward model calculation based on CUDA
* Common type of file read/write
* Gradient calculation (working)
* Image reconstruction (working)
"""

import sys

if sys.version_info.major < 3:
    raise ImportError("SSNP package only support python3")

import warnings

if sys.version_info < (3, 6):
    warnings.warn("Untested python version. Please use python>=3.6")

VERSION = '0.0.1rc4'
import pycuda
import pycuda.autoinit
from ssnp.data import read, write
from ssnp.beam import BeamArray
from ssnp.utils import Multipliers, config
