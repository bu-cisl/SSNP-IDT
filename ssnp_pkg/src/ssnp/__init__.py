"""
SSNP

=====

Features

* Forward model calculation based on CUDA
* Read/write data of common file type
* Gradient calculation
* Image reconstruction with regularization
"""

import sys

if sys.version_info.major < 3:
    raise ImportError("SSNP package only support python3")

import warnings

if sys.version_info < (3, 8):
    warnings.warn("Untested python version. Please use python>=3.8")

VERSION = '0.0.2'

import ssnp.utils.init_helper

from ssnp.beam import BeamArray
from ssnp.utils import Multipliers, config, read, write
