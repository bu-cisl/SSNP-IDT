import os
import platform

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
if platform.system() == 'Windows':
    os.environ['PATH'] = 'C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v10.1\\bin;' + os.environ['PATH']

import tensorflow as tf

N0 = 1.
PERIODIC_BOUNDARY = True
# RES = (0.315534, 0.315534, 0.236686)
RES = (0.25, 0.25, 0.1)
"""PixelSize = RES * FreeSpaceWavelength"""
SIZE = (256, 256, 0)  # better to be even numbers
"""3D (only x,y is used?) volume size (pixels)"""
DATA_TYPE = tf.complex128
"""Default complex tensorflow datatype for computation"""
EPS = 1E-6
"""Small number avoid divided by zero"""


DATA_TYPE_R = {tf.complex128: tf.float64, tf.complex64: tf.float32}[DATA_TYPE]

# DATA_TYPE_R = tf.float32

# K0 = 2 * math.pi / WAVELENGTH
# N0 = 1
