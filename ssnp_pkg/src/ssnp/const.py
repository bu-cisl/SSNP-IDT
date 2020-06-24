import tensorflow as tf

VERSION = '0.0.1rc2'
EPS = 1E-6
"""Small number avoid divided by zero"""
COMPLEX_TYPE = tf.complex128
"""Default complex tensorflow datatype for computation"""
REAL_TYPE = {tf.complex128: tf.float64, tf.complex64: tf.float32}[COMPLEX_TYPE]
"""Default real tensorflow datatype for computation"""
