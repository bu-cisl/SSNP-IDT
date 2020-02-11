import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ['PATH'] = 'C:\\Users\\zjb\\AppData\\Roaming\\Mathematica\\Paclets\\Repository\\' \
                     'CUDAResources-Win64-12.0.359\\CUDAToolkit\\bin;' + os.environ['PATH']
import tensorflow as tf

N0 = 1.
RES = (0.5, 0.5, 0.1)  # pixel_size = RES * lambda0
SIZE = (256, 256, 0)  # better to be even numbers
DATA_TYPE = tf.complex64
EPS = 1E-10
# DATA_TYPE_R = tf.float32

# K0 = 2 * math.pi / WAVELENGTH
# N0 = 1
