import numpy as np
import os
from time import perf_counter as time
import platform
import pycuda.compiler
t = time()

if platform.system() == 'Windows':
    os.environ['PATH'] += r";C:\Program Files (x86)\Microsoft Visual Studio\2019\Community" \
                          r"\VC\Tools\MSVC\14.25.28610\bin\Hostx64\x64"
    pycuda.compiler.DEFAULT_NVCC_FLAGS = ['-Xcompiler', '/wd 4819']

import ssnp
import ssnp.calc

ssnp.config.res = (0.1, 0.1, 0.1)
n = ssnp.read("bb.tiff", np.double)
n *= 0.01
NA = 0.65
u = ssnp.read("plane", np.complex128, shape=n.shape[1:], gpu=False)
u = np.broadcast_to(u, (8, *u.shape))
beam = ssnp.BeamArray(u)

for num in range(8):
    xy_theta = num / 8 * 2 * np.pi
    c_ab = NA * np.cos(xy_theta), NA * np.sin(xy_theta)
    beam.forward[num] *= beam.multiplier.tilt(c_ab, trunc=True, gpu=True)

beam.backward = 0
beam.ssnp(1, n)
beam.ssnp(-len(n) / 2)
beam.backward = None
fourier = ssnp.calc.get_funcs(beam.forward).fourier
with fourier(beam.forward) as f_beam:  # todo fix a_mul
    for fi in f_beam:
        fi *= beam.multiplier.binary_pupil(1.0001 * NA, gpu=True)

measurements = beam.forward.get()
print(time() - t)

ssnp.write("cudatest.tiff", measurements, scale=0.5, pre_operator=lambda x: np.abs(x))
