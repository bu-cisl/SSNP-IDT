import numpy as np
import os
from time import perf_counter as time
import platform
import pycuda.compiler
t = time()

if platform.system() == 'Windows':
    # eliminate "non-UTF8 char" warnings
    pycuda.compiler.DEFAULT_NVCC_FLAGS = ['-Xcompiler', '/wd 4819']
    # remove code below if you have valid C compiler in `PATH` already
    import glob
    CL_PATH = max(glob.glob(r"C:\Program Files (x86)\Microsoft Visual Studio"
                            r"\*\*\VC\Tools\MSVC\*\bin\Hostx64\x64\cl.exe"))
    os.environ['PATH'] += ";" + CL_PATH[:-7]

import ssnp

ssnp.config.res = (0.1, 0.1, 0.1)
n = ssnp.read("bb.tiff", np.double)
n *= 0.01
NA = 0.65
steps = []
u = ssnp.read("plane", np.complex128, shape=n.shape[1:])
beam = ssnp.BeamArray(u)

for num in range(8):
    xy_theta = num / 8 * 2 * np.pi
    c_ab = NA * np.cos(xy_theta), NA * np.sin(xy_theta)
    ui = u * beam.multiplier.tilt(c_ab, trunc=True, gpu=True)
    beam.forward = ui
    beam.backward = 0
    beam.ssnp(1, n)
    beam.ssnp(-len(n) / 2)
    beam.backward = None
    beam.binary_pupil(1.0001 * NA)
    steps.append(beam.forward.get())
print(time() - t)

ssnp.write("cudatest.tiff", np.stack(steps), scale=0.5, pre_operator=lambda x: np.abs(x))
