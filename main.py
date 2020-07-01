import numpy as np
import os
from time import time
import platform
import pycuda.compiler

if platform.system() == 'Windows':
    os.environ['PATH'] += r";C:\Program Files (x86)\Microsoft Visual Studio\2019\Community" \
                          r"\VC\Tools\MSVC\14.25.28610\bin\Hostx64\x64"
    pycuda.compiler.DEFAULT_NVCC_FLAGS = ['-Xcompiler', '/wd 4819']

import ssnp

n = ssnp.read("3ball.tiff")
n *= 0.01
NA = 0.65
steps = []
t = time()
# u = ssnp.read("plane", np.complex128, shape=n.shape[1:])
for num in range(8):
    xy_theta = num / 8 * 2 * np.pi
    c_ab = NA * np.cos(xy_theta), NA * np.sin(xy_theta)
    u = ssnp.read("plane", np.complex128, shape=n.shape[1:])
    u, c_ab_trunc = ssnp.tilt(u, c_ab, trunc=True)
    u_d = ssnp.pure_forward_d(u)

    for i in range(len(n)):
        ssnp.ssnp_step(u, u_d, 1, n[i])
    ssnp.ssnp_step(u, u_d, -len(n) / 2)
    ssnp.split_prop(u, u_d)
    ssnp.binary_pupil(u, 0.6501)
    steps.append(u.get())
print(time() - t)

ssnp.write("cudatest.tiff", np.stack(steps), scale=0.5, pre_operator=lambda x: np.abs(x))
