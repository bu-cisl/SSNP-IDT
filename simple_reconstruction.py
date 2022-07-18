import numpy as np
import os
from time import time
import platform
import pycuda.compiler
from pycuda import gpuarray

if platform.system() == 'Windows':
    os.environ['PATH'] += r";C:\Program Files (x86)\Microsoft Visual Studio\2019\Community" \
                          r"\VC\Tools\MSVC\14.25.28610\bin\Hostx64\x64"
    pycuda.compiler.DEFAULT_NVCC_FLAGS = ['-Xcompiler', '/wd 4819']

import ssnp
from ssnp import BeamArray

ssnp.config.res = (0.1, 0.1, 0.1)
n = ssnp.read("bb.tiff", dtype=np.double)
n *= 0.
ng = gpuarray.empty_like(n)

NA = 0.65

u_list = []
for num in range(8):
    u = ssnp.read("plane", np.complex128, shape=n.shape[1:])
    xy_theta = num / 8 * 2 * np.pi
    c_ab = NA * np.cos(xy_theta), NA * np.sin(xy_theta)
    u, c_ab_trunc = ssnp.tilt(u, c_ab, trunc=True)
    u_list.append(u)
beam = BeamArray(u_list[0])
mea = ssnp.read("meabb.tiff", np.double)
mea *= 2

t = time()
for step in range(3):
    print(f"Step: {step}")
    for num in range(8):
        beam.forward = u_list[num]
        beam.bpm(1, n, track=True)
        beam.bpm(-len(n) / 2, track=True)
        beam.binary_pupil(0.6501)
        loss = beam.forward_mse_loss(mea[num], track=True)
        print(f"dir {num}, loss = {loss}")
        beam.n_grad(ng)
        ng *= 0.001
        n -= ng
print(time() - t)

ssnp.write("bpm_recbb.tiff", n, scale=0.5, pre_operator=lambda x: np.abs(x))
