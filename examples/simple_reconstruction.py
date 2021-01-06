import numpy as np
import os
from time import perf_counter as time
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
ng_total = gpuarray.empty_like(n)
ANGLE_NUM = 8
NA = 0.65

u_list = []
u_plane = ssnp.read("plane", np.complex128, shape=n.shape[1:])
beam = BeamArray(u_plane, total_ops=len(n))
pupil = beam.multiplier.binary_pupil(0.6501, gpu=True)

for num in range(ANGLE_NUM):
    xy_theta = num / ANGLE_NUM * 2 * np.pi
    c_ab = NA * np.cos(xy_theta), NA * np.sin(xy_theta)
    u = u_plane * beam.multiplier.tilt(c_ab, trunc=True, gpu=True)
    u_list.append(u)
mea = ssnp.read("meabb.tiff", np.double)
mea *= 2

t = time()
for step in range(5):
    print(f"Step: {step}")
    ng_total *= 0
    for num in range(ANGLE_NUM):
        beam.forward = u_list[num]
        beam.backward = 0
        beam.ssnp(1, n, track=True)
        beam.ssnp(-len(n) / 2, track=True)
        beam.a_mul(pupil, track=True)
        loss = beam.forward_mse_loss(mea[num], track=True)
        print(f"dir {num}, loss = {loss}")
        beam.n_grad(ng)
        ng *= 0.005
        ng_total += ng

    n -= ng_total
    n = gpuarray.maximum(n, 0, out=n)
print(time() - t)

ssnp.write("ssnp_recbb.tiff", n)
