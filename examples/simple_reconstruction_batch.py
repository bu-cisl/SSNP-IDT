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
from ssnp import BeamArray, calc

ssnp.config.res = (0.1, 0.1, 0.1)
n = ssnp.read("bb.tiff", dtype=np.double)
n *= 0.
ng = gpuarray.empty_like(n)
# ng_total = gpuarray.empty_like(n)

NA = 0.65

u_list = []
u_plane = ssnp.read("plane", np.complex128, shape=n.shape[1:], gpu=False)
u_plane = np.broadcast_to(u_plane, (8, *u_plane.shape)).copy()
beam = BeamArray(u_plane, total_ops=len(n))
beam_in = BeamArray(u_plane, total_ops=len(n))

pupil = beam.multiplier.binary_pupil(0.6501, gpu=True)

for num in range(8):
    xy_theta = num / 8 * 2 * np.pi
    c_ab = NA * np.cos(xy_theta), NA * np.sin(xy_theta)
    beam_in.forward[num] *= beam.multiplier.tilt(c_ab, trunc=True, gpu=True)

mea = ssnp.read("meabb.tiff", np.double)
mea *= 2

t = time()
for step in range(5):
    print(f"Step: {step}")
    # ng_total *= 0
    beam.forward = beam_in.forward
    beam.backward = 0
    beam.ssnp(1, n, track=True)
    beam.ssnp(-len(n) / 2, track=True)
    beam.a_mul(pupil, track=True)
    loss = beam.forward_mse_loss(mea, track=True)
    print(f"{loss = :f}")
    loss = [calc.reduce_mse(beam.forward[i], mea[i]) for i in range(8)]
    print(f"loss detail: {', '.join([f'{i:6.1f}' for i in loss])}")
    beam.n_grad(ng)
    ng *= 0.005
    n -= ng
    n = gpuarray.maximum(n, 0, out=n)
print(time() - t)

ssnp.write("ssnp_recbb.tiff", n)