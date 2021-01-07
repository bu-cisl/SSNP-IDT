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
from ssnp.utils.fista import prox_tv, get_q, tv_cost

ssnp.config.res = (0.1, 0.1, 0.1)
ZERO = np.zeros((), dtype=np.double)
s = ssnp.read("bb.tiff", dtype=np.double)
s *= 0.0
# s.fill(ZERO)
x_1 = s.copy()
grad = gpuarray.empty_like(s)
sum_grad = gpuarray.empty_like(s)

NA = 0.65
gamma = 0.001
tau = 2.

u_list = []
u_plane = ssnp.read("plane", np.complex128, shape=s.shape[1:])
beam = BeamArray(u_plane, total_ops=len(s))
pupil = beam.multiplier.binary_pupil(0.6501, gpu=True)

for num in range(8):
    xy_theta = num / 8 * 2 * np.pi
    c_ab = NA * np.cos(xy_theta), NA * np.sin(xy_theta)
    u = u_plane * beam.multiplier.tilt(c_ab, trunc=True, gpu=True)
    u_list.append(u)
mea = ssnp.read("meabb.tiff", np.double)
mea *= 2

t = time()

for step in range(15):
    print(f"Step: {step}")

    sum_grad.fill(ZERO)
    for num in range(8):
        beam.forward = u_list[num]
        beam.backward = 0
        with beam.track():
            beam.ssnp(1, s)
            beam.ssnp(-len(s) / 2)
            beam.a_mul(pupil)
            # beam.relation = BeamArray.BACKWARD
            loss = beam.forward_mse_loss(mea[num])
        print(f"dir {num}, loss = {loss}")
        sum_grad += beam.n_grad(grad)
    sum_grad *= gamma
    s -= sum_grad
    s = gpuarray.maximum(s, 0, out=s)
    if step % 3 != 2:
        continue

    z = s
    x = grad  # only memory assignment
    tv_loss_ori = tv_loss_new = 0

    for i in range(len(z)):
        tv_loss_ori += tv_cost(z[i])
        x[i].set(prox_tv(z[i], gamma * tau))
        # x[i].set(z[i])
        tv_loss_new += tv_cost(x[i])
    print(f"TV loss: prev = {tv_loss_ori}, new = {tv_loss_new}")
    s.set(x)
    s -= x_1
    s *= (get_q(step // 3) - 1) / get_q(step // 3 + 1)
    s += x
    x_1.set(x)
print(time() - t)

ssnp.write("ssnp_fista_recbb.tiff", x_1, scale=1, pre_operator=lambda x: x)
