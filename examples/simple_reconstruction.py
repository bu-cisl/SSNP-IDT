import env_init  # This module is in current dir
import numpy as np
from time import perf_counter as time
from pycuda import gpuarray

import ssnp
from ssnp import BeamArray

ssnp.config.res = (0.1, 0.1, 0.1)
n = ssnp.read("sample.tiff", dtype=np.double, gpu=True)
n *= 0.
ng = gpuarray.empty_like(n)
ng_total = gpuarray.empty_like(n)
ANGLE_NUM = 8
NA = 0.65

u_list = []
u_plane = ssnp.read("plane", np.complex128, shape=n.shape[1:], gpu=True)
beam = BeamArray(u_plane, total_ops=len(n))
pupil = beam.multiplier.binary_pupil(0.6501, gpu=True)

for num in range(ANGLE_NUM):
    xy_theta = num / ANGLE_NUM * 2 * np.pi
    c_ab = NA * np.cos(xy_theta), NA * np.sin(xy_theta)
    u = u_plane * beam.multiplier.tilt(c_ab, trunc=True, gpu=True)
    u_list.append(u)
mea = ssnp.read("meas_sim.tiff", np.double, gpu=True)
mea *= 2

t = time()
for step in range(5):
    print(f"Step: {step}")
    ng_total *= 0
    for num in range(ANGLE_NUM):
        beam.forward = u_list[num]
        beam.backward = 0
        beam.merge_prop()
        with beam.track():
            beam.ssnp(1, n)
            beam.ssnp(-len(n) / 2)
            beam.a_mul(pupil)
            loss = beam.forward_mse_loss(mea[num])
        print(f"dir {num}, {loss = :f}")
        beam.n_grad(ng)
        ng_total += ng

    ng_total *= 350
    n -= ng_total
    n = gpuarray.maximum(n, 0, out=n)
print(time() - t)

ssnp.write("ssnp_rec.tiff", n)
