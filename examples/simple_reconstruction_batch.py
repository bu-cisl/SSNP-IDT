import env_init  # This module is in current dir
import numpy as np
from time import perf_counter as time
from pycuda import gpuarray

import ssnp
from ssnp import BeamArray, calc

n = ssnp.read("bb.tiff", dtype=np.double, gpu=True)
n.fill(0)  # zero initialization
ng = gpuarray.empty_like(n)

mea = ssnp.read("meabb.tiff", np.double, gpu=True)
mea *= 2  # restore data (divided by 2 before save)

ssnp.config.res = (0.1, 0.1, 0.1)
NA = 0.65
STEPS = 5
STEP_SIZE = 350
ANGLE_NUM = len(mea)

u_list = []
u_plane = ssnp.read("plane", np.complex128, shape=n.shape[1:], gpu=False)
u_plane = np.broadcast_to(u_plane, (ANGLE_NUM, *u_plane.shape)).copy()
beam = BeamArray(u_plane, total_ops=len(n))
beam_in = BeamArray(u_plane, total_ops=len(n))

pupil = beam.multiplier.binary_pupil(NA * 1.001, gpu=True)

for num in range(ANGLE_NUM):
    xy_theta = num / ANGLE_NUM * 2 * np.pi
    c_ab = NA * np.cos(xy_theta), NA * np.sin(xy_theta)
    beam_in.forward[num] *= beam.multiplier.tilt(c_ab, trunc=True, gpu=True)

t = time()
for step in range(STEPS):
    print(f"Step: {step}")
    beam.forward = beam_in.forward
    beam.backward = 0
    beam.merge_prop()
    with beam.track():
        beam.ssnp(1, n)
        beam.ssnp(-len(n) / 2)
        beam.a_mul(pupil)
        loss = beam.forward_mse_loss(mea)
    print(f"{loss = :f}")
    loss = [calc.reduce_mse(beam.forward[i], mi) for i, mi in enumerate(mea)]
    print(f"loss detail: {', '.join([f'{i:.2e}' for i in loss])}")
    beam.n_grad(ng)
    ng *= STEP_SIZE
    n -= ng
    n = gpuarray.maximum(n, 0, out=n)  # positive regularization
print(time() - t)

ssnp.write("ssnp_recbb.tiff", n)
