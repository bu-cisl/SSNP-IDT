import env_init  # This module is in current dir
import numpy as np
from time import perf_counter as time

t = time()

import ssnp

ANGLE_NUM = 8
ssnp.config.res = (0.1, 0.1, 0.1)
n = ssnp.read("bb.tiff", np.double, gpu=True)
n *= 0.01
NA = 0.65
u = ssnp.read("plane", np.complex128, shape=n.shape[1:], gpu=False)
u = np.broadcast_to(u, (ANGLE_NUM, *u.shape)).copy()
beam = ssnp.BeamArray(u)

for num in range(ANGLE_NUM):
    xy_theta = num / ANGLE_NUM * 2 * np.pi
    c_ab = NA * np.cos(xy_theta), NA * np.sin(xy_theta)
    beam.forward[num] *= beam.multiplier.tilt(c_ab, trunc=True, gpu=True)

beam.backward = 0
beam.ssnp(1, n)
beam.ssnp(-len(n) / 2)
beam.backward = None
beam.binary_pupil(1.0001 * NA)

measurements = beam.forward.get()
print(time() - t)

ssnp.write("cudatest.tiff", measurements, scale=0.5, pre_operator=lambda x: np.abs(x))
