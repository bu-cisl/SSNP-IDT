from pycuda import gpuarray


def ssnp_step(u: gpuarray, u_d: gpuarray, dz: float, n: gpuarray = None) -> tuple: ...
