from pycuda.gpuarray import GPUArray


def tilt(img: GPUArray, c_ab: tuple, *, trunc: bool = False, copy: bool = False): ...


def param_check(**kwargs): ...
