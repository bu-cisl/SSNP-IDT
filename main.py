# from const import *
from data_io import *
import numpy as np
from time import time
from tf_func import nature_d, ssnp_step, bpm_step, n_ball


class SizeWarning(RuntimeWarning):
    pass


def main():
    # n = tf.constant(0.8, shape=(150, 256, 256), dtype=DATA_TYPE)
    n = n_ball(0.01, 5, z_empty=(3, 3))
    img_in = tiff_import("tilt45.tiff", (0, np.pi/4))
    # {0: 40/nz/350, 45: 30/nz/250, 60: resz0.1/30/nz/400}
    u_d = nature_d(img_in)
    step_list = [img_in]
    t = time()
    for i in range(30):
        img_in, u_d = ssnp_step(img_in, u_d, 1)
        step_list.append(img_in)
    for n_zi in n:
        img_in, u_d = ssnp_step(img_in, u_d, 1, n_zi)
        step_list.append(img_in)
    for i in range(250):
        img_in, u_d = ssnp_step(img_in, u_d, 1)
        step_list.append(img_in)
    # for i in range(30):
    #     img_in = bpm_step(img_in, 1)
    #     step_list.append(img_in)
    # for n_zi in n:
    #     img_in = bpm_step(img_in, 1, n_zi)
    #     step_list.append(img_in)
    # for i in range(250):
    #     img_in = bpm_step(img_in, 1)
    #     step_list.append(img_in)
    print(time() - t)

    # tiff_export("re.tiff", step_list, pre_operator=lambda x: np.real(x)/4+0.5, scale=1)
    # tiff_export("im.tiff", step_list, pre_operator=lambda x: np.imag(x)/4+0.5, scale=1)
    tiff_export("u.tiff", step_list, pre_operator=lambda x: np.abs(x), scale=0.5, dtype=np.uint16)
    return step_list


if __name__ == '__main__':
    img = main()
