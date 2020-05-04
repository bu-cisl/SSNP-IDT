# from const import *
from data_io import *
import numpy as np
from time import time
from tf_func import nature_d, ssnp_step, bpm_step, n_ball, binary_pupil


class SizeWarning(RuntimeWarning):
    pass


def main():
    # n = tf.constant(0.05, shape=(30, 256, 256), dtype=DATA_TYPE)
    n = n_ball(0.01, 5, z_empty=(3, 3))
    img_in, _ = tiff_illumination("plane", (0.65, 0))
    # {0: 40/nz/350, 45: 75/nz/250, 60: resz0.1/30/nz/200}
    # n = tiff_import("n_idt.tiff")
    u_d = nature_d(img_in)
    step_list = tf.TensorArray(DATA_TYPE, size=0, dynamic_size=True)
    step_idx = 0
    step_list = step_list.write(step_idx, img_in)

    t = time()
    for i in range(30):
        img_in, u_d = ssnp_step(img_in, u_d, 1)
        step_idx += 1
        step_list = step_list.write(step_idx, img_in)
    for n_zi in n:
        img_in, u_d = ssnp_step(img_in, u_d, 1, n_zi)
        step_idx += 1
        step_list = step_list.write(step_idx, img_in)
    for i in range(50):
        img_in, u_d = ssnp_step(img_in, u_d, 1)
        step_idx += 1
        step_list = step_list.write(step_idx, img_in)
    img_in = binary_pupil(img_in, 0.66)
    step_idx += 1
    step_list = step_list.write(step_idx, img_in)
    step_list = step_list.stack()
    # for i in range(30):
    #     img_in = bpm_step(img_in, 1)
    #     step_list.append(img_in)
    # for n_zi in n:
    #     img_in = bpm_step(img_in, 1, n_zi)
    #     step_list.append(img_in)
    # for i in range(250):
    #     img_in = bpm_step(img_in, 1)
    #     step_list.append(img_in)
    # step_list.append(pupil(img_in, 0.66))
    print(time() - t)

    # tiff_export("re.tiff", step_list, pre_operator=lambda x: np.real(x)/4+0.5, scale=1)
    # tiff_export("im.tiff", step_list, pre_operator=lambda x: np.imag(x)/4+0.5, scale=1)
    tiff_export("u.tiff", step_list, pre_operator=lambda x: np.abs(x), scale=0.5, dtype=np.uint16)
    return step_list


if __name__ == '__main__':
    img = main()
