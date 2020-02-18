from const import *
from data_io import *
import numpy as np
from time import time
from tf_func import nature_d, ssnp_step, n_ball


class SizeWarning(RuntimeWarning):
    pass


def main():
    import tifffile
    # u_d = tf.constant(np.zeros((1024, 1024)), dtype=tf.complex64)
    # n = tf.constant(np.zeros((26, 512, 512)), dtype=tf.complex64)
    n = tf.constant(0.8, shape=(150, 256, 256), dtype=DATA_TYPE)
    # n = n_ball(0.03, 9, z_empty=(3, 3))
    img_in = tiff_import("small.tiff", (0, np.pi/4))
    # img_in = tf.constant(np.ones((512, 512)), dtype=tf.complex64)
    u_d = nature_d(img_in)
    step_list = [img_in]
    t = time()
    for i in range(40):
        img_in, u_d = ssnp_step(img_in, u_d, 1)
        step_list.append(img_in)
    for n_zi in n:
        img_in, u_d = ssnp_step(img_in, u_d, 1, n_zi)
        step_list.append(img_in)
    for i in range(0):
        img_in, u_d = ssnp_step(img_in, u_d, 1)
        step_list.append(img_in)
    print(time() - t)

    tiff_export("re.tiff", step_list, pre_operator=lambda x: np.real(x)/4+0.5, scale=1)
    tiff_export("im.tiff", step_list, pre_operator=lambda x: np.imag(x)/4+0.5, scale=1)
    return step_list


if __name__ == '__main__':
    img = main()
