from const import *
from data_io import *
import numpy as np
from time import time
from tf_func import nature_d, step, n_ball


def main():
    import tifffile
    # u: Tensor = tf.constant(np.random.rand(1024, 1024), dtype=tf.complex64)
    # u_d = tf.constant(np.zeros((1024, 1024)), dtype=tf.complex64)
    # n = tf.constant(np.zeros((26, 512, 512)), dtype=tf.complex64)
    # n = tf.constant(0, shape=(400, 512, 512), dtype=DATA_TYPE)
    n = n_ball(0.01, 10, z_empty=(5, 300))
    img_in = tifffile.imread("a.tiff")
    img_in = tf.constant(img_in, dtype=tf.complex64)
    # img_in = tf.constant(np.ones((512, 512)), dtype=tf.complex64)
    u_d = nature_d(img_in)
    step_list = [img_in]
    t = time()
    for n_zi in n:
        img_in, u_d = step(img_in, u_d, n_zi, 1)
        step_list.append(img_in)
    print(time() - t)

    tiff_write("b.tiff", step_list, pre_operator=lambda x: np.abs(x))


    return step_list


if __name__ == '__main__':
    img = main()
