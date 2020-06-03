import os
import platform
from time import time
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
if platform.system() == 'Windows':
    os.environ['PATH'] = 'C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v10.1\\bin;' + os.environ['PATH']

# from data_io import *
import tensorflow as tf
from tf_func import pure_forward_d, ssnp_step, bpm_step, n_ball, binary_pupil, n_grating, split_forward
import ssnp_pkg

NA = 0.65


@tf.function
def model(n, u):
    u_d = pure_forward_d(u)
    n = tf.maximum(n, 0)
    n = ssnp_pkg.tftool.real_to_complex(n)
    for ni in n:
        u, u_d = ssnp_step(u, u_d, 1, ni)
    u, u_d = ssnp_step(u, u_d, -len(n) / 2)
    u = split_forward(u, u_d)
    u = tf.abs(binary_pupil(u, 0.65001))
    return u


def main():
    n = ssnp_pkg.read('ball.tiff', tf.float64)
    n = tf.multiply(n, 0.01)
    # n = ssnp_pkg.read('rec.tiff', tf.float64)
    # n = n - 0.5
    out_list = tf.TensorArray(tf.float64, size=0, dynamic_size=True)
    for num in range(8):
        xy_theta = num / 8 * 2 * np.pi
        c_ab = NA * np.cos(xy_theta), NA * np.sin(xy_theta)
        img_in = ssnp_pkg.read("plane", tf.complex128, shape=n.shape[1:])
        img_in, c_ab_trunc = ssnp_pkg.tilt(img_in, c_ab, trunc=True)
        out = model(n, img_in)
        out_list = out_list.write(num, out)
    out_list = out_list.stack()

    # step_list.append(pupil(img_in, 0.66))
    # print(time() - t)

    ssnp_pkg.write("measurements.tiff", out_list, scale=0.5)
    return out_list


@tf.function
def model_loss(n, u, u0):
    loss = tf.zeros((), tf.float64)
    for i in range(8):
        loss = loss + tf.reduce_sum(tf.square(tf.abs(u0[i] - model(n, u[i]))))
    return loss


def rec():
    n = tf.zeros_like(ssnp_pkg.read('3beads.tiff', tf.float64))
    # n = ssnp_pkg.read('3beads.tiff', tf.complex128)
    n = tf.multiply(n, 0.01)
    u0 = ssnp_pkg.read("measurements.tiff", tf.float64)
    u0 = tf.multiply(u0, 2)
    in_list = []
    for num in range(8):
        xy_theta = num / 8 * 2 * np.pi
        c_ab = NA * np.cos(xy_theta), NA * np.sin(xy_theta)
        img_in = ssnp_pkg.read("plane", tf.complex128, shape=u0.shape[1:])
        img_in, _ = ssnp_pkg.tilt(img_in, c_ab, trunc=True)
        in_list.append(img_in)
    in_list = tf.stack(in_list)
    for it in range(10):
        print(model_loss(n, in_list, u0))
        with tf.GradientTape() as g:
            g.watch(n)
            loss = model_loss(n, in_list, u0)
        n = n - g.gradient(loss, n) * 0.003

    ssnp_pkg.write("rec.tiff", n, pre_operator=lambda x: x + 0.5)
    return n


if __name__ == '__main__':
    img = rec()
