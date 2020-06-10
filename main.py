import os
import platform
from time import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
if platform.system() == 'Windows':
    os.environ['PATH'] = 'C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v10.1\\bin;' + os.environ['PATH']

import numpy as np
import tensorflow as tf
from tf_func import pure_forward_d, ssnp_step, binary_pupil, split_forward
import ssnp

NA = 0.65


# @tf.function
def model(n, u):
    u_d = pure_forward_d(u)
    n = tf.maximum(n, 0)
    n = ssnp.tftool.real_to_complex(n)
    # u, u_d = ssnp_step(u, u_d, 0.5)
    for ni in n:
        u, u_d = ssnp_step(u, u_d, 1, ni)
    u, u_d = ssnp_step(u, u_d, -len(n) / 2)
    u = split_forward(u, u_d)
    u = tf.abs(binary_pupil(u, 0.6501))
    return u


def main():
    n = ssnp.read('ball.tiff', tf.float64)
    n = tf.multiply(n, 0.01)
    # n = ssnp.read('rec.tiff', tf.float64)
    out_list = tf.TensorArray(tf.float64, size=0, dynamic_size=True)
    for num in range(8):
        xy_theta = num / 8 * 2 * np.pi
        c_ab = NA * np.cos(xy_theta), NA * np.sin(xy_theta)
        img_in = ssnp.read("plane", tf.complex128, shape=n.shape[1:])
        img_in, c_ab_trunc = ssnp.tilt(img_in, c_ab, trunc=True)
        out = model(n, img_in)
        out_list = out_list.write(num, out)
    out_list = out_list.stack()

    # step_list.append(pupil(img_in, 0.66))
    # print(time() - t)

    ssnp.write("mea3b.tiff", out_list, scale=0.5)
    return out_list


@tf.function
def model_loss(n, u, u0):
    loss = tf.zeros((), tf.float64)
    for i in range(len(u)):
        loss = loss + tf.reduce_sum(tf.square(tf.abs(u0[i] - model(n, u[i]))))
    return loss


@tf.function
def model_loss_single(n, u, u0):
    loss = tf.reduce_sum(tf.square(tf.abs(u0 - model(n, u))))
    return loss


@tf.function
def tv_loss(n):
    loss = (tf.reduce_sum(tf.abs(n[:, :, 1:] - n[:, :, :-1])) +
            tf.reduce_sum(tf.abs(n[:, 1:, :] - n[:, :-1, :])) +
            tf.reduce_sum(tf.abs(n[1:, :, :] - n[:-1, :, :])))
    return loss


def rec():
    n = tf.zeros_like(ssnp.read('ball.tiff', tf.float64))
    # n = ssnp.read('ball.tiff', tf.float64)
    # n = tf.multiply(n, 0.01)
    u0 = ssnp.read("mea3b.tiff", dtype=tf.float64)
    u0 = tf.multiply(u0, 2)
    in_list = []
    for num in range(8):
        xy_theta = num / 8 * 2 * np.pi
        c_ab = NA * np.cos(xy_theta), NA * np.sin(xy_theta)
        img_in = ssnp.read("plane", tf.complex128, shape=u0.shape[1:])
        img_in, _ = ssnp.tilt(img_in, c_ab, trunc=True)
        in_list.append(img_in)
    # in_list = tf.stack(in_list)
    # print(model_loss(n, in_list, u0))
    for it in range(10):
        print(model_loss(n, in_list, u0))
        # print(tv_loss(n))
        for i in range(8):
            with tf.GradientTape() as g:
                g.watch(n)
                loss = model_loss_single(n, in_list[i], u0[i])
            n = tf.maximum(n - g.gradient(loss, n) * 0.002 * np.exp(it / 5), 0)

    ssnp.write("rec.tiff", n)
    return n


if __name__ == '__main__':
    img = rec()
