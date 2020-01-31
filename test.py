@tf.function
def test(x):
    a = tf.constant(np.random.rand(1024, 1024), dtype=tf.complex128)
    for i in range(200):
        a = tf.signal.fft2d(a)
        a = tf.signal.ifft2d(a)
    return x + a


a = tf.constant(np.random.rand(1024, 1024), dtype=tf.complex128)
print(tf.add(1, 2))
t = time()
test(a)
print(time() - t)