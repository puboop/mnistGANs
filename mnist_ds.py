from tensorflow import keras
import tensorflow as tf
import os
import numpy as np

MNIST_PATH = "./mnist.npz"


def load_mnist(path):
    if os.path.isfile(path):
        with np.load(path, allow_pickle=True) as f:
            x_train, y_train = f['x_train'], f['y_train']
            x_test, y_test = f['x_test'], f['y_test']
        return (x_train, y_train), (x_test, y_test)
    return keras.datasets.mnist.load_data(MNIST_PATH)


def get_half_batch_ds(batch_size):
    return get_ds(batch_size // 2)


def get_ds(batch_size):
    (x, y), _ = load_mnist(MNIST_PATH)
    x = _process_x(x)
    y = tf.cast(y, tf.int32)
    ds = tf.data \
        .Dataset \
        .from_tensor_slices((x, y)) \
        .cache() \
        .shuffle(1024) \
        .batch(batch_size) \
        .prefetch(tf.data.experimental.AUTOTUNE)
    return ds


def get_test_x():
    (_, _), (x, _) = load_mnist(MNIST_PATH)
    x = _process_x(x)
    return x


def get_test_69():
    _, (x, y) = load_mnist(MNIST_PATH)
    return _process_x(x[y == 6]), _process_x(x[y == 9])


def get_train_x():
    (x, _), _ = load_mnist(MNIST_PATH)
    x = _process_x(x)
    return x


def _process_x(x):
    """
    这段代码在TensorFlow中用于对输入张量x进行归一化处理，使其值的范围从[0, 255]缩放到[-1, 1]。下面是逐步解释：
    tf.cast(x, tf.float32):
    将输入张量x转换为浮点型 tf.float32。这一步是为了确保后续的计算不会因为整数除法而导致精度丢失。
    tf.expand_dims(..., axis=3):
    在指定的轴（这里是轴3）上增加一个维度。假设x是一个形状为 [batch_size, height, width] 的张量，tf.expand_dims会将其转换为 [batch_size, height, width, 1]。
    这个操作通常用于为图像数据增加通道维度，特别是在处理单通道（灰度）图像时。
    / 255.:
    将图像像素值从[0, 255]范围缩放到[0, 1]。这是因为在大多数神经网络中，输入数据的范围通常需要归一化到[0, 1]或[-1, 1]。
    * 2 - 1:
    将归一化后的像素值范围从[0, 1]缩放到[-1, 1]。具体地：
    乘以2将范围扩展到[0, 2]。
    减去1将范围调整为[-1, 1]。
    """
    return tf.expand_dims(tf.cast(x, tf.float32), axis=3) / 255. * 2 - 1


def get_69_ds():
    (x, y), _ = load_mnist(MNIST_PATH)
    x6, x9 = x[y == 6], x[y == 9]
    return _process_x(x6), _process_x(x9)


def downsampling(imgs, to_shape):
    s = to_shape[:2]
    imgs = tf.random.normal(imgs.shape, 0, 0.2) + imgs
    return tf.image.resize(imgs, size=s)
